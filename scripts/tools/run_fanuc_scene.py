"""Standalone Fanuc/Sharpa scene verification; no PPO, ROS or existing task edits."""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import uuid

from fanuc_scene_data import load_bundle, sample_motion


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--mode', choices=['reference', 'dynamic'], default='reference')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--duration', type=float, help='Short smoke run in simulation seconds')
    parser.add_argument('--physics-hz', type=int, default=120)
    parser.add_argument('--control-hz', type=int, default=50)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--collision-visuals', action='store_true',
                        help='Display DAE arm links using their collision meshes; physics unchanged')
    parser.add_argument('--disable-fabric', action='store_true',
                        help='Diagnostic USD synchronization path instead of Fabric')
    parser.add_argument('--video-fps', type=int, default=30)
    parser.add_argument('--self-collision', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--collider', choices=['convex_hull', 'convex_decomposition'],
                        default='convex_decomposition')
    parser.add_argument('--object-mass', type=float, default=.1,
                        help='Nominal kg; provisional assumption, not measured hammer mass')
    parser.add_argument('--mass-scale', type=float, default=1.)
    parser.add_argument('--friction', type=float, default=2.,
                        help='Static/dynamic coefficient on robots, hammer and table')
    parser.add_argument('--arm-stiffness', type=float, default=1000.)
    parser.add_argument('--arm-damping', type=float, default=50.)
    parser.add_argument('--hand-stiffness', type=float, default=8.)
    parser.add_argument('--hand-damping', type=float, default=.3)
    parser.add_argument('--check-only', action='store_true', help='Validate bundle without Isaac Lab')
    # Check-only must work on the local data environment, without Isaac imports.
    import sys
    if '--check-only' not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for name in ('physics_hz', 'control_hz', 'video_fps', 'object_mass', 'mass_scale'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be positive and finite')
    for name in ('friction', 'arm_stiffness', 'arm_damping', 'hand_stiffness', 'hand_damping'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be nonnegative and finite')
    if args.control_hz > args.physics_hz or args.video_fps > args.physics_hz:
        parser.error('Control/video rates cannot exceed physics rate')
    if args.video and args.physics_hz % args.video_fps:
        parser.error('Video FPS must divide physics rate to preserve video duration')
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error('duration must be positive and finite')
    return args


def run(args, app, data, output):
    import h5py
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from fanuc_scene_runtime import build_scene, scene_visual_report

    manifest, q, poses, fps, vertices, faces = data
    duration = (len(q)-1)/fps
    if args.duration is not None:
        duration = min(duration, args.duration)
    dt = 1./args.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, device=args.device, use_fabric=not args.disable_fabric))
    eye, look = [2.1, -2., 1.6], [.45, -.45, .05]
    sim.set_camera_view(eye, look)
    robots, hammer = build_scene(sim, args.bundle.resolve(), manifest, q, poses,
                                 vertices, faces, args, output)
    camera = writer = None
    if args.video:
        print('[fanuc] Creating review camera', flush=True)
        import imageio.v2 as imageio
        from isaaclab.sensors import Camera, CameraCfg
        camera = Camera(CameraCfg(
            prim_path='/World/ReviewCamera', update_period=0., width=960, height=720,
            data_types=['rgb'], spawn=sim_utils.PinholeCameraCfg(
                focal_length=24., horizontal_aperture=24., clipping_range=(.01, 100.))))
        writer = imageio.get_writer(str(output / 'video.mp4'), fps=args.video_fps)
    print('[fanuc] Exporting stage snapshot', flush=True)
    sim.stage.GetRootLayer().Export(str(output / 'scene.usda'))
    print('[fanuc] Starting sim.reset()', flush=True)
    sim.reset()
    print('[fanuc] sim.reset() complete', flush=True)
    for side, robot in robots.items():
        actual = robot.data.root_state_w[0, :7]
        expected = torch.as_tensor(manifest['bases_xyz_wxyz'][side],
                                   dtype=actual.dtype, device=sim.device)
        if (not torch.allclose(actual[:3], expected[:3], atol=1e-4, rtol=0)
                or abs(float(torch.dot(actual[3:], expected[3:]))) < 1.-1e-5):
            raise RuntimeError(f'{side}: imported root pose differs from recorded NPZ base')
    if camera is not None:
        camera.set_world_poses_from_view(torch.tensor([eye], device=sim.device),
                                         torch.tensor([look], device=sim.device))
    ids = {}
    for side, robot in robots.items():
        ids[side], found = robot.find_joints(manifest['joint_names'][side], preserve_order=True)
        if found != manifest['joint_names'][side] or len(robot.joint_names) != 28:
            raise RuntimeError(f'{side}: imported joint names/order differ from bundle')
        print(f'[fanuc] {side}: 28 named joints verified', flush=True)
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float32, device=sim.device).unsqueeze(0)

    def write_reference(time, direct):
        target, pose = sample_motion(q, poses, fps, time)
        for index, (side, robot) in enumerate(robots.items()):
            values = tensor(target[index*28:(index+1)*28])
            bounds = robot.data.joint_pos_limits[:, ids[side], :]
            bounded = torch.maximum(torch.minimum(values, bounds[..., 1]), bounds[..., 0])
            if float(torch.max(torch.abs(values-bounded))) > 2e-5:
                raise RuntimeError(f'{side}: imported joint bounds disagree with trajectory by >2e-5 rad')
            values = bounded
            if direct:
                robot.write_joint_state_to_sim(values, torch.zeros_like(values), joint_ids=ids[side])
            robot.set_joint_position_target(values, joint_ids=ids[side])
            robot.write_data_to_sim()
        if direct:
            hammer.write_root_pose_to_sim(tensor(pose))
            hammer.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        return target

    write_reference(0., True)
    for robot in robots.values():
        robot.reset()
    hammer.reset()
    # Reference playback pins the written state through one real physics step
    # (no gravity, no collisions, zero velocity, target == state). Writing state
    # and calling only sim.forward() left the renderer at the USD zero pose.
    reference = args.mode == 'reference'
    if reference:
        sim.step(render=False)
    else:
        sim.forward()
    for robot in robots.values():
        robot.update(dt)
    hammer.update(dt)
    for index, (side, robot) in enumerate(robots.items()):
        error = float(np.max(np.abs(
            robot.data.joint_pos[0, ids[side]].detach().cpu().numpy() - q[0, index*28:(index+1)*28])))
        print(f'[fanuc] {side}: initial joint readback error {error:.2e} rad', flush=True)
        if error > 1e-3:
            raise RuntimeError(f'{side}: written initial joints were not applied ({error:.3f} rad)')
    visual_report = scene_visual_report(sim.stage)
    visual_report['camera_eye'] = eye
    visual_report['camera_target'] = look
    (output / 'visual_report.json').write_text(json.dumps(visual_report, indent=2), encoding='utf-8')
    # Flush renderer only; never advance dynamics while warming a camera.
    if camera is not None:
        for _ in range(4):
            sim.render()
    rows = {'time': [], 'joint_position': [], 'joint_velocity': [], 'object_state': [],
            'reference_object_pose': [], 'reference_joint_position': []}
    targets = []
    finished = False

    def record(time):
        reference_q, reference_pose = sample_motion(q, poses, fps, time)
        rows['time'].append(time)
        rows['joint_position'].append(np.concatenate([
            robot.data.joint_pos[0, ids[side]].detach().cpu().numpy().copy()
            for side, robot in robots.items()]))
        rows['joint_velocity'].append(np.concatenate([
            robot.data.joint_vel[0, ids[side]].detach().cpu().numpy().copy()
            for side, robot in robots.items()]))
        rows['object_state'].append(hammer.data.root_state_w[0].detach().cpu().numpy().copy())
        rows['reference_object_pose'].append(reference_pose)
        rows['reference_joint_position'].append(reference_q)

    try:
        record(0.)
        last_tick = -1
        target = q[0]
        steps = math.ceil(duration/dt)
        for step in range(steps):
            if not app.is_running():
                raise RuntimeError('Simulation interrupted; partial data will be marked incomplete')
            time_before, time_after = step*dt, (step+1)*dt
            if args.mode == 'reference':
                target = write_reference(min(time_after, duration), True)
                sim.step(render=False)  # Pinned state only; no grasp claim.
            else:
                tick = math.floor(time_before*args.control_hz + 1e-8)
                if tick != last_tick:
                    target = write_reference(tick/args.control_hz, False)
                    last_tick = tick
                sim.step(render=False)
            for robot in robots.values():
                robot.update(dt)
            hammer.update(dt)
            targets.append(target.copy())
            record(time_after)
            if writer is not None and (step+1) % (args.physics_hz//args.video_fps) == 0:
                sim.render()
                camera.update(dt=1./args.video_fps, force_recompute=True)
                rgb = camera.data.output['rgb'][0, :, :, :3].detach().cpu().numpy()
                writer.append_data(rgb)
                if not (output / 'first_frame.png').exists():
                    imageio.imwrite(str(output / 'first_frame.png'), rgb)
            elif not args.headless:
                sim.render()
            if (step+1) % args.physics_hz == 0:
                print(f'[fanuc] {time_after:.1f}/{duration:.1f} simulated seconds', flush=True)
            if not np.isfinite(rows['joint_position'][-1]).all() or not np.isfinite(rows['object_state'][-1]).all():
                raise RuntimeError('Non-finite simulated state')
        finished = True
    finally:
        if writer is not None:
            writer.close()
        with h5py.File(output / 'scene_rollout.h5', 'w') as hf:
            hf.attrs['complete'] = finished
            hf.attrs['mode'] = args.mode
            hf.attrs['schema'] = 'fanuc_scene_review_v1'
            hf.attrs['quaternion_order'] = 'wxyz'
            hf.attrs['is_policy_collection'] = False
            hf.create_dataset('joint_names', data=np.asarray(
                ['left/'+n for n in manifest['joint_names']['left']] +
                ['right/'+n for n in manifest['joint_names']['right']], dtype='S80'))
            for key, values in rows.items():
                hf.create_dataset(key, data=np.asarray(values), compression='gzip')
            hf.create_dataset('commanded_joint_position', data=np.asarray(targets).reshape(-1, 56), compression='gzip')
        actual = np.asarray(rows['joint_position'])
        desired = np.asarray(rows['reference_joint_position'])
        object_state = np.asarray(rows['object_state'])
        ref_object = np.asarray(rows['reference_object_pose'])
        metrics = {'complete': finished, 'mode': args.mode,
                   'joint_tracking_rmse_rad': float(np.sqrt(np.mean((actual-desired)**2))),
                   'object_final_position_error_m': float(np.linalg.norm(object_state[-1, :3]-ref_object[-1, :3])),
                   'object_initial_z_m': float(object_state[0, 2]),
                   'object_max_origin_z_m': float(object_state[:, 2].max()),
                   'success_assessed': False}
        (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f'Finished. Outputs: {output}', flush=True)


def main():
    args = parse_args()
    data = load_bundle(args.bundle)
    print(f"[fanuc] Bundle verified: {data[0]['frames']} frames, {data[0]['duration_seconds']:.3f}s")
    if args.check_only:
        return
    output = args.output or Path('outputs/fanuc_scene') / (
        args.mode + '_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['bundle_manifest'] = data[0]
    config['controller_note'] = 'Provisional force-based PD; dynamic targets held at 50 Hz, not a ROS driver emulator.'
    config['mass_note'] = 'Nominal object mass is an unmeasured assumption, unrelated to NPZ geometry.'
    (output / 'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'[fanuc] Mode: {args.mode}; outputs: {output}', flush=True)
    from isaaclab.app import AppLauncher
    if args.video:
        args.enable_cameras = True
    import faulthandler
    faulthandler.enable()
    launcher = AppLauncher(args)
    failed = False
    try:
        run(args, launcher.app, data, output)
    except Exception as error:
        import traceback
        import sys
        failed = True
        detail = traceback.format_exc()
        (output / 'FAILED.txt').write_text(detail, encoding='utf-8')
        print(detail, file=sys.stderr, flush=True)
        print(f'[fanuc] FAILED. Details: {output / "FAILED.txt"}', file=sys.stderr, flush=True)
        # This server's SimulationApp shutdown terminates the process with code
        # zero, even while unwinding an exception. On failure, exit explicitly
        # after persisting/flushing diagnostics; the OS releases GPU resources.
        import os
        os._exit(1)
    else:
        # All outputs are already written and closed. SimulationApp.close() can
        # hang on this server after a camera run, so leave without it.
        import os
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    finally:
        try:
            launcher.app.close()
        finally:
            # SimulationApp.close can raise SystemExit(0), hiding a pending
            # exception. Preserve a failed status after normal Python cleanup.
            if failed:
                raise SystemExit(1)


if __name__ == '__main__':
    main()
