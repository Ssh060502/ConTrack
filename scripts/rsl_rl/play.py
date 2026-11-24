from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if p not in ("", str(repo_root))]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "source" / "ConTrack"))

from isaaclab.app import AppLauncher

from scripts.rsl_rl import cli_args

parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_folder", type=str, default=None, help="Optional video output directory.")
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--data", type=str, default=None)
parser.add_argument("--session", type=str, default=None)
parser.add_argument(
    "--export_rollout",
    type=str,
    default=None,
    help=(
        "Path to save env 0's rolled-out joint trajectory as an h5 file with the same "
        "qpos/urdf_name/is_rhand/fps schema as a reference dataset. Run with --num_envs 1 "
        "for a clean single-episode recording; recording stops at env 0's first episode end. "
        "Implies --deterministic_start."
    ),
)
parser.add_argument(
    "--deterministic_start",
    action="store_true",
    default=False,
    help=(
        "Force env 0's first episode to start at reference frame 0 instead of the "
        "training curriculum's random start frame (see force_start_frame0 in "
        "mdp/state.py). Useful for --video too, not just --export_rollout, so a "
        "recording shows one full attempt from the true beginning instead of "
        "jumping mid-clip when the curriculum resets."
    ),
)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--collect_trajectories", type=str, default=None, help="Episode H5 output (single Sharpa env).")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--collection_max_steps", type=int, default=100000)
parser.add_argument("--trace_reset", action="store_true", help="Collection: trace full reset state and first-control physics substeps.")
parser.add_argument("--diagnostic_no_self_collisions", action="store_true",
                    help="Collection-only diagnostic: disable robot self collisions, not hand-object collisions.")
parser.add_argument("--pose_break_mode", choices=("strict", "log-only"), default="strict",
                    help="Collection only: log-only records pose deviations without terminating on them.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

session_dir = Path(args_cli.session).expanduser() if args_cli.session else None
session_reference_path = ""
if session_dir is not None:
    for line in (session_dir / "params" / "env.yaml").read_text().splitlines():
        s = line.strip()
        if s.startswith("reference_path:"):
            session_reference_path = s.split(":", 1)[1].strip()
            break
    ref = Path(session_reference_path)
    if not ref.is_file():
        ref = sorted((repo_root / "data").rglob(ref.name))[0]
    session_reference_path = str(ref.resolve())

if "Crx-Sharpa" in args_cli.task:
    _data_variable = "CONTRACK_CRX_SHARPA_DATA"
elif "Rby1a-Sharpa" in args_cli.task:
    _data_variable = "CONTRACK_RBY1A_SHARPA_DATA"
else:
    _data_variable = "CONTRACK_XARM_XHAND_DATA"
if args_cli.data is not None:
    os.environ[_data_variable] = args_cli.data
elif session_reference_path:
    os.environ[_data_variable] = session_reference_path

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch
import yaml

from rsl_rl_contrack.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks
import ConTrack.tasks


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    cfg = (
        yaml.safe_load((session_dir / "params" / "agent.yaml").read_text())
        if session_dir is not None
        else cli_args.update_rsl_rl_cfg(agent_cfg, args_cli).to_dict()
    )
    if args_cli.seed is not None:
        cfg["seed"] = args_cli.seed
    cfg["device"] = args_cli.device if args_cli.device is not None else cfg["device"]
    resume_path = args_cli.checkpoint
    log_dir = str(session_dir) if session_dir is not None else str(Path(resume_path).parent)

    env_cfg.seed = cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.log_dir = log_dir
    if args_cli.diagnostic_no_self_collisions and not args_cli.collect_trajectories:
        raise ValueError("Self-collision diagnostic requires --collect_trajectories")

    if args_cli.collect_trajectories:
        if "Rby1a-Sharpa" not in args_cli.task or env_cfg.scene.num_envs != 1:
            raise ValueError("Trajectory collection requires Rby1a-Sharpa and --num_envs 1")
        if args_cli.export_rollout:
            raise ValueError("Use collection without --export_rollout")
        if args_cli.episodes <= 0 or args_cli.collection_max_steps <= 0:
            raise ValueError("Episode and step limits must be positive")
        from ConTrack.tasks.manager_based.mimic_rby1a_sharpa import mdp as sharpa_mdp
        # Avoid the training reset library and threshold curriculum changing the
        # initial-state distribution or failure criterion between episodes.
        env_cfg.events.reset_to_demo.func = sharpa_mdp.reset_reference_episode_first_frame
        env_cfg.threshold_curriculum_enabled = False
        from scripts.tools.trajectory_collection import diagnostic_pose_term
        env_cfg.terminations.pose_break.func = diagnostic_pose_term(
            env_cfg.terminations.pose_break.func, args_cli.pose_break_mode)
        if args_cli.diagnostic_no_self_collisions:
            env_cfg.scene.rby1a_sharpa.spawn.articulation_props.enabled_self_collisions = False
            print("[DIAGNOSTIC ONLY] Robot self collisions disabled; do not pool this run with normal training data.")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Report resolved PhysX values, not just the requested density/material settings.
    # The collection view flattens environments and objects; ranges cover all of them.
    physics_report = {}
    if "Rby1a-Sharpa" in args_cli.task:
        objects = env.unwrapped.scene["objects"]
        view = objects.root_physx_view
        for key, values in (
            ("mass_kg", view.get_masses()),
            ("inertia_kg_m2", view.get_inertias()),
            ("material_static_dynamic_restitution", view.get_material_properties()),
        ):
            values = values.detach().cpu().numpy()
            physics_report[key] = values.tolist()
            print(f"[ConTrack Sharpa physics] {key}: shape={values.shape}, "
                  f"min={values.min():.6g}, max={values.max():.6g}")
        physics_report["requested_mass_scale"] = float(
            os.environ.get("CONTRACK_SHARPA_OBJECT_MASS_SCALE", "1.0")
        )
        friction = os.environ.get("CONTRACK_SHARPA_OBJECT_FRICTION")
        physics_report["requested_object_friction"] = None if friction is None else float(friction)

    if args_cli.export_rollout or args_cli.deterministic_start:
        # The env's curriculum resets to a random frame of the reference clip by
        # default (see reset_reference_episode in mdp/state.py), which is what
        # training wants but makes a recorded rollout/video start from an
        # arbitrary mid-clip point instead of the beginning. Force frame 0 here
        # and re-reset so playback starts from the true beginning of the motion.
        env.unwrapped.force_start_frame0 = True
        env.reset()

    if args_cli.video and not args_cli.collect_trajectories:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_folder or os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            name_prefix=Path(resume_path).stem,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner_class = cfg["class_name"]
    runner = (
        OnPolicyRunner(env, cfg, log_dir=None, device=cfg["device"])
        if runner_class == "OnPolicyRunner"
        else DistillationRunner(env, cfg, log_dir=None, device=cfg["device"])
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    if args_cli.collect_trajectories:
        from scripts.tools.trajectory_collection import collect
        try:
            collect(env, policy, runner.alg.policy, simulation_app,
                    args_cli.collect_trajectories, args_cli.episodes,
                    args_cli.collection_max_steps, physics_report,
                    {"checkpoint": str(Path(resume_path).resolve()),
                     "reference_path": str(env_cfg.reference_path),
                     "seed": int(cfg["seed"]), "task": args_cli.task,
                     "pose_break_mode": args_cli.pose_break_mode,
                     "robot_self_collisions": bool(env_cfg.scene.rby1a_sharpa.spawn.articulation_props.enabled_self_collisions),
                     "diagnostic_no_self_collisions": args_cli.diagnostic_no_self_collisions},
                    video_folder=(args_cli.video_folder or
                                  str(Path(args_cli.collect_trajectories).parent / "videos"))
                    if args_cli.video else None, trace_reset=args_cli.trace_reset)
        finally:
            env.close()
        return
    obs = env.get_observations()
    timestep = 0
    episode_logs: dict[str, list[float]] = {}

    rollout_frames: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_object_pos: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_object_quat_xyzw: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_done = False
    rollout_base_pos = []
    rollout_base_quat_xyzw = []
    export_rby_base = all(Path(name).name == "rby1a_sharpa.urdf" for name in env.unwrapped.urdf_name)
    try:
        env.unwrapped.scene["objects"]
        has_objects = True
    except KeyError:
        has_objects = False
    if rollout_frames is not None and not has_objects:
        print("[warn] --export_rollout: no 'objects' asset in this scene; the saved h5 "
              "will have no object_tracks group.")
    if rollout_frames is not None and env.unwrapped.num_envs != 1:
        print(
            f"[warn] --export_rollout records only env 0, but num_envs={env.unwrapped.num_envs}. "
            "Pass --num_envs 1 for a clean single-episode recording."
        )

    while simulation_app.is_running() and timestep < args_cli.video_length:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            runner.alg.policy.reset(dones)
        if "log" in extras:
            for k, v in extras["log"].items():
                episode_logs.setdefault(k, []).append(float(v))
        # Isaac Lab may already have reset this environment inside step().
        # Exclude the terminal step from every exported channel together.
        if rollout_frames is not None and bool(dones[0]):
            rollout_done = True
        if rollout_frames is not None and not rollout_done:
            base_env = env.unwrapped
            if export_rby_base:
                robot = base_env.scene[base_env.ref_hands[0][0]]
                body_id = robot.body_names.index("base")
                rollout_base_pos.append(
                    (robot.data.body_pos_w[0, body_id] - base_env.scene.env_origins[0]).cpu().numpy().copy()
                )
                base_quat = robot.data.body_quat_w[0, body_id].cpu().numpy()
                rollout_base_quat_xyzw.append(base_quat[[1, 2, 3, 0]].copy())
            frame = np.zeros(
                (len(base_env.ref_hands), base_env.unused_default_qpos.shape[0]), dtype=np.float32
            )
            for name, joint_ids, h_idx in base_env.ref_hands:
                frame[h_idx] = base_env.scene[name].data.joint_pos[0, joint_ids].cpu().numpy()
            rollout_frames.append(frame)
            if has_objects:
                objects = base_env.scene["objects"]
                obj_pos = (
                    objects.data.object_pos_w[0] - base_env.scene.env_origins[0]
                ).cpu().numpy()  # (O, 3), relative to this env's own origin
                obj_quat_wxyz = objects.data.object_quat_w[0].cpu().numpy()  # (O, 4)
                rollout_object_pos.append(obj_pos)
                rollout_object_quat_xyzw.append(obj_quat_wxyz[:, [1, 2, 3, 0]])
            if bool(dones[0]):
                rollout_done = True
        timestep += 1

    if rollout_frames is not None:
        if not rollout_frames:
            print("[warn] --export_rollout given but no frames were recorded")
        else:
            base_env = env.unwrapped
            qpos = np.stack(rollout_frames, axis=0)
            urdf_name = np.array([s.encode("utf-8") for s in base_env.urdf_name])
            is_rhand = base_env.is_rhand.cpu().numpy().astype(np.uint8)
            fps = np.full((qpos.shape[1],), float(base_env.ref_fps), dtype=np.float32)
            out_path = Path(args_cli.export_rollout).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as hf:
                if physics_report:
                    import json
                    hf.attrs["object_physics_json"] = json.dumps(physics_report)
                hf.create_dataset("qpos", data=qpos)
                hf.create_dataset("urdf_name", data=urdf_name)
                hf.create_dataset("is_rhand", data=is_rhand)
                hf.create_dataset("fps", data=fps)
                if rollout_base_pos:
                    hf.create_dataset("robot_base/translations", data=np.stack(rollout_base_pos))
                    hf.create_dataset("robot_base/orientations_xyzw", data=np.stack(rollout_base_quat_xyzw))
                    hf.attrs["robot_base_frame"] = "base"
                hf.attrs["pose_coordinates"] = "environment origin; orientations_xyzw"
                hf.attrs["terminal_step_excluded"] = True
                if has_objects and rollout_object_pos:
                    object_pos = np.stack(rollout_object_pos, axis=0)  # (T, O, 3)
                    object_quat_xyzw = np.stack(rollout_object_quat_xyzw, axis=0)  # (T, O, 4)
                    for obj_idx in range(object_pos.shape[1]):
                        grp = hf.create_group(f"object_tracks/object_{obj_idx}")
                        grp.create_dataset("translations", data=object_pos[:, obj_idx])
                        grp.create_dataset("orientations_xyzw", data=object_quat_xyzw[:, obj_idx])
            print(
                f"[info] saved rollout qpos to {out_path} "
                f"(T={qpos.shape[0]}, H={qpos.shape[1]}, stopped_early={rollout_done}, "
                f"objects_saved={has_objects})"
            )

    env.close()

    if episode_logs:
        print("\n=== Episode statistics over this rollout ===")
        for k in sorted(episode_logs.keys()):
            vals = episode_logs[k]
            print(f"{k:40s} mean={sum(vals) / len(vals):.4f}  n={len(vals)}")
    else:
        print("\n[warn] no episodes finished during this rollout; increase --video_length or --num_envs")


if __name__ == "__main__":
    main()
    simulation_app.close()
