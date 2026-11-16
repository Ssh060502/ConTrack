"""Single-environment episode collector; imported after Isaac Sim starts.

State arrays have T+1 entries; observations and transition arrays have T entries.
Terminal observations are deliberately not synthesized from post-reset observations.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch


def numpy(value):
    return value.detach().cpu().numpy().copy()


def diagnostic_pose_term(original):
    """Wrap the configured termination term at its actual evaluation point."""
    # Isaac Lab validates the signature literally; a **kwargs parameter is
    # interpreted as an unconfigured required term parameter in this version.
    # The Sharpa obj_pose_exceeded term takes only env.
    def evaluate(env):
        result = original(env)
        idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
        objects = env.scene["objects"].data
        position = objects.object_pos_w - env.scene.env_origins.unsqueeze(1)
        reference = env.ref_object_pos[idx]
        pos_error = torch.norm(position - reference, dim=-1)
        dot = torch.clamp(torch.abs(torch.sum(
            objects.object_quat_w * env.ref_object_quat[idx], dim=-1)), 0., 1.)
        rot_error = 2. * torch.arccos(dot)
        env._collection_pose_diagnostic = {
            "reference_frame": numpy(idx[0]),
            "object_position": numpy(position[0]),
            "reference_position": numpy(reference[0]),
            "position_error": numpy(pos_error[0]),
            "rotation_error_rad": numpy(rot_error[0]),
            "position_threshold": numpy(env.object_pos_threshold[0]),
            "rotation_threshold_rad": numpy(env.object_rot_threshold[0]),
            "position_exceeded": numpy(pos_error[0] > env.object_pos_threshold[0]),
            "rotation_exceeded": numpy(rot_error[0] > env.object_rot_threshold[0]),
            "pose_break": numpy(result[0]),
        }
        return result
    return evaluate


def policy_observation(obs, policy_module):
    """Use the policy's group selection/order, before its observation normalizer.

    RSL-RL returns a TensorDict, not a flat tensor. Indexing obs[0] selects
    a TensorDict batch entry and its numpy() yields an object array.
    """
    if hasattr(policy_module, "get_actor_obs"):
        value = policy_module.get_actor_obs(obs)
    elif hasattr(policy_module, "get_student_obs"):
        value = policy_module.get_student_obs(obs)
    elif hasattr(obs, "keys"):
        raise TypeError("Observation mapping requires a policy observation accessor")
    else:
        value = obs
    array = numpy(value)
    if array.dtype.kind not in "fiu" or array.ndim != 2 or array.shape[0] != 1:
        raise TypeError(f"Expected numeric actor observation (1, D), got {array.shape}, {array.dtype}")
    return array[0].copy()


def snapshot(env):
    result = {}
    for name, ids, h in env.ref_hands:
        robot = env.scene[name]
        result[f"hands/{h}/qpos"] = numpy(robot.data.joint_pos[0, ids])
        result[f"hands/{h}/qvel"] = numpy(robot.data.joint_vel[0, ids])
    objects = env.scene["objects"].data
    result["objects/position"] = numpy(objects.object_pos_w[0] - env.scene.env_origins[0])
    result["objects/quaternion_wxyz"] = numpy(objects.object_quat_w[0])
    result["objects/linear_velocity"] = numpy(objects.object_lin_vel_w[0])
    result["objects/angular_velocity"] = numpy(objects.object_ang_vel_w[0])
    result["reference_frame"] = numpy(env.frame_idx[0])
    idx = int(env.frame_idx[0])
    result["reference/object_position"] = numpy(env.ref_object_pos[idx])
    result["reference/object_quaternion_wxyz"] = numpy(env.ref_object_quat[idx])
    result["reference/joint_position"] = numpy(env.ref_qpos[idx])
    for side, keys in env.hand_contact_sensor_keys.items():
        for i, key in enumerate(keys):
            data = env.scene[key].data
            # Forces are simulated link contacts, not a tactile taxel model.
            result[f"contacts/{side}/{i}/net_force_w"] = numpy(data.net_forces_w[0])
            if data.force_matrix_w is not None:
                result[f"contacts/{side}/{i}/object_force_w"] = numpy(data.force_matrix_w[0])
            if data.contact_pos_w is not None:
                result[f"contacts/{side}/{i}/position_env"] = numpy(
                    data.contact_pos_w[0] - env.scene.env_origins[0])
    return result


def targets(env):
    return {f"hands/{h}/joint_position_target": numpy(env.scene[name].data.joint_pos_target[0, ids])
            for name, ids, h in env.ref_hands}


def write_episode(group, states, observations, transitions, completed):
    """Validate alignment before persisting an episode, including its final state."""
    count = len(transitions)
    if len(states) != count + 1 or len(observations) != count:
        raise ValueError("Episode state/action alignment is invalid")
    observation_array = np.stack(observations) if count else None
    if count and (observation_array.dtype.kind not in "fiu" or observation_array.ndim != 2):
        raise TypeError("Policy observations must be a numeric (T, D) array")
    group.attrs["completed"] = completed
    group.attrs["transition_count"] = count
    for key in states[0]:
        group.create_dataset(f"states/{key}", data=np.stack([s[key] for s in states]), compression="gzip")
    positions = np.stack([s["objects/position"] for s in states])
    references = np.stack([s["reference/object_position"] for s in states])
    group.create_dataset("metrics/object_height_change", data=positions[..., 2] - positions[0, ..., 2])
    group.create_dataset("metrics/object_position_error", data=np.linalg.norm(positions - references, axis=-1))
    q = np.stack([s["objects/quaternion_wxyz"] for s in states])
    qr = np.stack([s["reference/object_quaternion_wxyz"] for s in states])
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    qr = qr / np.maximum(np.linalg.norm(qr, axis=-1, keepdims=True), 1e-12)
    group.create_dataset("metrics/object_rotation_error_rad", data=2 * np.arccos(
        np.clip(np.abs(np.sum(q * qr, axis=-1)), 0, 1)))
    if count:
        dataset = group.create_dataset("policy_observation", data=observation_array, compression="gzip")
        dataset.attrs["representation"] = "policy-selected concatenated groups, before observation normalization"
        for key in transitions[0]:
            group.create_dataset(f"transitions/{key}", data=np.stack([t[key] for t in transitions]), compression="gzip")


def collect(env, policy, policy_module, app, output, episodes, max_steps, physics, metadata,
            video_folder=None):
    base = env.unwrapped
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    original_reset = base._reset_idx
    terminal = {}
    collecting = False
    video_writer = None

    def render_frame():
        frame = base.render()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise RuntimeError("Collection video requires an RGB array render")
        return frame.copy()

    def start_video(number):
        if video_folder is None:
            return None
        import imageio.v2 as imageio
        directory = Path(video_folder)
        directory.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(directory / f"episode_{number:06d}.mp4"),
                                    fps=round(1. / float(base.step_dt)), macro_block_size=1)
        writer.append_data(render_frame())
        return writer

    def capture_reset(ids):
        # ManagerBasedRLEnv calls this after physics/rewards/terminations and
        # before writing reset poses. Never use step()'s reset observation as s_T.
        if collecting and len(ids):
            terminal["state"] = snapshot(base)
            terminal["targets"] = targets(base)
            terminal["applied_action"] = numpy(base.action_manager.action[0])
            terminal["reasons"] = {
                name: numpy(base.termination_manager.get_term(name)[0])
                for name in base.termination_manager.active_terms}
            if video_writer is not None:
                # Render before reset writes new poses: the final video frame
                # belongs to this episode, not the following attempt.
                terminal["video_frame"] = render_frame()
        return original_reset(ids)

    base._reset_idx = capture_reset
    completed = 0
    steps = 0
    states, observations, transitions = [], [], []
    try:
        # Exclusive create: an existing dataset is never silently overwritten.
        with h5py.File(path, "x") as hf:
            hf.attrs["schema_version"] = "contrack_episode_v1"
            hf.attrs["metadata_json"] = json.dumps(metadata)
            hf.attrs["physics_json"] = json.dumps(physics)
            hf.attrs["control_dt"] = float(base.step_dt)
            hf.attrs["quaternion_order"] = "wxyz"
            hf.attrs["position_frame"] = "environment origin; velocities and forces use world axes"
            hf.attrs["start_reference_frame"] = 0
            hf.attrs["threshold_curriculum_enabled"] = False
            hf.attrs["object_pos_threshold"] = float(base.cfg.object_pos_threshold_max)
            hf.attrs["object_rot_threshold"] = float(base.cfg.object_rot_threshold_max)
            hf.attrs["hand_mapping_json"] = json.dumps([
                {"hand": int(h), "asset": name,
                 "joint_names": [base.scene[name].joint_names[int(j)] for j in ids]}
                for name, ids, h in base.ref_hands])
            hf.attrs["contact_mapping_json"] = json.dumps(base.hand_contact_sensor_keys)
            hf.attrs["policy_obs_groups_json"] = json.dumps(getattr(policy_module, "obs_groups", {}))
            hf.attrs["completed_episodes"] = 0
            hf.attrs["collection_complete"] = False
            hf.attrs["pose_diagnostics_version"] = 1
            hf.attrs["video_folder"] = "" if video_folder is None else str(video_folder)
            env.reset()
            policy_module.reset(torch.ones(1, device=base.device, dtype=torch.bool))
            obs = env.get_observations()
            states = [snapshot(base)]
            collecting = True
            video_writer = start_video(completed)
            while app.is_running() and completed < episodes and steps < max_steps:
                terminal.clear()
                base._collection_pose_diagnostic = None
                with torch.inference_mode():
                    observations.append(policy_observation(obs, policy_module))
                    action = policy(obs)
                    action_saved = numpy(action[0])
                    obs, reward, done, _ = env.step(action)
                ended = bool(done[0])
                diagnostic = base._collection_pose_diagnostic
                if diagnostic is None:
                    raise RuntimeError("Termination diagnostic hook did not run")
                if ended and "state" not in terminal:
                    raise RuntimeError("Terminal reset hook did not fire; refusing to save misaligned data")
                transition = {"policy_action": action_saved,
                              "applied_action": terminal["applied_action"] if ended else numpy(base.action_manager.action[0]),
                              "reward": numpy(reward[0]), "done": np.asarray(ended)}
                # Store actual processed joint targets before reset clears action buffers.
                transition.update(terminal["targets"] if ended else targets(base))
                transition.update({f"diagnostics/{key}": value for key, value in diagnostic.items()})
                for name in base.termination_manager.active_terms:
                    transition[f"termination/{name}"] = (
                        terminal["reasons"][name] if ended else
                        numpy(base.termination_manager.get_term(name)[0]))
                transitions.append(transition)
                states.append(terminal["state"] if ended else snapshot(base))
                if video_writer is not None:
                    video_writer.append_data(terminal["video_frame"] if ended else render_frame())
                policy_module.reset(done)
                steps += 1
                if ended:
                    group = hf.create_group(f"episodes/{completed:06d}")
                    write_episode(group, states, observations, transitions, True)
                    reasons = [k for k, v in terminal["reasons"].items() if bool(v)]
                    print(f"[collect] episode={completed} steps={len(transitions)} reasons={reasons}", flush=True)
                    print(f"[diagnostic] frame={int(diagnostic['reference_frame'])} "
                          f"pos_error_m={diagnostic['position_error'].tolist()} "
                          f"pos_threshold_m={float(diagnostic['position_threshold'])} "
                          f"rot_error_deg={np.degrees(diagnostic['rotation_error_rad']).tolist()} "
                          f"rot_threshold_deg={float(np.degrees(diagnostic['rotation_threshold_rad']))}", flush=True)
                    if video_writer is not None:
                        video_writer.close()
                        video_writer = None
                    completed += 1
                    hf.attrs["completed_episodes"] = completed
                    hf.flush()
                    states, observations, transitions = [snapshot(base)], [], []
                    if completed < episodes and steps < max_steps:
                        video_writer = start_video(completed)
            if transitions:
                write_episode(hf.create_group("partial_episode"), states, observations, transitions, False)
            hf.attrs["completed_episodes"] = completed
            hf.attrs["total_steps"] = steps
            hf.attrs["requested_episodes"] = episodes
            hf.attrs["collection_complete"] = completed == episodes
    finally:
        base._reset_idx = original_reset
        if video_writer is not None:
            video_writer.close()
    print(f"[collect] saved {completed}/{episodes} episodes to {path}", flush=True)
    if completed != episodes:
        raise RuntimeError("Collection stopped before requested episode count; partial data retained")
