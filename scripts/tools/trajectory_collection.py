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
        group.create_dataset("policy_observation", data=np.stack(observations), compression="gzip")
        for key in transitions[0]:
            group.create_dataset(f"transitions/{key}", data=np.stack([t[key] for t in transitions]), compression="gzip")


def collect(env, policy, policy_module, app, output, episodes, max_steps, physics, metadata):
    base = env.unwrapped
    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    original_reset = base._reset_idx
    terminal = {}
    collecting = False

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
            env.reset()
            policy_module.reset(torch.ones(1, device=base.device, dtype=torch.bool))
            obs = env.get_observations()
            states = [snapshot(base)]
            collecting = True
            while app.is_running() and completed < episodes and steps < max_steps:
                terminal.clear()
                observations.append(numpy(obs[0]))
                with torch.inference_mode():
                    action = policy(obs)
                    action_saved = numpy(action[0])
                    obs, reward, done, _ = env.step(action)
                ended = bool(done[0])
                if ended and "state" not in terminal:
                    raise RuntimeError("Terminal reset hook did not fire; refusing to save misaligned data")
                transition = {"policy_action": action_saved,
                              "applied_action": terminal["applied_action"] if ended else numpy(base.action_manager.action[0]),
                              "reward": numpy(reward[0]), "done": np.asarray(ended)}
                # Store actual processed joint targets before reset clears action buffers.
                transition.update(terminal["targets"] if ended else targets(base))
                for name in base.termination_manager.active_terms:
                    transition[f"termination/{name}"] = (
                        terminal["reasons"][name] if ended else
                        numpy(base.termination_manager.get_term(name)[0]))
                transitions.append(transition)
                states.append(terminal["state"] if ended else snapshot(base))
                policy_module.reset(done)
                steps += 1
                if ended:
                    group = hf.create_group(f"episodes/{completed:06d}")
                    write_episode(group, states, observations, transitions, True)
                    reasons = [k for k, v in terminal["reasons"].items() if bool(v)]
                    print(f"[collect] episode={completed} steps={len(transitions)} reasons={reasons}", flush=True)
                    completed += 1
                    hf.attrs["completed_episodes"] = completed
                    hf.flush()
                    states, observations, transitions = [snapshot(base)], [], []
            if transitions:
                write_episode(hf.create_group("partial_episode"), states, observations, transitions, False)
            hf.attrs["completed_episodes"] = completed
            hf.attrs["total_steps"] = steps
            hf.attrs["requested_episodes"] = episodes
            hf.attrs["collection_complete"] = completed == episodes
    finally:
        base._reset_idx = original_reset
    print(f"[collect] saved {completed}/{episodes} episodes to {path}", flush=True)
    if completed != episodes:
        raise RuntimeError("Collection stopped before requested episode count; partial data retained")
