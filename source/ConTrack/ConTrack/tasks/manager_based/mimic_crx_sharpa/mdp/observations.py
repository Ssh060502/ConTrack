from __future__ import annotations

import torch


def all_qpos(env) -> torch.Tensor:
    """Concatenate current CRX-5iA+Sharpa joint positions and object poses.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands`` and a ``objects`` rigid object collection.

    Returns
    -------
    torch.Tensor, shape=(num_envs, 28*num_hands + 7*num_objects), dtype=float32
        Flattened vector ``[hands_qpos, objects_pos(3), objects_quat_wxyz(4)]`` in environment frame.
    """
    hand_qpos = torch.cat(
        [
            env.scene[name].data.joint_pos[:, joint_ids]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    objects = env.scene["objects"]
    obj_pos = objects.data.object_pos_w - env.scene.env_origins.unsqueeze(1)
    obj_quat = objects.data.object_quat_w
    obj_pose = torch.cat((obj_pos, obj_quat), dim=-1).reshape(env.num_envs, -1)
    return torch.cat((hand_qpos, obj_pose), dim=-1)


def all_qvel(env) -> torch.Tensor:
    """Concatenate current CRX-5iA+Sharpa joint velocities and object root velocities.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands`` and a ``objects`` rigid object collection.

    Returns
    -------
    torch.Tensor, shape=(num_envs, 28*num_hands + 6*num_objects), dtype=float32
        Flattened vector ``[hands_qvel, objects_lin_vel(3), objects_ang_vel(3)]``.
    """
    hand_qvel = torch.cat(
        [
            env.scene[name].data.joint_vel[:, joint_ids]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    objects = env.scene["objects"]
    obj_vel = torch.cat(
        (objects.data.object_lin_vel_w, objects.data.object_ang_vel_w), dim=-1
    ).reshape(env.num_envs, -1)
    return torch.cat((hand_qvel, obj_vel), dim=-1)


def delayed_all_qpos(env, delay_range: tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Return ``all_qpos`` with a per-env randomized observation delay.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``device``, ``num_envs``, and (when stepping) ``reset_buf`` (num_envs,).
        The function stores latency state in:
        - ``policy_obs_delay_steps`` (num_envs,) long
        - ``policy_all_qpos_hist`` (num_envs, max_delay+1, D) float
        where ``D`` matches the output dim of :func:`all_qpos`.
    delay_range : tuple[int, int]
        Inclusive uniform integer range ``(min_delay, max_delay)`` in environment steps.

    Returns
    -------
    torch.Tensor, shape=(num_envs, 28*num_hands + 7*num_objects), dtype=float32
        Delayed ``all_qpos`` using ``policy_obs_delay_steps`` per environment (0 means no delay).
    """
    x = all_qpos(env)
    max_d = int(delay_range[1])
    if max_d == 0:
        return x
    if not hasattr(env, "policy_obs_delay_steps"):
        lo, hi = int(delay_range[0]), int(delay_range[1])
        env.policy_obs_delay_steps = torch.randint(
            lo, hi + 1, (env.num_envs,), device=env.device, dtype=torch.long
        )
    if not hasattr(env, "policy_all_qpos_hist"):
        env.policy_all_qpos_hist = x.unsqueeze(1).repeat(1, max_d + 1, 1)
    reset_ids = (
        env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if hasattr(env, "reset_buf")
        else None
    )
    if reset_ids is not None and int(reset_ids.numel()) > 0:
        lo, hi = int(delay_range[0]), int(delay_range[1])
        env.policy_obs_delay_steps[reset_ids] = torch.randint(
            lo, hi + 1, (int(reset_ids.shape[0]),), device=env.device
        )
        env.policy_all_qpos_hist[reset_ids] = (
            x[reset_ids].unsqueeze(1).repeat(1, max_d + 1, 1)
        )
    env.policy_all_qpos_hist = torch.cat(
        [x.unsqueeze(1), env.policy_all_qpos_hist[:, :-1]], dim=1
    )
    d = torch.clamp(env.policy_obs_delay_steps, 0, max_d)
    return env.policy_all_qpos_hist[torch.arange(env.num_envs, device=env.device), d]


def delayed_all_qvel(env, delay_range: tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Return ``all_qvel`` with the same per-env observation delay as :func:`delayed_all_qpos`.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``device``, ``num_envs``, and ``policy_obs_delay_steps`` (num_envs,) long.
        The function stores history in ``policy_all_qvel_hist`` (num_envs, max_delay+1, Dv) float where ``Dv``
        matches the output dim of :func:`all_qvel`.
    delay_range : tuple[int, int]
        Inclusive delay range in steps. Only ``max_delay`` is used to size the history buffer (sampling happens in
        :func:`delayed_all_qpos`).

    Returns
    -------
    torch.Tensor, shape=(num_envs, 28*num_hands + 6*num_objects), dtype=float32
        Delayed ``all_qvel``.
    """
    x = all_qvel(env)
    max_d = int(delay_range[1])
    if max_d == 0:
        return x
    if not hasattr(env, "policy_all_qvel_hist"):
        env.policy_all_qvel_hist = x.unsqueeze(1).repeat(1, max_d + 1, 1)
    reset_ids = (
        env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if hasattr(env, "reset_buf")
        else None
    )
    if reset_ids is not None and int(reset_ids.numel()) > 0:
        env.policy_all_qvel_hist[reset_ids] = (
            x[reset_ids].unsqueeze(1).repeat(1, max_d + 1, 1)
        )
    env.policy_all_qvel_hist = torch.cat(
        [x.unsqueeze(1), env.policy_all_qvel_hist[:, :-1]], dim=1
    )
    d = torch.clamp(env.policy_obs_delay_steps, 0, max_d)
    return env.policy_all_qvel_hist[torch.arange(env.num_envs, device=env.device), d]


def ref_all_qpos(env, num_future: int = 5) -> torch.Tensor:
    """Stack reference CRX-5iA+Sharpa qpos and object poses for current and future frames.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``frame_idx`` (num_envs,), ``ref_len`` (int) and reference buffers.
    num_future : int
        Number of future frames to include in addition to the current frame.

    Returns
    -------
    torch.Tensor, shape=(num_envs, (num_future+1)*(28*num_hands + 7*num_objects)), dtype=float32
        Flattened reference poses ordered by time then by hands then by objects.
    """
    offsets = torch.arange(0, num_future + 1, device=env.device) * env.ref_step_frames
    base = torch.clamp(env.frame_idx_f, max=env.ref_len - 1).unsqueeze(-1)
    idx = torch.clamp((base + offsets).to(torch.long), max=env.ref_len - 1)

    hands = env.ref_qpos[idx].reshape(env.num_envs, num_future + 1, -1)
    objs = torch.cat(
        (env.ref_object_pos[idx], env.ref_object_quat[idx]), dim=-1
    ).reshape(env.num_envs, num_future + 1, -1)
    return torch.cat((hands, objs), dim=-1).reshape(env.num_envs, -1)


def phase(env) -> torch.Tensor:
    """Normalized phase for reference playback.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``frame_idx`` (num_envs,) and ``ref_len`` (int).

    Returns
    -------
    torch.Tensor, shape=(num_envs, 1), dtype=float32
        Phase in ``[0, 1]`` computed as ``frame_idx / (ref_len - 1)``.
    """
    return (env.frame_idx.float() / float(env.ref_len - 1)).unsqueeze(-1)


def ref_hand_contact(env) -> torch.Tensor:
    """Concatenate reference per-link contact flags at the current frame.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``frame_idx`` (num_envs,), ``ref_len`` (int), ``ref_hands`` and
        ``ref_hand_is_contact`` (T, num_objects, num_hands, 10).

    Returns
    -------
    torch.Tensor, shape=(num_envs, num_hands*10*num_objects), dtype=float32
        Flattened contact indicators in ``{0, 1}`` ordered by hands, then objects, then the 10 tracked links.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    out = []
    for _, _, h in env.ref_hands:
        c = env.ref_hand_is_contact[idx][:, :, h]
        out.append(c.reshape(env.num_envs, -1))
    return torch.cat(out, dim=-1).to(torch.float32)


def hand_contact(env) -> torch.Tensor:
    """Concatenate per-link contact flags for the current Sharpa hands.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands`` and ``hand_contact_sensor_keys``. Each keyed ContactSensor is
        expected to provide ``data.contact_pos_w`` with shape ``(num_envs, 1, num_objects, 3)`` where NaN means no
        contact.

    Returns
    -------
    torch.Tensor, shape=(num_envs, num_hands*10*num_objects), dtype=float32
        Flattened contact indicators in ``{0, 1}`` ordered by hands, then objects, then the 10 tracked links.
    """
    out = []
    for _, _, h in env.ref_hands:
        side = "right" if bool(env.is_rhand[h]) else "left"
        c = []
        for key in env.hand_contact_sensor_keys[side]:
            pos_w = env.scene[key].data.contact_pos_w[:, 0]
            c.append(torch.isfinite(pos_w).all(dim=-1))
        out.append(torch.stack(c, dim=-1).reshape(env.num_envs, -1))
    return torch.cat(out, dim=-1).to(torch.float32)

