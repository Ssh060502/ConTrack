from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply_inverse

QVEL_PENALTY_SIGMA = 1.0
QACC_PENALTY_SIGMA = 50.0


def tracking_arm(env) -> torch.Tensor:
    """Arm joint position tracking reward as exp(-MSE / 0.5^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_qpos`` (T, H, 28), ``frame_idx`` (num_envs,) and ``ref_hands``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean exp(-mse / 0.5^2) over arm joints for all hands, where mse is mean squared error in joint units^2.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    loss = []
    for name, joint_ids, h in env.ref_hands:
        cur = env.scene[name].data.joint_pos[:, joint_ids[:6]]
        ref = env.ref_qpos[idx][:, h, :6]
        err = cur - ref
        loss.append(torch.mean(err * err, dim=-1))
    return torch.exp(-4.0 * torch.mean(torch.stack(loss, dim=-1), dim=-1))


def tracking_finger(env) -> torch.Tensor:
    """Finger joint position tracking reward as exp(-MSE / 0.5^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_qpos`` (T, H, 28), ``frame_idx`` (num_envs,) and ``ref_hands``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean exp(-mse / 0.5^2) over all finger joints and hands, where mse is mean squared error in joint units^2.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    loss = []
    for name, joint_ids, h in env.ref_hands:
        cur = env.scene[name].data.joint_pos[:, joint_ids[6:]]
        ref = env.ref_qpos[idx][:, h, 6:]
        err = cur - ref
        loss.append(torch.mean(err * err, dim=-1))
    return torch.exp(-4.0 * torch.mean(torch.stack(loss, dim=-1), dim=-1))


def tracking_obj_pos(env, std: float) -> torch.Tensor:
    """Object position tracking reward as exp(-MSE / std^2) in environment frame.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_object_pos`` (T, O, 3) and a ``objects`` rigid object collection.
    std : float
        Gaussian kernel standard deviation in meters.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean exp(-mse / std^2) over all object translations, where mse is mean squared error in meters^2.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    objects = env.scene["objects"]
    cur = objects.data.object_pos_w - env.scene.env_origins.unsqueeze(1)
    err = cur - env.ref_object_pos[idx]
    return torch.exp(-(1.0 / (std * std)) * torch.mean(err * err, dim=(-1, -2)))


def tracking_obj_rot(env, std: float) -> torch.Tensor:
    """Object rotation tracking reward as exp(-theta^2 / std^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_object_quat`` (T, O, 4) and a ``objects`` rigid object collection.
    std : float
        Gaussian kernel standard deviation in radians.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean exp(-theta^2 / std^2) over all objects, where theta is the quaternion angle error in radians.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    objects = env.scene["objects"]
    quat_dot = torch.clamp(
        torch.abs(torch.sum(objects.data.object_quat_w * env.ref_object_quat[idx], dim=-1)),
        0.0,
        1.0,
    )
    theta = torch.mean(2.0 * torch.arccos(quat_dot), dim=-1)
    return torch.exp(-(1.0 / (std * std)) * theta * theta)


def qvel_penalty_arm(env) -> torch.Tensor:
    """Penalize arm joint velocities as 1 - exp(-mean(v^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(v^2) / sigma^2) over CRX arm joint velocities, where sigma=1.0.
    """
    v = torch.cat(
        [
            env.scene[name].data.joint_vel[:, joint_ids[:6]]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    x = torch.mean(v * v, dim=-1)
    return 1.0 - torch.exp(-x / (QVEL_PENALTY_SIGMA * QVEL_PENALTY_SIGMA))


def qvel_penalty_finger(env) -> torch.Tensor:
    """Penalize finger joint velocities as 1 - exp(-mean(v^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(v^2) / sigma^2) over finger joint velocities, where sigma=1.0.
    """
    v = torch.cat(
        [
            env.scene[name].data.joint_vel[:, joint_ids[6:]]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    x = torch.mean(v * v, dim=-1)
    return 1.0 - torch.exp(-x / (QVEL_PENALTY_SIGMA * QVEL_PENALTY_SIGMA))


def qvel_penalty_obj_pos(env) -> torch.Tensor:
    """Penalize object linear velocities as 1 - exp(-mean(v^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing a ``objects`` rigid object collection.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(v^2) / sigma^2) over all object linear velocities, where sigma=1.0.
    """
    v = env.scene["objects"].data.object_lin_vel_w
    x = torch.mean(v * v, dim=(-1, -2))
    return 1.0 - torch.exp(-x / (QVEL_PENALTY_SIGMA * QVEL_PENALTY_SIGMA))


def qvel_penalty_obj_rot(env) -> torch.Tensor:
    """Penalize object angular velocities as 1 - exp(-mean(v^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing a ``objects`` rigid object collection.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(v^2) / sigma^2) over all object angular velocities, where sigma=1.0.
    """
    v = env.scene["objects"].data.object_ang_vel_w
    x = torch.mean(v * v, dim=(-1, -2))
    return 1.0 - torch.exp(-x / (QVEL_PENALTY_SIGMA * QVEL_PENALTY_SIGMA))


def qacc_penalty_arm(env) -> torch.Tensor:
    """Penalize arm accelerations as 1 - exp(-mean(a^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``prev_arm_qvel`` (num_envs, 6*H), ``ref_hands``, ``cfg.sim.dt`` and
        ``cfg.decimation``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(a^2) / sigma^2) over CRX arm joint accelerations, where sigma=50.0.
    """
    v = torch.cat(
        [
            env.scene[name].data.joint_vel[:, joint_ids[:6]]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    a = (v - env.prev_arm_qvel) * (1.0 / (float(env.cfg.sim.dt) * float(env.cfg.decimation)))
    x = torch.mean(a * a, dim=-1)
    return 1.0 - torch.exp(-x / (QACC_PENALTY_SIGMA * QACC_PENALTY_SIGMA))


def qacc_penalty_finger(env) -> torch.Tensor:
    """Penalize finger accelerations as 1 - exp(-mean(a^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``prev_finger_qvel`` (num_envs, 22*H), ``ref_hands``, ``cfg.sim.dt`` and
        ``cfg.decimation``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(a^2) / sigma^2) over finger joint accelerations, where sigma=50.0.
    """
    v = torch.cat(
        [
            env.scene[name].data.joint_vel[:, joint_ids[6:]]
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    a = (v - env.prev_finger_qvel) * (1.0 / (float(env.cfg.sim.dt) * float(env.cfg.decimation)))
    x = torch.mean(a * a, dim=-1)
    return 1.0 - torch.exp(-x / (QACC_PENALTY_SIGMA * QACC_PENALTY_SIGMA))


def qacc_penalty_obj_pos(env) -> torch.Tensor:
    """Penalize object linear accelerations as 1 - exp(-mean(a^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``prev_object_lin_vel`` (num_envs, O, 3), ``cfg.sim.dt`` and ``cfg.decimation``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(a^2) / sigma^2) over all object linear accelerations, where sigma=50.0.
    """
    v = env.scene["objects"].data.object_lin_vel_w
    a = (v - env.prev_object_lin_vel) * (1.0 / (float(env.cfg.sim.dt) * float(env.cfg.decimation)))
    x = torch.mean(a * a, dim=(-1, -2))
    return 1.0 - torch.exp(-x / (QACC_PENALTY_SIGMA * QACC_PENALTY_SIGMA))


def qacc_penalty_obj_rot(env) -> torch.Tensor:
    """Penalize object angular accelerations as 1 - exp(-mean(a^2) / sigma^2).

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``prev_object_ang_vel`` (num_envs, O, 3), ``cfg.sim.dt`` and ``cfg.decimation``.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean 1 - exp(-mean(a^2) / sigma^2) over all object angular accelerations, where sigma=50.0.
    """
    v = env.scene["objects"].data.object_ang_vel_w
    a = (v - env.prev_object_ang_vel) * (1.0 / (float(env.cfg.sim.dt) * float(env.cfg.decimation)))
    x = torch.mean(a * a, dim=(-1, -2))
    return 1.0 - torch.exp(-x / (QACC_PENALTY_SIGMA * QACC_PENALTY_SIGMA))


def contact_reward(env) -> torch.Tensor:
    """Reward matching object contacts between reference annotations and simulation.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing:
            - ``frame_idx`` (num_envs,) long
            - ``ref_len`` (int)
            - ``ref_hands`` (list[tuple[str, list[int], int]]) with CRX-5iA+Sharpa asset name and reference hand index
            - ``ref_hand_is_contact`` (T, O, H, 10) bool
            - ``hand_contact_sensor_keys`` (dict[str, list[str]]) mapping ``{"left","right"}`` to 10 ContactSensor keys
            - ``scene`` sensors at those keys with ``data.contact_pos_w`` (num_envs, 1, O, 3) where NaN means no contact

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean match fraction over hands for the 10 tracked links. Each link contributes 1 if the simulation
        contacts at least one same object as the reference contact annotation at the current frame, else 0, so the
        per-hand range is ``[0, 1]``.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    out = []
    for _, _, h in env.ref_hands:
        side = "right" if bool(env.is_rhand[h]) else "left"
        ref_c = env.ref_hand_is_contact[idx][:, :, h]
        sim_c = []
        for key in env.hand_contact_sensor_keys[side]:
            pos_w = env.scene[key].data.contact_pos_w[:, 0]
            sim_c.append(torch.isfinite(pos_w).all(dim=-1))
        sim_c = torch.stack(sim_c, dim=-1)
        out.append(torch.mean(((ref_c & sim_c).any(dim=1)).to(torch.float32), dim=-1))
    return torch.mean(torch.stack(out, dim=-1), dim=-1)


def contact_distance_reward(env) -> torch.Tensor:
    """Reward proximity between simulated and reference contact points when both are in contact.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing:
            - ``frame_idx`` (num_envs,) long
            - ``ref_len`` (int)
            - ``ref_hands`` (list[tuple[str, list[int], int]]) with CRX-5iA+Sharpa asset name and reference hand index
            - ``ref_hand_is_contact`` (T, O, H, 10) bool
            - ``ref_hand_contact_points`` (T, O, H, 10, 3) float32 in object local frame (meters)
            - ``hand_contact_sensor_keys`` (dict[str, list[str]]) mapping ``{"left","right"}`` to 10 ContactSensor keys
            - ``scene`` sensors at those keys with ``data.contact_pos_w`` (num_envs, 1, O, 3) in world frame
            - ``scene["objects"].data.object_pos_w`` (num_envs, O, 3) float32
            - ``scene["objects"].data.object_quat_w`` (num_envs, O, 4) float32 in (w,x,y,z)

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=float32
        Mean exp(-d^2 / 0.03^2) over links that are jointly in contact, where d is the per-link mean contact-point
        distance in meters over objects. Returns 0 if no links are jointly in contact.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    objects = env.scene["objects"]
    obj_pos_w = objects.data.object_pos_w
    obj_quat_w = objects.data.object_quat_w
    out = []
    for _, _, h in env.ref_hands:
        side = "right" if bool(env.is_rhand[h]) else "left"
        ref_c = env.ref_hand_is_contact[idx][:, :, h]
        ref_p_local = env.ref_hand_contact_points[idx][:, :, h]
        total = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
        cnt = torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device)
        for i, key in enumerate(env.hand_contact_sensor_keys[side]):
            pos_w_raw = env.scene[key].data.contact_pos_w[:, 0]
            m_sim = torch.isfinite(pos_w_raw).all(dim=-1)
            m = ref_c[:, :, i] & m_sim
            pos_w = torch.nan_to_num(pos_w_raw, nan=0.0)
            sim_local = quat_apply_inverse(obj_quat_w, pos_w - obj_pos_w)
            diff = sim_local - ref_p_local[:, :, i]
            err = torch.sum(diff * diff, dim=-1) * m.to(torch.float32)
            c = m.sum(dim=-1).to(torch.float32)
            total += torch.exp(
                -(1.0 / (0.03 * 0.03)) * err.sum(dim=-1) / c.clamp(min=1)
            ) * (c > 0)
            cnt += (c > 0).to(torch.float32)
        out.append(total / cnt.clamp(min=1) * (cnt > 0))
    return torch.mean(torch.stack(out, dim=-1), dim=-1).to(torch.float32)

