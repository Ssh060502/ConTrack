from __future__ import annotations

import torch


def obj_pose_exceeded(env) -> torch.Tensor:
    """Terminate when any object deviates too far from the reference.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``frame_idx`` (num_envs,), ``ref_len`` (int), ``ref_object_pos`` (T, O, 3),
        ``ref_object_quat`` (T, O, 4), ``scene.env_origins`` (num_envs, 3) and ``scene["objects"]`` buffers.
        Also expects ``object_pos_threshold`` (num_envs,) and ``object_rot_threshold`` (num_envs,) float buffers.

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=bool
        Boolean mask where True signals termination.
    """
    idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
    objects = env.scene["objects"]
    pos = objects.data.object_pos_w - env.scene.env_origins.unsqueeze(1)
    pos_err = torch.norm(pos - env.ref_object_pos[idx], dim=-1)
    quat_dot = torch.clamp(
        torch.abs(torch.sum(objects.data.object_quat_w * env.ref_object_quat[idx], dim=-1)),
        0.0,
        1.0,
    )
    rot_err = 2.0 * torch.arccos(quat_dot)
    return torch.any(
        (pos_err > env.object_pos_threshold.unsqueeze(1))
        | (rot_err > env.object_rot_threshold.unsqueeze(1)),
        dim=-1,
    )


def time_out(env) -> torch.Tensor:
    """Terminate when tracking reaches the last reference frame.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``frame_idx`` (num_envs,) and ``ref_len`` (int).

    Returns
    -------
    torch.Tensor, shape=(num_envs,), dtype=bool
        Time-out mask where True indicates the end of the clip.
    """
    return env.frame_idx >= (env.ref_len - 1)

