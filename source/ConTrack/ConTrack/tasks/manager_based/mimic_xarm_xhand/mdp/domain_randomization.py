from __future__ import annotations

import torch


def perturb_objects_xy(env, env_ids, xy_range: float):
    """Apply a random XY translation to all objects after reset.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``scene["objects"]``, ``ref_hands`` and ``device``. Expected object buffers
        are ``objects.data.object_pos_w`` (num_envs, O, 3), ``objects.data.object_quat_w`` (num_envs, O, 4),
        ``objects.data.object_lin_vel_w`` (num_envs, O, 3), and ``objects.data.object_ang_vel_w`` (num_envs, O, 3).
    env_ids : Sequence[int] | None, shape=(N,)
        Environment ids to perturb; ``None`` targets all environments.
    xy_range : float
        Uniform sampling range in meters. For each environment, ``dx, dy ~ U(-xy_range, xy_range)`` and the same
        ``(dx, dy)`` is applied to all objects in that environment.

    Returns
    -------
    None
        Writes updated object states to simulation with unchanged rotation and velocities.
    """
    objects = env.scene["objects"]
    if env_ids is None:
        env_ids = env.scene[env.ref_hands[0][0]]._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    dxy = (
        torch.rand((int(env_ids.shape[0]), 2), device=env.device) * 2.0 - 1.0
    ) * xy_range
    pos = objects.data.object_pos_w[env_ids].clone()
    pos[..., :2] += dxy.unsqueeze(1)
    objects.write_object_state_to_sim(
        torch.cat(
            [
                pos,
                objects.data.object_quat_w[env_ids],
                objects.data.object_lin_vel_w[env_ids],
                objects.data.object_ang_vel_w[env_ids],
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )


def apply_object_push(env, env_ids, force_magnitude: float, force_frame: int):
    """Apply a brief, random-direction horizontal push to the object at one specific reference frame.

    Meant to simulate a sudden disturbance mid-grasp (e.g. a bump). Runs every step (register with
    ``mode="interval"``, ``interval_range_s=(0.0, 0.0)``, matching ``advance_frame``); each step it writes zero
    force to every env except the ones whose ``frame_idx`` currently equals ``force_frame``, so the push only
    acts for the single physics step at which each env crosses that frame.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``scene["objects"]`` (RigidObjectCollection), ``frame_idx`` (num_envs,),
        ``num_objects`` (int) and ``device``.
    env_ids : Sequence[int] | None
        Ignored (the event manager passes this for interval-mode events; the push targets whichever envs are
        currently at ``force_frame``, independent of ``env_ids``).
    force_magnitude : float
        Push force in Newtons. ``<= 0`` disables the push entirely (the function returns immediately).
    force_frame : int
        The ``frame_idx`` value at which to fire the push, per env.

    Returns
    -------
    None
        Writes a per-env, per-object external force (zero almost everywhere, non-zero for one step per env) to
        the object rigid bodies. Torque is always zero (pure translational push).
    """
    if force_magnitude <= 0.0:
        return
    objects = env.scene["objects"]
    forces = torch.zeros((env.num_envs, env.num_objects, 3), device=env.device)
    hit = (env.frame_idx == force_frame).nonzero(as_tuple=False).squeeze(-1)
    if hit.numel() > 0:
        directions = torch.randn((hit.shape[0], 3), device=env.device)
        directions[:, 2] = 0.0
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        forces[hit] = (directions * force_magnitude).unsqueeze(1)
    torques = torch.zeros_like(forces)
    objects.set_external_force_and_torque(forces, torques)


def randomize_joint_pd_gains(
    env,
    env_ids,
    stiffness_mult_range: tuple[float, float] = (1.0, 1.0),
    damping_mult_range: tuple[float, float] = (1.0, 1.0),
):
    """Randomize per-joint implicit PD gains (stiffness, damping) on reset.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance exposing ``ref_hands`` and ``scene[name]`` articulations. Each articulation is expected
        to provide ``data.default_joint_stiffness`` (num_envs, num_joints) and ``data.default_joint_damping``
        (num_envs, num_joints), and writers ``write_joint_stiffness_to_sim`` / ``write_joint_damping_to_sim``.
    env_ids : Sequence[int] | None, shape=(N,)
        Environment ids to randomize; ``None`` targets all environments.
    stiffness_mult_range : tuple[float, float]
        Uniform sampling range for stiffness multipliers ``(lo, hi)``. Samples are per-env per-joint.
    damping_mult_range : tuple[float, float]
        Uniform sampling range for damping multipliers ``(lo, hi)``. Samples are per-env per-joint.

    Returns
    -------
    None
        Writes randomized joint stiffness/damping to simulation for the xArm+xHand joints used in the reference.
    """
    if env_ids is None:
        env_ids = env.scene[env.ref_hands[0][0]]._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    seen = set()
    s_lo, s_hi = (float(x) for x in stiffness_mult_range)
    d_lo, d_hi = (float(x) for x in damping_mult_range)
    for name, joint_ids, _ in env.ref_hands:
        if name in seen:
            continue
        seen.add(name)
        hand = env.scene[name]
        n = int(env_ids.shape[0])
        m = len(joint_ids)
        s_mult = torch.rand((n, m), device=env.device) * (s_hi - s_lo) + s_lo
        d_mult = torch.rand((n, m), device=env.device) * (d_hi - d_lo) + d_lo
        hand.write_joint_stiffness_to_sim(
            hand.data.default_joint_stiffness[env_ids][:, joint_ids] * s_mult,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        hand.write_joint_damping_to_sim(
            hand.data.default_joint_damping[env_ids][:, joint_ids] * d_mult,
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
