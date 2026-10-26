from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.sim.utils import get_current_stage
from pxr import Sdf, Usd, UsdPhysics

from ConTrack.robots.rby1a_sharpa import (
    DEFAULT_RBY1A_QPOS,
    RBY1A_LEFT_JOINT_NAMES,
    RBY1A_RIGHT_JOINT_NAMES,
    SHARPA_LEFT_JOINT_NAMES,
    SHARPA_RIGHT_JOINT_NAMES,
)

_REF_CPU: Dict[str, dict[str, Any]] = {}
_REF_CACHE: Dict[tuple[str, str], dict[str, Any]] = {}
_HAND_CONTACT_ORDER = (
    "index_link1",
    "index_link2",
    "middle_link1",
    "middle_link2",
    "ring_link1",
    "ring_link2",
    "pinky_link1",
    "pinky_link2",
    "thumb_link1",
    "thumb_link2",
)

DELTA = 100
EMA_ALPHA = 0.05
HARD_RESET_TOPK = 64
EPSILON_GREEDY = 0.5
TAU = 1.0

_RIGHT_FILTERED_PAIRS = (
    (
        "right_hand_C_MC",
        (
            "right_thumb_MC",
            "right_index_PP",
            "right_middle_PP",
            "right_ring_PP",
            "right_pinky_PP",
        ),
    ),
    ("right_index_PP", ("right_hand_C_MC",)),
    ("right_middle_PP", ("right_hand_C_MC",)),
    ("right_ring_PP", ("right_hand_C_MC",)),
    ("right_pinky_MC", ("right_pinky_PP",)),
    ("right_pinky_MCP_VL", ("right_pinky_PP",)),
    ("right_pinky_PP", ("right_hand_C_MC", "right_pinky_MC")),
    ("right_thumb_MC", ("right_hand_C_MC", "right_thumb_PP")),
    ("right_thumb_PP", ("right_thumb_MC",)),
)
_SHARPA_FILTERED_PAIRS = _RIGHT_FILTERED_PAIRS + tuple(
    (
        src.replace("right_", "left_"),
        tuple(tgt.replace("right_", "left_") for tgt in tgts),
    )
    for src, tgts in _RIGHT_FILTERED_PAIRS
)


def filter_sharpa_self_collisions(env, env_ids):
    """Filter Sharpa self-collision pairs that should stay ignored inside ``Rby1aSharpa``.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance (unused).
    env_ids : Sequence[int] | None
        Environment ids (unused).

    Returns
    -------
    None
        Authors ``UsdPhysics.FilteredPairsAPI`` relationships on the Sharpa link prims under each
        ``*/Rby1aSharpa`` articulation in the current stage.
    """
    stage = get_current_stage()
    link_names = {src for src, _ in _SHARPA_FILTERED_PAIRS} | {
        tgt for _, tgts in _SHARPA_FILTERED_PAIRS for tgt in tgts
    }
    for prim in stage.Traverse():
        if prim.GetName() != "Rby1aSharpa":
            continue
        link_paths = {
            p.GetName(): p.GetPath()
            for p in Usd.PrimRange(prim)
            if p.GetName() in link_names
        }
        for src, tgts in _SHARPA_FILTERED_PAIRS:
            if src not in link_paths or any(tgt not in link_paths for tgt in tgts):
                continue
            UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(str(link_paths[src]))
            ).CreateFilteredPairsRel().SetTargets(
                [Sdf.Path(str(link_paths[tgt])) for tgt in tgts]
            )


def load_reference(path: str) -> dict[str, np.ndarray]:
    """Load an RB-Y1A+Sharpa reference clip with joint, object tracks, and object contact annotations.

    Parameters
    ----------
    path : str
        Absolute path to the reference file. Expected keys are ``qpos`` (T, H, 29), ``urdf_name`` (H,),
        ``is_rhand`` (H,), ``fps`` (H,), and ``object_tracks`` with per-object ``translations`` (T, 3) and
        ``orientations_xyzw`` (T, 4). Contact annotations are expected in ``contacts`` with per-object datasets
        ``is_contact`` (T, H, 15) and ``points`` (T, H, 15, 3).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping with ``qpos`` (T, H, 29), ``object_pos`` (T, O, 3), ``object_quat_wxyz`` (T, O, 4), ``fps`` (H,)
        in ``float32``, ``urdf_name`` as ``list[str]`` and ``is_rhand`` (H,) as ``bool``.
        Contact targets are returned as ``hand_is_contact`` (T, O, H, 10) in ``uint8`` and
        ``hand_contact_points`` (T, O, H, 10, 3) in ``float32`` in the object-local frame where the 10 links follow:
        ``("index_link1","index_link2","middle_link1","middle_link2","ring_link1","ring_link2","pinky_link1","pinky_link2","thumb_link1","thumb_link2")``.
    """
    file_path = Path(path)
    with h5py.File(file_path, "r") as f:
        qpos = f["qpos"][:].astype(np.float32)
        fps = f["fps"][:].astype(np.float32)
        urdf_name = [s.decode("utf-8") for s in np.asarray(f["urdf_name"])]
        is_rhand = np.asarray(f["is_rhand"], np.uint8).astype(bool)
        obj_keys = list(f["object_tracks"].keys())
        if all(k.isdigit() for k in obj_keys):
            obj_keys = sorted(obj_keys, key=lambda s: int(s))
        obj_pos = np.stack(
            [
                np.asarray(f["object_tracks"][k]["translations"], np.float32)
                for k in obj_keys
            ],
            axis=1,
        ).astype(np.float32)
        obj_q = np.stack(
            [
                np.asarray(f["object_tracks"][k]["orientations_xyzw"], np.float32)
                for k in obj_keys
            ],
            axis=1,
        ).astype(np.float32)
        seg_names = [s.decode("utf-8") for s in np.asarray(f["contacts/segment_names"])]
        c_keys = sorted([k for k in f["contacts"].keys() if k.isdigit()], key=int)
        contact_is = np.stack(
            [np.asarray(f["contacts"][k]["is_contact"], np.uint8) for k in c_keys],
            axis=1,
        )
        contact_pts = np.stack(
            [np.asarray(f["contacts"][k]["points"], np.float32) for k in c_keys], axis=1
        ).astype(np.float32)
        contact_pts = np.nan_to_num(contact_pts, nan=0.0)
    obj_quat_wxyz = np.concatenate([obj_q[..., 3:4], obj_q[..., :3]], axis=-1).astype(
        np.float32
    )
    seg = {n: i for i, n in enumerate(seg_names)}
    link_segs = (
        (seg["index_proximal"],),
        (seg["index_middle"], seg["index_distal"]),
        (seg["middle_proximal"],),
        (seg["middle_middle"], seg["middle_distal"]),
        (seg["ring_proximal"],),
        (seg["ring_middle"], seg["ring_distal"]),
        (seg["pinky_proximal"],),
        (seg["pinky_middle"], seg["pinky_distal"]),
        (seg["thumb_proximal"],),
        (seg["thumb_middle"], seg["thumb_distal"]),
    )
    hand_is = []
    hand_pts = []
    for seg_ids in link_segs:
        m = contact_is[..., list(seg_ids)].astype(np.float32)
        hand_is.append((m > 0).any(axis=3).astype(np.uint8))
        w = m[..., None]
        p = (contact_pts[..., list(seg_ids), :] * w).sum(axis=3)
        c = m.sum(axis=3)
        hand_pts.append((p / np.maximum(c, 1.0)[..., None]).astype(np.float32))
    hand_is = np.stack(hand_is, axis=3).astype(np.uint8)
    hand_pts = np.stack(hand_pts, axis=3).astype(np.float32)
    return {
        "qpos": qpos,
        "object_pos": obj_pos,
        "object_quat_wxyz": obj_quat_wxyz,
        "fps": fps,
        "urdf_name": urdf_name,
        "is_rhand": is_rhand,
        "hand_is_contact": hand_is,
        "hand_contact_points": hand_pts,
    }


def reference_tensors(device: str, reference_path: str) -> dict[str, Any]:
    """Return cached reference tensors on the requested device.

    Parameters
    ----------
    device : str
        Torch device string, e.g. ``"cuda:0"``.
    reference_path : str
        Absolute HDF5 path used as the cache key.

    Returns
    -------
    dict[str, Any]
        Mapping with tensors ``qpos`` (T, H, 29), ``object_pos`` (T, O, 3), ``object_quat_wxyz`` (T, O, 4),
        ``fps`` (H,) on ``device`` and ``urdf_name`` (list[str]) and ``is_rhand`` (H,) bool.
        Contact targets are returned as ``hand_is_contact`` (T, O, H, 10) ``uint8`` and
        ``hand_contact_points`` (T, O, H, 10, 3) ``float32`` in object local frame.
    """
    key_cpu = Path(reference_path).as_posix()
    if key_cpu not in _REF_CPU:
        ref_np = load_reference(reference_path)
        q = torch.from_numpy(ref_np["object_quat_wxyz"])
        flip = torch.where(
            torch.sum(q[1:] * q[:-1], dim=-1, keepdim=True) < 0.0, -1.0, 1.0
        )
        sign = torch.ones((q.shape[0], q.shape[1], 1))
        sign[1:] = torch.cumprod(flip, dim=0)
        q = q * sign
        _REF_CPU[key_cpu] = {
            "qpos": torch.from_numpy(ref_np["qpos"]),
            "object_pos": torch.from_numpy(ref_np["object_pos"]),
            "object_quat_wxyz": q,
            "fps": torch.from_numpy(ref_np["fps"]),
            "urdf_name": ref_np["urdf_name"],
            "is_rhand": torch.from_numpy(ref_np["is_rhand"]),
            "hand_is_contact": torch.from_numpy(ref_np["hand_is_contact"]),
            "hand_contact_points": torch.from_numpy(ref_np["hand_contact_points"]),
        }
    key = (key_cpu, device)
    if key not in _REF_CACHE:
        _REF_CACHE[key] = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in _REF_CPU[key_cpu].items()
        }
    return _REF_CACHE[key]


def prepare_reference_state(env, env_ids):
    """Attach reference tensors, joint indices and per-env frame counters to ``env``.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance with scene asset ``rby1a_sharpa``.
    env_ids : Sequence[int] | None
        Environment ids (unused; included for event manager signature compatibility).

    Returns
    -------
    None
        Adds ``ref_qpos`` (T, H, 29), ``urdf_name`` (H,), ``is_rhand`` (H,),
        ``ref_hands`` (H,), ``ref_object_pos`` (T, O, 3), ``ref_object_quat`` (T, O, 4),
        ``ref_hand_is_contact`` (T, O, H, 10), ``ref_hand_contact_points`` (T, O, H, 10, 3),
        ``unused_joint_ids`` (list[int]) and ``unused_default_qpos`` (29,) for the missing side when the clip
        contains only one hand (otherwise ``unused_joint_ids=[]``),
        ``hand_contact_sensor_keys`` (dict[str, list[str]]),
        ``ref_fps`` (float), ``ref_step_frames`` (float),
        ``num_objects`` (int), ``ref_len`` (int), ``frame_idx`` (num_envs,), ``frame_idx_f`` (num_envs,),
        ``prev_arm_qvel`` (num_envs, 7*H), ``prev_finger_qvel`` (num_envs, 22*H),
        ``prev_object_lin_vel`` (num_envs, O, 3), ``prev_object_ang_vel`` (num_envs, O, 3),
        ``object_pos_threshold`` (num_envs,), ``object_rot_threshold`` (num_envs,),
        ``threshold_curriculum_success_ema`` ().
        The reference is resampled to action-step rate such that one reference index corresponds to one action step
        (i.e. ``ref_step_frames=1`` and ``ref_fps=1/(sim.dt*decimation)``).
        Initializes reset library buffers:
        - ``reset_lib_state`` (T, 58*H + 13*O) float32 storing per-frame hand+object reset states with finite-difference
          joint/object velocities (ref_fps assumed)
        - ``reset_lib_track_len`` (T,) float32 storing EMA remaining length per frame (in reference frames)
        - ``reset_lib_track_len_best`` (T,) float32 storing decayed best remaining length per frame (update criterion)
        - ``reset_lib_terminate_index_count`` (T,) long storing terminate-frame histogram since last log/reset (counts)
        - ``reset_lib_reset_index_count`` (T,) long storing reset-start-frame histogram since last log/reset (counts)
        - ``reset_hand_dim`` (int) = 58*H and ``reset_obj_dim`` (int) = 13*O
        - ``episode_states`` (num_envs, T, 58*H + 13*O) float32 storing per-env tracked states
        - ``episode_start_frame`` (num_envs,) long storing per-env episode start frame
        - ``episode_valid`` (num_envs,) bool indicating whether the env has a completed episode to contribute
        - ``episode_filled`` (num_envs, T) bool indicating which episode offsets were written
    """
    if hasattr(env, "ref_hands"):
        return
    ref = reference_tensors(env.device, env.cfg.reference_path)
    raw_len = int(ref["qpos"].shape[0])
    raw_fps = float(torch.max(ref["fps"]).item())
    action_fps = 1.0 / (float(env.cfg.sim.dt) * float(env.cfg.decimation))
    action_len = int((raw_len - 1) * action_fps / raw_fps) + 1
    t = torch.arange(action_len, device=env.device, dtype=torch.float32) * (
        raw_fps / action_fps
    )
    i0 = t.to(torch.long)
    i1 = torch.clamp(i0 + 1, max=raw_len - 1)
    w = (t - i0.to(torch.float32)).view(-1, 1, 1)
    env.ref_qpos = ref["qpos"][i0] * (1.0 - w) + ref["qpos"][i1] * w
    env.ref_object_pos = ref["object_pos"][i0] * (1.0 - w) + ref["object_pos"][i1] * w
    q0 = ref["object_quat_wxyz"][i0]
    q1 = ref["object_quat_wxyz"][i1]
    q1 = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -q1, q1)
    q = q0 * (1.0 - w) + q1 * w
    env.ref_object_quat = q / torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-8)
    nn = torch.clamp((t + 0.5).to(torch.long), max=raw_len - 1)
    env.ref_hand_is_contact = ref["hand_is_contact"][nn].to(torch.bool)
    env.ref_hand_contact_points = ref["hand_contact_points"][nn]
    env.urdf_name = ref["urdf_name"]
    env.is_rhand = ref["is_rhand"].to(torch.bool)
    env.num_objects = int(env.ref_object_pos.shape[1])
    env.ref_len = int(env.ref_qpos.shape[0])
    env.ref_fps = action_fps
    env.ref_step_frames = 1.0

    h = int(env.ref_qpos.shape[1])
    fps = float(env.ref_fps)
    hand_qpos = env.ref_qpos
    hand_qvel = torch.zeros_like(hand_qpos)
    hand_qvel[1:-1] = 0.5 * (hand_qpos[2:] - hand_qpos[:-2]) * fps
    hand_qvel[0] = (hand_qpos[1] - hand_qpos[0]) * fps
    hand_qvel[-1] = (hand_qpos[-1] - hand_qpos[-2]) * fps
    hand_state = torch.cat([hand_qpos, hand_qvel], dim=-1).view(env.ref_len, -1)

    obj_pos = env.ref_object_pos
    obj_lin_vel = torch.zeros_like(obj_pos)
    obj_lin_vel[1:-1] = 0.5 * (obj_pos[2:] - obj_pos[:-2]) * fps
    obj_lin_vel[0] = (obj_pos[1] - obj_pos[0]) * fps
    obj_lin_vel[-1] = (obj_pos[-1] - obj_pos[-2]) * fps

    q = env.ref_object_quat
    q_prev = torch.empty_like(q)
    q_next = torch.empty_like(q)
    q_prev[0], q_next[0] = q[0], q[1]
    q_prev[-1], q_next[-1] = q[-2], q[-1]
    q_prev[1:-1], q_next[1:-1] = q[:-2], q[2:]
    w0, x0, y0, z0 = q_prev.unbind(-1)
    w1, x1, y1, z1 = q_next.unbind(-1)
    rel = torch.stack(
        [
            w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0,
            -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0,
            -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0,
            -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0,
        ],
        dim=-1,
    )
    rel = torch.where(rel[..., :1] < 0.0, -rel, rel)
    w = rel[..., 0]
    v = rel[..., 1:]
    s = torch.linalg.norm(v, dim=-1)
    a = 2.0 * torch.atan2(s, w.clamp(min=1e-8))
    rotvec = v * (a / s.clamp(min=1e-8)).unsqueeze(-1)
    scale = torch.full((env.ref_len, 1, 1), 0.5 * fps, device=env.device)
    scale[0] = fps
    scale[-1] = fps
    obj_ang_vel = rotvec * scale

    obj_state = torch.cat(
        [obj_pos, env.ref_object_quat, obj_lin_vel, obj_ang_vel], dim=-1
    ).view(env.ref_len, -1)
    env.reset_hand_dim = int(58 * h)
    env.reset_obj_dim = int(13 * env.num_objects)
    env.reset_lib_state = torch.cat([hand_state, obj_state], dim=-1)
    env.reset_lib_track_len = torch.zeros((env.ref_len,), device=env.device)
    env.reset_lib_track_len_best = torch.zeros(
        (env.ref_len,), dtype=torch.float32, device=env.device
    )
    env.reset_lib_terminate_index_count = torch.zeros(
        (env.ref_len,), dtype=torch.long, device=env.device
    )
    env.reset_lib_reset_index_count = torch.zeros(
        (env.ref_len,), dtype=torch.long, device=env.device
    )
    env.episode_states = torch.empty(
        (env.num_envs, env.ref_len, env.reset_lib_state.shape[1]), device=env.device
    )
    env.episode_start_frame = torch.zeros(
        (env.num_envs,), dtype=torch.long, device=env.device
    )
    env.episode_valid = torch.zeros(
        (env.num_envs,), dtype=torch.bool, device=env.device
    )
    env.episode_filled = torch.zeros(
        (env.num_envs, env.ref_len), dtype=torch.bool, device=env.device
    )

    env.ref_hands = []
    hand: Articulation = env.scene["rby1a_sharpa"]
    for h_idx, is_r in enumerate(env.is_rhand.tolist()):
        if is_r:
            ids = [
                int(i)
                for i in hand.find_joints(
                    RBY1A_RIGHT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES,
                    preserve_order=True,
                )[0]
            ]
            env.ref_hands.append(("rby1a_sharpa", ids, h_idx))
        else:
            ids = [
                int(i)
                for i in hand.find_joints(
                    RBY1A_LEFT_JOINT_NAMES + SHARPA_LEFT_JOINT_NAMES, preserve_order=True
                )[0]
            ]
            env.ref_hands.append(("rby1a_sharpa", ids, h_idx))
    has_right = bool(env.is_rhand.any().item())
    has_left = bool((~env.is_rhand).any().item())
    env.unused_joint_ids = []
    if has_left and not has_right:
        env.unused_joint_ids = [
            int(i)
            for i in hand.find_joints(
                RBY1A_RIGHT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES, preserve_order=True
            )[0]
        ]
    elif has_right and not has_left:
        env.unused_joint_ids = [
            int(i)
            for i in hand.find_joints(
                RBY1A_LEFT_JOINT_NAMES + SHARPA_LEFT_JOINT_NAMES, preserve_order=True
            )[0]
        ]
    env.unused_default_qpos = torch.tensor(
        DEFAULT_RBY1A_QPOS + [0.0] * 22, device=env.device, dtype=torch.float32
    )
    env.hand_contact_sensor_keys = {
        "right": [f"contact_right_{k}" for k in _HAND_CONTACT_ORDER],
        "left": [f"contact_left_{k}" for k in _HAND_CONTACT_ORDER],
    }
    env.frame_idx = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
    env.frame_idx_f = torch.zeros(
        (env.num_envs,), dtype=torch.float32, device=env.device
    )
    env.prev_arm_qvel = torch.zeros(
        (env.num_envs, 7 * len(env.ref_hands)), device=env.device
    )
    env.prev_finger_qvel = torch.zeros(
        (env.num_envs, 22 * len(env.ref_hands)), device=env.device
    )
    env.prev_object_lin_vel = torch.zeros(
        (env.num_envs, env.num_objects, 3), device=env.device
    )
    env.prev_object_ang_vel = torch.zeros(
        (env.num_envs, env.num_objects, 3), device=env.device
    )
    env.object_pos_threshold = torch.full(
        (env.num_envs,),
        float(env.cfg.object_pos_threshold_max),
        dtype=torch.float32,
        device=env.device,
    )
    env.object_rot_threshold = torch.full(
        (env.num_envs,),
        float(env.cfg.object_rot_threshold_max),
        dtype=torch.float32,
        device=env.device,
    )
    env.threshold_curriculum_success_ema = torch.zeros(
        (), dtype=torch.float32, device=env.device
    )


def init_reference_buffers(env, env_ids):
    """Startup event that prepares shared reference buffers.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance.
    env_ids : Sequence[int] | None
        Environment ids (unused).

    Returns
    -------
    None
        Calls :func:`prepare_reference_state`.
    """
    prepare_reference_state(env, env_ids)


def reset_reference_episode(env, env_ids):
    """Reset hands and objects from the per-frame reset library.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance with prepared reference buffers.
        Expected buffers:
        - ``ref_qpos`` (T, H, 29) float
        - ``ref_object_pos`` (T, O, 3) float
        - ``ref_object_quat`` (T, O, 4) float
        - ``frame_idx`` (num_envs,) long
        - ``frame_idx_f`` (num_envs,) float
        - ``reset_lib_state`` (T, 58*H + 13*O) float
        - ``reset_lib_track_len`` (T,) float
        - ``reset_lib_track_len_best`` (T,) float
        - ``reset_lib_terminate_index_count`` (T,) long
        - ``reset_lib_reset_index_count`` (T,) long
        - ``episode_states`` (num_envs, T, 58*H + 13*O) float
        - ``episode_start_frame`` (num_envs,) long
        - ``episode_valid`` (num_envs,) bool
    env_ids : Sequence[int] | None, shape=(N,)
        Environment ids to reset; ``None`` targets all environments.

    Returns
    -------
    None
        Before resetting, updates the reset library with the per-frame states from the just-finished episodes.
        Also updates threshold curriculum EMA and object pose-break thresholds.
        The per-frame ``track_len`` is defined as the remaining length to termination (frames), and is updated with
        EMA. The cached reset state is updated when ``len_rep`` exceeds a decayed per-frame best criterion. Reset
        frames are sampled by selecting the K most positive gradients of
        ``track_len / (ref_len-k)`` and (with probability ``EPSILON_GREEDY``) resetting ``DELTA`` frames earlier,
        otherwise sampling with softmax on ``-track_len / (ref_len-k)`` (temperature ``TAU``).
    """
    prepare_reference_state(env, env_ids)
    objects = env.scene["objects"]
    if env_ids is None:
        env_ids = env.scene[env.ref_hands[0][0]]._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids.shape[0])

    upd_env_ids = env_ids[env.episode_valid[env_ids]]
    if int(upd_env_ids.shape[0]) > 0:
        hands = torch.cat(
            [
                torch.cat(
                    [
                        env.scene[name].data.joint_pos[upd_env_ids][:, joint_ids],
                        env.scene[name].data.joint_vel[upd_env_ids][:, joint_ids],
                    ],
                    dim=-1,
                )
                for name, joint_ids, _ in env.ref_hands
            ],
            dim=-1,
        )
        obj_pos = (
            objects.data.object_pos_w[upd_env_ids]
            - env.scene.env_origins[upd_env_ids].unsqueeze(1)
        )
        obj = torch.cat(
            [
                obj_pos,
                objects.data.object_quat_w[upd_env_ids],
                objects.data.object_lin_vel_w[upd_env_ids],
                objects.data.object_ang_vel_w[upd_env_ids],
            ],
            dim=-1,
        ).view(int(upd_env_ids.shape[0]), -1)
        start0 = env.episode_start_frame[upd_env_ids]
        end0 = env.frame_idx[upd_env_ids]
        offs0 = end0 - start0
        env.episode_states[upd_env_ids, offs0] = torch.cat([hands, obj], dim=-1)
        env.episode_filled[upd_env_ids, offs0] = True
        if env.cfg.threshold_curriculum_enabled:
            success = (end0 >= env.ref_len - 1).to(torch.float32).mean()
            a = env.cfg.threshold_curriculum_success_ema_alpha
            env.threshold_curriculum_success_ema = (
                env.threshold_curriculum_success_ema * (1.0 - a) + success * a
            )
            w = torch.clamp(
                (
                    env.threshold_curriculum_success_ema
                    - env.cfg.threshold_curriculum_sr_low
                )
                / (
                    env.cfg.threshold_curriculum_sr_high
                    - env.cfg.threshold_curriculum_sr_low
                ),
                0.0,
                1.0,
            )
            env.object_pos_threshold[:] = (
                env.cfg.object_pos_threshold_max * (1.0 - w)
                + env.cfg.object_pos_threshold_min * w
            )
            env.object_rot_threshold[:] = (
                env.cfg.object_rot_threshold_max * (1.0 - w)
                + env.cfg.object_rot_threshold_min * w
            )
        env.reset_lib_terminate_index_count.add_(
            torch.bincount(end0, minlength=env.ref_len)
        )
        track_len = end0 - start0 + 1
        total = int(track_len.sum().item())
        base = torch.cumsum(track_len, dim=0) - track_len
        env_rep = torch.repeat_interleave(upd_env_ids, track_len)
        offs_rep = torch.arange(
            total, device=env.device, dtype=torch.long
        ) - torch.repeat_interleave(base, track_len)
        frames = torch.repeat_interleave(start0, track_len) + offs_rep
        len_rep = torch.repeat_interleave(track_len, track_len) - offs_rep
        filled = env.episode_filled[env_rep, offs_rep]
        frames, env_rep, offs_rep, len_rep = (
            frames[filled],
            env_rep[filled],
            offs_rep[filled],
            len_rep[filled],
        )
        len_rep_f = len_rep.to(torch.float32)
        sum_f = torch.bincount(frames, weights=len_rep_f, minlength=env.ref_len)
        cnt_f = torch.bincount(frames, minlength=env.ref_len).to(torch.float32)
        m = cnt_f > 0
        env.reset_lib_track_len[m] = (
            env.reset_lib_track_len[m] * (1.0 - EMA_ALPHA)
            + (sum_f[m] / cnt_f[m]) * EMA_ALPHA
        )
        env.reset_lib_track_len_best.mul_(0.99999)
        better = (frames != 0) & (len_rep_f > env.reset_lib_track_len_best[frames])
        frames, env_rep, offs_rep, len_rep, len_rep_f = (
            frames[better],
            env_rep[better],
            offs_rep[better],
            len_rep[better],
            len_rep_f[better],
        )
        key = frames * (env.ref_len + 1) - len_rep
        order = torch.argsort(key)
        frames, env_rep, offs_rep, len_rep_f = (
            frames[order],
            env_rep[order],
            offs_rep[order],
            len_rep_f[order],
        )
        keep = torch.ones((int(frames.shape[0]),), dtype=torch.bool, device=env.device)
        keep[1:] = frames[1:] != frames[:-1]
        frames, env_rep, offs_rep, len_rep_f = (
            frames[keep],
            env_rep[keep],
            offs_rep[keep],
            len_rep_f[keep],
        )
        env.reset_lib_state[frames] = env.episode_states[env_rep, offs_rep]
        env.reset_lib_track_len_best[frames] = len_rep_f

    l = env.reset_lib_track_len[: env.ref_len - 1]
    h = (env.ref_len - torch.arange(env.ref_len - 1, device=env.device)).to(
        torch.float32
    )
    ratio = l / h
    grad = torch.zeros_like(ratio)
    grad[:-1] = ratio[1:] - ratio[:-1]
    k = min(HARD_RESET_TOPK, int(grad.shape[0]))
    worst = torch.topk(grad, k, largest=True).indices
    if not (worst == 0).any().item():
        worst[-1] = 0
    start = torch.multinomial(torch.softmax(-ratio / TAU, dim=0), n, replacement=True)
    hard = torch.rand((n,), device=env.device) < EPSILON_GREEDY
    start[hard] = torch.clamp(
        worst[torch.randint(0, k, (int(hard.sum().item()),), device=env.device)]
        - DELTA,
        min=0,
    )
    if getattr(env, "force_start_frame0", False):
        # Opt-in override for deterministic full-length playback (e.g. play.py
        # --export_rollout). Leaves the curriculum sampling above untouched so
        # training is unaffected; only the start frame actually used is replaced.
        start = torch.zeros((n,), dtype=torch.long, device=env.device)

    env.reset_lib_reset_index_count.add_(torch.bincount(start, minlength=env.ref_len))
    state = env.reset_lib_state[start]
    num_hands = int(env.reset_hand_dim // 58)
    hands = state[:, : env.reset_hand_dim].view(n, num_hands, 58)
    hand_qpos = hands[..., :29]
    hand_qvel = hands[..., 29:]
    for name, joint_ids, h_idx in env.ref_hands:
        env.scene[name].write_joint_state_to_sim(
            hand_qpos[:, h_idx],
            hand_qvel[:, h_idx],
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
    if env.unused_joint_ids:
        qpos = env.unused_default_qpos.unsqueeze(0).expand(n, -1)
        env.scene["rby1a_sharpa"].write_joint_state_to_sim(
            qpos,
            torch.zeros_like(qpos),
            joint_ids=env.unused_joint_ids,
            env_ids=env_ids,
        )
    obj_state = state[:, env.reset_hand_dim :].view(n, env.num_objects, 13)
    env.prev_object_lin_vel[env_ids] = obj_state[..., 7:10]
    env.prev_object_ang_vel[env_ids] = obj_state[..., 10:13]
    obj_state_sim = obj_state.clone()
    obj_state_sim[..., 0:3] += env.scene.env_origins[env_ids].unsqueeze(1)
    objects.write_object_state_to_sim(obj_state_sim, env_ids=env_ids)

    env.frame_idx[env_ids] = start
    env.frame_idx_f[env_ids] = start.float()
    env.prev_arm_qvel[env_ids] = hand_qvel[..., :7].reshape(n, -1)
    env.prev_finger_qvel[env_ids] = hand_qvel[..., 7:].reshape(n, -1)
    env.episode_filled[env_ids] = False
    env.episode_states[env_ids, 0] = state
    env.episode_filled[env_ids, 0] = True
    env.episode_start_frame[env_ids] = start
    env.episode_valid[env_ids] = True


def reset_reference_episode_first_frame(env, env_ids):
    """Reset hands and objects to the first reference frame with reference velocities.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedEnv
        Environment instance with prepared reference buffers.
        Expected buffers:
        - ``ref_qpos`` (T, H, 29) float
        - ``ref_object_pos`` (T, O, 3) float
        - ``ref_object_quat`` (T, O, 4) float
        - ``frame_idx`` (num_envs,) long
        - ``frame_idx_f`` (num_envs,) float
        Also expects reset-library buffers prepared by :func:`prepare_reference_state`:
        - ``reset_lib_state`` (T, 58*H + 13*O) float
        - ``reset_hand_dim`` (int) and ``reset_obj_dim`` (int)
        - ``prev_arm_qvel`` (num_envs, 7*H), ``prev_finger_qvel`` (num_envs, 22*H)
        - ``prev_object_lin_vel`` (num_envs, O, 3), ``prev_object_ang_vel`` (num_envs, O, 3)
    env_ids : Sequence[int] | None, shape=(N,)
        Environment ids to reset; ``None`` targets all environments.

    Returns
    -------
    None
        Writes joint/object pose and velocities from the first frame in ``reset_lib_state`` (finite differences of the
        reference). Sets ``frame_idx`` / ``frame_idx_f`` to ``0`` and initializes previous-velocity buffers.
    """
    prepare_reference_state(env, env_ids)
    objects = env.scene["objects"]
    if env_ids is None:
        env_ids = env.scene[env.ref_hands[0][0]]._ALL_INDICES
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    n = int(env_ids.shape[0])

    state = env.reset_lib_state[0].unsqueeze(0).expand(n, -1)
    num_hands = int(env.reset_hand_dim // 58)
    hands = state[:, : env.reset_hand_dim].view(n, num_hands, 58)
    hand_qpos = hands[..., :29]
    hand_qvel = hands[..., 29:]
    for name, joint_ids, h_idx in env.ref_hands:
        env.scene[name].write_joint_state_to_sim(
            hand_qpos[:, h_idx],
            hand_qvel[:, h_idx],
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
    if env.unused_joint_ids:
        qpos = env.unused_default_qpos.unsqueeze(0).expand(n, -1)
        env.scene["rby1a_sharpa"].write_joint_state_to_sim(
            qpos,
            torch.zeros_like(qpos),
            joint_ids=env.unused_joint_ids,
            env_ids=env_ids,
        )
    obj_state = state[:, env.reset_hand_dim :].view(n, env.num_objects, 13)
    obj_state_sim = obj_state.clone()
    obj_state_sim[..., 0:3] += env.scene.env_origins[env_ids].unsqueeze(1)
    objects.write_object_state_to_sim(obj_state_sim, env_ids=env_ids)
    env.frame_idx[env_ids] = 0
    env.frame_idx_f[env_ids] = 0.0
    env.prev_arm_qvel[env_ids] = hand_qvel[..., :7].reshape(n, -1)
    env.prev_finger_qvel[env_ids] = hand_qvel[..., 7:].reshape(n, -1)
    env.prev_object_lin_vel[env_ids] = obj_state[..., 7:10]
    env.prev_object_ang_vel[env_ids] = obj_state[..., 10:13]


def advance_reference_frame(env, env_ids):
    """Advance the per-env reference frame index after each environment step.

    Parameters
    ----------
    env : isaaclab.envs.ManagerBasedRLEnv
        Environment instance exposing ``frame_idx`` (num_envs,), ``reset_buf`` (num_envs,) and ``ref_len`` (int).
    env_ids : Sequence[int]
        Environment ids provided by the event manager (unused).

    Returns
    -------
    None
        Updates cached previous velocities and increments ``frame_idx`` for environments that were not reset in the
        current step.
    """
    env_ids = (env.reset_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if int(env_ids.numel()) == 0:
        return
    hands = torch.cat(
        [
            torch.cat(
                [
                    env.scene[name].data.joint_pos[env_ids][:, joint_ids],
                    env.scene[name].data.joint_vel[env_ids][:, joint_ids],
                ],
                dim=-1,
            )
            for name, joint_ids, _ in env.ref_hands
        ],
        dim=-1,
    )
    objects = env.scene["objects"]
    obj_pos = objects.data.object_pos_w[env_ids] - env.scene.env_origins[
        env_ids
    ].unsqueeze(1)
    obj = torch.cat(
        [
            obj_pos,
            objects.data.object_quat_w[env_ids],
            objects.data.object_lin_vel_w[env_ids],
            objects.data.object_ang_vel_w[env_ids],
        ],
        dim=-1,
    ).view(int(env_ids.shape[0]), -1)
    offs = env.frame_idx[env_ids] - env.episode_start_frame[env_ids]
    env.episode_states[env_ids, offs] = torch.cat([hands, obj], dim=-1)
    env.episode_filled[env_ids, offs] = True

    env.prev_arm_qvel[env_ids].copy_(
        torch.cat(
            [
                env.scene[name].data.joint_vel[env_ids][:, joint_ids[:7]]
                for name, joint_ids, _ in env.ref_hands
            ],
            dim=-1,
        )
    )
    env.prev_finger_qvel[env_ids].copy_(
        torch.cat(
            [
                env.scene[name].data.joint_vel[env_ids][:, joint_ids[7:]]
                for name, joint_ids, _ in env.ref_hands
            ],
            dim=-1,
        )
    )
    env.prev_object_lin_vel[env_ids].copy_(objects.data.object_lin_vel_w[env_ids])
    env.prev_object_ang_vel[env_ids].copy_(objects.data.object_ang_vel_w[env_ids])

    env.frame_idx_f[env_ids] = torch.clamp(
        env.frame_idx_f[env_ids] + env.ref_step_frames, max=env.ref_len - 1
    )
    env.frame_idx[env_ids] = env.frame_idx_f[env_ids].to(torch.long)

