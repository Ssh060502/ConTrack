from __future__ import annotations

from collections.abc import Callable
import hashlib
import os
from pathlib import Path

import h5py
import isaaclab.sim as sim_utils
import numpy as np
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from ConTrack.robots.xarm_xhand_left import XARM_XHAND_LEFT_CFG
from ConTrack.robots.xarm_xhand_right import XARM_XHAND_RIGHT_CFG
from ConTrack.tasks.manager_based.mimic_xarm_xhand import mdp

REPO_ROOT = Path(__file__).resolve().parents[6]
DATA_PATH = Path(
    os.environ.get(
        "CONTRACK_XARM_XHAND_DATA",
        REPO_ROOT / "data" / "dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5",
    )
).resolve()

NUM_ENVS = 64
ENV_SPACING = 2.0
DECIMATION = 2
SIM_DT = 1.0 / 120.0
TARGET_GRAVITY = (0.0, 0.0, -9.81)

ACTION_SCALE = (0.01, 0.25)
NUM_OBS_REFERENCE_POSE = 5

# Domain randomization
OBJECT_XY_RAND_RANGE = 0.0
JOINT_STIFFNESS_MULT_RANGE = (1.0, 1.0)
JOINT_DAMPING_MULT_RANGE = (1.0, 1.0)
POLICY_OBS_DELAY_STEPS_RANGE = (0, 0)
POLICY_ALL_QPOS_NOISE_STD = 0.0
POLICY_ALL_QVEL_NOISE_STD = 0.0

# Rewards
TRACKING_WEIGHT_ARM = 10.0
TRACKING_WEIGHT_FINGER = 10.0

TRACKING_WEIGHT_OBJ_POS = 500.0
TRACKING_WEIGHT_OBJ_ROT = 500.0
TRACKING_STD_OBJ_POS = 0.05
TRACKING_STD_OBJ_ROT = 0.5

CONTACT_REWARD_WEIGHT = 100.0
CONTACT_DISTANCE_WEIGHT = 100.0

QVEL_PENALTY_WEIGHT_ARM = 1.0
QVEL_PENALTY_WEIGHT_FINGER = 1.0
QVEL_PENALTY_WEIGHT_OBJ_POS = 1.0
QVEL_PENALTY_WEIGHT_OBJ_ROT = 0.1

QACC_PENALTY_WEIGHT_ARM = 10.0
QACC_PENALTY_WEIGHT_FINGER = 10.0
QACC_PENALTY_WEIGHT_OBJ_POS = 100.0
QACC_PENALTY_WEIGHT_OBJ_ROT = 10.0

# Termination
OBJECT_POS_THRESHOLD = 0.1
OBJECT_ROT_THRESHOLD = 1.0
OBJECT_POS_THRESHOLD_MIN = 0.05
OBJECT_ROT_THRESHOLD_MIN = 0.5
THRESHOLD_CURRICULUM_SUCCESS_EMA_ALPHA = 0.05
THRESHOLD_CURRICULUM_SR_LOW = 0.9
THRESHOLD_CURRICULUM_SR_HIGH = 0.999

# Physics
OBJECT_DENSITY = 800.0
# Absolute center-of-mass coordinate in the object's local mesh frame (meters). (0, 0, 0) means "let PhysX
# auto-compute it from uniform density + shape" (skips the override). Non-zero biases mass toward that point,
# e.g. toward a hammer head instead of the shape's natural (density-weighted) centroid.
OBJECT_COM_OFFSET = (-0.09, 0.0, 0.0)
OBJECT_CONTACT_OFFSET = 1e-2
OBJECT_REST_OFFSET = 0.0
USD_CACHE_DIR = (REPO_ROOT / "assets" / "usd_cache" / DATA_PATH.stem).resolve()
USD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_MAX_CONTACT_DATA_COUNT_PER_PRIM = 32
CONVEX_DECOMP_HULL_VERTEX_LIMIT = 32
CONVEX_DECOMP_MAX_CONVEX_HULLS = 8
CONVEX_DECOMP_VOXEL_RESOLUTION = 100000
CONVEX_DECOMP_ERROR_PERCENTAGE = 25.0


@clone
def spawn_ref_usd_mesh(
    prim_path: str,
    cfg: SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a cached USD mesh as an instanceable reference for object_tracks.

    Parameters
    ----------
    prim_path : str
        Prim path (or regex) to spawn the rigid body Xform, e.g. ``"{ENV_REGEX_NS}/object_0"``.
    cfg : SpawnerCfg
        Spawner configuration. Expected to have ``usd_path`` (str) pointing to a cached USD asset.
    translation : tuple[float, float, float] | None
        Initial translation applied to the Xform.
    orientation : tuple[float, float, float, float] | None
        Initial quaternion orientation as ``(w, x, y, z)``.

    Returns
    -------
    pxr.Usd.Prim
        The spawned root prim at ``prim_path``.
    """
    stage = get_current_stage()
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    sim_utils.standardize_xform_ops(
        prim, translation=translation, orientation=orientation, scale=(1.0, 1.0, 1.0)
    )
    prim.GetReferences().AddReference(str(cfg.usd_path))
    prim.SetInstanceable(True)
    rb = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb.CreateRigidBodyEnabledAttr().Set(True)
    rb.CreateKinematicEnabledAttr().Set(False)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateDensityAttr().Set(OBJECT_DENSITY)
    if OBJECT_COM_OFFSET != (0.0, 0.0, 0.0):
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*OBJECT_COM_OFFSET))
    return prim


@clone
def spawn_ref_articulation_joints(
    prim_path: str,
    cfg: SpawnerCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn UsdPhysics.RevoluteJoint prims for all HDF5 ``articulation_i_j`` pairs.

    Parameters
    ----------
    prim_path : str
        Prim path (or regex) to spawn an Xform container under each environment, e.g.
        ``"{ENV_REGEX_NS}/articulations"``.
    cfg : SpawnerCfg
        Spawner configuration (unused; joint parameters are cached at module load time).
    translation : tuple[float, float, float] | None
        Ignored.
    orientation : tuple[float, float, float, float] | None
        Ignored.

    Returns
    -------
    pxr.Usd.Prim
        The spawned container prim at ``prim_path`` holding revolute joints. Each joint uses per-clip angular limits
        (degrees) estimated from the reference motion, and disables collisions between its two linked bodies.
    """
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    sim_utils.standardize_xform_ops(root)
    for i, j, p0, p1, q0, q1, lower, upper in _articulation_joints:
        joint = UsdPhysics.RevoluteJoint.Define(
            stage, f"{prim_path}/articulation_{i}_{j}"
        )
        joint.CreateBody0Rel().SetTargets([f"../../object_{i}"])
        joint.CreateBody1Rel().SetTargets([f"../../object_{j}"])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(float(p0[0]), float(p0[1]), float(p0[2]))
        )
        joint.CreateLocalPos1Attr().Set(
            Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2]))
        )
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(float(q0[3]), Gf.Vec3f(float(q0[0]), float(q0[1]), float(q0[2])))
        )
        joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(float(q1[3]), Gf.Vec3f(float(q1[0]), float(q1[1]), float(q1[2])))
        )
        joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)
        joint.CreateLowerLimitAttr().Set(float(lower))
        joint.CreateUpperLimitAttr().Set(float(upper))
        joint.CreateCollisionEnabledAttr().Set(False)
    return root


@configclass
class UsdMeshCfg(SpawnerCfg):
    """Spawner cfg for a cached USD mesh rigid object.

    Parameters
    ----------
    usd_path : str
        Absolute path to a cached USD asset whose default prim is a rigid body Xform with a mesh child.
    """

    func: Callable = spawn_ref_usd_mesh
    usd_path: str = ""


with h5py.File(DATA_PATH, "r") as _f:
    _ref_len = int(_f["qpos"].shape[0])
    _fps_max = float(np.max(np.asarray(_f["fps"], np.float32)))
    _urdf_name = [s.decode("utf-8") for s in np.asarray(_f["urdf_name"])]
    _is_rhand = np.asarray(_f["is_rhand"], np.uint8).astype(bool)
    _has_right = bool(np.any(_is_rhand))
    _has_left = bool(np.any(~_is_rhand))
    _action_asset_name = "xarm_xhand_right" if bool(_is_rhand[0]) else "xarm_xhand_left"
    _base_t = np.asarray(_f["base_translation"], np.float32)
    _right_base_t = (
        _base_t[_is_rhand][0] if _has_right else np.zeros((3,), dtype=np.float32)
    )
    _left_base_t = (
        _base_t[~_is_rhand][0] if _has_left else np.zeros((3,), dtype=np.float32)
    )

    _obj_keys = list(_f["object_tracks"].keys())
    if all(k.isdigit() for k in _obj_keys):
        _obj_keys = sorted(_obj_keys, key=lambda s: int(s))
    _rigid_objects = {}
    for _k in _obj_keys:
        _g = _f["object_tracks"][_k]
        _v = np.asarray(_g["vertices"], np.float32)
        _tri = np.asarray(_g["faces"], np.uint32)
        _h = hashlib.sha1(
            _v.tobytes()
            + _tri.tobytes()
            + f"{OBJECT_DENSITY}_{OBJECT_CONTACT_OFFSET}_{OBJECT_REST_OFFSET}".encode()
            + f"_{CONVEX_DECOMP_HULL_VERTEX_LIMIT}_{CONVEX_DECOMP_MAX_CONVEX_HULLS}_{CONVEX_DECOMP_VOXEL_RESOLUTION}_{CONVEX_DECOMP_ERROR_PERCENTAGE}".encode()
            + f"_{OBJECT_COM_OFFSET}".encode()
        )
        _usd_path = USD_CACHE_DIR / f"{_h.hexdigest()[:16]}.usd"
        if not _usd_path.exists():
            _stage = Usd.Stage.CreateNew(str(_usd_path))
            _stage.SetMetadata("metersPerUnit", 1.0)
            _stage.SetMetadata("upAxis", "Z")
            _root = UsdGeom.Xform.Define(_stage, "/Object").GetPrim()
            _stage.SetDefaultPrim(_root)
            _mesh = UsdGeom.Mesh.Define(_stage, "/Object/mesh")
            _mesh.CreatePointsAttr([Gf.Vec3f(*v) for v in _v.tolist()])
            _mesh.CreateFaceVertexCountsAttr([3] * int(_tri.shape[0]))
            _mesh.CreateFaceVertexIndicesAttr(_tri.reshape(-1).tolist())
            _mesh.CreateSubdivisionSchemeAttr("none")
            _rb = UsdPhysics.RigidBodyAPI.Apply(_root)
            _rb.CreateRigidBodyEnabledAttr().Set(True)
            _rb.CreateKinematicEnabledAttr().Set(False)
            _mass_api = UsdPhysics.MassAPI.Apply(_root)
            _mass_api.CreateDensityAttr().Set(OBJECT_DENSITY)
            if OBJECT_COM_OFFSET != (0.0, 0.0, 0.0):
                _mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*OBJECT_COM_OFFSET))
            _c = UsdPhysics.CollisionAPI.Apply(_mesh.GetPrim())
            _c.CreateCollisionEnabledAttr().Set(True)
            _pc = PhysxSchema.PhysxCollisionAPI.Apply(_mesh.GetPrim())
            _pc.CreateContactOffsetAttr().Set(OBJECT_CONTACT_OFFSET)
            _pc.CreateRestOffsetAttr().Set(OBJECT_REST_OFFSET)
            UsdPhysics.MeshCollisionAPI.Apply(
                _mesh.GetPrim()
            ).CreateApproximationAttr().Set("convexDecomposition")
            _cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(
                _mesh.GetPrim()
            )
            sim_utils.safe_set_attribute_on_usd_schema(
                _cd, "hull_vertex_limit", CONVEX_DECOMP_HULL_VERTEX_LIMIT, True
            )
            sim_utils.safe_set_attribute_on_usd_schema(
                _cd, "max_convex_hulls", CONVEX_DECOMP_MAX_CONVEX_HULLS, True
            )
            sim_utils.safe_set_attribute_on_usd_schema(
                _cd, "voxel_resolution", CONVEX_DECOMP_VOXEL_RESOLUTION, True
            )
            sim_utils.safe_set_attribute_on_usd_schema(
                _cd, "error_percentage", CONVEX_DECOMP_ERROR_PERCENTAGE, True
            )
            _stage.GetRootLayer().Save()
        _t0 = np.asarray(_g["translations"][0], np.float32)
        _q0 = np.asarray(_g["orientations_xyzw"][0], np.float32)
        _rigid_objects[f"object_{_k}"] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/object_{_k}",
            spawn=UsdMeshCfg(usd_path=str(_usd_path)),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(float(_t0[0]), float(_t0[1]), float(_t0[2])),
                rot=(float(_q0[3]), float(_q0[0]), float(_q0[1]), float(_q0[2])),
            ),
        )
    _articulation_joints = []
    for _i, _j in [
        tuple(map(int, k.split("_")[1:]))
        for k in _f.keys()
        if k.startswith("articulation_")
    ]:
        _g0, _g1 = _f["object_tracks"][str(_i)], _f["object_tracks"][str(_j)]
        _t0 = np.asarray(_g0["translations"], np.float32)
        _t1 = np.asarray(_g1["translations"], np.float32)
        _q0 = np.asarray(_g0["orientations_xyzw"], np.float32)
        _q1 = np.asarray(_g1["orientations_xyzw"], np.float32)
        _r0 = R.from_quat(_q0).as_matrix().astype(np.float32)
        _r1 = R.from_quat(_q1).as_matrix().astype(np.float32)
        _r_rel = np.matmul(_r0.transpose(0, 2, 1), _r1)
        _p_rel = np.matmul(
            _r0.transpose(0, 2, 1), (_t1 - _t0).astype(np.float32)[..., None]
        )[..., 0]
        _a = np.concatenate(
            [np.tile(np.eye(3, dtype=np.float32), (_p_rel.shape[0], 1, 1)), -_r_rel],
            axis=2,
        ).reshape(-1, 6)
        _x_y = np.linalg.lstsq(_a, _p_rel.reshape(-1), rcond=None)[0].astype(np.float32)
        _rotvec = R.from_matrix(_r_rel).as_rotvec().astype(np.float32)
        _axis0 = _rotvec[np.argmax(np.linalg.norm(_rotvec, axis=1))].astype(np.float32)
        _axis0 /= np.linalg.norm(_axis0)
        _axis_w = _r0[0] @ _axis0
        _axis_w /= np.linalg.norm(_axis_w)
        _tmp = (
            np.array([1.0, 0.0, 0.0], np.float32)
            if abs(float(_axis_w[0])) < 0.9
            else np.array([0.0, 1.0, 0.0], np.float32)
        )
        _x_w = np.cross(_tmp, _axis_w)
        _x_w /= np.linalg.norm(_x_w)
        _r_joint_w = np.stack([_x_w, np.cross(_axis_w, _x_w), _axis_w], axis=1).astype(
            np.float32
        )
        _q0_local = R.from_matrix(_r0[0].T @ _r_joint_w).as_quat().astype(np.float32)
        _q1_local = R.from_matrix(_r1[0].T @ _r_joint_w).as_quat().astype(np.float32)
        _lim = (_rotvec @ _axis0) * (180.0 / np.pi)
        _articulation_joints.append(
            (
                _i,
                _j,
                _x_y[:3],
                _x_y[3:],
                _q0_local,
                _q1_local,
                float(np.min(_lim)),
                float(np.max(_lim)),
            )
        )

CONTACT_FILTER_EXPR = [f"{{ENV_REGEX_NS}}/object_{k}" for k in _obj_keys]
_XHAND_CONTACT_PRIM_SUFFIX = (
    ("index_link1", "index_rota_link1"),
    ("index_link2", "index_rota_link2"),
    ("middle_link1", "mid_link1"),
    ("middle_link2", "mid_link2"),
    ("ring_link1", "ring_link1"),
    ("ring_link2", "ring_link2"),
    ("pinky_link1", "pinky_link1"),
    ("pinky_link2", "pinky_link2"),
    ("thumb_link1", "thumb_rota_link1"),
    ("thumb_link2", "thumb_rota_link2"),
)


@configclass
class XarmXhandSceneCfg(InteractiveSceneCfg):
    """Scene with ground, object tracks and the xArm+xHand present in the reference file."""

    ground = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0))
    )
    xarm_xhand_right: ArticulationCfg | None = (
        XARM_XHAND_RIGHT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/XarmXhandRight",
            spawn=XARM_XHAND_RIGHT_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=XARM_XHAND_RIGHT_CFG.init_state.replace(
                pos=(
                    float(_right_base_t[0]),
                    float(_right_base_t[1]),
                    float(_right_base_t[2]),
                )
            ),
        )
        if _has_right
        else None
    )
    xarm_xhand_left: ArticulationCfg | None = (
        XARM_XHAND_LEFT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/XarmXhandLeft",
            spawn=XARM_XHAND_LEFT_CFG.spawn.replace(activate_contact_sensors=True),
            init_state=XARM_XHAND_LEFT_CFG.init_state.replace(
                pos=(
                    float(_left_base_t[0]),
                    float(_left_base_t[1]),
                    float(_left_base_t[2]),
                )
            ),
        )
        if _has_left
        else None
    )
    objects: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects=_rigid_objects
    )
    articulations: AssetBaseCfg | None = (
        AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/articulations",
            spawn=SpawnerCfg(func=spawn_ref_articulation_joints),
        )
        if _articulation_joints
        else None
    )
    for _side, _hand, _has in (
        ("right", "XarmXhandRight", _has_right),
        ("left", "XarmXhandLeft", _has_left),
    ):
        for _k, _suffix in _XHAND_CONTACT_PRIM_SUFFIX:
            __annotations__[f"contact_{_side}_{_k}"] = ContactSensorCfg | None
            locals()[f"contact_{_side}_{_k}"] = (
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/{_hand}/{_side}_hand_{_suffix}",
                    track_contact_points=True,
                    filter_prim_paths_expr=CONTACT_FILTER_EXPR,
                    max_contact_data_count_per_prim=CONTACT_MAX_CONTACT_DATA_COUNT_PER_PRIM,
                )
                if _has
                else None
            )
    del _side, _hand, _has, _k, _suffix
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1800.0),
    )


@configclass
class ActionsCfg:
    """Residual actions around the current reference qpos."""

    residual = mdp.XarmXhandResidualActionCfg(
        class_type=mdp.XarmXhandResidualAction,
        asset_name=_action_asset_name,
        reference_path=str(DATA_PATH),
        action_scale=ACTION_SCALE,
    )


@configclass
class ActionsPlaybackCfg:
    """Playback actions that ignore policy input."""

    playback = mdp.H5XarmXhandPlaybackActionCfg(
        class_type=mdp.H5XarmXhandPlaybackAction,
        asset_name=_action_asset_name,
        reference_path=str(DATA_PATH),
    )


@configclass
class ObservationsCfg:
    """Observation groups for asymmetric actor-critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations."""

        all_qpos = ObsTerm(
            func=mdp.delayed_all_qpos,
            params={"delay_range": POLICY_OBS_DELAY_STEPS_RANGE},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=POLICY_ALL_QPOS_NOISE_STD),
        )
        all_qvel = ObsTerm(
            func=mdp.delayed_all_qvel,
            params={"delay_range": POLICY_OBS_DELAY_STEPS_RANGE},
            noise=AdditiveGaussianNoiseCfg(mean=0.0, std=POLICY_ALL_QVEL_NOISE_STD),
        )
        xhand_contact = ObsTerm(func=mdp.xhand_contact)
        ref_all_qpos = ObsTerm(
            func=mdp.ref_all_qpos, params={"num_future": NUM_OBS_REFERENCE_POSE}
        )
        ref_xhand_contact = ObsTerm(func=mdp.ref_xhand_contact)
        phase = ObsTerm(func=mdp.phase)
        enable_corruption = True
        concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations."""

        all_qpos = ObsTerm(func=mdp.all_qpos)
        all_qvel = ObsTerm(func=mdp.all_qvel)
        xhand_contact = ObsTerm(func=mdp.xhand_contact)
        ref_all_qpos = ObsTerm(
            func=mdp.ref_all_qpos, params={"num_future": NUM_OBS_REFERENCE_POSE}
        )
        ref_xhand_contact = ObsTerm(func=mdp.ref_xhand_contact)
        phase = ObsTerm(func=mdp.phase)
        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Startup, reset and per-step frame advance."""

    startup_cache = EventTerm(func=mdp.init_reference_buffers, mode="startup")
    reset_to_demo = EventTerm(func=mdp.reset_reference_episode, mode="reset")
    object_xy_rand = EventTerm(
        func=mdp.perturb_objects_xy,
        mode="reset",
        params={"xy_range": OBJECT_XY_RAND_RANGE},
    )
    joint_pd_rand = EventTerm(
        func=mdp.randomize_joint_pd_gains,
        mode="reset",
        params={
            "stiffness_mult_range": JOINT_STIFFNESS_MULT_RANGE,
            "damping_mult_range": JOINT_DAMPING_MULT_RANGE,
        },
    )
    advance_frame = EventTerm(
        func=mdp.advance_reference_frame, mode="interval", interval_range_s=(0.0, 0.0)
    )


@configclass
class EventCfgFirstFrame(EventCfg):
    """Reset variant that always starts tracking at frame zero."""

    reset_to_demo = EventTerm(
        func=mdp.reset_reference_episode_first_frame, mode="reset"
    )


@configclass
class RewardsCfg:
    """Dense tracking rewards and motion penalties."""

    tracking_arm = RewTerm(func=mdp.tracking_arm, weight=TRACKING_WEIGHT_ARM)
    tracking_finger = RewTerm(func=mdp.tracking_finger, weight=TRACKING_WEIGHT_FINGER)
    tracking_obj_pos = RewTerm(
        func=mdp.tracking_obj_pos,
        weight=TRACKING_WEIGHT_OBJ_POS,
        params={"std": TRACKING_STD_OBJ_POS},
    )
    tracking_obj_rot = RewTerm(
        func=mdp.tracking_obj_rot,
        weight=TRACKING_WEIGHT_OBJ_ROT,
        params={"std": TRACKING_STD_OBJ_ROT},
    )

    qvel_penalty_arm = RewTerm(
        func=mdp.qvel_penalty_arm, weight=-QVEL_PENALTY_WEIGHT_ARM
    )
    qvel_penalty_finger = RewTerm(
        func=mdp.qvel_penalty_finger, weight=-QVEL_PENALTY_WEIGHT_FINGER
    )
    qvel_penalty_obj_pos = RewTerm(
        func=mdp.qvel_penalty_obj_pos, weight=-QVEL_PENALTY_WEIGHT_OBJ_POS
    )
    qvel_penalty_obj_rot = RewTerm(
        func=mdp.qvel_penalty_obj_rot, weight=-QVEL_PENALTY_WEIGHT_OBJ_ROT
    )

    qacc_penalty_arm = RewTerm(
        func=mdp.qacc_penalty_arm, weight=-QACC_PENALTY_WEIGHT_ARM
    )
    qacc_penalty_finger = RewTerm(
        func=mdp.qacc_penalty_finger, weight=-QACC_PENALTY_WEIGHT_FINGER
    )
    qacc_penalty_obj_pos = RewTerm(
        func=mdp.qacc_penalty_obj_pos, weight=-QACC_PENALTY_WEIGHT_OBJ_POS
    )
    qacc_penalty_obj_rot = RewTerm(
        func=mdp.qacc_penalty_obj_rot, weight=-QACC_PENALTY_WEIGHT_OBJ_ROT
    )
    contact_reward = RewTerm(func=mdp.contact_reward, weight=CONTACT_REWARD_WEIGHT)
    contact_distance_reward = RewTerm(
        func=mdp.contact_distance_reward, weight=CONTACT_DISTANCE_WEIGHT
    )


@configclass
class TerminationsCfg:
    """Terminate at the end of the HDF5 clip."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    pose_break = DoneTerm(func=mdp.obj_pose_exceeded)


@configclass
class XarmXhandEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based env that tracks a reference with residual actions in physics."""

    scene: XarmXhandSceneCfg = XarmXhandSceneCfg(
        num_envs=NUM_ENVS, env_spacing=ENV_SPACING, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    reference_path: str = str(DATA_PATH)
    threshold_curriculum_enabled: bool = True
    object_pos_threshold_max: float = OBJECT_POS_THRESHOLD
    object_rot_threshold_max: float = OBJECT_ROT_THRESHOLD
    object_pos_threshold_min: float = OBJECT_POS_THRESHOLD_MIN
    object_rot_threshold_min: float = OBJECT_ROT_THRESHOLD_MIN
    threshold_curriculum_success_ema_alpha: float = (
        THRESHOLD_CURRICULUM_SUCCESS_EMA_ALPHA
    )
    threshold_curriculum_sr_low: float = THRESHOLD_CURRICULUM_SR_LOW
    threshold_curriculum_sr_high: float = THRESHOLD_CURRICULUM_SR_HIGH
    cmdp_reward_groups: dict[str, list[str]] = {
        "rg": ["tracking_obj_pos", "tracking_obj_rot"],
        "rp": [
            "qvel_penalty_arm",
            "qvel_penalty_finger",
            "qvel_penalty_obj_pos",
            "qvel_penalty_obj_rot",
            "qacc_penalty_arm",
            "qacc_penalty_finger",
            "qacc_penalty_obj_pos",
            "qacc_penalty_obj_rot",
        ],
        "rs": [
            "tracking_arm",
            "tracking_finger",
            "contact_reward",
            "contact_distance_reward",
        ],
    }

    def __post_init__(self):
        """Set simulation parameters for playback.

        Parameters
        ----------
        self : XarmXhandEnvCfg
            Environment configuration instance.

        Returns
        -------
        None
            Updates time-step, gravity, friction, PhysX GPU buffers, and viewer settings in-place.
        """
        self.decimation = DECIMATION
        self.episode_length_s = float(_ref_len) / _fps_max
        self.sim.dt = SIM_DT
        self.sim.gravity = TARGET_GRAVITY
        self.sim.render_interval = DECIMATION
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        )
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.min_position_iteration_count = 2
        self.sim.physx.min_velocity_iteration_count = 2
        self.sim.physx.enable_stabilization = True
        self.sim.physx.solve_articulation_contact_last = True
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
        self.viewer.eye = (2.2, 2.2, 1.3)


@configclass
class XarmXhandPlayEnvCfg(XarmXhandEnvCfg):
    """Play-only env that always starts tracking at the first reference frame."""

    events: EventCfgFirstFrame = EventCfgFirstFrame()
