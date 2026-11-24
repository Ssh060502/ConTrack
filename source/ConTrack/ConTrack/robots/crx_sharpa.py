"""Dual FANUC CRX-5iA + Sharpa Wave: one articulation holding both arms.

Asset: ``assets/usd/crx_sharpa/crx_sharpa.usd``, converted from
``assets/urdf/crx_sharpa.urdf`` (``scripts/tools/build_crx_sharpa_urdf.py``).
The two arm base poses are baked into that asset, so the articulation root
stays at the scene origin (the left arm base).

Hand actuation reuses the calibrated Sharpa values of the RB-Y1A config; it is
the same hand. Arm stiffness/damping are the RB-Y1A starting values, not an
identified model of the CRX servo. Arm effort and velocity limits are the
CRX-5iA URDF limits.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from ConTrack.robots import resolve_asset_path
from ConTrack.robots.rby1a_sharpa import (
    SHARPA_LEFT_JOINT_NAMES,
    SHARPA_RIGHT_JOINT_NAMES,
    _SHARPA_ARMATURE,
    _SHARPA_DAMPING,
    _SHARPA_EFFORT_LIMIT_SIM,
    _SHARPA_FRICTION,
    _SHARPA_STIFFNESS,
    _SHARPA_VELOCITY_LIMIT_SIM,
)

__all__ = [
    "CRX_SHARPA_CFG",
    "CRX_LEFT_JOINT_NAMES",
    "CRX_RIGHT_JOINT_NAMES",
    "SHARPA_LEFT_JOINT_NAMES",
    "SHARPA_RIGHT_JOINT_NAMES",
    "DEFAULT_CRX_QPOS",
    "ARM_DOF",
]

_USD_PATH = resolve_asset_path("usd", "crx_sharpa", "crx_sharpa.usd")

ARM_DOF = 6
CRX_LEFT_JOINT_NAMES = [f"left_J{i}" for i in range(1, 7)]
CRX_RIGHT_JOINT_NAMES = [f"right_J{i}" for i in range(1, 7)]

# dual_crx_control config/initial_pose.yaml, ROS/URDF convention (not pendant J3).
_DEFAULT_CRX_DEG = {
    "left": (0.0, 30.0, -30.0, 0.0, 60.0, 0.0),
    "right": (-90.0, -30.0, 210.0, 0.0, -60.0, 0.0),
}
DEFAULT_CRX_QPOS = {
    side: [math.radians(v) for v in values] for side, values in _DEFAULT_CRX_DEG.items()
}

# CRX-5iA URDF limits: J1-J3 160 N m, J4-J6 40 N m; 150/150/180/225/225/225 deg/s.
_CRX_EFFORT_LIMIT_SIM = {".*_J[1-3]": 160.0, ".*_J[4-6]": 40.0}
_CRX_VELOCITY_LIMIT_SIM = {
    ".*_J[1-2]": math.radians(150.0),
    ".*_J3": math.radians(180.0),
    ".*_J[4-6]": math.radians(225.0),
}

_DEFAULT_JOINT_STATE = {}
for _side, _names in (("left", CRX_LEFT_JOINT_NAMES), ("right", CRX_RIGHT_JOINT_NAMES)):
    _DEFAULT_JOINT_STATE.update(
        {name: value for name, value in zip(_names, DEFAULT_CRX_QPOS[_side], strict=True)}
    )
_DEFAULT_JOINT_STATE.update(
    {name: 0.0 for name in SHARPA_LEFT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES}
)

CRX_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_USD_PATH),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=_DEFAULT_JOINT_STATE,
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "crx": ImplicitActuatorCfg(
            joint_names_expr=["left_J[1-6]", "right_J[1-6]"],
            effort_limit_sim=_CRX_EFFORT_LIMIT_SIM,
            velocity_limit_sim=_CRX_VELOCITY_LIMIT_SIM,
            stiffness=1000.0,
            damping=50.0,
        ),
        "sharpa": ImplicitActuatorCfg(
            joint_names_expr=SHARPA_LEFT_JOINT_NAMES + SHARPA_RIGHT_JOINT_NAMES,
            effort_limit_sim=_SHARPA_EFFORT_LIMIT_SIM,
            stiffness=_SHARPA_STIFFNESS,
            damping=_SHARPA_DAMPING,
            armature=_SHARPA_ARMATURE,
            friction=_SHARPA_FRICTION,
            velocity_limit_sim=_SHARPA_VELOCITY_LIMIT_SIM,
        ),
    },
)
