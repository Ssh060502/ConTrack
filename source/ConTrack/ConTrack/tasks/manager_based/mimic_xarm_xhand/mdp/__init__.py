"""MDP terms for xArm+xHand HDF5 physics playback."""

from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.actions import (
    H5XarmXhandPlaybackAction,
    H5XarmXhandPlaybackActionCfg,
    XarmXhandResidualAction,
    XarmXhandResidualActionCfg,
)
from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.observations import (
    all_qpos,
    all_qvel,
    delayed_all_qpos,
    delayed_all_qvel,
    phase,
    ref_all_qpos,
    ref_xhand_contact,
    xhand_contact,
)
from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.rewards import (
    contact_distance_reward,
    contact_reward,
    qacc_penalty_arm,
    qacc_penalty_finger,
    qacc_penalty_obj_pos,
    qacc_penalty_obj_rot,
    qvel_penalty_arm,
    qvel_penalty_finger,
    qvel_penalty_obj_pos,
    qvel_penalty_obj_rot,
    tracking_arm,
    tracking_finger,
    tracking_obj_pos,
    tracking_obj_rot,
)
from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.domain_randomization import (
    apply_object_push,
    perturb_objects_xy,
    randomize_joint_pd_gains,
)
from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.state import (
    advance_reference_frame,
    init_reference_buffers,
    reset_reference_episode,
    reset_reference_episode_first_frame,
)
from ConTrack.tasks.manager_based.mimic_xarm_xhand.mdp.terminations import (
    obj_pose_exceeded,
    time_out,
)

__all__ = [
    "H5XarmXhandPlaybackAction",
    "H5XarmXhandPlaybackActionCfg",
    "XarmXhandResidualAction",
    "XarmXhandResidualActionCfg",
    "advance_reference_frame",
    "all_qpos",
    "apply_object_push",
    "all_qvel",
    "delayed_all_qpos",
    "delayed_all_qvel",
    "contact_distance_reward",
    "contact_reward",
    "init_reference_buffers",
    "obj_pose_exceeded",
    "phase",
    "perturb_objects_xy",
    "randomize_joint_pd_gains",
    "qacc_penalty_arm",
    "qacc_penalty_finger",
    "qacc_penalty_obj_pos",
    "qacc_penalty_obj_rot",
    "qvel_penalty_arm",
    "qvel_penalty_finger",
    "qvel_penalty_obj_pos",
    "qvel_penalty_obj_rot",
    "ref_all_qpos",
    "ref_xhand_contact",
    "reset_reference_episode",
    "reset_reference_episode_first_frame",
    "time_out",
    "tracking_arm",
    "tracking_finger",
    "tracking_obj_pos",
    "tracking_obj_rot",
    "xhand_contact",
]
