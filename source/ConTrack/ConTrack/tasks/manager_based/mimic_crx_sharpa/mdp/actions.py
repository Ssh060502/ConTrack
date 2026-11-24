from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ConTrack.tasks.manager_based.mimic_crx_sharpa.mdp.state import (
    prepare_reference_state,
)


@configclass
class H5CrxSharpaPlaybackActionCfg(ActionTermCfg):
    """Replay an CRX-5iA+Sharpa HDF5 clip as joint position targets.

    Parameters
    ----------
    reference_path : str
        Absolute path to the HDF5 file containing ``qpos`` (T, H, 28) and object tracks.
    """

    class_type: type[ActionTerm] = None
    reference_path: str = ""


class H5CrxSharpaPlaybackAction(ActionTerm):
    """Set joint position targets from the HDF5 reference and ignore policy actions."""

    cfg: H5CrxSharpaPlaybackActionCfg

    def __init__(self, cfg: H5CrxSharpaPlaybackActionCfg, env):
        """Initialize the playback term and cache joint indices.

        Parameters
        ----------
        cfg : H5CrxSharpaPlaybackActionCfg
            Action term configuration.
        env : isaaclab.envs.ManagerBasedEnv
            Environment instance providing the scene and device.

        Returns
        -------
        None
            Initializes internal buffers in-place.
        """
        super().__init__(cfg, env)
        env.cfg.reference_path = cfg.reference_path
        prepare_reference_state(env, None)
        self._hands: list[tuple[Articulation, list[int], int]] = [
            (env.scene[name], joint_ids, h) for name, joint_ids, h in env.ref_hands
        ]
        self._raw_actions = torch.zeros((env.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros((env.num_envs, 0), device=self.device)
        self._targets = [
            torch.zeros((env.num_envs, len(joint_ids)), device=self.device)
            for _, joint_ids, _ in self._hands
        ]

    @property
    def action_dim(self) -> int:
        """Return the action dimension.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Zero, since this term ignores policy actions.
        """
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        """Return raw actions.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor, shape=(num_envs, 0), dtype=float32
            Raw action buffer (always empty).
        """
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Return processed actions.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor, shape=(num_envs, 0), dtype=float32
            Processed action buffer (always empty).
        """
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Compute per-env joint targets from the current reference frame.

        Parameters
        ----------
        actions : torch.Tensor, shape=(num_envs, 0), dtype=float32
            Ignored.

        Returns
        -------
        None
            Updates cached target tensors in-place.
        """
        self._raw_actions = actions
        env = self._env
        idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
        qpos = env.ref_qpos[idx]
        for i, (_, _, h) in enumerate(self._hands):
            self._targets[i] = qpos[:, h]

    def apply_actions(self):
        """Write position targets to all arms/hands present in the HDF5.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Calls ``set_joint_position_target`` on the CRX-5iA+Sharpa articulation.
        """
        env = self._env
        for (hand, joint_ids, _), targets in zip(self._hands, self._targets):
            hand.set_joint_position_target(targets, joint_ids=joint_ids)
        if env.unused_joint_ids:
            env.scene["crx_sharpa"].set_joint_position_target(
                env.unused_default_qpos.unsqueeze(0).expand(env.num_envs, -1),
                joint_ids=env.unused_joint_ids,
            )

    def reset(self, env_ids=None):
        """Reset internal target buffers for selected environments.

        Parameters
        ----------
        env_ids : Sequence[int] | None
            Environment ids to reset; ``None`` targets all.

        Returns
        -------
        None
            Zeros the cached target tensors for the selected indices.
        """
        if env_ids is None:
            env_ids = self._asset._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for targets in self._targets:
            targets[env_ids] = 0.0


H5CrxSharpaPlaybackActionCfg.class_type = H5CrxSharpaPlaybackAction


@configclass
class CrxSharpaResidualActionCfg(ActionTermCfg):
    """Residual joint targets around the current reference pose.

    Parameters
    ----------
    reference_path : str
        Absolute path to the reference file used to determine which hands and joints exist.
    action_scale : tuple[float, float]
        Per-joint action scales ``(arm, finger)`` applied to policy actions before adding to the reference joint
        positions.
    """

    class_type: type[ActionTerm] = None
    reference_path: str = ""
    action_scale: tuple[float, float] = (1.0, 1.0)


class CrxSharpaResidualAction(ActionTerm):
    """Set joint position targets as ``ref_qpos_t + action`` for each hand in the clip."""

    cfg: CrxSharpaResidualActionCfg

    def __init__(self, cfg: CrxSharpaResidualActionCfg, env):
        """Initialize residual action term and cache joint indices.

        Parameters
        ----------
        cfg : CrxSharpaResidualActionCfg
            Action term configuration.
        env : isaaclab.envs.ManagerBasedEnv
            Environment instance providing the scene and device.

        Returns
        -------
        None
            Initializes internal buffers in-place.
        """
        super().__init__(cfg, env)
        env.cfg.reference_path = cfg.reference_path
        prepare_reference_state(env, None)
        self._hands: list[tuple[Articulation, list[int], int]] = [
            (env.scene[name], joint_ids, h) for name, joint_ids, h in env.ref_hands
        ]
        self._sizes = [len(joint_ids) for _, joint_ids, _ in env.ref_hands]
        arm_s, finger_s = (float(x) for x in self.cfg.action_scale)
        self._action_scales = [
            torch.tensor([arm_s] * 6 + [finger_s] * 22, device=self.device)
            for _ in self._sizes
        ]
        self._raw_actions = torch.zeros(
            (env.num_envs, sum(self._sizes)), device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._targets = [
            torch.zeros((env.num_envs, n), device=self.device) for n in self._sizes
        ]

    @property
    def action_dim(self) -> int:
        """Return action dimension.

        Parameters
        ----------
        None

        Returns
        -------
        int
            Total number of controlled joints across all hands in the clip.
        """
        return int(self._raw_actions.shape[1])

    @property
    def raw_actions(self) -> torch.Tensor:
        """Return raw action tensor.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor, shape=(num_envs, action_dim), dtype=float32
            Raw action buffer in ``[-1, 1]``.
        """
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Return processed joint targets flattened across hands.

        Parameters
        ----------
        None

        Returns
        -------
        torch.Tensor, shape=(num_envs, action_dim), dtype=float32
            Joint position targets in the same order as actions.
        """
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Compute joint position targets from the current reference frame and residual actions.

        Parameters
        ----------
        actions : torch.Tensor, shape=(num_envs, action_dim), dtype=float32
            Residual joint deltas added to the reference joint positions.

        Returns
        -------
        None
            Updates cached target tensors in-place.
        """
        env = self._env
        actions = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions = actions
        idx = torch.clamp(env.frame_idx, max=env.ref_len - 1)
        qpos = env.ref_qpos[idx]
        start = 0
        for i, (_, joint_ids, h) in enumerate(self._hands):
            end = start + len(joint_ids)
            target = qpos[:, h] + actions[:, start:end] * self._action_scales[i]
            self._targets[i] = target
            self._processed_actions[:, start:end] = target
            start = end

    def apply_actions(self):
        """Write computed position targets to all hands in the clip.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Calls ``set_joint_position_target`` on the CRX-5iA+Sharpa articulation.
        """
        env = self._env
        for (hand, joint_ids, _), targets in zip(self._hands, self._targets):
            hand.set_joint_position_target(targets, joint_ids=joint_ids)
        if env.unused_joint_ids:
            env.scene["crx_sharpa"].set_joint_position_target(
                env.unused_default_qpos.unsqueeze(0).expand(env.num_envs, -1),
                joint_ids=env.unused_joint_ids,
            )

    def reset(self, env_ids=None):
        """Reset internal buffers for selected environments.

        Parameters
        ----------
        env_ids : Sequence[int] | None
            Environment ids to reset; ``None`` targets all.

        Returns
        -------
        None
            Zeros action and target buffers for the selected indices.
        """
        if env_ids is None:
            env_ids = self._asset._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        for targets in self._targets:
            targets[env_ids] = 0.0


CrxSharpaResidualActionCfg.class_type = CrxSharpaResidualAction

