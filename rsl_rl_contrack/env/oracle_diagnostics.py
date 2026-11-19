"""Condition-specific rollout diagnostics; does not change rewards or actions."""
from __future__ import annotations
import torch


class OracleDiagnosticsWrapper:
    def __init__(self, env, conditions, condition_ids, arm_indices, finger_indices):
        self.env = env
        self.conditions = conditions
        self.labels = torch.as_tensor(condition_ids, device=env.device)
        self.arm_indices, self.finger_indices = arm_indices, finger_indices
        self.policy = None
        self.pending = []
        self.completed = []
        # task, style, penalty, environment reward and length; spans rollouts.
        self.episode_sums = torch.zeros((env.num_envs, 5), device=env.device)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def it_step_count(self):
        return self.env.it_step_count

    @it_step_count.setter
    def it_step_count(self, value):
        self.env.it_step_count = value

    def get_observations(self):
        return self.env.get_observations()

    def reset(self):
        self.episode_sums.zero_()
        self.pending.clear()
        self.completed.clear()
        return self.env.reset()

    @torch.no_grad()
    def step(self, actions):
        if self.policy is None:
            raise RuntimeError("Attach adapter policy before collecting training diagnostics")
        parts = self.policy.last_action_components
        base, correction, mean = (parts[k] for k in ("base", "correction", "mean"))
        metrics = {
            "correction_abs_mean": correction.abs().mean(-1),
            "correction_abs_peak": correction.abs().amax(-1),
            "correction_near_limit_fraction": (correction.abs() >= .95*self.policy.correction_scale).float().mean(-1),
            "base_boundary_fraction": (base.abs() >= .999).float().mean(-1),
            "mean_outside_bounds_fraction": (mean.abs() > 1.).float().mean(-1),
            "sample_outside_bounds_fraction": (actions.abs() > 1.).float().mean(-1),
            "sample_clip_abs_mean": (actions-actions.clamp(-1., 1.)).abs().mean(-1),
            "exploration_abs_mean": (actions-mean).abs().mean(-1),
        }
        for label, indices in (("arm", self.arm_indices), ("finger", self.finger_indices)):
            if indices:
                metrics[f"correction_{label}_abs_mean"] = correction[:, indices].abs().mean(-1)
        obs, rewards, dones, extra = self.env.step(actions)
        terms = torch.stack([extra["rg"], extra["rs"], extra["rp"], rewards,
                             torch.ones_like(rewards)], dim=-1)
        self.episode_sums += terms
        ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if ids.numel():
            self.completed.append((self.labels[ids].clone(), self.episode_sums[ids].clone()))
            self.episode_sums[ids] = 0.
        # Post-step data on done environments belongs to reset, not terminal.
        # Explicitly exclude it from tracking metrics (episode rewards above
        # still include the true final transition).
        raw = self.env.unwrapped
        objects = raw.scene["objects"]
        frame = raw.frame_idx.clamp(max=raw.ref_len-1)
        pos = objects.data.object_pos_w - raw.scene.env_origins.unsqueeze(1)
        pos_error = (pos-raw.ref_object_pos[frame]).norm(dim=-1).mean(-1)
        quat = objects.data.object_quat_w
        refquat = raw.ref_object_quat[frame]
        dot = (quat*refquat).sum(-1).abs().clamp(0., 1.)
        rot_error = (2.*torch.acos(dot)).mean(-1)
        valid = ~(dones > 0)
        metrics["position_error_m"] = pos_error
        metrics["rotation_error_rad"] = rot_error
        reference_elevated = ((raw.ref_object_pos[frame][..., 2]-raw.ref_object_pos[0][..., 2]) > .03).any(-1)
        metrics["carrying_phase_position_error_m"] = pos_error
        metrics["carrying_phase_rotation_error_rad"] = rot_error
        for key, value in metrics.items():
            mask = valid & reference_elevated if key.startswith("carrying_phase_") else valid if key in ("position_error_m", "rotation_error_rad") else None
            self.pending.append((key, value.detach(), mask))
        return obs, rewards, dones, extra

    @torch.no_grad()
    def drain_diagnostics(self):
        result = {}
        for cid, (mass, friction) in enumerate(self.conditions):
            prefix = f"Oracle/mass{mass:g}_friction{friction:g}"
            selected = self.labels == cid
            accum = {}
            for key, values, valid in self.pending:
                mask = selected if valid is None else selected & valid
                if key not in accum:
                    accum[key] = [torch.zeros((), device=self.device), torch.zeros((), device=self.device)]
                if key.endswith("_peak"):
                    accum[key][0] = torch.maximum(accum[key][0], values[mask].amax() if mask.any() else values.new_zeros(()))
                else:
                    accum[key][0] += values[mask].sum()
                accum[key][1] += mask.sum()
            for key, (total, count) in accum.items():
                if count.item():
                    result[f"{prefix}/{key}"] = total.item() if key.endswith("_peak") else (total/count).item()
            finished = [values[labels == cid] for labels, values in self.completed]
            finished = torch.cat(finished) if finished else self.episode_sums[:0]
            result[f"{prefix}/completed_episodes"] = len(finished)
            if len(finished):
                for i, key in enumerate(("task_return", "style_return", "penalty_return", "environment_return", "episode_steps")):
                    result[f"{prefix}/{key}"] = finished[:, i].mean().item()
                result[f"{prefix}/task_plus_penalty_return"] = (finished[:,0]+finished[:,2]).mean().item()
        self.pending.clear()
        self.completed.clear()
        return result
