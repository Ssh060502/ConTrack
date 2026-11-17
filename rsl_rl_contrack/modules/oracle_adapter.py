"""Frozen ConTrack actor with a physical-parameter-conditioned residual policy."""
from __future__ import annotations
import copy
import hashlib
import math
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl_contrack.modules.actor_critic import ActorCritic


def frozen_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class OracleAdapterActorCritic(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions, *, base_checkpoint,
                 base_policy_cfg, base_obs_groups, correction_scale=.5,
                 init_noise_std=.05, **kwargs):
        if not math.isfinite(correction_scale) or correction_scale <= 0:
            raise ValueError("correction_scale must be finite and positive")
        if init_noise_std <= 0 or not math.isfinite(init_noise_std):
            raise ValueError("init_noise_std must be finite and positive")
        super().__init__(obs, obs_groups, num_actions, init_noise_std=init_noise_std,
                         noise_std_type="log", state_dependent_std=False, **kwargs)
        self.correction_scale = float(correction_scale)
        self.base = ActorCritic(obs, copy.deepcopy(base_obs_groups), num_actions,
                                **copy.deepcopy(base_policy_cfg))
        checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        self.base.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.base.requires_grad_(False)
        self.base.eval()
        last = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "base"):
            self.base.eval()
        return self

    def correction(self, obs):
        value = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.correction_scale * torch.tanh(self.actor(value))

    def mean_action(self, obs):
        with torch.no_grad():
            base_action = self.base.act_inference(obs)
        return base_action + self.correction(obs)

    def act(self, obs, **kwargs):
        mean = self.mean_action(obs)
        std = self.log_std.clamp(-10., 0.).exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        # PPO stores/log-probs the unbounded Gaussian sample. Existing action
        # clipping remains in the environment, as in the original implementation.
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.mean_action(obs).clamp(-1., 1.)

    def evaluate(self, obs, **kwargs):
        # PPO passes recurrent placeholder kwargs even to feed-forward policies.
        return super().evaluate(obs)
