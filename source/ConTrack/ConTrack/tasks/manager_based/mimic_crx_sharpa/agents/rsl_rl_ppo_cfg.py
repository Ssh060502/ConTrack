# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoAlgorithmCmdpCfg(RslRlPpoAlgorithmCfg):
    cmdp_alpha = 0.9
    cmdp_lambda_lr = 5e-2
    cmdp_warmup_iters = 0
    cmdp_vg_window = 100
    cmdp_vg_star_window = 20
    cmdp_lambda_init = 0.0


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 100_000
    save_interval = 1_000
    video_length = 500
    experiment_name = "CrxSharpa-Mimic"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCmdpCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=10,
        num_mini_batches=16,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        cmdp_alpha=0.9,
        cmdp_lambda_lr=5e-2,
        cmdp_warmup_iters=0,
        cmdp_vg_window=100,
        cmdp_vg_star_window=20,
        cmdp_lambda_init=0.0,
    )

