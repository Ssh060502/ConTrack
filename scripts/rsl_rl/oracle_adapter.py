"""Train/evaluate a frozen-base oracle adapter; run inside env_isaaclab."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback
import uuid

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "source" / "ConTrack"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
parser.add_argument("--session", default=str(repo / "logs/rsl_rl/sharpa_mug_sweep/grab-s4_mug_pass_1-rby1a_sharpa-2026_10_14-03_57_03-sharpa_mug_sweep_trial"))
parser.add_argument("--base_checkpoint", default=None)
parser.add_argument("--data", default="/mnt/ssd1/shenghe/datasets/GRAB_contrack/grab-s4_mug_pass_1-rby1a_sharpa.h5")
parser.add_argument("--conditions", default="1:2,3:2,6:2,1:0.7,1:0.2")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--rollout_steps", type=int, default=32)
parser.add_argument("--correction_scale", type=float, default=.5)
parser.add_argument("--policy", choices=("adapter", "base", "zero"), default="adapter")
parser.add_argument("--adapter_checkpoint", default=None, help="Resume training or evaluate this adapter checkpoint")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--video", action="store_true")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
parser.add_argument("--project", default="ConTrack-adapter")
parser.add_argument("--output_root", default="logs/oracle_adapter")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

session = Path(args.session).expanduser().resolve()
base_checkpoint = Path(args.base_checkpoint).expanduser().resolve() if args.base_checkpoint else session / "model_1500.pt"
data = Path(args.data).expanduser().resolve()
for path in (session/"params/agent.yaml", base_checkpoint, data):
    if not path.is_file():
        parser.error(f"Required file missing: {path}")
if min(args.num_envs, args.max_iterations, args.save_interval, args.rollout_steps, args.episodes) <= 0:
    parser.error("Environment/iteration/episode counts must be positive")
if args.mode == "train" and (args.policy != "adapter" or args.video):
    parser.error("Train uses --policy adapter without --video")
if args.mode == "evaluate" and args.num_envs != 1:
    parser.error("Evaluation requires --num_envs 1 and exactly one condition")
if args.mode == "evaluate" and args.policy == "adapter" and not args.adapter_checkpoint:
    parser.error("Use --adapter_checkpoint, or --policy base/zero for initial checks")
if args.policy in ("base", "zero") and args.adapter_checkpoint:
    parser.error("base/zero checks must not load an adapter checkpoint")
os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = str(data)
# Establish the original nominal mass before per-environment scaling. Do not let
# shell variables silently change both the nominal mass and the scale labels.
os.environ.pop("CONTRACK_SHARPA_OBJECT_MASS_SCALE", None)
os.environ.pop("CONTRACK_SHARPA_OBJECT_FRICTION", None)
args.enable_cameras = bool(args.video)
launcher = AppLauncher(args)

import gymnasium as gym
import torch
import yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import isaaclab_tasks
import ConTrack.tasks
from ConTrack.tasks.manager_based.mimic_rby1a_sharpa import mdp
from rsl_rl_contrack.env.cons_mimic_reward_wrapper import ConsMimicRewardWrapper
from rsl_rl_contrack.env.oracle_physics import parse_conditions, configure_physics, OracleContextWrapper
from rsl_rl_contrack.modules.oracle_adapter import OracleAdapterActorCritic, frozen_digest
from rsl_rl_contrack.runners import on_policy_runner as runner_module
from rsl_rl_contrack.utils import resolve_obs_groups
from scripts.tools.trajectory_collection import diagnostic_pose_term, collect


def never_pose_break(env):
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


class AdapterRunner(runner_module.OnPolicyRunner):
    def save(self, path, infos=None):
        if frozen_digest(self.alg.policy.base) != self.adapter_info["frozen_base_digest"]:
            raise RuntimeError("Frozen base changed before checkpoint save")
        return super().save(path, infos=infos or {"oracle_adapter":self.adapter_info})


def main():
    conditions = parse_conditions(args.conditions)
    if args.mode == "evaluate" and len(conditions) != 1:
        raise ValueError("Evaluate one condition at a time")
    task = "Isaac-Rby1a-Sharpa-Mimic-Manager-v0"
    device = args.device or "cuda:0"
    cfg = parse_env_cfg(task, device=device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.events.reset_to_demo.func = mdp.reset_reference_episode_first_frame
    cfg.threshold_curriculum_enabled = False
    cfg.terminations.pose_break.func = (diagnostic_pose_term(mdp.obj_pose_exceeded, "log-only")
                                      if args.mode == "evaluate" else never_pose_break)
    # Original self-collisions retained; no warm-up or discarded first episode.
    raw = gym.make(task, cfg=cfg, render_mode="rgb_array" if args.video else None)
    try:
        original_cfg = yaml.safe_load((session/"params/agent.yaml").read_text())
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=original_cfg.get("clip_actions"))
        context, physics = configure_physics(raw.unwrapped, conditions)
        env = OracleContextWrapper(ConsMimicRewardWrapper(wrapped, raw.unwrapped), context)
        env.reset()
        obs = env.get_observations()
        base_groups = resolve_obs_groups(obs, copy.deepcopy(original_cfg.get("obs_groups") or {}), ["critic"])
        policy_cfg = copy.deepcopy(original_cfg["policy"])
        if policy_cfg.pop("class_name") != "ActorCritic":
            raise ValueError("Prototype supports the feed-forward ActorCritic mug checkpoint only")
        config = copy.deepcopy(original_cfg)
        config.update(device=device, seed=args.seed, logger=args.logger, wandb_project=args.project,
                      num_steps_per_env=args.rollout_steps, save_interval=args.save_interval)
        config["obs_groups"] = {key: list(base_groups[key])+["oracle"] for key in ("policy", "critic")}
        config["policy"] = dict(class_name="OracleAdapterActorCritic", base_checkpoint=str(base_checkpoint),
                                base_policy_cfg=policy_cfg, base_obs_groups=base_groups,
                                correction_scale=args.correction_scale, actor_obs_normalization=True,
                                critic_obs_normalization=True, actor_hidden_dims=[128,128],
                                critic_hidden_dims=[128,128], activation="elu", init_noise_std=.05)
        config["algorithm"].update(num_mini_batches=4, num_learning_epochs=5, learning_rate=3e-4,
                                   entropy_coef=.001)
        if config["algorithm"].get("rnd_cfg") or config["algorithm"].get("symmetry_cfg"):
            raise ValueError("RND/symmetry are not supported in this adapter prototype")
        saved = None
        if args.adapter_checkpoint:
            saved = torch.load(args.adapter_checkpoint, map_location="cpu", weights_only=False)
            run = saved.get("infos", {}).get("oracle_adapter") if isinstance(saved.get("infos"), dict) else None
            if run is None:
                raise ValueError("Not an oracle-adapter checkpoint")
            for key in ("policy", "obs_groups", "algorithm"):
                config[key] = copy.deepcopy(run["runner_config"][key])
            # The frozen base is embedded in adapter checkpoints, but the original
            # file is still checked during construction for architecture validation.
            config["policy"]["base_checkpoint"] = str(base_checkpoint)
        directory = Path(args.output_root).expanduser().resolve() / (
            f"mug_{args.mode}_{args.policy}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}")
        directory.mkdir(parents=True)
        cfg.log_dir = str(directory)
        config["wandb_run_name"] = directory.name
        config["experiment_name"] = "oracle_adapter_mug"
        pristine = copy.deepcopy(config)
        (directory/"runner_config.json").write_text(json.dumps(pristine, indent=2))
        metadata = {"base_checkpoint":str(base_checkpoint), "reference_path":str(data),
                    "seed":args.seed, "mode":args.mode, "evaluation_policy":args.policy,
                    "conditions":conditions, "reset":"reference frame 0, no curriculum, pose log-only",
                    "reward":"original task/style/penalty terms, unchanged", "robot_self_collisions":True,
                    "physics_assignment":"fixed per parallel environment"}
        (directory/"run_metadata.json").write_text(json.dumps(metadata, indent=2))
        (directory/"physics.json").write_text(json.dumps(physics, indent=2))
        runner_module.OracleAdapterActorCritic = OracleAdapterActorCritic
        runner = AdapterRunner(env, config, log_dir=str(directory) if args.mode=="train" else None, device=device)
        actor = runner.alg.policy
        source_digest = frozen_digest(actor.base)
        # Only trainable parameters are in optimizer groups, including on resume.
        runner.alg.optimizer = torch.optim.Adam(
            [p for p in actor.parameters() if p.requires_grad], lr=runner.alg.learning_rate)
        if args.adapter_checkpoint:
            runner.load(args.adapter_checkpoint, load_optimizer=args.mode=="train", map_location=device)
            if frozen_digest(actor.base) != source_digest:
                raise ValueError("Adapter checkpoint frozen base differs from the supplied original checkpoint")
        else:
            actor.eval()
            with torch.inference_mode():
                torch.testing.assert_close(actor.act_inference(obs), actor.base.act_inference(obs), rtol=0, atol=0)
            print("[adapter] Zero-initialized adapter exactly matches frozen base inference.", flush=True)
        before = frozen_digest(actor.base)
        runner.adapter_info = {"runner_config":pristine, "metadata":metadata, "frozen_base_digest":before}
        print(f"[adapter] Trainable parameters: {sum(p.numel() for p in actor.parameters() if p.requires_grad)}", flush=True)
        print(f"[adapter] Outputs: {directory}", flush=True)
        if args.mode == "train":
            runner.learn(args.max_iterations, init_at_random_ep_len=False)
            if frozen_digest(actor.base) != before:
                raise RuntimeError("Frozen base weights/normalizer changed")
            # Save configuration alongside every normal runner checkpoint as well
            # as in a self-describing final checkpoint.
            runner.save(str(directory/"adapter_final.pt"), infos={"oracle_adapter":{
                "runner_config":pristine, "metadata":metadata, "frozen_base_digest":before}})
            print("[adapter] Frozen base hash verified unchanged.", flush=True)
        else:
            actor.eval()
            chosen = actor.base if args.policy=="base" else actor
            metadata.update(checkpoint=str(args.adapter_checkpoint or base_checkpoint), task=task,
                            pose_break_mode="log-only")
            collect(env, chosen.act_inference, chosen, launcher.app, directory/"episodes.h5",
                    args.episodes, 100000, physics, metadata,
                    video_folder=directory/"videos" if args.video else None)
            if frozen_digest(actor.base) != before:
                raise RuntimeError("Frozen base changed during evaluation")
        print(f"Finished. Outputs: {directory}", flush=True)
    except BaseException:
        # Print before Isaac cleanup: SimulationApp.close() may terminate the
        # process before Python displays an otherwise unhandled exception.
        traceback.print_exc()
        sys.stderr.flush()
        if "directory" in locals():
            (directory / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        raw.close()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
    try:
        launcher.app.close()
    finally:
        os._exit(1)
finally:
    launcher.app.close()
