from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if p not in ("", str(repo_root))]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "source" / "ConTrack"))

from isaaclab.app import AppLauncher

from scripts.rsl_rl import cli_args

parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--data", type=str, default=None)
parser.add_argument("--session", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

session_dir = Path(args_cli.session).expanduser() if args_cli.session else None
session_reference_path = ""
if session_dir is not None:
    for line in (session_dir / "params" / "env.yaml").read_text().splitlines():
        s = line.strip()
        if s.startswith("reference_path:"):
            session_reference_path = s.split(":", 1)[1].strip()
            break
    ref = Path(session_reference_path)
    if not ref.is_file():
        ref = sorted((repo_root / "data").rglob(ref.name))[0]
    session_reference_path = str(ref.resolve())

if args_cli.data is not None:
    if "Rby1a-Sharpa" in args_cli.task:
        os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = args_cli.data
    else:
        os.environ["CONTRACK_XARM_XHAND_DATA"] = args_cli.data
elif session_reference_path:
    if "Rby1a-Sharpa" in args_cli.task:
        os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = session_reference_path
    else:
        os.environ["CONTRACK_XARM_XHAND_DATA"] = session_reference_path

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import yaml

from rsl_rl_contrack.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks
import ConTrack.tasks


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    cfg = (
        yaml.safe_load((session_dir / "params" / "agent.yaml").read_text())
        if session_dir is not None
        else cli_args.update_rsl_rl_cfg(agent_cfg, args_cli).to_dict()
    )
    if args_cli.seed is not None:
        cfg["seed"] = args_cli.seed
    cfg["device"] = args_cli.device if args_cli.device is not None else cfg["device"]
    resume_path = args_cli.checkpoint
    log_dir = str(session_dir) if session_dir is not None else str(Path(resume_path).parent)

    env_cfg.seed = cfg["seed"]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
            name_prefix=Path(resume_path).stem,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=cfg["clip_actions"])
    runner_class = cfg["class_name"]
    runner = (
        OnPolicyRunner(env, cfg, log_dir=None, device=cfg["device"])
        if runner_class == "OnPolicyRunner"
        else DistillationRunner(env, cfg, log_dir=None, device=cfg["device"])
    )
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    timestep = 0
    episode_logs: dict[str, list[float]] = {}

    while simulation_app.is_running() and timestep < args_cli.video_length:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            runner.alg.policy.reset(dones)
        if "log" in extras:
            for k, v in extras["log"].items():
                episode_logs.setdefault(k, []).append(float(v))
        timestep += 1

    env.close()

    if episode_logs:
        print("\n=== Episode statistics over this rollout ===")
        for k in sorted(episode_logs.keys()):
            vals = episode_logs[k]
            print(f"{k:40s} mean={sum(vals) / len(vals):.4f}  n={len(vals)}")
    else:
        print("\n[warn] no episodes finished during this rollout; increase --video_length or --num_envs")


if __name__ == "__main__":
    main()
    simulation_app.close()
