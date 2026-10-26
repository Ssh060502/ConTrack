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
parser.add_argument(
    "--export_rollout",
    type=str,
    default=None,
    help=(
        "Path to save env 0's rolled-out joint trajectory as an h5 file with the same "
        "qpos/urdf_name/is_rhand/fps schema as a reference dataset. Run with --num_envs 1 "
        "for a clean single-episode recording; recording stops at env 0's first episode end."
    ),
)
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
import h5py
import numpy as np
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

    if args_cli.export_rollout:
        # The env's curriculum resets to a random frame of the reference clip by
        # default (see reset_reference_episode in mdp/state.py), which is what
        # training wants but makes a recorded rollout start from an arbitrary
        # mid-clip point instead of the beginning. Force frame 0 here and
        # re-reset so the export starts from the true beginning of the motion.
        env.unwrapped.force_start_frame0 = True
        env.reset()

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

    rollout_frames: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_object_pos: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_object_quat_xyzw: list[np.ndarray] | None = [] if args_cli.export_rollout else None
    rollout_done = False
    try:
        env.unwrapped.scene["objects"]
        has_objects = True
    except KeyError:
        has_objects = False
    if rollout_frames is not None and not has_objects:
        print("[warn] --export_rollout: no 'objects' asset in this scene; the saved h5 "
              "will have no object_tracks group.")
    if rollout_frames is not None and env.unwrapped.num_envs != 1:
        print(
            f"[warn] --export_rollout records only env 0, but num_envs={env.unwrapped.num_envs}. "
            "Pass --num_envs 1 for a clean single-episode recording."
        )

    while simulation_app.is_running() and timestep < args_cli.video_length:
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            runner.alg.policy.reset(dones)
        if "log" in extras:
            for k, v in extras["log"].items():
                episode_logs.setdefault(k, []).append(float(v))
        if rollout_frames is not None and not rollout_done:
            base_env = env.unwrapped
            frame = np.zeros(
                (len(base_env.ref_hands), base_env.unused_default_qpos.shape[0]), dtype=np.float32
            )
            for name, joint_ids, h_idx in base_env.ref_hands:
                frame[h_idx] = base_env.scene[name].data.joint_pos[0, joint_ids].cpu().numpy()
            rollout_frames.append(frame)
            if has_objects:
                objects = base_env.scene["objects"]
                obj_pos = (
                    objects.data.object_pos_w[0] - base_env.scene.env_origins[0]
                ).cpu().numpy()  # (O, 3), relative to this env's own origin
                obj_quat_wxyz = objects.data.object_quat_w[0].cpu().numpy()  # (O, 4)
                rollout_object_pos.append(obj_pos)
                rollout_object_quat_xyzw.append(obj_quat_wxyz[:, [1, 2, 3, 0]])
            if bool(dones[0]):
                rollout_done = True
        timestep += 1

    if rollout_frames is not None:
        if not rollout_frames:
            print("[warn] --export_rollout given but no frames were recorded")
        else:
            base_env = env.unwrapped
            qpos = np.stack(rollout_frames, axis=0)
            urdf_name = np.array([s.encode("utf-8") for s in base_env.urdf_name])
            is_rhand = base_env.is_rhand.cpu().numpy().astype(np.uint8)
            fps = np.full((qpos.shape[1],), float(base_env.ref_fps), dtype=np.float32)
            out_path = Path(args_cli.export_rollout).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as hf:
                hf.create_dataset("qpos", data=qpos)
                hf.create_dataset("urdf_name", data=urdf_name)
                hf.create_dataset("is_rhand", data=is_rhand)
                hf.create_dataset("fps", data=fps)
                if has_objects and rollout_object_pos:
                    object_pos = np.stack(rollout_object_pos, axis=0)  # (T, O, 3)
                    object_quat_xyzw = np.stack(rollout_object_quat_xyzw, axis=0)  # (T, O, 4)
                    for obj_idx in range(object_pos.shape[1]):
                        grp = hf.create_group(f"object_tracks/object_{obj_idx}")
                        grp.create_dataset("translations", data=object_pos[:, obj_idx])
                        grp.create_dataset("orientations_xyzw", data=object_quat_xyzw[:, obj_idx])
            print(
                f"[info] saved rollout qpos to {out_path} "
                f"(T={qpos.shape[0]}, H={qpos.shape[1]}, stopped_early={rollout_done}, "
                f"objects_saved={has_objects})"
            )

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
