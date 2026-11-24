# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import os
import re
import argparse
import signal
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path = [p for p in sys.path if p not in ("", str(repo_root))]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "source" / "ConTrack"))

from isaaclab.app import AppLauncher

from scripts.rsl_rl import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--data", type=str, default=None, help="Path to the HDF5 reference trajectory file."
)
parser.add_argument(
    "--session",
    type=str,
    default=None,
    help="Run name or run directory under logs/rsl_rl/* to resume from (uses env.yaml reference_path and latest model_*.pt).",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations."
)
parser.add_argument(
    "--wandb_description",
    type=str,
    default=None,
    help="Optional suffix appended to the log directory basename (logged to wandb config as 'description').",
)
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--video_num_envs",
    type=int,
    default=9,
    help="Number of environments used by the render-only play process (0 disables video recording).",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

session_dir = None
session_reference_path = ""
if args_cli.session:
    p = Path(args_cli.session).expanduser()
    if p.is_dir():
        session_dir = p
    else:
        p = (repo_root / "logs" / "rsl_rl" / args_cli.session).expanduser()
        if p.is_dir():
            session_dir = p
        else:
            c = sorted((repo_root / "logs" / "rsl_rl").glob(f"*/{args_cli.session}"))
            if len(c) != 1:
                raise ValueError(f"session matches: {[str(x) for x in c]}")
            session_dir = c[0]
    for line in (session_dir / "params" / "env.yaml").read_text().splitlines():
        s = line.strip()
        if s.startswith("reference_path:"):
            session_reference_path = s.split(":", 1)[1].strip()
            break
    ref = Path(session_reference_path)
    if not ref.is_file():
        ref = sorted((repo_root / "data").rglob(ref.name))[0]
    session_reference_path = str(ref.resolve())

if "Crx-Sharpa" in args_cli.task:
    _data_variable = "CONTRACK_CRX_SHARPA_DATA"
elif "Rby1a-Sharpa" in args_cli.task:
    _data_variable = "CONTRACK_RBY1A_SHARPA_DATA"
else:
    _data_variable = "CONTRACK_XARM_XHAND_DATA"
if args_cli.data is not None:
    os.environ[_data_variable] = args_cli.data
elif session_reference_path:
    os.environ[_data_variable] = session_reference_path

# never enable cameras in the training process (videos are rendered by a separate play process)
args_cli.enable_cameras = False

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
_interrupt_requested = False


def _early_request_stop(signum, frame) -> None:
    global _interrupt_requested
    if _interrupt_requested:
        os._exit(130)
    _interrupt_requested = True
    print("\nTraining interrupted by Ctrl-C.", flush=True)


signal.signal(signal.SIGINT, _early_request_stop)
signal.signal(signal.SIGTERM, _early_request_stop)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [
            r".\isaaclab.bat",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    else:
        cmd = [
            "./isaaclab.sh",
            "-p",
            "-m",
            "pip",
            "install",
            f"rsl-rl-lib=={RSL_RL_VERSION}",
        ]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import subprocess
import torch
import threading
import time
from datetime import datetime

os.environ.setdefault("WANDB_IGNORE_GLOBS", "*.pt,*.pth")

from rsl_rl_contrack.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import ConTrack.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _start_wandb_video_uploader(
    log_dir: str,
    num_steps_per_env: int,
    video_subdir: str = os.path.join("videos", "train"),
    poll_interval_s: int = 30,
) -> tuple[threading.Event, threading.Thread]:
    """Start a daemon thread that logs new ``.mp4`` videos in ``log_dir/video_subdir`` to Weights & Biases.

    Parameters
    ----------
    log_dir : str
        Experiment log directory.
    num_steps_per_env : int
        PPO rollout horizon used to map video env-step ``N`` to PPO iteration ``N // num_steps_per_env`` for wandb.
    video_subdir : str
        Subdirectory under ``log_dir`` that contains ``.mp4`` videos.
    poll_interval_s : int
        Poll interval in seconds.

    Returns
    -------
    stop_event : threading.Event
        Stop signal for the uploader thread.
    thread : threading.Thread
        Daemon thread that performs the uploads.
    """
    import wandb

    stop_event = threading.Event()
    logged_paths = set()
    pending_stats = {}
    video_dir = Path(log_dir) / video_subdir

    def _worker() -> None:
        while not stop_event.is_set():
            if wandb.run is None:
                stop_event.wait(poll_interval_s)
                continue
            for video_path in video_dir.glob("*.mp4"):
                video_key = str(video_path)
                if video_key in logged_paths:
                    continue
                st = video_path.stat()
                cur = (st.st_size, st.st_mtime_ns)
                if pending_stats.get(video_key) != cur:
                    pending_stats[video_key] = cur
                    continue
                wandb.log({"video": wandb.Video(video_key, format="mp4")})
                logged_paths.add(video_key)
            stop_event.wait(poll_interval_s)

    thread = threading.Thread(target=_worker, name="wandb_video_uploader", daemon=True)
    thread.start()
    return stop_event, thread


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Train with RSL-RL agent."""
    if _interrupt_requested:
        raise KeyboardInterrupt
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    video_enabled = args_cli.video_num_envs > 0
    video_master = video_enabled and (not args_cli.distributed or app_launcher.local_rank == 0)
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    if session_dir and agent_cfg.resume:
        log_dir = str(session_dir)
        print(f"[INFO] Resuming session in directory: {log_dir}")
    else:
        # specify directory for logging runs: {dataset}-{time-stamp}_{run_name}
        dataset_name = Path(env_cfg.reference_path).stem
        log_dir = f"{dataset_name}-{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
        if args_cli.wandb_description:
            log_dir += f"-{args_cli.wandb_description}"
        # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
        print(f"Exact experiment name requested from command line: {log_dir}")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    if _interrupt_requested:
        raise KeyboardInterrupt
    # create isaac environment
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode=None,
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if session_dir:
            if args_cli.checkpoint is None:
                ckpt = sorted(
                    session_dir.glob("model_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]),
                )[-1]
            elif args_cli.checkpoint.isdigit():
                ckpt = session_dir / f"model_{args_cli.checkpoint}.pt"
            elif (session_dir / args_cli.checkpoint).is_file():
                ckpt = session_dir / args_cli.checkpoint
            else:
                ckpt = Path(args_cli.checkpoint).expanduser()
            resume_path = str(ckpt)
        else:
            resume_path = get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )

    if video_master:
        vdir = Path(log_dir) / "videos" / "train"
        vdir.mkdir(parents=True, exist_ok=True)
        print(
            f"[INFO] Video rendering runs in a separate play process (num_envs={args_cli.video_num_envs}): {vdir}"
        )

    isaac_env = env.unwrapped

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if _interrupt_requested:
        raise KeyboardInterrupt
    from rsl_rl_contrack.env.cons_mimic_reward_wrapper import ConsMimicRewardWrapper

    env = ConsMimicRewardWrapper(env, isaac_env)

    # create runner from rsl-rl
    cfg = agent_cfg.to_dict()
    base_name = Path(log_dir).name
    m = re.search(r"\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}", base_name)
    cfg["start_timestamp"] = m.group(0) if m else ""
    desc = ""
    if m:
        rest = base_name[m.end() :]
        if rest.startswith("-"):
            desc = rest[1:]
            if agent_cfg.run_name and desc.endswith(f"_{agent_cfg.run_name}"):
                desc = desc[: -(len(agent_cfg.run_name) + 1)]
    cfg["description"] = desc
    cfg["wandb_run_name"] = Path(env_cfg.reference_path).stem + (f"-{desc}" if desc else "")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    reset_lib_dir = Path(log_dir) / "reset_library"
    reset_lib_l_dir = reset_lib_dir / "l"
    reset_lib_ratio_dir = reset_lib_dir / "ratio"
    reset_lib_grad_dir = reset_lib_dir / "gradient"
    reset_lib_terminate_index_dir = reset_lib_dir / "terminate_index"
    reset_lib_reset_index_dir = reset_lib_dir / "reset_index"
    reset_lib_l_dir.mkdir(parents=True, exist_ok=True)
    reset_lib_ratio_dir.mkdir(parents=True, exist_ok=True)
    reset_lib_grad_dir.mkdir(parents=True, exist_ok=True)
    reset_lib_terminate_index_dir.mkdir(parents=True, exist_ok=True)
    reset_lib_reset_index_dir.mkdir(parents=True, exist_ok=True)
    old_save = runner.save
    video_proc = None
    video_log_path = None
    video_error_event = threading.Event()
    play_task = (
        args_cli.task
        if "-Play-" in args_cli.task
        else args_cli.task.replace("-v0", "-Play-v0")
    )

    def _read_video_log(path: Path) -> str:
        text = path.read_text()
        return text if text else "<empty render log>\n"

    def _fail_on_video_error() -> None:
        if video_proc is None:
            return
        ret = video_proc.poll()
        if ret is None or ret == 0:
            return
        video_error_event.set()
        print(
            f"\nVideo render subprocess failed with exit code {ret}.\n"
            f"Render log: {video_log_path}\n"
            f"{_read_video_log(video_log_path)}",
            file=sys.stderr,
            flush=True,
        )
        raise RuntimeError("video render subprocess failed")

    def save(path: str, infos: dict | None = None) -> None:
        """Save checkpoint and a reset library track-length plot.

        Parameters
        ----------
        path : str
            Checkpoint output path.
        infos : dict[str, object] | None
            Optional checkpoint metadata.

        Returns
        -------
        None
            Writes ``path`` and reset-library plots under ``reset_library/``:
            - stores reset-library tensors inside checkpoint ``infos["handmimic_reset_lib"]`` (CPU tensors):
              - ``reset_lib_state`` (T, D) float32
              - ``reset_lib_track_len`` (T,) float32
              - ``reset_lib_track_len_best`` (T,) float32
              - ``reset_lib_terminate_index_count`` (T,) int64
              - ``reset_lib_reset_index_count`` (T,) int64
              - (optional) ``threshold_curriculum_success_ema`` () float32, ``object_pos_threshold`` (num_envs,) float32,
                ``object_rot_threshold`` (num_envs,) float32
            - ``l/l_<iter>.png``: remaining length ``L[k]`` (frames)
            - ``l/l_<iter>.csv``: remaining length ``L[k]`` (frames), CSV header ``k,l``
            - ``ratio/ratio_<iter>.png``: survival ratio ``L[k] / H[k]`` (unitless), ``H[k]=ref_len-k`` (frames)
            - ``ratio/ratio_<iter>.csv``: survival ratio ``L[k] / H[k]`` (unitless), CSV header ``k,ratio``
            - ``gradient/gradient_<iter>.png``: forward difference of survival ratio (1/frame)
            - ``gradient/gradient_<iter>.csv``: forward difference of survival ratio (1/frame), CSV header ``k,gradient``
            - ``terminate_index/terminate_index_<iter>.png``: terminate frame index histogram since last save_interval (counts)
            - ``terminate_index/terminate_index_<iter>.csv``: terminate frame index histogram since last save_interval (counts), CSV header ``k,count``
            - ``reset_index/reset_index_<iter>.png``: reset start frame index histogram since last save_interval (counts)
            - ``reset_index/reset_index_<iter>.csv``: reset start frame index histogram since last save_interval (counts), CSV header ``k,count``
        """
        nonlocal video_proc, video_log_path
        if video_error_event.is_set():
            raise RuntimeError("video render subprocess failed")
        _fail_on_video_error()
        infos = {} if infos is None else infos
        infos["handmimic_reset_lib"] = {
            "reset_lib_state": runner.env.unwrapped.reset_lib_state.detach().cpu(),
            "reset_lib_track_len": runner.env.unwrapped.reset_lib_track_len.detach().cpu(),
            "reset_lib_track_len_best": runner.env.unwrapped.reset_lib_track_len_best.detach().cpu(),
            "reset_lib_terminate_index_count": runner.env.unwrapped.reset_lib_terminate_index_count.detach().cpu(),
            "reset_lib_reset_index_count": runner.env.unwrapped.reset_lib_reset_index_count.detach().cpu(),
        }
        if hasattr(runner.env.unwrapped, "threshold_curriculum_success_ema"):
            infos["handmimic_reset_lib"]["threshold_curriculum_success_ema"] = (
                runner.env.unwrapped.threshold_curriculum_success_ema.detach().cpu()
            )
            infos["handmimic_reset_lib"]["object_pos_threshold"] = (
                runner.env.unwrapped.object_pos_threshold.detach().cpu()
            )
            infos["handmimic_reset_lib"]["object_rot_threshold"] = (
                runner.env.unwrapped.object_rot_threshold.detach().cpu()
            )
        old_save(path, infos)
        if video_master:
            if video_proc is None or video_proc.poll() is not None:
                video_log_path = Path(log_dir) / "videos" / "train" / f"{Path(path).stem}_render.log"
                with open(video_log_path, "w") as video_log:
                    video_proc = subprocess.Popen(
                        [
                            sys.executable,
                            "-u",
                            "-m",
                            "scripts.rsl_rl.play",
                            "--task",
                            play_task,
                            "--session",
                            log_dir,
                            "--checkpoint",
                            path,
                            "--num_envs",
                            str(args_cli.video_num_envs),
                            "--video",
                            "--video_length",
                            str(agent_cfg.video_length),
                            "--headless",
                        ],
                        cwd=str(repo_root),
                        stdout=video_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        l = (
            runner.env.unwrapped.reset_lib_track_len.detach()
            .to("cpu")
            .to(torch.float32)
        )
        ref_len = int(l.shape[0])
        l = l[: ref_len - 1]
        h = ref_len - torch.arange(ref_len - 1, dtype=torch.float32)
        ratio = l / h
        grad = torch.zeros_like(ratio)
        grad[:-1] = ratio[1:] - ratio[:-1]
        it = runner.current_learning_iteration
        (reset_lib_l_dir / f"l_{it}.csv").write_text(
            "k,l\n" + "\n".join(f"{i},{v}" for i, v in enumerate(l.tolist()))
        )
        (reset_lib_ratio_dir / f"ratio_{it}.csv").write_text(
            "k,ratio\n" + "\n".join(f"{i},{v}" for i, v in enumerate(ratio.tolist()))
        )
        (reset_lib_grad_dir / f"gradient_{it}.csv").write_text(
            "k,gradient\n" + "\n".join(f"{i},{v}" for i, v in enumerate(grad.tolist()))
        )

        plt.figure(figsize=(12, 4))
        plt.plot(l.tolist(), linewidth=1.0)
        plt.xlabel("reference frame index k (frame)")
        plt.ylabel("remaining length L[k] (frames)")
        plt.title("reset library: L[k]")
        plt.tight_layout()
        plt.savefig(reset_lib_l_dir / f"l_{it}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(ratio.tolist(), linewidth=1.0)
        plt.xlabel("reference frame index k (frame)")
        plt.ylabel("survival ratio L[k] / H[k] (unitless)")
        plt.title("reset library: L[k] / H[k] (H[k]=ref_len-k)")
        plt.tight_layout()
        plt.savefig(reset_lib_ratio_dir / f"ratio_{it}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(grad.tolist(), linewidth=1.0)
        plt.xlabel("reference frame index k (frame)")
        plt.ylabel("Δ(L/H) = (L[k+1]/H[k+1]) - (L[k]/H[k]) (1/frame)")
        plt.title("reset library: GRADIENT(L[k]/H[k])")
        plt.tight_layout()
        plt.savefig(reset_lib_grad_dir / f"gradient_{it}.png", dpi=150)
        plt.close()

        terminate_cnt = (
            runner.env.unwrapped.reset_lib_terminate_index_count.detach()
            .to("cpu")
            .to(torch.float32)
        )
        reset_cnt = (
            runner.env.unwrapped.reset_lib_reset_index_count.detach()
            .to("cpu")
            .to(torch.float32)
        )
        (reset_lib_terminate_index_dir / f"terminate_index_{it}.csv").write_text(
            "k,count\n"
            + "\n".join(f"{i},{v}" for i, v in enumerate(terminate_cnt.tolist()))
        )
        (reset_lib_reset_index_dir / f"reset_index_{it}.csv").write_text(
            "k,count\n"
            + "\n".join(f"{i},{v}" for i, v in enumerate(reset_cnt.tolist()))
        )

        plt.figure(figsize=(12, 4))
        plt.plot(terminate_cnt.tolist(), linewidth=1.0)
        plt.xlabel("reference frame index k (frame)")
        plt.ylabel("count (resets)")
        plt.title("reset library: terminate frame index (save_interval)")
        plt.tight_layout()
        plt.savefig(
            reset_lib_terminate_index_dir / f"terminate_index_{it}.png", dpi=150
        )
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(reset_cnt.tolist(), linewidth=1.0)
        plt.xlabel("reference frame index k (frame)")
        plt.ylabel("count (resets)")
        plt.title("reset library: reset start frame index (save_interval)")
        plt.tight_layout()
        plt.savefig(reset_lib_reset_index_dir / f"reset_index_{it}.png", dpi=150)
        plt.close()

        runner.env.unwrapped.reset_lib_terminate_index_count.zero_()
        runner.env.unwrapped.reset_lib_reset_index_count.zero_()

    runner.save = save
    old_log = runner.log

    def log(locs: dict, width: int = 80, pad: int = 35) -> None:
        """Log training stats and reset library mean track length.

        Parameters
        ----------
        locs : dict
            Runner locals dict containing ``it`` (int) used as the step index.
        width : int
            Terminal print width.
        pad : int
            Terminal print left padding.

        Returns
        -------
        None
            Logs default RSL-RL metrics and additionally ``Train/reset_lib_track_len_mean`` (scalar) to the writer.
        """
        old_log(locs, width=width, pad=pad)
        runner.writer.add_scalar(
            "Train/reset_lib_track_len_mean",
            float(
                runner.env.unwrapped.reset_lib_track_len.to(torch.float32).mean().item()
            ),
            locs["it"],
        )

    runner.log = log
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        resume_infos = runner.load(resume_path) or {}
        s = resume_infos.get("handmimic_reset_lib")
        if s is not None:
            runner.env.unwrapped.reset_lib_state.copy_(
                s["reset_lib_state"].to(env_cfg.sim.device)
            )
            runner.env.unwrapped.reset_lib_track_len.copy_(
                s["reset_lib_track_len"].to(env_cfg.sim.device)
            )
            runner.env.unwrapped.reset_lib_track_len_best.copy_(
                s["reset_lib_track_len_best"].to(env_cfg.sim.device)
            )
            runner.env.unwrapped.reset_lib_terminate_index_count.copy_(
                s["reset_lib_terminate_index_count"].to(env_cfg.sim.device)
            )
            runner.env.unwrapped.reset_lib_reset_index_count.copy_(
                s["reset_lib_reset_index_count"].to(env_cfg.sim.device)
            )
            if "threshold_curriculum_success_ema" in s:
                runner.env.unwrapped.threshold_curriculum_success_ema.copy_(
                    s["threshold_curriculum_success_ema"].to(env_cfg.sim.device)
                )
                runner.env.unwrapped.object_pos_threshold.copy_(
                    s["object_pos_threshold"].to(env_cfg.sim.device)
                )
                runner.env.unwrapped.object_rot_threshold.copy_(
                    s["object_rot_threshold"].to(env_cfg.sim.device)
                )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    uploader_stop = None
    uploader_thread = None
    if video_master and agent_cfg.logger == "wandb":
        uploader_stop, uploader_thread = _start_wandb_video_uploader(
            log_dir, agent_cfg.num_steps_per_env
        )

    interrupted = False
    video_failed = False
    failed = False

    def _request_stop(signum, frame) -> None:
        nonlocal interrupted
        global _interrupt_requested
        if _interrupt_requested:
            os._exit(130)
        _interrupt_requested = True
        interrupted = True
        print("\nTraining interrupted by Ctrl-C.", flush=True)

    if _interrupt_requested:
        interrupted = True
        raise KeyboardInterrupt

    prev_sigint = signal.getsignal(signal.SIGINT)
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
            stop_requested=lambda: _interrupt_requested,
        )
        if video_master and video_proc is not None:
            while video_proc.poll() is None:
                if _interrupt_requested:
                    raise KeyboardInterrupt
                time.sleep(0.5)
            _fail_on_video_error()
    except KeyboardInterrupt:
        interrupted = True
    except RuntimeError as err:
        if str(err) != "video render subprocess failed":
            failed = True
            raise
        video_failed = True
        failed = True
    except Exception:
        failed = True
        raise
    finally:
        if video_master and video_proc is not None and video_proc.poll() is None:
            os.killpg(video_proc.pid, signal.SIGINT)
            try:
                video_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(video_proc.pid, signal.SIGTERM)
                video_proc.wait()
        if uploader_stop is not None and uploader_thread is not None:
            uploader_stop.set()
            uploader_thread.join()
        if runner.writer is not None:
            if hasattr(runner.writer, "stop"):
                runner.writer.stop(exit_code=255 if interrupted else int(failed))
            else:
                runner.writer.close()
        env.close()
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
    if interrupted:
        raise KeyboardInterrupt
    if video_failed:
        raise RuntimeError("video render subprocess failed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        simulation_app.close()
