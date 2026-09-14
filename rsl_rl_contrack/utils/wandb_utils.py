# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import atexit
import os
import uuid
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Wandb is required to log to Weights and Biases."
    ) from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """Initialize a wandb run that is stable across restarts.

        Parameters
        ----------
        log_dir : str
            Log directory used by the runner (also used as wandb ``dir``).
        flush_secs : int
            TensorBoard flush interval in seconds.
        cfg : dict
            Runner config dict. Must contain ``wandb_project`` (str). Optional keys:

            - ``wandb_run_name`` (str): override wandb run name (default: ``basename(log_dir)``)
            - ``start_timestamp`` (str): logged to wandb config as a separate column
            - ``description`` (str): logged to wandb config as a separate column

        Returns
        -------
        None
            Initializes a wandb run whose id is stored in ``{log_dir}/wandb_run_id`` so that restarting training
            and reusing the same ``log_dir`` resumes the same wandb run.
        """
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = cfg.get("wandb_run_name") or os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError(
                "Please specify wandb_project in the runner config, e.g. legged_gym."
            ) from None

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        run_id_path = os.path.join(log_dir, "wandb_run_id")
        run_id = open(run_id_path).read().strip() if os.path.isfile(run_id_path) else ""
        if not run_id:
            for base in [
                os.path.join(log_dir, "wandb"),
                os.path.join(os.getcwd(), "wandb"),
            ]:
                if os.path.isdir(base):
                    for d in sorted(os.listdir(base), reverse=True):
                        if d.startswith("run-") and os.path.isfile(
                            os.path.join(base, d, "files", "config.yaml")
                        ):
                            if (
                                log_dir
                                in open(
                                    os.path.join(base, d, "files", "config.yaml")
                                ).read()
                            ):
                                run_id = d.split("-")[-1]
                                break
                    if run_id:
                        break
        if not run_id:
            run_id = uuid.uuid4().hex[:8]
        if not os.path.isfile(run_id_path):
            open(run_id_path, "w").write(run_id)

        # Initialize wandb
        wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            id=run_id,
            resume="allow",
            dir=log_dir,
        )
        atexit.register(wandb.finish)

        wandb.config.update(
            {
                "log_dir": log_dir,
                "start_timestamp": cfg.get("start_timestamp"),
                "description": cfg.get("description"),
            },
            allow_val_change=True,
        )
        wandb.run.summary["start_timestamp"] = cfg.get("start_timestamp")
        wandb.run.summary["description"] = cfg.get("description")

    def store_config(
        self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict
    ) -> None:
        wandb.config.update({"runner_cfg": runner_cfg}, allow_val_change=True)
        wandb.config.update({"policy_cfg": policy_cfg}, allow_val_change=True)
        wandb.config.update({"alg_cfg": alg_cfg}, allow_val_change=True)
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()}, allow_val_change=True)
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)}, allow_val_change=True)

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def stop(self, exit_code: int = 0) -> None:
        super().close()
        wandb.finish(exit_code=exit_code)

    def log_config(
        self, env_cfg: dict | object, runner_cfg: dict, alg_cfg: dict, policy_cfg: dict
    ) -> None:
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path: str, iter: int) -> None:
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        wandb.save(path, base_path=os.path.dirname(path))
