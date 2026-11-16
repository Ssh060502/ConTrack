"""Render reference robot/object poses directly, without policy or physics rollout."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / "source" / "ConTrack"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True, help="New MP4 output path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
data_path = Path(args.data).expanduser().resolve()
output = Path(args.output).expanduser().resolve()
if not data_path.is_file():
    parser.error(f"Data not found: {data_path}")
if output.exists():
    parser.error(f"Output already exists: {output}")
os.environ["CONTRACK_RBY1A_SHARPA_DATA"] = str(data_path)
args.enable_cameras = True
launcher = AppLauncher(args)

import gymnasium as gym
import imageio.v2 as imageio
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import ConTrack.tasks


def main():
    cfg = parse_env_cfg("Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0",
                        device=args.device or "cuda:0", num_envs=1)
    env = gym.make("Isaac-Rby1a-Sharpa-Mimic-Manager-Play-v0", cfg=cfg, render_mode="rgb_array")
    try:
        env.reset()
        base = env.unwrapped
        output.parent.mkdir(parents=True, exist_ok=True)
        fps = round(1. / float(base.step_dt))
        print(f"Reference only: {base.ref_len} frames at {fps} FPS; no policy, no dynamics.", flush=True)
        with imageio.get_writer(str(output), fps=fps, macro_block_size=1) as writer:
            with torch.inference_mode():
                for frame in range(base.ref_len):
                    if not launcher.app.is_running():
                        raise RuntimeError("Reference replay interrupted")
                    for name, ids, h in base.ref_hands:
                        q = base.ref_qpos[frame, h].unsqueeze(0)
                        base.scene[name].write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids)
                        base.scene[name].set_joint_position_target(q, joint_ids=ids)
                    objects = base.scene["objects"]
                    position = base.ref_object_pos[frame].unsqueeze(0) + base.scene.env_origins.unsqueeze(1)
                    quaternion = base.ref_object_quat[frame].unsqueeze(0)
                    velocity = torch.zeros((*position.shape[:-1], 6), device=base.device)
                    objects.write_object_state_to_sim(torch.cat([position, quaternion, velocity], dim=-1))
                    base.frame_idx[:] = frame
                    base.frame_idx_f[:] = frame
                    # Update articulation transforms without advancing physics.
                    base.sim.forward()
                    base.sim.render()
                    base.scene.update(base.step_dt)
                    rgb = env.render()
                    if rgb is None:
                        raise RuntimeError("Renderer did not return an RGB frame")
                    writer.append_data(rgb)
        print(f"Saved reference video: {output}", flush=True)
    finally:
        env.close()


try:
    main()
finally:
    launcher.app.close()
