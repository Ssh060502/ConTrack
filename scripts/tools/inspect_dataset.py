"""Inspect a ConTrack reference HDF5 clip without launching Isaac Sim.

Prints the dataset structure/metadata and can optionally save joint-angle plots,
object-trajectory plots, and an offline 3D animation of the object motion (MP4).
Only needs h5py/numpy/scipy/matplotlib/imageio, all already installed alongside
ConTrack, so this runs on the CPU without touching a GPU or Isaac Sim.

Examples
--------
Print structure/metadata only:
    python scripts/tools/inspect_dataset.py --data data/xhand/dexterhand-Cylinder_00-fps_20-xarm7_xhand-3600_3700.h5

Also save joint-angle and object-trajectory plots:
    python scripts/tools/inspect_dataset.py --data <path.h5> --plots

Also render an offline animation of the object motion (no robot mesh, no Isaac Sim):
    python scripts/tools/inspect_dataset.py --data <path.h5> --video --video-stride 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def summarize(f: h5py.File, out_dir: Path) -> None:
    """Print dataset shapes/dtypes and key metadata for a reference clip, and save it to ``summary.txt``.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle for a ConTrack reference clip.
    out_dir : Path
        Directory to save ``summary.txt`` into.
    """
    lines: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        lines.append(line)

    emit("=== Dataset structure ===")

    def show(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            emit(f"{name:55s} shape={obj.shape} dtype={obj.dtype}")

    f.visititems(show)

    emit()
    emit("=== Metadata ===")
    urdf_name = [s.decode("utf-8") for s in np.asarray(f["urdf_name"])]
    is_rhand = np.asarray(f["is_rhand"], dtype=np.uint8).astype(bool)
    fps = np.asarray(f["fps"], dtype=np.float32)
    num_frames = int(f["qpos"].shape[0])
    obj_keys = list(f["object_tracks"].keys())
    emit(f"hands            : {urdf_name} (is_rhand={is_rhand.tolist()})")
    emit(f"fps per hand     : {fps.tolist()}")
    emit(f"num frames       : {num_frames}")
    emit(f"clip duration(s) : {num_frames / float(np.max(fps)):.2f}")
    emit(f"num objects      : {len(obj_keys)} -> {obj_keys}")
    if "contacts" in f:
        seg_names = [s.decode("utf-8") for s in np.asarray(f["contacts/segment_names"])]
        emit(f"contact segments : {seg_names}")

    out_path = out_dir / "summary.txt"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[saved] {out_path}")


def plot_joint_trajectories(f: h5py.File, out_dir: Path) -> None:
    """Save a per-hand plot of joint angles over time.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle exposing ``qpos`` (T, H, 19).
    out_dir : Path
        Directory to save ``joint_trajectories.png`` into.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qpos = np.asarray(f["qpos"])
    num_frames, num_hands, num_joints = qpos.shape
    fig, axes = plt.subplots(num_hands, 1, figsize=(10, 4 * num_hands), squeeze=False)
    t = np.arange(num_frames)
    for h in range(num_hands):
        ax = axes[h, 0]
        for j in range(num_joints):
            group = "arm" if j < 7 else "finger"
            ax.plot(t, qpos[:, h, j], linewidth=0.8, label=f"{group}_{j}" if j in (0, 7) else None)
        ax.set_title(f"Hand {h} joint trajectories")
        ax.set_xlabel("frame")
        ax.set_ylabel("joint angle (rad)")
        ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    out_path = out_dir / "joint_trajectories.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_object_trajectories(f: h5py.File, out_dir: Path) -> None:
    """Save a 3D plot of every object's translation trajectory.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle exposing ``object_tracks/<key>/translations`` (T, 3).
    out_dir : Path
        Directory to save ``object_trajectories.png`` into.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obj_keys = list(f["object_tracks"].keys())
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    for k in obj_keys:
        pos = np.asarray(f["object_tracks"][k]["translations"], dtype=np.float32)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], label=f"object_{k}")
        ax.scatter(*pos[0], marker="o", s=30)
        ax.scatter(*pos[-1], marker="x", s=30)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Object trajectories (o=start, x=end)")
    ax.legend()
    out_path = out_dir / "object_trajectories.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def render_video(f: h5py.File, out_dir: Path, fps: float, stride: int) -> None:
    """Render an offline 3D animation of the object mesh motion to MP4.

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle exposing per-object ``vertices``, ``faces``, ``translations``,
        and ``orientations_xyzw``.
    out_dir : Path
        Directory to save ``object_motion.mp4`` (or a PNG frame sequence as a fallback) into.
    fps : float
        Playback frame rate for the output video.
    stride : int
        Render every ``stride``-th frame (use >1 to speed up long clips).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R

    obj_keys = list(f["object_tracks"].keys())
    meshes = []
    for k in obj_keys:
        v = np.asarray(f["object_tracks"][k]["vertices"], dtype=np.float32)
        tri = np.asarray(f["object_tracks"][k]["faces"], dtype=np.int64)
        pos = np.asarray(f["object_tracks"][k]["translations"], dtype=np.float32)
        quat_xyzw = np.asarray(f["object_tracks"][k]["orientations_xyzw"], dtype=np.float32)
        meshes.append((v, tri, pos, quat_xyzw))

    num_frames = meshes[0][2].shape[0]
    frame_ids = list(range(0, num_frames, max(1, stride)))

    all_pos = np.concatenate([m[2] for m in meshes], axis=0)
    center = all_pos.mean(axis=0)
    radius = float(np.max(np.linalg.norm(all_pos - center, axis=-1))) + 0.15

    frames = []
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    for t in frame_ids:
        ax.cla()
        for v, tri, pos, quat_xyzw in meshes:
            rot = R.from_quat(quat_xyzw[t]).as_matrix()
            verts_w = (rot @ v.T).T + pos[t]
            ax.plot_trisurf(verts_w[:, 0], verts_w[:, 1], verts_w[:, 2], triangles=tri, linewidth=0.1, antialiased=True)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(f"frame {t}/{num_frames}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(img)
    plt.close(fig)

    try:
        import imageio.v2 as imageio

        out_path = out_dir / "object_motion.mp4"
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"[saved] {out_path}")
    except Exception as exc:  # noqa: BLE001 - fall back to a PNG sequence if the mp4 codec is unavailable
        print(f"[warn] failed to write MP4 ({exc}); saving PNG frames instead")
        frame_dir = out_dir / "object_motion_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        for i, img in enumerate(frames):
            imageio.imwrite(frame_dir / f"frame_{i:05d}.png", img)
        print(f"[saved] {frame_dir} ({len(frames)} frames)")


def main() -> None:
    """Parse CLI arguments and dump the requested inspection outputs for one reference clip."""
    parser = argparse.ArgumentParser(description="Inspect a ConTrack reference HDF5 clip.")
    parser.add_argument("--data", type=str, required=True, help="Path to the reference .h5 file.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for plots/video (default: '<data-file-stem>_inspect' next to the .h5 file).",
    )
    parser.add_argument("--plots", action="store_true", help="Save joint-angle and object-trajectory plots (PNG).")
    parser.add_argument(
        "--video",
        action="store_true",
        help="Render an offline 3D animation of the object motion (MP4, no Isaac Sim / GPU needed).",
    )
    parser.add_argument(
        "--video-stride",
        type=int,
        default=1,
        help="Use every Nth frame when rendering the video (speeds up rendering for long clips).",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else data_path.parent / f"{data_path.stem}_inspect"
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data_path, "r") as f:
        summarize(f, out_dir)
        if args.plots:
            plot_joint_trajectories(f, out_dir)
            plot_object_trajectories(f, out_dir)
        if args.video:
            fps = float(np.max(np.asarray(f["fps"], dtype=np.float32)))
            render_video(f, out_dir, fps=fps, stride=args.video_stride)

    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
