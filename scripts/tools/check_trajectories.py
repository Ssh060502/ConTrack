"""Validate collector H5 alignment without Isaac Sim; print episode lengths/reasons."""
import argparse
import h5py
import numpy as np


def check(path):
    with h5py.File(path) as hf:
        if hf.attrs.get("schema_version") != "contrack_episode_v1":
            raise ValueError("Not a ConTrack episode collection")
        print("Collection complete:", bool(hf.attrs["collection_complete"]))
        print("Completed episodes:", int(hf.attrs["completed_episodes"]))
        groups = list(hf.get("episodes", {}).items())
        if "partial_episode" in hf:
            groups.append(("partial_episode", hf["partial_episode"]))
        for name, group in groups:
            t = int(group.attrs["transition_count"])

            def aligned(expected):
                def visitor(key, dataset):
                    if isinstance(dataset, h5py.Dataset) and dataset.shape[0] != expected:
                        raise ValueError(f"{name}/{key}: expected {expected} rows, got {dataset.shape}")
                return visitor

            group["states"].visititems(aligned(t + 1))
            if t:
                group["transitions"].visititems(aligned(t))
                observation = group["policy_observation"]
                if observation.shape[0] != t or observation.ndim != 2 or observation.dtype.kind not in "fiu":
                    raise ValueError("Observation/action alignment mismatch")
                done = group["transitions/done"][:].astype(bool)
                if done[:-1].any() or bool(done[-1]) != bool(group.attrs["completed"]):
                    raise ValueError("Episode crosses a reset or has an invalid final done flag")
                reasons = [k for k, v in group["transitions/termination"].items() if bool(v[-1])]
                if "diagnostics" in group["transitions"]:
                    diagnostic = group["transitions/diagnostics"]
                    position_hit = diagnostic["position_error"][:] > diagnostic["position_threshold"][: , None]
                    rotation_hit = diagnostic["rotation_error_rad"][:] > diagnostic["rotation_threshold_rad"][:, None]
                    expected = np.any(position_hit | rotation_hit, axis=-1)
                    if not np.array_equal(expected, diagnostic["pose_break"][:]):
                        raise ValueError("Pose diagnostic differs from termination evaluation")
                    if not np.array_equal(expected, group["transitions/termination/pose_break"][:]):
                        raise ValueError("Recorded termination flag differs from diagnostic")
                    print("  termination frame:", diagnostic["reference_frame"][-1],
                          "position error:", diagnostic["position_error"][-1],
                          "rotation error (deg):", np.degrees(diagnostic["rotation_error_rad"][-1]))
            else:
                reasons = []
            pos = group["states/objects/position"][:]
            if not np.isfinite(pos).all():
                raise ValueError("Nonfinite object position")
            print(name, "steps=", t, "reasons=", reasons)
        print("Alignment checks passed (not a physical-validity or success-rate assessment).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    check(parser.parse_args().path)
