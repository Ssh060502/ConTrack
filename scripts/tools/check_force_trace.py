"""Inspect first-control contact forces without loading Isaac Sim."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def summarize(group):
    result = {"forces_N": {}, "joints": {}}
    def visit(name, item):
        if not isinstance(item, h5py.Dataset):
            return
        if name.startswith("contact/") and item.shape and item.shape[-1] == 3:
            values = item[:]
            result["forces_N"][name] = float(np.linalg.norm(values, axis=-1).max()) if values.size else 0.0
    group.visititems(visit)
    if "robots" in group:
        for name, robot in group["robots"].items():
            q = robot["physx/get_dof_positions"][:]
            v = robot["physx/get_dof_velocities"][:]
            target = robot["buffered/joint_pos_target"][:]
            result["joints"][name] = {
                "max_abs_velocity_rad_s": float(np.abs(v).max()),
                "max_abs_position_target_error_rad": float(np.abs(q-target).max()),
            }
    return result


def check(path):
    path = Path(path)
    if path.is_dir():
        path = path / "episodes.h5"
    report = {}
    with h5py.File(path) as hf:
        for name, episode in hf["episodes"].items():
            if "reset_trace" not in episode:
                raise ValueError("No reset_trace: collect with --trace-reset first")
            trace = episode["reset_trace"]
            phases = [p for p in ("reset_after", "before_policy") if p in trace]
            for step in sorted(trace["substeps"]):
                phases.extend(f"substeps/{step}/{phase}" for phase in ("before", "after"))
            phases.append("first_control_end")
            report[name] = {phase: summarize(trace[phase]) for phase in phases}
            report[name]["unavailable_fields"] = json.loads(trace.attrs["unavailable_fields_json"])
            for phase in phases:
                forces = report[name][phase]["forces_N"]
                fresh = {k:v for k,v in forces.items() if "/physx/get_net_contact_forces" in k}
                peak = max(fresh, key=fresh.get) if fresh else None
                print(f"episode={name} {phase}: " + (f"{peak}={fresh[peak]:.2f} N" if peak else "fresh contact readback unavailable"))
    output = path.parent / "force_trace_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Report:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Collection directory or episodes.h5")
    check(parser.parse_args().path)
