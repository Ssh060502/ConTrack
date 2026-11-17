"""Compare --trace-reset H5 files offline, taking the first episode as baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def read_phase(group):
    values = {}
    def visit(key, value):
        if isinstance(value, h5py.Dataset):
            values[key] = value[()]
    group.visititems(visit)
    return values


def differences(baseline, values):
    rows = []
    for key in sorted(set(baseline) | set(values)):
        if key not in baseline or key not in values:
            rows.append({"field": key, "reason": "missing in one trace"})
            continue
        a, b = np.asarray(baseline[key]), np.asarray(values[key])
        if a.shape != b.shape:
            rows.append({"field": key, "reason": "shape differs"})
        elif not np.array_equal(a, b, equal_nan=True):
            finite = np.isfinite(a) & np.isfinite(b)
            delta = np.abs(a.astype(float)[finite] - b.astype(float)[finite])
            rows.append({"field": key, "max_abs_difference": float(delta.max()) if delta.size else None,
                         "nan_mask_differs": not np.array_equal(np.isnan(a), np.isnan(b))})
    return rows


def compare(paths):
    baseline = None
    reports = []
    for item in paths:
        path = Path(item).expanduser()
        if path.is_dir():
            matches = list(path.rglob("episodes.h5"))
            if len(matches) != 1:
                raise ValueError(f"Provide one run at a time: {path} contains {len(matches)} H5 files")
            path = matches[0]
        with h5py.File(path) as hf:
            if not hf.attrs.get("reset_trace_enabled", False):
                raise ValueError(f"{path}: missing --trace-reset data")
            for episode, group in hf["episodes"].items():
                trace = group["reset_trace"]
                count = int(trace.attrs["decimation"])
                if len(trace["substeps"]) != count:
                    raise ValueError(f"{path}/{episode}: incomplete substep trace")
                phases = ["reset_after", "before_policy", "after_policy_before_step"]
                phases += [f"substeps/{i:03d}/{phase}" for i in range(count) for phase in ("before", "after")]
                phases += ["first_control_end"]
                values = {phase: read_phase(trace[phase]) for phase in phases}
                if baseline is None:
                    baseline = values
                result = {phase: differences(baseline.get(phase, {}), values[phase]) for phase in phases}
                report = {"file": str(path.resolve()), "episode": episode, "differences": result,
                          "unavailable_fields": json.loads(trace.attrs.get("unavailable_fields_json", "{}"))}
                reports.append(report)
                print(f"\n{path} episode={episode}")
                for phase, rows in result.items():
                    print(f"  {phase}: {len(rows)} differing fields")
                    for row in rows[:8]:
                        print("   ", row)
                if report["unavailable_fields"]:
                    print("  Unavailable optional fields:", list(report["unavailable_fields"]))
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="H5 paths or individual run directories")
    parser.add_argument("--output", help="Optional comparison JSON path")
    args = parser.parse_args()
    result = compare(args.paths)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
