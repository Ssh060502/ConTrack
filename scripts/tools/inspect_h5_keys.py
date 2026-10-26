"""Print every dataset's path, shape, and dtype in an h5 file, plus any top-level
attributes. Works on any h5 file (GRAB reference, play.py --export_rollout output, or
anything else) without assuming a particular schema -- useful for checking exactly
which fields a given file does or doesn't have.

Usage:
    python scripts/tools/inspect_h5_keys.py /path/to/file.h5
"""

from __future__ import annotations

import argparse

import h5py


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to the .h5 file to inspect")
    args = parser.parse_args()

    with h5py.File(args.path, "r") as f:
        print(f"File: {args.path}")
        if f.attrs:
            print("Top-level attrs:")
            for key, value in f.attrs.items():
                print(f"  {key} = {value}")

        def visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"{name}/ (group)")

        f.visititems(visitor)


if __name__ == "__main__":
    main()
