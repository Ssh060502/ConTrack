"""Create a portable scene bundle without Isaac Lab, ROS or robot connections."""
import argparse
from pathlib import Path
from fanuc_scene_data import prepare_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retargeting-root', type=Path, required=True)
    parser.add_argument('--trajectory', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--table-bounds', type=float, nargs=4,
                        default=[.18358936, .88358936, -1.15578132, .24421868],
                        metavar=('XMIN', 'XMAX', 'YMIN', 'YMAX'),
                        help='Approximate review footprint; not a measured calibration')
    args = parser.parse_args()
    result = prepare_bundle(args.retargeting_root, args.trajectory, args.reference,
                            args.output, args.table_bounds)
    print(f"Prepared {args.output.resolve()}: {result['frames']} frames, "
          f"{result['duration_seconds']:.3f}s; source hashes recorded")


if __name__ == '__main__':
    main()
