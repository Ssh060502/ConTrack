"""Load the converted dual-arm CRX + Sharpa USD in PhysX and report what it contains.

Checks the things the training task depends on: one fixed-base articulation,
56 named joints, the finger links that carry contact sensors, and the baked
arm base poses. No training, no ROS, no hardware.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--usd', type=Path, default=REPO / 'assets/usd/crx_sharpa/crx_sharpa.usd')
parser.add_argument('--record', type=Path, default=REPO / 'assets/urdf/crx_sharpa.json',
                    help='Written by build_crx_sharpa_urdf.py')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
launcher = AppLauncher(args)


def main():
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    usd = args.usd.resolve()
    if not usd.is_file():
        raise FileNotFoundError(usd)
    record = json.loads(args.record.read_text(encoding='utf-8'))
    expected = record['joint_names']['left'] + record['joint_names']['right']
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device=args.device))
    robot = Articulation(ArticulationCfg(
        prim_path='/World/CrxSharpa',
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)),
        actuators={'all': ImplicitActuatorCfg(joint_names_expr=['.*'], stiffness=0., damping=0.)}))
    sim.reset()
    robot.update(1/120)
    joints, bodies = list(robot.joint_names), list(robot.body_names)
    problems = []
    print(f'[crx] fixed base: {robot.is_fixed_base}; joints: {len(joints)}; bodies: {len(bodies)}')
    if not robot.is_fixed_base:
        problems.append('articulation is not fixed-base')
    if sorted(joints) != sorted(expected):
        problems.append(f'joint names differ: missing {sorted(set(expected)-set(joints))}, '
                        f'unexpected {sorted(set(joints)-set(expected))}')
    print('[crx] joint order in PhysX (tasks must look joints up by name):')
    for start in range(0, len(joints), 6):
        print('       ' + ', '.join(joints[start:start+6]))
    contact = [f'{side}_{finger}_{part}' for side in ('left', 'right')
               for finger in ('thumb', 'index', 'middle', 'ring', 'pinky') for part in ('PP', 'DP')]
    missing = [name for name in contact if name not in bodies]
    print(f'[crx] contact-sensor links missing: {missing}')
    if missing:
        problems.append(f'missing contact links {missing}')
    print(f"[crx] palm links kept as separate bodies: {[b for b in bodies if 'C_MC' in b]}")
    print(f'[crx] root body: {bodies[0]}')
    # Zero-pose link positions against the base poses recorded when the URDF was built.
    positions = robot.data.body_pos_w[0].detach().cpu().numpy()
    for side in ('left', 'right'):
        base = np.asarray(record['bases_xyz_rpy'][side][:3])
        # J1's origin sits 0.185 m above its base along the (near-vertical) base z axis.
        actual = positions[bodies.index(f'{side}_J1_link')]
        offset = float(np.linalg.norm(actual - base))
        print(f'[crx] {side}_J1_link at {np.round(actual, 4).tolist()}; '
              f'{offset:.4f} m from recorded base {np.round(base, 4).tolist()} (expect 0.185)')
        if abs(offset - .185) > 2e-3:
            problems.append(f'{side} base pose in USD does not match {args.record.name}')
    limits = robot.data.joint_pos_limits[0].detach().cpu().numpy()
    for name in ('left_J3', 'right_J3', 'right_index_MCP_FE'):
        low, high = limits[joints.index(name)]
        print(f'[crx] {name} limits: [{low:.4f}, {high:.4f}] rad')
    if problems:
        print('[crx] FAILED:')
        for problem in problems:
            print('       - ' + problem)
    else:
        print('[crx] OK: asset matches what the CRX task expects')
    return 1 if problems else 0


try:
    code = main()
except Exception:
    import traceback
    traceback.print_exc()
    code = 1
# SimulationApp.close() can hang on this server; results are already printed.
sys.stdout.flush()
sys.stderr.flush()
import os
os._exit(code)
