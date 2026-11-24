# CRX-5iA + LEAP Hand (Paxini fingertips)

Single CRX arm plus the existing 16-joint LEAP subtree, 22 commanded joints:
`J1`–`J6`, then `joint_0`–`joint_15`. This is a URDF/kinematic embodiment;
there is no CRX MJCF, dynamics, collision-aware planner, or hardware adapter.

## Frames and initialization

The default `world -> base_link` transform is identity. CRX geometry is not
mirrored or modified for the right side. The configured arm initial pose is
**`[-90, -30, 210, 0, -60, 0]` in ROS / URDF degrees**, matching CRX + Sharpa
and `dual_crx_control/config/initial_pose.yaml`. It replaces the earlier
`[-90, 0, 180, 0, 90, 0]` home. J6 remains within its source limits of
[-225, 225] degrees.
This is a model initialization, not a
measurement or command to the physical robot. LEAP initial joints follow
`panda_leap_paxini`.

```text
world -> base_link -> J1_link ... J6_link -> flange -> palm_lower -> wrist
             |                               |         +-> LEAP fingers
             +-> wbase                       +-> ee_mount
                                             +-> fanuc_flange
```

- `base_link -> wbase`: translation `[0, 0, 0.185]` m.
- `J6_link -> flange`: translation `[0.145, 0, 0]` m.
- `flange -> ee_mount`: identity.
- `flange -> fanuc_flange`: zero translation, fixed-axis RPY `[180, -90, 0]` degrees.
- **Physical-test** `flange -> palm_lower`: XYZ `[0.01, -0.03, -0.065]` m,
  RPY `[-pi, 1.56, 0]` radians. This is the 180-degree remount merged on
  2026-09-09. The exact pitch is retained (about 89.381 degrees, not exactly
  90). It supersedes the earlier static-viewer transform; physical metrology is
  still pending.
- `palm_lower -> wrist`: XYZ `[-0.16, -0.04, -0.01]` m,
  RPY `[0, pi/2, 0]`. This is the existing hand's optimization frame.

Tune `flange_to_leap` to match the physical bracket later. Do not change the
Quest normalization or confuse `flange` with `fanuc_flange` to compensate for
mounting differences. J3/J5 rotate about local -Y; J4/J6 about local -X.
Keep URDF joint coordinates and limits, including legal angles outside [-pi, pi].

The following placements are reference setup data, not baked into this model:

| dual_crx right-arm setup | table-from-base XYZ (m) | fixed-axis RPY (rad) |
| --- | --- | --- |
| Mock YAML | `[0, -0.3, 0]` | `[0, 0, -1.5707963267948966]` |
| Physical calibration YAML | `[0.2559220458, -0.5374404144, 0.0621767901]` | `[0.0028907952, -0.0022817890, -0.0000199954]` |

Sources: `dual_crx_description/config/robot_placement{,_physical}.yaml` at
`dad54b1`. `launch_real.sh --arm both` uses the physical calibration; its
single-arm path only sets `origin_y` by default. The physical file's -35 mm
contact correction is in the measurement frame, not a flange-local tool offset.
Do not apply it or the 185 mm offset again to a solved table-from-base placement.

## Retargeting configuration

Select `retargeting_profiles=vector_wrist_joint_crx5ia_leap_paxini`.
The existing vector/wrist solver, fingertip targets, benchmark metadata and
execution flow are reused. Arm DOF is 6; finger limit indices shift from the
Panda profile by one. Arm posture weights start at zero instead of copying
Panda's joint-specific preferences. The arm temporal weights remain 0.1;
command speed settings are 0.2 rad/s for CRX joints, but the existing execution
policy does not apply a continuous speed limit after startup interpolation.
This is an offline integration profile, not a validated physical control policy.

The existing soft hand-vector objective trades wrist pose against fingertip
matching. A finite 22-value result does not imply a <=1 mm / <=1 degree wrist
IK solution, continuity guarantee, collision clearance, or physical tracking.

```bash
# Headless offline input, solver, kinematic backend, and diagnostic artifact.
env -u PYTHONPATH /home/howard/.venvs/retargeting_crx/bin/python \
  scripts/crx5ia_leap_offline_demo.py --frames 20

# Standard app path, also headless by default.
env -u PYTHONPATH /home/howard/.venvs/retargeting_crx/bin/python \
  -m retargeting_apps.main app=teleop_exe \
  retargeting_profiles=vector_wrist_joint_crx5ia_leap_paxini \
  teleoperation_modes=offline_kinematic input.end=19
```

After explicit authorization for live tracking/viewer, use the same profile
with `teleoperation_modes=online_quest3_kinematic viewer.enabled=true`.
Use the machine's ADB path via `input.adb`. Do not select
`online_quest3_mujoco`: this embodiment has no simulation model.
The standard Viser URDF path requires `yourdfpy`, and FANUC DAE visuals require
`pycollada`. On 2026-09-07, following Howard's installation authorization,
`yourdfpy==0.0.60` and `pycollada==0.9.3` were installed in the local project
environment. The headless visual/collision mesh loading test passed and
`pip check` reported no broken requirements. The viewer itself was not started.

## Source and reproducible generation

FANUC source: `FANUC-CORPORATION/fanuc_description`, commit
`fb40c9803a826ba68c7c8e28ba904a25efa7fcd2`, CRX package version 2.3.0.
`provenance/` retains the original macro, package metadata, repository README,
and Apache-2.0 license. The source README and package declare Apache-2.0;
the macro credits FANUC America Corporation and FANUC CORPORATION, 2025–2026.
The imported mesh directory has no additional per-file notice files. DAE
files retain their contributor/material metadata. LEAP assets are reused
from the existing repository bundle with its existing provenance.

The manifest records hashes of imported source files and the LEAP URDF.
Generation expands only the pinned macro with identity base placement,
rewrites FANUC mesh URIs, and appends the existing hand subtree with the
Howard-adjusted flange mount. It preserves source geometry, inertials, joint
limits and axes. Collision STL scale remains 0.001; visual DAE units are metres.
Unlike the standalone upstream display wrapper, this model does not add 0.75 m
to its world origin. All runtime resources resolve within this repository;
LEAP meshes are shared with `panda_leap_paxini`, not copied or symlinked.

```bash
# Development-time regeneration only; requires an existing xacro installation.
# This does not launch or source ROS. No Xacro dependency at runtime or in tests.
env -u PYTHONPATH /home/howard/.venvs/retargeting_crx/bin/python \
  scripts/import_crx5ia_leap.py \
  --fanuc-source /home/howard/ws_fanuc/src/fanuc_description \
  --xacro-python-path /opt/ros/jazzy/lib/python3.12/site-packages
```

For comparison without overwriting tracked assets, pass `--output /tmp/crx-import`.
The importer rejects a mismatched commit or dirty FANUC checkout. Nothing in
runtime configuration refers to that checkout or requires a ROS package lookup.

## Verification checkpoints

The 2026-09-07 user-tested setup used flange-to-palm XYZ
`[0.01, 0.03, 0.065]` m, RPY `[0, -1.56, 0]` rad, and initial J6 = 180
degrees. Howard reported that the live Quest + kinematic viewer test worked.
The current 2026-09-09 physical-test setup instead uses XYZ
`[0.01, -0.03, -0.065]` m, RPY `[-pi, 1.56, 0]` rad, and initial J6 = 0.
The earlier visual result does not validate this remount. Both checkpoints use
the shared Quest `world_to_robot.rotation_euler_xyz_deg = [90, 0, 0]`. This is
user-reported visual evidence, not a new quantitative pose accuracy or
physical-control measurement.
The shared Quest YAML now uses this CRX setup; Panda users should explicitly
restore their previous `[90, 0, -90]` world calibration if appropriate.

For a static installation check, run `scripts/view_robot_initial.py`; it only
displays configured initial joints and base/flange/palm/wrist axes, without
an input, solver, or robot backend. Restart it after editing the URDF/config.

The numeric replay/solver results below were recorded with the original
provisional mount; they are historical and do not measure the adjusted mount.

- Full headless suite: `173 passed, 1 skipped` (optional viewer loader absent).
  `python -m compileall -q src tests scripts` and `git diff --check` passed.
  After installing viewer dependencies, the previously skipped mesh test passed
  separately (`1 passed`); upstream dependency deprecation warnings remain.
- Identity-base zero, both reference home postures and 20 seeded random legal
  postures match independently transcribed upstream serial-chain FK at `1e-10`
  matrix tolerance. Nonidentity placement is applied exactly once.
- Wrist linear/angular Jacobians match central differences at nonzero joints
  with `1e-8` tolerance, including zero wrist influence from the 16 finger joints.
- Twenty nonsingular reachable robot-vector targets, seeded within 0.03 rad
  of their source joints, solve to maximum wrist errors approximately
  `1.072e-6 m` / `0.0513 degrees`. This numerical capability check uses
  `ftol_abs=1e-10` and exact robot-vector references without human pinch
  rescaling or posture/temporal penalties; it is not the default live objective.
- Default-profile 20-frame archived AVP demo: maximum wrist residual
  `0.0171222 m` / `9.1401 degrees`, average solve time about `0.025 s` on this
  machine. Results and metadata are in `outputs/crx5ia_leap_offline/`.
- Fresh generation into `/tmp` reproduced the URDF and manifest byte-for-byte.
  All 14 FANUC mesh hashes match their source; collision STL metric extents
  were checked. DAE units, geometry arrays, and absence of external textures
  are checked statically; optional full viewer mesh loading is explicitly skipped
  when its dependencies are unavailable.
- Remaining work: physical mount tuning, visual/Quest checks, unreachable and
  near-singular behavior characterization and continuous branch-change checks.
  No full wrist-IK success/failure contract was added to the existing solver.
  No GUI, sensor, ROS, or physical robot was started for this implementation.
