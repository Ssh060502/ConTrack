Fanuc + Sharpa scene verification
================================

This is a separate standalone Isaac Lab scene, not an edit to an RBY1A task,
not a registered RL environment, and not a trained Fanuc policy. No ROS is
started and no hardware commands are sent. Existing PPO/adapter entry points,
RBY1A assets, checkpoints and task registrations are unchanged.

Prepare locally
---------------

The preparation program needs NumPy and h5py, not Isaac Lab. Run from ConTrack::

  .venv-replay-check/Scripts/python.exe scripts/tools/prepare_fanuc_scene.py --retargeting-root D:/Visiting/MSc/retargeting_crx_sharpa --trajectory D:/Visiting/MSc/retargeting_crx_sharpa/outputs/hammer/both_tabletop_z175_slow7.npz --reference grab-s1_hammer_use_1-rby1a_sharpa.h5 --output outputs/fanuc_sharpa/hammer_z175_v1

The output directory must be new. Referenced robot meshes are copied with
relative URDF paths, explicit scales retained. Missing meshes and Git LFS
pointers fail preparation. Licenses/provenance accompany the bundle. H5
contains the hammer mesh; NPZ supplies transformed object poses, joint motion,
base transforms and table height. These files have different time bases and
are never paired by frame index. Source and bundled-file SHA256 hashes are saved.

The supplied recording has 357 frames at 8.57142857 Hz: 41.533 seconds from
first to last sample, preserving 7x slowdown. Its right-base transform takes
precedence over current upstream launch defaults. Hand mounting is inherited
from the generated retargeting URDF, with mount_calibrated=false retained.

The default tabletop footprint comes from the earlier approximate review
bounds, not a fresh physical measurement. Thickness 3 cm is also an assumption.
Change preparation's --table-bounds XMIN XMAX YMIN YMAX if needed.

Offline checks
--------------

From the local workspace::

  .venv-replay-check/Scripts/python.exe -m unittest discover -s tests -p test_fanuc_scene.py -v
  .venv-replay-check/Scripts/python.exe scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --check-only

These test coordinate/quaternion conventions, temporal interpolation, asset
portability/integrity, joint limits, and rejection of bad IK/baked base offsets.
They do NOT test PhysX import, rendering, contact dynamics or grasp success.

Move the prepared bundle to the Linux server
------------------------------------------

The generated directory is ignored by git, so git pull does not transfer it.
An archive created locally contains a top-level hammer_z175_v1/ directory.
Copy it using the same SSH host/IP normally used for that server, then::

  cd ~/ConTrack
  mkdir -p outputs/fanuc_sharpa
  unzip /path/to/hammer_z175_v1.zip -d outputs/fanuc_sharpa
  conda activate env_isaaclab
  python scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --check-only

Also transfer the four new files in scripts/tools/: fanuc_scene_data.py,
fanuc_scene_runtime.py, prepare_fanuc_scene.py, and run_fanuc_scene.py.
The original source H5 and external
retargeting checkout are not needed on the server once the bundle is prepared.

The delivered fanuc_scene_hammer_z175_v1.zip already includes these four scripts,
the test, this guide and outputs/fanuc_sharpa/hammer_z175_v1/. It can instead be
extracted directly into ~/ConTrack; do not use the -d outputs/fanuc_sharpa option
for this combined code-and-resource archive. It has no RBY1A files.

Server smoke tests, then full replay
----------------------------------

First run a short geometry-only check::

  python scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --mode reference --duration 2 --video --headless --device cuda:0

Then remove --duration 2 for the full 41.533-second reference. In reference mode,
robot joints and object pose are directly written every physics step, with
collisions and gravity disabled and zero velocity, then stepped once so the
renderer follows. A visible grasp here is only geometry playback. The run
prints an initial joint readback error and fails if the written pose is absent.

Check dynamic import/contact behavior briefly, then run the full trajectory::

  python scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --mode dynamic --duration 2 --video --headless --device cuda:0
  python scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --mode dynamic --video --headless --device cuda:0

Dynamic mode initializes joints/object once and then runs physics with gravity.
Object pose is NOT overwritten after initialization. The robot receives recorded
joint-position targets at 50 Hz with zero-order hold, integrated at 120 Hz.
This is not yet an emulation of the 500 Hz ROS interpolator/physical servos.
Noninteger control/physics ratios quantize target delivery to physics steps.
Neither mode consumes an RBY1A policy checkpoint or trains PPO.

Self collision defaults on. --no-self-collision is an explicit diagnostic switch.
Both modes verify imported joint names/counts and root poses against the bundle.
Joint lookup uses names, not the importer-dependent joint array ordering.

Provisional dynamics
--------------------

Default object mass is 0.1 kg, explicitly an unmeasured engineering assumption.
--object-mass sets nominal kg; --mass-scale multiplies it. --friction sets static
and dynamic friction on BOTH robots, object and table, with average combination.
Default coefficient is 2.0; it is not a measured material value.

Example, after baseline scene verification::

  python scripts/tools/run_fanuc_scene.py --bundle outputs/fanuc_sharpa/hammer_z175_v1 --mode dynamic --object-mass 0.1 --mass-scale 3 --friction 0.7 --video --headless --device cuda:0

Arm PD defaults: 1000 Nm/rad, 50 Nm/(rad/s). Hand PD defaults: 8 Nm/rad,
0.3 Nm/(rad/s). Effort and velocity limits come from each URDF joint. These
are provisional force-based PD settings, NOT a conversion of SDK MIT gains.
CLI options --arm-stiffness/--arm-damping/--hand-stiffness/--hand-damping expose
them for later identification. The same hand gains are initially applied to
all hand joints. Articulation contact response has not been calibrated.

Mesh collision cooking defaults to convex decomposition; --collider convex_hull
is an explicit approximation for diagnosis. Collision cooking and physical
success must be checked in Isaac Lab, especially fingertips and the hammer.
The recorded slowdown materially changes the manipulation dynamics.

Outputs and scope
-----------------

Every invocation gets a unique outputs/fanuc_scene/MODE_TIMESTAMP_ID/ directory::

  run_config.json       Arguments, assumptions and source bundle manifest
  scene.usda           Stage snapshot referencing this run's USD cache
  usd_cache/           Server-generated robot/object USD files
  video.mp4            Optional 30 FPS recording in simulation time
  scene_rollout.h5      Named joint states, object states, references and commands
  metrics.json         Tracking errors; success_assessed is always false
  FAILED.txt           Exception details if a run fails

H5 state/reference arrays have T+1 rows; commanded_joint_position has T rows.
Object state is XYZ + WXYZ + linear/angular velocity (13 values). Joint arrays
are left arm+hand then right arm+hand (56). No tactile data or policy actions are
fabricated. This format is scene review data, not the existing episode collector
schema and not a history-encoder training dataset. Partial runs have complete=false
when the recording loop has started. Rendering/import failures may precede H5 creation.

Before RL work, verify the full reference placement/video, imported collision
geometry, initial penetration, dynamic tracking and object mass/inertia. A later
Fanuc RL task still needs reference conversion/loading, observations, actions,
reward/termination design and a Fanuc base policy; these are not claimed here.

Implementation uses Isaac Lab's public URDF, articulation and camera interfaces:
https://github.com/isaac-sim/IsaacLab/tree/v2.3.0/scripts/tutorials
Local offline tests do not establish compatibility with an untested server build.
