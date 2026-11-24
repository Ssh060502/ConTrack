# Reset and first physics step diagnostics

This is instrumentation, not a reset fix. It does not add warm-up steps, change
targets, skip failed attempts or adjust rewards. It temporarily wraps sim.step
to read inputs/outputs, then restores it. Readbacks cost time; compare results to
the uninstrumented baseline to check for instrumentation effects.

Sync scripts/collect_mug.sh, scripts/rsl_rl/play.py, and these tools:
trajectory_collection.py, reset_trace.py, compare_reset_traces.py.

Start with a consecutive three-episode run:

```bash
bash scripts/collect_mug.sh --gpu 0 --episodes 3 --pose-break-mode log-only --video --trace-reset
```

Then execute the following command three separate times, retaining each directory:

```bash
bash scripts/collect_mug.sh --gpu 0 --episodes 1 --pose-break-mode log-only --video --trace-reset
```

The additional H5 group episodes/000000/reset_trace contains:

- reset_before: entry to the environment reset method (previous terminal state
  on automatic reset; startup state on the first explicit reset).
- reset_after: immediately after that method, before outer step/reset bookkeeping.
- before_policy: before the first policy evaluation, after initial-state recording
  and optional initial-frame rendering.
- after_policy_before_step: after policy evaluation, before env.step processes
  the action. Manager buffers can still hold reset/previous values at this point.
- substeps/000/before and after, and later substeps: around real sim.step calls,
  after normal action application and scene writes. The after sample precedes
  normal scene.update.
- first_control_end: after env.step returns. If the first action terminates,
  this phase is post-reset; the physics substeps still precede reset.

Full articulation buffers include all joints (task and unused), root state,
position/velocity/effort targets, and available torque fields. Joint names are
attributes. Raw PhysX getters use their native layout and world coordinates;
no quaternion conversion is performed. Buffered data may be stale immediately
after sim.step: use physx DOF positions/velocities for fresh substep comparisons.
Unavailable optional getters are recorded; missing required getters fail visibly.
Internal solver contact warm-start state is not exposed or claimed captured.

Version 2 adds required direct `get_link_transforms` and `get_link_velocities`
readbacks and `link_names_json`. The comparator prints per-link translation,
rotation and velocity differences as well as whole-field differences. Quaternion
rotation comparison treats q and -q as equivalent. The trace does not force a
kinematic refresh after teleporting joints: a stale link transform is itself a
diagnostic observation. This does not capture collision shape geometry, contact
manifold caches or solver warm-start data. Existing version 1 files are still
readable but cannot supply the added fields. Do not mix trace versions for a
new link-level comparison.

For a single-factor diagnostic only, add `--diagnostic-no-self-collisions` to
collect_mug.sh. It sets the robot articulation's self-collision flag to false
before environment creation, retains hand-object collision settings, records the
requested setting in H5 metadata, and adds noself_DIAGNOSTIC to the output name.
This is not a proposed physical model for training. Compare first-substep
differences against the original trace; a change in grasp success alone does
not isolate the root cause. This disables robot self-collisions broadly, so a
positive result does not identify a particular finger collision pair.

To collect version 2 data, only sync updated reset_trace.py and
compare_reset_traces.py if the earlier launcher and collector are already present.
Keep the same --trace-reset command. First validate the three-episode run, then
collect the three independent single-episode runs.

Version 2 adds required direct `get_link_transforms` and `get_link_velocities`
readbacks and `link_names_json`. The comparator prints per-link translation,
rotation and velocity differences as well as whole-field differences. Quaternion
rotation comparison treats q and -q as equivalent. The trace does not force a
kinematic refresh after teleporting joints: a stale link transform is itself a
diagnostic observation. This does not capture collision shape geometry, contact
manifold caches or solver warm-start data. Existing version 1 files are still
readable but cannot supply the added fields. Do not mix trace versions for a
new link-level comparison.

To collect version 2 data, only sync updated reset_trace.py and
compare_reset_traces.py if the earlier launcher and collector are already present.
Keep the same --trace-reset command. First validate the three-episode run, then
collect the three independent single-episode runs.

Existing trajectory checking still applies. Compare detailed traces with:

```bash
python scripts/tools/compare_reset_traces.py <fresh1-run> <fresh2-run> <fresh3-run> <three-episode-run> --output reset_trace_comparison.json
```

The first episode in the first input is the baseline. The report shows exact
differences phase by phase; differences alone do not identify causation. Inputs
remain unchanged. Old H5 files cannot acquire these fields retroactively.
