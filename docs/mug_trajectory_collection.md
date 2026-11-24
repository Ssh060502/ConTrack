# Closed-loop mug collection

This records the frozen ConTrack policy (`model_1500.pt` in the 2026-10-14
03:57:03 mug session), without training. Keep the session YAML files alongside
the checkpoint. The server paths are configured in `scripts/collect_mug.sh`.

Sync these files to the server, preserving paths:

- `scripts/collect_mug.sh`
- `scripts/rsl_rl/play.py`
- `scripts/tools/trajectory_collection.py`
- `scripts/tools/check_trajectories.py`

The previously updated Sharpa environment config must also be present.

First run a small smoke check inside `env_isaaclab`:

```bash
bash scripts/collect_mug.sh --gpu 0 --episodes 2
python scripts/tools/check_trajectories.py <printed-output-directory>/episodes.h5
```

For synchronized diagnostic video, use
`bash scripts/collect_mug.sh --gpu 0 --episodes 2 --video`.
To observe the entire reference horizon without pose-error truncation, use
`bash scripts/collect_mug.sh --gpu 0 --episodes 2 --video --pose-break-mode log-only`.
The default remains `strict`. Log-only preserves all threshold diagnostics but
returns false for the pose-break termination term; reference end/time-out still
ends an episode. A dropped object may remain dropped until that horizon ends.
`time_out` does not imply grasp success. Mode is stored in H5 metadata and the
output directory name; do not pool modes without accounting for the difference.
Each episode has its own `videos/episode_000000.mp4`. The collector renders the
terminal frame inside its pre-reset hook; video frame 0 is the initial state,
followed by one frame per action, including the terminal state. The original
reset behavior and thresholds are retained. This rendering path requires server
validation and imageio/FFmpeg support.

Every transition also stores `transitions/diagnostics/`: the reference frame,
actual/reference positions, position and rotation errors, effective thresholds,
per-object exceeded flags, raw `threshold_exceeded` and active `pose_break` flags. These are sampled
inside the termination function before reset and reference-frame advancement.
Use them to explain termination, rather than post-step `metrics/` whose reference
counter can already have advanced. The checker verifies that the diagnostic
threshold crossings agree with the recorded termination flags.

Then collect the four conditions separately:

```bash
bash scripts/collect_mug.sh --gpu 0 --episodes 10
bash scripts/collect_mug.sh --gpu 0 --episodes 10 --mass-scale 3.0
bash scripts/collect_mug.sh --gpu 0 --episodes 10 --mass-scale 5.0
bash scripts/collect_mug.sh --gpu 0 --episodes 10 --friction 0.2
```

Each run creates `outputs/physics_validation/mug_collect_mass.../` containing
`episodes.h5` and `run.log`, plus `videos/` when `--video` is enabled. Existing
`run_mug.sh` remains available for video comparisons.

## Semantics

- Single environment, deterministic inference, default seed 42. Repeat runs with
  identical settings may be duplicates. These are validation data, not yet a
  diverse training corpus or an independent success-rate estimate.
- Every reset uses reference frame zero and the original reference reset state.
  Training reset-library updates and termination-threshold curriculum are disabled.
  Position/rotation thresholds are fixed at the config maximums and recorded.
- Physics parameters stay fixed for a process and apply to all tracked objects.
- Each `episodes/000000` group is a complete episode, including failed attempts.
  States have T+1 entries, policy observations and transitions have T entries.
- A collector-local `_reset_idx` hook captures final physical state, joint targets,
  applied action and termination terms before automatic reset. The next episode
  starts with the reset state. If the hook fails to capture a terminal state,
  collection raises an error rather than writing a reset state as the final state.
- There is no terminal policy observation: only the exact observation actually
  supplied to each action is stored as a numeric vector using the actor's group
  selection and concatenation order, before observation normalization. Group
  selection is recorded in `policy_obs_groups_json`. Do not use next episode observations as the
  previous episode's next observation. Contact positions may contain NaNs for
  absent contacts; contact forces are simulation link contacts, not taxel tactile.
- `policy_action` precedes wrapper clipping, `applied_action` is the action manager
  input, and `hands/*/joint_position_target` is the processed position target.
- Positions are relative to the environment origin; forces/velocities use world
  axes; quaternions use wxyz. Reference frames are the actual environment counters
  at snapshot time (terminal/reset event ordering can affect counters).
- Metrics include object height change and reference position/orientation error.
  These are not grasp-success labels. Time-out alone is not declared success.
- The global step cap is 100000. At the cap or application shutdown, an unfinished
  attempt is saved under `partial_episode`, excluded from completed episode count;
  the launcher exits unsuccessfully if the requested count was not reached.
- Completed episodes are flushed to disk as they finish. Forced process termination
  may still lose the in-memory episode or damage the file; exit normally when possible.

Offline checks validate H5 alignment and synthetic reset-boundary handling. Actual
Isaac Lab reset timing, contact channels and physical validity require a server
smoke run before collecting training data.
