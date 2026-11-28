RBY1A waterbottle history study
=============================

Run in env_isaaclab on the server, from ~/ConTrack. No Fanuc assets are needed.
Existing oracle/history commands keep their defaults. D (joint fine-tuning) is
not implemented in this study. No real-robot commands are involved.

Methods
-------

* state (A): current base observations plus 64 constant zeros. No oracle labels
  or history. Zero padding matches B/C MLP dimensions, not effective information.
* random-history (B): randomly initialized, frozen Encoder with an isolated seed;
  schema, train-only normalization and architecture come from C's checkpoint.
* history (C): pretrained, frozen Encoder; residual receives z, not parameter predictions.
* oracle: log actual mass/friction. Its two context dimensions differ from A/B/C;
  treat as a privileged reference, not a parameter-count-matched ablation.
* base: evaluation only, no residual.

All learned policies freeze the base, train the residual online, retain existing
reward terms and use task-only PPO. Histories contain completed transitions,
actual manager actions, joint position targets and observed state changes. Done
environments clear history. Frozen Encoders remain in eval mode. The base keeps
its original observations: joints-only restricts only the history branch.

Data and offline Encoder
-----------------------

The default input list names four old base collection directories explicitly:
default, mass6, friction0.5, friction1. It does not scan all physics_validation.
Override with --inputs if you moved them. Missing/incomplete files fail visibly.
The local audit found 40 unique episodes, 10 per condition; joint state dimension
116, action/target dimension 116. Exact deduplication does not detect near-duplicates.
These are development data, previously inspected; reserve new batches/conditions
for final evaluation. Episode splits are not independent collection batches.

First audit and freeze the split::

  python scripts/run_waterbottle_study.py audit --output outputs/waterbottle_audit_v1.json
  python scripts/run_waterbottle_study.py prepare --prepare-only --output-dir outputs/history_encoder/waterbottle_prepare_v1

Two-epoch interface check, NOT a final research checkpoint::

  python scripts/run_waterbottle_study.py prepare --epochs 2 --device cuda:0 --split-manifest outputs/history_encoder/waterbottle_prepare_v1/manifest.json --output-dir outputs/history_encoder/waterbottle_encoder_smoke_v1

Actual pretraining (joints only, parameter SmoothL1 only, dynamics weight zero)::

  python scripts/run_waterbottle_study.py prepare --epochs 100 --device cuda:0 --split-manifest outputs/history_encoder/waterbottle_prepare_v1/manifest.json --output-dir outputs/history_encoder/waterbottle_encoder_v1

Labels are normalized log(actual mass in kg), log(object material friction).
Data from different conditions train one model, not separate mass/friction models.
history_best.pt is selected on validation loss. Repeatedly inspected test results
are development evidence, not new blind tests. Use --history 1 and a new output
directory for a short-history comparison with the same split manifest.

Online study
------------

First run all four train modes for two iterations to catch simulator integration
errors. This tests training plumbing, not task success. All children run serially::

  python scripts/run_waterbottle_study.py train --history-checkpoint outputs/history_encoder/waterbottle_encoder_smoke_v1/history_best.pt --iterations 2 --num-envs 12 --gpu 0

Formal training, same settings for all four methods::

  python scripts/run_waterbottle_study.py train --history-checkpoint outputs/history_encoder/waterbottle_encoder_v1/history_best.pt --conditions 1:2,3:2,1:0.7 --iterations 1000 --num-envs 252 --gpu 0

Defaults are 32 rollout steps, save every 100 iterations, correction scale 0.1,
seed42, encoder seed123. Use --methods to run a subset. --dry-run prints commands
without opening Isaac. Per-env physics is fixed, not resampled each episode.
Conditions here are provisional, not selected by held-out test performance.

The printed study directory contains study.json, method logs and per-method
directories with adapter_final.pt, model_*.pt, diagnostics.jsonl, physics.json,
run_metadata.json. B/C additionally save history_encoder_used.pt. Keep that file,
the source Encoder checkpoint, base checkpoint and normalization for reproduction.
Evaluation reconstructs B from the template and seed and verifies its weight hash.

Evaluate the same preselected model filename across methods::

  python scripts/run_waterbottle_study.py evaluate --study <training-study-directory> --conditions 1:2,3:2,1:0.7 --episodes 10 --gpu 0 --video

Without --video, collection skips video rendering/encoding. --model model_500.pt
chooses an earlier checkpoint for all methods. Choose it using validation, not
the final test. Do not combine incomplete training studies with default all-method
evaluation: specify --methods base state, for example, if only state was trained.

Each evaluation method/condition directory contains episodes.h5 and optional
videos/episode_*.mp4. The batch checks completion, condition/source identity and
physical readback consistency. Multiple seeds alone may reproduce identical
trajectories under the fixed reset; do not count them as independent trials.

Manual success assessment
-------------------------

Create a video annotation template, fill every outcome, then summarize::

  python scripts/tools/score_waterbottle_study.py --study <evaluation-study-directory> --create-labels
  python scripts/tools/score_waterbottle_study.py --study <evaluation-study-directory>

Outcomes: no_lift (no grasped lift), dropped (lift then loss of grasp), completed
(maintained grasp and completed reference operation), incomplete (grasped but did
not complete, without a clear drop), uncertain. A collision-induced toss is not
a grasped lift. Set the rubric before comparing methods; full completion is not
defined by time_out, final pose alone or total reward. Missing/uncertain labels
leave completion_rate null. Keep notes and report numerator/denominator and
uncertainty. This does not establish real-robot transfer.
