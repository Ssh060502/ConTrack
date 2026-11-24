# Frozen-base physical parameter adapter prototype

This is Stage 3's privileged-parameter control experiment, without a history
encoder or Transformer. It reuses existing PPO/CMDP training and the original
reward terms. It is not a claim of solved grasp success or sim-to-real transfer.

The base mug policy is model_1500.pt in the 2026-10-14 03:57:03 training session.
Its actor, critic and normalization statistics are loaded strictly and frozen.
A new 128/128 MLP actor and a new two-output critic are trained. Inputs append
log(actual object mass in kg), log(object material static friction). All shapes
use the same static and dynamic coefficient. These are privileged simulation
parameters, not effective contact-pair coefficients or measured real values.

The deterministic action is clamp(base_action + scale*tanh(adapter_output), -1, 1).
Adapter output starts at zero, so initial inference is exactly the base policy.
Default scale 0.5 is in normalized residual-action units; the environment still
converts residuals into reference-relative joint position targets. It is not a
0.5 rad or torque command. During PPO a Gaussian is sampled around the combined
mean; the environment performs the existing clipping. Only the mean correction
is bounded by scale, not the unbounded Gaussian sample. All 58 reference actions
can be corrected in this version; it is not a hand-only adapter.

## Server files and checks

Sync:

- scripts/run_mug_adapter.sh
- scripts/rsl_rl/oracle_adapter.py
- rsl_rl_contrack/modules/oracle_adapter.py
- rsl_rl_contrack/env/oracle_physics.py
- tests/test_oracle_adapter.py and tests/test_oracle_physics.py

The existing collection modules and Sharpa environment modifications must also
be present. No checkpoint needs to be downloaded to the local PC.

Run in the repo root with env_isaaclab activated:

```bash
python -m unittest discover -s tests -p 'test_oracle*.py' -v
bash scripts/run_mug_adapter.sh --gpu 0 --num_envs 20 --max_iterations 2 --save_interval 1 --logger tensorboard
```

The startup assertion should print `Zero-initialized adapter exactly matches
frozen base inference`. After training the frozen base hash must be unchanged.
The local development environment has no torch/tensordict; numerical physics
assignment tests run there with a NumPy backend double, while the actual network
gradient/normalization/checkpoint test must run on this server. Full Isaac
simulation, PPO updates and checkpoint evaluation require server smoke validation.

Outputs are under logs/oracle_adapter/mug_train_adapter_timestamp_id, including
runner_config.json, run_metadata.json, physics.json, model_*.pt and adapter_final.pt.
Every model checkpoint includes architecture/configuration metadata and the
frozen base. The launcher terminal log is alongside these directories.
Original checkpoints are never overwritten. Untrusted checkpoints should not be
loaded; this follows the repository's existing weights_only=False load path.

## Train

After the smoke check succeeds, start in tmux:

```bash
tmux new -s mug_adapter
conda activate env_isaaclab
cd ~/ConTrack
bash scripts/run_mug_adapter.sh --gpu 0 --num_envs 256 --max_iterations 1000 --project ConTrack-adapter
```

The default five conditions are 1:2,3:2,6:2,1:0.7,1:0.2, meaning mass multiplier
and object material coefficient. Assignment is balanced by environment index,
fixed for that environment for the entire process. Mass and inertia scale
together. Parameters are read back after setting; a mismatch aborts. This first
version does not randomize parameters at every reset, does not add initial-state
noise, and does not establish interpolation/extrapolation performance. It uses
one tracked object only. You can start with an easier subset, e.g.:

```bash
bash scripts/run_mug_adapter.sh --gpu 0 --conditions '1:2,3:2' --num_envs 256 --max_iterations 1000
```

Training uses reference-frame-0 resets, disables the reset-library/threshold
curriculum, leaves robot self-collisions enabled and uses log-only pose breaks.
No first episode is discarded and no settling/warm-up stage is added. The known
process-history sensitivity is not silently fixed or considered resolved here.
New adapter and frozen-base comparisons must use the same entry point/protocol.
The prototype retains the original rewards; evaluation must additionally assess
lift, carrying, loss and orientation. No reward increase is labeled success.

## Evaluate base, zero adapter and learned adapter

Evaluate exactly one condition and one environment per command. Fresh process
and subsequent episode results must be retained and reported separately as needed.

```bash
# Recreate the baseline using exactly the new physics-assignment/reset pipeline.
bash scripts/run_mug_adapter.sh --gpu 0 --mode evaluate --policy base --num_envs 1 --conditions '3:2' --episodes 10 --video

# Before-training adapter: deterministic actions must match base inference.
bash scripts/run_mug_adapter.sh --gpu 0 --mode evaluate --policy zero --num_envs 1 --conditions '3:2' --episodes 10 --video

# Trained adapter (replace the path).
bash scripts/run_mug_adapter.sh --gpu 0 --mode evaluate --policy adapter --adapter_checkpoint <training-directory>/adapter_final.pt --num_envs 1 --conditions '3:2' --episodes 10 --video
```

Evaluation outputs include episodes.h5 with original states/actions/contact
schema and per-episode videos. Their physics_json is the new per-environment
readback report; it should not be assumed identical to old collector metadata.
Run scripts/tools/check_trajectories.py on the evaluation H5. Compare full task
outcomes rather than only final pose errors. Finishing at time_out is not success.

Resume uses --adapter_checkpoint with --mode train; it restores the adapter,
critic, optimizer and PPO state using the existing runner semantics. Supply the
same original base checkpoint (default path above). A digest mismatch between
the supplied base and embedded frozen base aborts. --max_iterations on resume
means additional iterations. The base checkpoint is still needed at construction;
use --base_checkpoint and --session if paths moved.
