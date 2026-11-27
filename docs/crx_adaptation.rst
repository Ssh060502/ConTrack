CRX history adaptation (RBY1A remains supported)
===============================================

Code paths
----------

* Existing RBY1A defaults and commands remain valid.
* oracle_adapter.py accepts --robot crx and --tool hammer/mug. Other robot/tool
  combinations require an explicit --session. CRX defaults use the env8000
  model_1999.pt runs. --session/--base_checkpoint/--data override these defaults.
* The frozen-base residual policy is dimension-independent. CRX diagnostics
  recognize left_J1..J6/right_J1..J6 instead of treating all actions as fingers.
* train_history_encoder.py accepts --robot crx and --tool hammer/mug/etc.
  Defaults remain RBY1A/waterbottle. Collection metadata filters robot before
  feature extraction; incompatible clips/mappings/control rates are rejected.
* History latent control uses --context-source history --history-checkpoint FILE
  in the same adapter launcher. No oracle context group enters either actor or
  critic. The parameter prediction head's two outputs are NOT MLP inputs.

Pretraining objective
---------------------

The existing TransformerEncoder produces z from completed past transitions.
Token i = concat(normalized x_i, normalized u_i, (x_{i+1}-x_i)/state_std).
At decision t, tokens have i<t. State contains joint positions/velocities and
simulated net link contact forces; action contains the recorded manager action
and actuator joint targets (including the effects of action-term scaling/clipping).
It excludes object ground truth,
reference trajectory, phase, true physical labels and future actions.

The parameter head predicts two normalized log parameters::

  y = ( [log(mass_kg), log(object_static_friction)] - train_mean ) / train_std
  L_param = mean(SmoothL1(y_hat, y))
  L_delta = mean(SmoothL1(delta_hat, (x_next-x)/state_std))
  L = L_param + dynamics_weight * L_delta

The mean in L_param averages the mass and friction components. The dynamics
head is a small prediction MLP, not a Transformer text-style decoder. z is the
latent before both heads. There is no per-layer parameter supervision or
contrastive loss. --dynamics-weight 0 implements parameter-only pretraining.
Default 1 preserves the previous RBY1A objective.

Previous RBY1A waterbottle diagnostics found dynamics MSE worse than the zero-delta
baseline despite better-than-constant parameter MAE. This is unresolved; CRX
support alone does not resolve it. Start with parameter-only pretraining and
compare dynamics_weight=1 with the same episode split. Report parameter MAE
against constant baselines and task performance separately.

Data requirements and preparation
---------------------------------

Keep CRX and RBY1A reference data, base weights and history weights separate.
CRX has 56 joint actions, not RBY1A's 58. Feature/action dimensions are inferred
from H5, never copied from the old model. A CRX history checkpoint must have the
same tool, reference/calibration filename, named joints/contact sensors and
control_dt as the online environment.

First collect baseline/mass6/friction1, ten episodes each, using the CRX collector.
Collections need >=3 unique episodes per condition after exact deduplication.
Changed physical conditions, failures and successes are retained. Inspect videos
for reset artifacts and physical validity; alignment checks are insufficient.

Server commands, from ~/ConTrack with env_isaaclab activated::

  python -m unittest discover -s tests -p 'test_history*.py' -v
  python -m unittest discover -s tests -p 'test_adaptation_profiles.py' -v
  python scripts/train_history_encoder.py --robot crx --tool hammer --inputs outputs/physics_validation --prepare-only

The directory may contain old RBY1A files: these are skipped. Multiple CRX hammer
clips/calibrations fail explicitly; then provide only the intended collection
directories using --inputs DIR1 DIR2 DIR3.

Two-epoch neural-network smoke test, parameters only::

  python scripts/train_history_encoder.py --robot crx --tool hammer --inputs outputs/physics_validation --dynamics-weight 0 --epochs 2 --device cuda:0 --logger none

Then train a fresh run (choose a new --output-dir if this already exists)::

  python scripts/train_history_encoder.py --robot crx --tool hammer --inputs outputs/physics_validation --dynamics-weight 0 --epochs 100 --device cuda:0 --logger wandb --project ConTrack-history --output-dir outputs/history_encoder/crx_hammer_params_v1

Episode-level splitting shares collection batches; it is not unseen-condition or
independent-batch validation. --split-unit file requires >=3 independent collection
files per physical condition. Validation chooses history_best.pt; test is evaluated
once after model selection. The frozen manifest preserves deduplicated membership.
evaluate_history_encoder.py and run_history_comparison.py restore the robot from
the saved config (older checkpoints default to RBY1A).

Oracle adapter interface test (can run before encoder training finishes)
-----------------------------------------------------------------------

CRX defaults: conditions 1:2,6:2,1:1; correction_scale .1; task-only objective.
The friction values are object material coefficients, not measured effective
contact friction. Simulator mass/inertia/material readbacks are recorded.
No additional lift/hold reward has been added::

  bash scripts/run_crx_adapter.sh --gpu 0 --num_envs 8 --max_iterations 2 --save_interval 1 --logger tensorboard

After verifying this interface and base performance, start a separate training::

  bash scripts/run_crx_adapter.sh --gpu 0 --num_envs 256 --max_iterations 1000 --conditions '1:2,6:2,1:1' --project ConTrack-adapter

For mug, add --tool mug; the CRX mug env8000 model is selected. RBY1A commands
still use run_mug_adapter.sh/run_waterbottle_adapter.sh without --robot crx.

Frozen history latent + residual RL interface
--------------------------------------------

After history_best.pt exists, this does not require a trained oracle checkpoint::

  bash scripts/run_crx_adapter.sh --gpu 0 --context-source history --history-checkpoint outputs/history_encoder/crx_hammer_params_v1/history_best.pt --num_envs 8 --max_iterations 2 --save_interval 1 --logger tensorboard

Then use the same conditions/budget as the oracle experiment for a comparison::

  bash scripts/run_crx_adapter.sh --gpu 0 --context-source history --history-checkpoint outputs/history_encoder/crx_hammer_params_v1/history_best.pt --num_envs 256 --max_iterations 1000 --conditions '1:2,6:2,1:1' --project ConTrack-adapter

The encoder is frozen/eval; only the residual MLP, its normalizer, critic and
exploration std train. The base policy and base normalization are frozen. The
zero-initialized residual initially reproduces base inference. Outputs follow::

  a = clip(a_base + correction_scale*tanh(MLP(base_observation, z)))

The action is in policy residual-action space, not a direct torque correction.
Environment action scaling/clipping still applies. Training uses PPO task reward
plus the existing penalties, not action-label MSE. There is no successful-action
teacher dataset assumed and no inferred parameter fed to the MLP.

At cold start t=0, z=0 until one transition completes. On automatic reset, the
terminal/reset transition is discarded from history; that environment's entire
buffer is cleared, without clearing other environments. States/targets and applied
actions match the offline extractor. The history head's mass/friction outputs and
all simulator physical labels are excluded from the history context input.

This phase generates fresh closed-loop data because the residual changes the
trajectory distribution. It does not reuse base actions as correct-action labels.
Encoder digest is checked on checkpoint saves; base digest is also checked.
Keep history_best.pt/normalization/manifest together with the residual run. Loading
an oracle residual as a history residual, or changing encoder/robot/reference,
is rejected rather than silently mixing the experiments.

Evaluation uses --mode evaluate --num_envs 1 --conditions '6:2' --episodes 10 --video
and --adapter_checkpoint PATH_TO_RESIDUAL_MODEL. For history also supply the same
--context-source history --history-checkpoint as in training. --policy base with
context-source oracle provides the base comparison. Evaluation saves ten videos
and episodes.h5 and uses pose-break log-only, matching the collector protocol.

Limitations
-----------

Joint fine-tuning of the Transformer with PPO is not implemented here. It requires
storing raw histories/masks in rollout storage and recomputing z during PPO updates;
unfreezing a wrapper that precomputes detached z would not train the encoder.
This implementation deliberately provides the frozen-pretraining experiment first.
Random/frozen and shuffled-history ablations, per-layer losses and cross-attention
remain separate experiments, not claimed as implemented.

Simulator contact forces are not calibrated real tactile data. Matching CRX geometry
does not establish real sensor/actuator alignment. These additions provide code and
CPU contract tests; actual Isaac Lab training/evaluation must run on the server.

Joint-only input ablation
-------------------------

Add --state-features joints to train_history_encoder.py to remove force channels
from BOTH x and state-delta tokens. Past manager actions and actuator targets
remain unchanged. Default all preserves the original inputs and old checkpoints.
Use the original --split-manifest and identical epochs/history/seed/stride/batch
size to compare against the all-feature model. Raw episode hashes identify the
same underlying recordings even after selecting a feature subset. Joint-only
normalization is fit afresh on the same training episodes, without force values.

evaluate_history_encoder.py and diagnose_history_timeline.py restore the selected
input mode from the checkpoint automatically. Timeline force plots still audit
the original force recordings; those plotted forces are NOT joint-only model
inputs. Online history control also loads this feature mode from the checkpoint.
It does not read forces into joint-only history, although the existing base
policy, rewards and simulated physics have not been changed by this ablation.

This is a controlled input ablation, not a repair of reset collisions or a claim
that joint trajectories are unaffected by the startup force transient. Keep the
original recordings and all-feature checkpoints for comparison. Reusing already
inspected test episodes is descriptive analysis, not a fresh independent test.
