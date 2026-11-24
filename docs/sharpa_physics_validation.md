# Sharpa hammer physics validation (no retraining)

Use the existing hammer checkpoint with `scripts/rsl_rl/play.py`. This is a
closed-loop policy evaluation, not reference-only playback or PPO training.

## Server launcher

After activating `env_isaaclab`, run:

```bash
bash scripts/run_hammer.sh
bash scripts/run_hammer.sh --mass-scale 2.0
bash scripts/run_hammer.sh --friction 0.5
bash scripts/run_hammer.sh --mass-scale 2.0 --friction 0.5
bash scripts/run_hammer.sh --gpu 1 --mass-scale 2.0
```

The launcher uses the configured hammer checkpoint and dataset and ignores
inherited physics overrides. Omitting friction retains original inheritance.
Each run gets a unique directory under `outputs/physics_validation/`, with
mass/friction in its name, containing `rollout.h5`, `run.log` and `videos/`.
The log records settings. Paths are configured in `scripts/run_hammer.sh`.
The old `run_hammer_default.sh` forwards to the new launcher.

`--gpu INDEX` selects one CUDA-visible GPU (default 0), recorded in `run.log`.
The selected device is remapped to `cuda:0` inside the process, so the launcher
explicitly passes `--device cuda:0` for simulation and policy inference.
Use `nvidia-smi` on the server to inspect device availability and utilization;
video rendering uses Isaac Sim's graphics configuration and should also be
checked there when running concurrent video jobs.

## Settings

| Environment variable | Default | Meaning |
| --- | --- | --- |
| `CONTRACK_SHARPA_OBJECT_MASS_SCALE` | `1.0` | Multiply the original object density of 200 kg/m^3. For unchanged collision geometry, PhysX computes scaled mass and inertia. Must be finite and positive. |
| `CONTRACK_SHARPA_OBJECT_FRICTION` | unset | Optional object collision material override. Sets both static and dynamic friction. Must be finite and nonnegative. |

The settings apply to all tracked objects and are fixed for the process. They
are read when the environment module is imported. Start a fresh process for
each condition. No center-of-mass override is introduced.

With neither variable set, the original density, global material, material
inheritance and object USD cache key are retained. In particular, explicitly
setting friction to 2.0 creates an object material and is **not** the same
configuration as leaving the override unset. The robot and ground are unchanged.

Object friction overrides are bound to the collision mesh in the cached USD.
The object material requests the existing `multiply` combination mode. These
are material coefficients, not a guarantee of the effective hand-object contact
coefficient: inspect the other material and its combination mode as well.
Density and optional material overrides are included in cache keys.

## PowerShell example

Run inside the environment that has Isaac Lab / Isaac Sim installed. Replace
the two paths below with the original training session and its checkpoint.

```powershell
$session = "<hammer training session directory>"
$checkpoint = "<hammer model_XXXXX.pt path>"
$hammerData = "data/sharpa/grab-s1_hammer_use_1-rby1a_sharpa-120_840.h5"

# Original baseline: remove overrides, including any left in this terminal.
Remove-Item Env:CONTRACK_SHARPA_OBJECT_MASS_SCALE -ErrorAction SilentlyContinue
Remove-Item Env:CONTRACK_SHARPA_OBJECT_FRICTION -ErrorAction SilentlyContinue
python scripts/rsl_rl/play.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 --session $session --checkpoint $checkpoint --data $hammerData --num_envs 1 --seed 42 --video_length 1000 --headless --export_rollout outputs/physics_validation/hammer_default.h5

# Mass-only comparison: double density, leave friction inheritance unchanged.
$env:CONTRACK_SHARPA_OBJECT_MASS_SCALE = "2.0"
python scripts/rsl_rl/play.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 --session $session --checkpoint $checkpoint --data $hammerData --num_envs 1 --seed 42 --video_length 1000 --headless --export_rollout outputs/physics_validation/hammer_mass2.h5

# Friction-only comparison: original mass, explicit object friction material.
$env:CONTRACK_SHARPA_OBJECT_MASS_SCALE = "1.0"
$env:CONTRACK_SHARPA_OBJECT_FRICTION = "0.5"
python scripts/rsl_rl/play.py --task Isaac-Rby1a-Sharpa-Mimic-Manager-v0 --session $session --checkpoint $checkpoint --data $hammerData --num_envs 1 --seed 42 --video_length 1000 --headless --export_rollout outputs/physics_validation/hammer_friction05.h5

# Restore original defaults for subsequent commands, including training.
Remove-Item Env:CONTRACK_SHARPA_OBJECT_MASS_SCALE -ErrorAction SilentlyContinue
Remove-Item Env:CONTRACK_SHARPA_OBJECT_FRICTION -ErrorAction SilentlyContinue
```

For Linux, use command-scoped variables instead, for example
`CONTRACK_SHARPA_OBJECT_MASS_SCALE=2.0 python scripts/rsl_rl/play.py ...`.

`play.py` prints the resolved PhysX object mass, inertia and material-property
array ranges. When exporting a rollout it also stores these arrays and the
requested overrides in the H5 attribute `object_physics_json`. First compare
default and mass-only runs: resolved mass and inertia should scale together.
Material properties are static friction, dynamic friction, restitution along
the last axis; they describe each object's shapes, not the effective contact pair.

The existing exporter saves joint positions and object poses for environment 0's
first episode; it omits terminal/reset frames. It does not yet save full
state/action/contact histories or compute a multi-episode success rate. These
initial comparisons validate parameter wiring and inspect motion; they do not
establish statistical robustness.

Local checks cover syntax and parameter validation only. Actual USD binding,
PhysX readback and policy rollouts must be verified in the Isaac Lab environment.
