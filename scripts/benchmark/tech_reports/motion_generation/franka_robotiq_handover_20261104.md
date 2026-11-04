# Franka + Robotiq ARG2F 140 HandOver qualification — 2026-11-04

The HandOver benchmark primary metric is `physical_skill_success`, meaning all
required physical stages complete. Over 50 measured environments per condition
and backend, TOPPRA task success improves **34/50 (68%) → 41/50 (82%)**;
cuRobo improves **33/50 (66%) → 42/50 (84%)**. End-to-end validity is a separate
diagnostic, not the primary task-success metric.

Both conditions use #948's Franka arm and the same
`Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf`, assembly, deployment, role
bindings, task, sampling budgets, protocols and scoring. The baseline uses the
frozen component; the treatment is the complete HandOver-specific calibration
bundle, including home, FEP, tool/contact frame, commands, drives and sampler
settings. No individual-parameter ablation or causal attribution was performed.

The recorded baseline source is `9d7170e52`, calibrated source `6221f4da0`.
The packaged component has the exact effective RobotCfg of the measured treatment;
null/default FEP fields were removed, and only profile identities/resource locations
were renamed. The packaged coverage suite retains the measured protocol, domains
and seeds. The default frozen component remains unchanged.

## Time & Memory

Compilation-call measurements only. Other work shared the machine, so no speed
or memory-efficiency improvement is claimed. CPU columns are RSS changes and GPU
columns cover Torch allocations, excluding native simulator/renderer allocations.

| variant | backend | measured calls | cost_time_ms median | cpu_delta_mb median | gpu_delta_mb median | peak_gpu_mb max |
|---|---|---:|---:|---:|---:|---:|
| without_ours | toppra | 14 | 14630.209 | 0.000 | 0.000 | 8.125 |
| without_ours | curobo | 14 | 7831.393 | 0.000 | 0.000 | 6586.414 |
| with_ours | toppra | 14 | 8153.447 | -0.500 | 0.000 | 8.125 |
| with_ours | curobo | 14 | 6173.317 | 0.000 | 0.000 | 6586.172 |

## Success & Other Metrics

There are 8 domains: nominal and mirrored roles have one row each; the other
six domains each use seeds 11/23 with B=4. All 50 rows per backend enter the
aggregate denominator, including every failed row. Per-domain native reports
macro-average their equally sized seed cases; the ALL rows below count the 50
environment outcomes directly rather than averaging the eight domain rates.

| domain | backend | Frozen task success | Calibrated task success | delta (pp) | Frozen end-to-end | Calibrated end-to-end |
|---|---|---:|---:|---:|---:|---:|
| nominal | toppra | 1/1 (100.0%) | 1/1 (100.0%) | +0.0 | 1/1 (100.0%) | 1/1 (100.0%) |
| nominal | curobo | 1/1 (100.0%) | 1/1 (100.0%) | +0.0 | 1/1 (100.0%) | 1/1 (100.0%) |
| object_xy | toppra | 5/8 (62.5%) | 5/8 (62.5%) | +0.0 | 4/8 (50.0%) | 4/8 (50.0%) |
| object_xy | curobo | 4/8 (50.0%) | 5/8 (62.5%) | +12.5 | 3/8 (37.5%) | 4/8 (50.0%) |
| object_yaw | toppra | 6/8 (75.0%) | 7/8 (87.5%) | +12.5 | 6/8 (75.0%) | 7/8 (87.5%) |
| object_yaw | curobo | 7/8 (87.5%) | 8/8 (100.0%) | +12.5 | 7/8 (87.5%) | 8/8 (100.0%) |
| target_relative_pose | toppra | 5/8 (62.5%) | 8/8 (100.0%) | +37.5 | 5/8 (62.5%) | 8/8 (100.0%) |
| target_relative_pose | curobo | 7/8 (87.5%) | 8/8 (100.0%) | +12.5 | 7/8 (87.5%) | 8/8 (100.0%) |
| task_frame_planar | toppra | 5/8 (62.5%) | 6/8 (75.0%) | +12.5 | 0/8 (0.0%) | 0/8 (0.0%) |
| task_frame_planar | curobo | 3/8 (37.5%) | 6/8 (75.0%) | +37.5 | 0/8 (0.0%) | 0/8 (0.0%) |
| object_target_position | toppra | 4/8 (50.0%) | 6/8 (75.0%) | +25.0 | 0/8 (0.0%) | 0/8 (0.0%) |
| object_target_position | curobo | 3/8 (37.5%) | 6/8 (75.0%) | +37.5 | 0/8 (0.0%) | 0/8 (0.0%) |
| target_relative_shell | toppra | 8/8 (100.0%) | 8/8 (100.0%) | +0.0 | 8/8 (100.0%) | 8/8 (100.0%) |
| target_relative_shell | curobo | 8/8 (100.0%) | 8/8 (100.0%) | +0.0 | 8/8 (100.0%) | 8/8 (100.0%) |
| bilateral_role_mirror | toppra | 0/1 (0.0%) | 0/1 (0.0%) | +0.0 | 0/1 (0.0%) | 0/1 (0.0%) |
| bilateral_role_mirror | curobo | 0/1 (0.0%) | 0/1 (0.0%) | +0.0 | 0/1 (0.0%) | 0/1 (0.0%) |
| ALL | toppra | 34/50 (68.0%) | 41/50 (82.0%) | +14.0 | 24/50 (48.0%) | 28/50 (56.0%) |
| ALL | curobo | 33/50 (66.0%) | 42/50 (84.0%) | +18.0 | 26/50 (52.0%) | 29/50 (58.0%) |

## Leaderboard

Rank by the native primary task metric, with all evaluated backend/condition
pairs retained. These are results for this workload, not a broad robot ranking.

| rank | variant / backend | completed rows | physical success | end-to-end success |
|---:|---|---:|---:|---:|
| 1 | with_ours/curobo | 50/50 | 42/50 (84.0%) | 29/50 (58.0%) |
| 2 | with_ours/toppra | 50/50 | 41/50 (82.0%) | 28/50 (56.0%) |
| 3 | without_ours/toppra | 50/50 | 34/50 (68.0%) | 24/50 (48.0%) |
| 4 | without_ours/curobo | 50/50 | 33/50 (66.0%) | 26/50 (52.0%) |

## Reproduction and evidence

Use the packaged `atomic_dual_franka_robotiq_hand_over_spatial_generalization`
suite. The benchmark README documents both calibrated and frozen-component
commands. Physics runs on CPU; cuRobo uses CUDA with CUDA graphs disabled.
Hardware/software: RTX 4090, Python 3.11.16, Torch 2.10.0+cu128, CUDA 12.8,
cuRobo 0.8.0, DexSim 0.5.1rc1. OMP/OPENBLAS threads=1, serial execution,
zero warmups, one measured repeat. Cold calls and the nominal preflight are
excluded. This is a quality experiment; compilation/replay time is not a
performance acceptance criterion.

Both conditions share the `atomic-skill-v3` target-specific final-sample finger
contact scoring from #967's scoring prerequisite (5 mm signed separation),
explicitly configured finger links, and arm-only tracking scope at 0.05 rad.
Neither scoring nor tracking-scope changes count toward the measured gain.
The baseline's complete raw artifacts were reused without rewriting; their
hashes are retained. An interrupted calibrated attempt is retained locally and
excluded; treatment counts come from one complete restart, with no stitched rows.

The [evidence snapshot](franka_robotiq_handover_20261104.json) contains all 200
compact measured outcomes, raw artifact hashes, source/configuration pins,
software environment, calibration deltas, and the 14-batch paired-input audit.
Both conditions have 100 environment rows and use the same scoring protocol.
Input object/target poses and logical roles pass pairing checks; default model
geometry/provenance remain unchanged, and all 14,874 calibrated geometry points
match the prior implementation exactly. Native solver outcomes may vary on
rerun; no confidence or population-wide generalization claim is made.

## Remaining failures

Geometry validity is **30/50 in both conditions and both backends**, with exactly
the same 20 failing rows. The planar-frame and joint object/target perturbation
domains still have zero end-to-end success. Role mirroring fails **0/1** for
both backends. Task-success gains therefore do not establish full motion
validity or full Atomic benchmark acceptance. Qualification covers one soda-can
asset, two seeds, one repetition, and this Franka/Robotiq assembly only.
