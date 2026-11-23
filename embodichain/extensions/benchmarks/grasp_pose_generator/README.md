# Physical grasp validation

From the repository root, with the simulation dependencies and CUDA available:

```bash
d=embodichain/extensions/benchmarks/grasp_pose_generator
embodichain benchmark grasp-pose-generator validate --case "$d/block.yaml" --poses "$d/candidates.jsonl" --num-envs 9 --output outputs/benchmarks/grasp-validation
embodichain benchmark grasp-pose-generator run --suite "$d/suite.yaml" --num-envs 30 --output outputs/benchmarks/grasp-suite --resume
```

Both commands accept `--physics default` (the default GPU backend) and
`--physics newton` (Newton with the DexUni solver). Each invocation evaluates
one backend. Use distinct output directories to keep both aggregate reports.
Checkpoint identities include the backend and solver, so a resumed run
reuses results only from the matching physics configuration. Default uses a
stationary kinematic table for GPU contact actor identification. DexUni uses
two-way coupling. Both paths use positive impulses for measured contact rows.
For geometry-only DexUni rows, a finite signed separation at or below zero
qualifies contact. Reports record the contact policy alongside the solver.

Use the same YAML/JSONL inputs with the installed CLI. Each JSON line contains a
4×4 rigid transform `T_object_tcp` in metres, ordered best first. A case selects
a native parallel-jaw gripper by catalog ID and a target through an ordinary
dynamic RigidObjectCfg. The flat support surface is at arena Z=0. The stand
sets the requested wrist orientation at reset and executes the fixed-orientation
trajectory with three force-limited prismatic drives and native finger drives.

The suite crosses the object list, grippers, placements and seeds. Object
entries declare a stable ID, CAD family and boolean development holdout.
Optional named `placements` override top-level rigid config fields.
`algorithms` maps method names to ranked JSONL filename templates, supporting
`{object}`, `{gripper}`, `{placement}` and `{seed}`. A null value runs the
built-in antipodal generator on the object's semantic mesh. Use an empty file
to record an algorithm's abstention. Missing files are input errors.

Settling checks the target pose five times in the last 0.5 s before the snapshot,
with tolerances of 1 mm translation and 0.005 rad rotation.
Each scene checkpoint stores the frozen candidates and measured starting pose.
Each condition starts from the same settled target pose with cleared velocities.
The base trial approaches 8 cm, closes, lifts 10 cm and holds for 3 seconds.
The independent X and Y trials also travel 3 cm horizontally and back in two
0.30 s minimum-jerk legs. The commanded acceleration peak is 1.92 m/s².
Labels use collision clearance, support contact, continuous retention and
terminal bilateral pad contact. Relative COM motion and orientation are
diagnostics. A window records whether the target entered airborne, and whether its drop time was censored at the observation horizon.
Missing contact evidence is retried from checkpoints. Native worker failure
does not remove its scene from the intended population.

The trials directory contains typed Parquet parts keyed by frozen case identity.
The JSON report includes the case inventory, each gripper's four development
holdout cells, family-cluster bootstrap intervals and paired method differences.
The Markdown leaderboard averages double-held-out scores across hands. Folds
describe evaluation membership. For a pretrained method, its training inventory
must be verified separately before claiming training-set novelty. Fixed
Top-10 precision charges missing slots as failures. Conditional X/Y retention
uses only trials that qualified to start that disturbance. Run with a fresh
output directory after changing externally referenced asset dependencies.

For fold-specific model tuning, run each frozen model with `held_grippers`
set to its held-out hand and include all comparison hands in `grippers`.
For a single frozen generator the default reports every hand as a separate
holdout. User-supplied candidate metadata (for example score and opening) is
preserved in the checkpoints.
