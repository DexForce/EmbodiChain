# Skill boundary diagnosis and controlled stops

Native configured expansion jobs enable `tracking.boundary_stop` in their owned
execution-policy snapshot. Authored policy files, physical/contact validators,
scene samplers and grasp/release gates are unchanged. Scalar component overrides
and inline policy declarations use the existing canonical resolver.

```yaml
tracking:
  kind: timed
  boundary_stop: true
  boundary_position_tolerance: 0.01
  boundary_velocity_tolerance: 0.005
  boundary_timeout: 1.5
```

Explicit `false` retains historical behavior. Direct Atomic users remain opt-in.
Position tolerance uses each joint's coordinate units (rad or m); velocity uses
those units per second. Configure tighter/application-specific values when the
controller and physical task support them. Defaults accommodate the observed
steady reference error of the Default UR5 pick-place setup; they are not universal
robot safety limits or an acceleration/jerk guarantee.

After the nominal final command, the session continues sending the same command
until two distinct measured control boundaries satisfy target-position error and
finite-difference position speed. Hand targets are retained; no trajectory sample
is modified. This reduces residual physical motion before the next skill captures
its actual start state. The remaining reference-to-measurement error can still
cause a bounded target jump. This feature does not implement fly-through blending.

Native qvel is validated finite and logged separately. This matters for backends
whose post-substep velocity differs from control-grid displacement. Existing
terminal acceptance remains an additional requirement. Timeout and invalid data
reject only affected rows and preserve ordinary safe-stop behavior. New frames
consume wall/step budgets and are recorded as actual actions with T+1 observations.

`boundary_diagnostic` runtime events contain the bound joint IDs, error, native
and control-grid measured speeds, elapsed time, readiness, timeout and tolerances.
The first control-speed sample is unknown and serialized as null. Failed and
cancelled runs must not be pooled into successful episode statistics.

## Reproduce comparisons

Use separate baseline and modified checkouts with the same Python environment,
CPU affinity and visible GPU. Run each pair serially in before/after and then
after/before order, keeping all settings identical:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SDL_VIDEODRIVER=offscreen CUDA_VISIBLE_DEVICES=1 \
python -m scripts.benchmark.expansion.task_pose_coverage \
  --worker --group A --scene open --rows 4 --rollouts 16 \
  --candidates 512 --seconds 600 --output /absolute/fresh-result
```

Repeat for group D. The task owns Default physics; `collision_backend: newton`
is the independent collision checker, not a switch of the physical backend.
Retain configuration, source manifest, attempts, rejected evidence and receipts.
The benchmark source manifest includes Atomic and Gym Task Program code.

`scripts/benchmark/expansion/skill_boundaries.py` consumes a comparison directory
with `runs.json` and `{A,D}-{0,1}-{before,after}` worker directories. Each run entry
contains case, group, revision and metrics (wall_s, committed, executed). It outputs
per-boundary JSON, percentile summaries and six-joint position/velocity PNG/PDF:

```bash
python -m scripts.benchmark.expansion.skill_boundaries --input /absolute/comparison
```

The plotted pair is chosen by largest baseline target jump among successful
same-ordinal pairs whose initial object poses match within 1e-4 componentwise.
The plot reports raw positions and control-grid finite differences, aligned to
first Place command, without postprocessing smoothing. Aggregate results include
all successful boundaries; success rates retain all executed attempts. The source
artifacts and pair selection are recorded for independent inspection.

## Recorded Default-backend qualification

Baseline: `fb96705b0`. UR5 pick-place, open scene, seed 7, B=4, 16 rollout
budget and 512 proposal budget, groups A (bounded/FIFO) and D
(workspace/coverage), each run serially in before/after then after/before order.
Physics is Default CPU; independent geometry checking uses Newton CPU. Visible
GPU 1 is held constant; eight CPU cores and one numerical thread are used.
Measured wall time includes setup, execution, original acceptance and export,
not Python process import. Full source/config manifests are retained per run.

| Group | Confirmed / executed before → after | Mean wall seconds before → after | Boundary command jump P95 (rad) before → after | Measured control-speed P95 (rad/s) before → after |
| --- | --- | --- | --- | --- |
| A | 30/32 → 30/32 | 54.12 → 56.47 | 0.009904 → 0.009540 | 0.021962 → 0.003043 |
| D | 20/26 → 18/26 | 48.71 → 49.19 | 0.006324 → 0.005695 | 0.021708 → 0.003002 |

D did not fill the requested quota in either version: proposal budgets were
exhausted. The new gate rejects one additional previously accepted sample per
repeat: terminal residual approximately 0.01066 rad exceeds the explicit 0.01
threshold after 1.5 s, despite zero control-grid speed. No existing predicate
was relaxed. D's successful-set boundary statistics therefore have a different
population; they are not an isolated causal improvement estimate. Effective
accepted throughput decreases approximately 4.2% for A and 10.9% for D. Two
orders/one seed are insufficient for a general performance claim.

The mechanism demonstrably reduces residual measured motion, but does not remove
reference jumps from steady tracking bias. It should not be advertised as a
complete cure for command discontinuities. A future boundary-conditioned planner
is required for strict continuous references. The evidence shown below deliberately
retains the visible command spikes.

![Paired six-joint position and velocity traces](assets/skill-boundaries.png)

The pair is A, repeat 0, candidate ordinal 15: largest baseline target jump among
matching successful pairs, initial object pose components differ by at most
1.06e-6. Blue is before, orange after; solid is command, dashed measured. Speeds
are finite differences on the 40 ms control grid. Time zero is each recording's
first Place command; recorded signals are not smoothed.

A collection driver is also available:

```bash
CUDA_VISIBLE_DEVICES=1 python -m scripts.benchmark.expansion.skill_boundaries \
  --collect --baseline /absolute/baseline-checkout \
  --revision /absolute/modified-checkout --input /absolute/new-comparison
```
