# Affordance-constrained kinematic refinement

This example qualifies same-robot, same-asset Cartesian prototype repair for
Pick, MoveHeldObject and Place. Its sources and injected errors are synthetic.
It does not estimate real-device calibration, contact forces or unknown shape.

The source provider resolves an explicit/configured grasp or a nearby candidate
from the existing antipodal mesh affordance. Candidates must fit the physical
parallel-jaw envelope and pass endpoint IK. The source path is aligned within
explicit correction budgets and fitted on its unchanged control grid. Close and
release are stationary geometric constraints, lift/hold preserves the warped
reference, and two samples at each phase edge protect derivative neighborhoods.
The resulting path still passes continuous-seed IK, joint limits, sampled
collision checks and the existing measured task/effect validation.

The held transform committed by Pick is used by later calls. Place verifies the
whole footprint and signed release gap using actual physical box geometry. A
failed refinement aborts the candidate; the host does not replay the raw input.
Current fitting uses a bounded local rotation chart and rejects its cut. It is
a sparse pose fit plus existing IK validation, not a joint-space SQP collision
optimizer. General meshes still need a correspondingly qualified collection host.

## Verification ladder

| Layer | Independent check | Acceptance |
|---|---|---|
| Numerical | Analytic lines, exact rotations and sensor outliers | Exact anchors; bounded Euclidean/geodesic corrections; lower outlier error |
| Skill integration | Existing action fixture with biased TCP | Correct grasp/held relation; no source mutation; no native fallback |
| Geometry | Analytic box support with rotated footprints | Whole footprint inside support; bounded signed release gap; invalid geometry rejected |
| Provenance | Real sink with original/refined typed arrays | Matching baseline digest, source grid and receipt; tampering rejected before commit |
| Ablation benchmark | Replay vs rigid anchor vs sparse fit on identical data | Report oracle RMSE, anchor/rotation error, success, latency and retained RSS |
| Simulation | Fixed 9 layouts x 6 recipes with injected TCP bias | Measured lift, held relation, place/support/stability, collision, attachments and receipts |
| Negative simulation | Raw biased replay and 5 cm over-budget source | Explicit rejection, no confirmed episode and retained failure reason |

The analytic benchmark uses a separately constructed straight line, a 3 degree
yaw transform, a millimetre translation and 1.5 mm high-frequency sensor jitter.
It does not supply clean oracle arrays to the refiner. The common benchmark
thresholds are position RMSE below 0.8 mm, anchor error below 1 micrometre and
orientation error below 1e-5 radians. These are fixture thresholds, not universal
real-robot accuracy promises. Warmup is excluded; CPU RSS is a separate retained
delta and does not measure transient peak memory.

```bash
python -m scripts.benchmark.motion.kinematic_refinement \
  --output /tmp/refinement-numerical --repeats 7

python examples/expansion/trajectory/real2sim/refinement_acceptance.py \
  --output /tmp/refinement-ablation-and-sequence

python examples/expansion/trajectory/real2sim/prepare_refinement.py \
  --output /tmp/refinement-bias --error bias --refine

python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  /tmp/refinement-bias/task.yaml --json

embodichain run-task --gym-config /tmp/refinement-bias/task.yaml \
  --real2sim-profile examples/expansion/trajectory/real2sim/collection.tabletop.single.yaml \
  --dataset-dir /tmp/refinement-bias/dataset --device cpu --headless \
  --renderer no-render --disable-sensor --max_episodes 1

python examples/expansion/trajectory/real2sim/matrix_run_task.py \
  --task /tmp/refinement-bias/task.yaml \
  --profile examples/expansion/trajectory/real2sim/collection.tabletop.yaml \
  --output /tmp/refinement-matrix --device cpu --jobs 3
```

Use an installed checkout, or set ``PYTHONPATH`` to this checkout and its bundled
task/robot extension roots. Output directories must be new. ``--error drift``
adds a slowly varying free-phase deviation; ``--error over_budget`` injects
5 cm. Omit ``--refine`` for the replay ablation. The authored cube grasp derives
from fixture geometry/TCP calibration; runtime never receives clean benchmark
trajectories. Raw input bundles remain separately hashed and immutable.

The source profile also owns the maximum correction, reference smoothness,
candidate/iteration budgets, support margin and allowed release gap. Authority
is explicit; qpos and hybrid bindings cannot silently become refinable. Example
sources are ``synthetic_reference`` and injected TCP poses are ``transformed``.

Hardware qualification, per-sample source uncertainty estimation, general joint
trajectory optimization and continuous collision certificates remain separate
work. Missing or rejected evidence must not be counted as accepted coverage.
