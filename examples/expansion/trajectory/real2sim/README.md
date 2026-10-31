# Synthetic Real2Sim reference

For bounded affordance/geometry-based trajectory repair, deterministic error
injection, ablations and the verification ladder, see [REFINEMENT.md](REFINEMENT.md).

This example uses the existing UR5 and DH-PGI primitive cube/platform scene.
`sample_geometry_v3/` contains a simulator-generated, verified nominal
reference with pinned robot, object and support geometry. The three portable
prototype bundles include the original source JSON/NPZ; the directory also
contains the actual T+1 state, T submitted commands and a capture report.
Its origin is `synthetic_reference`; TCP poses in the source are `fk_derived`.
It is not a real-robot qualification dataset. `capture.py` can generate a
fresh reference and a TCP comparison plot locally.

The Pick close phase is long enough for its commanded finger velocity to fit
the explicit simulation budget.
The profile uses the URDF arm speed bounds and a conservative synthetic
finger-speed budget; acceleration values are simulation assumptions. The
[manufacturer's PGI-140-80 product data](https://www.dh-robotics.com/wp-content/uploads/2022/01/PGI-Series.pdf)
states full-stroke opening/closing time, but does not specify the joint
acceleration limits used here. These budgets do not establish real-device
qualification.

`acceptance_report_pr.json` summarizes the later 54-case run against the
recorded source commit. `sequence_acceptance_report_pr.json` records three
additional repeated-program variants and their per-call validation. Their full
datasets and audit outputs are generated locally rather than bundled here.
`reproducibility_report_v3.json`, `reproducibility_headroom_report.json` and
`negative_report_v3.json` record earlier same-seed repeats and an unreachable
object failure. These results describe the recorded runs, not every pose in
the surrounding workspace or continuous-time motion.

The B=1 source-qualified `run-task` path selects one layout and recipe per
episode, validates sampled plan and actual-state collision geometry, and writes
the actual T+1 observations, submitted actions, source bundles and trace to a
synchronous LeRobot dataset. A matching receipt is published before reset.
The profile declares all 9 layouts and 6 recipes in their fixed order:

```bash
embodichain run-task \
  --gym-config examples/expansion/trajectory/real2sim/task.ur5.recorded.yaml \
  --real2sim-profile examples/expansion/trajectory/real2sim/collection.tabletop.yaml \
  --dataset-dir /tmp/real2sim-dataset --device cpu --headless \
  --renderer no-render --disable-sensor --max_episodes 54

python examples/expansion/trajectory/real2sim/audit.py \
  --dataset /tmp/real2sim-dataset/ur5_tabletop_real2sim_000 \
  --profile examples/expansion/trajectory/real2sim/collection.tabletop.yaml \
  --output /tmp/real2sim-acceptance-report.json
```

The collision result is sampled, not continuous-time. Grasp closure is
geometrically projected at first sampled contact for plan validation and the
executed robot/object states are checked independently. The current tabletop
owner accepts a fixed-base, scalar-joint robot, a dynamic box, a static box and
the default ground plane; unsupported physical topology is rejected.
The tabletop owner accepts sequential Pick, MoveHeldObject and Place calls,
including single calls and bounded repeats, when every occurrence has a pinned
prototype binding. Each Pick checks its measured lift, each held interval checks
grasp drift, and each Place checks its own release target, support and stability.
Move and Place require a confirmed held relation. A repeated Pick from the target
support may adapt to its measured height and up to 2° tilt only after the host
verifies the object's support footprint; lift remains gravity-aligned.
The current profile also locks joints whose nominal speed or acceleration is
at least 95% of its effective limit before sampling joint residuals. Other
authorized joints retain the configured residual scale, and each call records
its locked joint indices and names. For task-space variants, the same threshold
also triggers a bounded search for lower-acceleration IK references in authorized
free phases. Contact, hold and protected boundary samples stay fixed; FK error,
motion limits and sampled collision checks still gate every candidate.
Use the dataset path printed by `run-task`; the example suffix `000` applies
when the destination is initially empty. The independent audit checks schedule completeness, receipts, asset and source
hashes, actual T+1 arrays, hard-event mappings, per-phase duration, measured
placement and stored IK and motion-limit metrics. Protected same-grid samples
are enforced by the plan adapter and covered by focused software tests.

For independent B=1 shards, with the six recipes still collected sequentially
within each layout, run:

```bash
python examples/expansion/trajectory/real2sim/matrix_run_task.py \
  --task examples/expansion/trajectory/real2sim/task.ur5.recorded.yaml \
  --profile examples/expansion/trajectory/real2sim/collection.tabletop.yaml \
  --output /tmp/real2sim-sharded-acceptance --device cpu --jobs 3
```

The resulting report links nine separately persisted datasets.

The following deployments exercise one Pick and two bounded Pick → Move →
Place cycles, using an explicit single-episode profile:

```bash
embodichain run-task \
  --gym-config examples/expansion/trajectory/real2sim/task.ur5.pick_only.yaml \
  --real2sim-profile examples/expansion/trajectory/real2sim/collection.tabletop.single.yaml \
  --dataset-dir /tmp/real2sim-pick --device cpu --headless \
  --renderer no-render --disable-sensor --max_episodes 1

embodichain run-task \
  --gym-config examples/expansion/trajectory/real2sim/task.ur5.repeated.yaml \
  --real2sim-profile examples/expansion/trajectory/real2sim/collection.tabletop.single.yaml \
  --dataset-dir /tmp/real2sim-repeat --device cpu --headless \
  --renderer no-render --disable-sensor --max_episodes 1

python examples/expansion/trajectory/real2sim/audit.py \
  --dataset /tmp/real2sim-repeat/ur5_tabletop_real2sim_000 \
  --profile examples/expansion/trajectory/real2sim/collection.tabletop.single.yaml \
  --output /tmp/real2sim-repeat-audit.json
```

The source binding belongs under the existing `runtime_services` mapping of an
integration file. For example, the first compiled call in this checked-in
reference can be declared as follows (paths are relative to that integration
file):

```yaml
nominal_motion_sources:
  - call_index: 0
    program_fingerprint: 091319bf0fd5ab5f838e075f2a6bc52b86e571c4eb890ed424b7433f1edacc70
    provider_id: real2sim.prototype
    revision: '1'
    prototype_manifest: sample_geometry_v3/prototypes/0/prototype.json
    prototype_sha256: 0c967257993dd06a44e03c2ffc52fd7c06991178f00b298e75da4358e2db7d9b
    object_entity_id: cube
    authority: eef_pose
    gripper_mode: semantic
    require_collision_validation: true
```

Add separate entries for the transport and place occurrences when deploying a
complete program. Each entry is pinned by the compiled program and prototype
hash; Cartesian transfer also declares the target entity and `eef_pose`
authority. Ordinary configured sources require a backend collision checker;
the tabletop collection path pairs its configured sources with a host-owned
phase-aware guard. The older diagnostic runners use programmatic bindings and
keep their collision opt-out explicit.

From an installed checkout with DexSim and the bundled task/robot extensions:

```bash
embodichain real2sim validate examples/expansion/trajectory/real2sim/sample_geometry_v3/prototypes/0/prototype.json
embodichain real2sim prepare examples/expansion/trajectory/real2sim/sample_geometry_v3/prototypes/0/source/trajectory.json --joints examples/expansion/trajectory/real2sim/joints.ur5.json --control-dt 0.04 --output /tmp/real2sim-prepared-pick
python examples/expansion/trajectory/real2sim/capture.py --output-dir /tmp/real2sim-reference --device cpu
python examples/expansion/trajectory/real2sim/replay.py --reference-dir /tmp/real2sim-reference --output-dir /tmp/real2sim-replay --diagnostic-no-collision-check
python examples/expansion/trajectory/real2sim/layout_replay.py --reference-dir /tmp/real2sim-reference --output-dir /tmp/real2sim-one --layout object_x_plus --recipe joint_7 --diagnostic-no-collision-check
python examples/expansion/trajectory/real2sim/matrix.py --reference-dir /tmp/real2sim-reference --output-dir /tmp/real2sim-matrix --diagnostic-no-collision-check
```

The matrix runner executes all nine requested layouts and all six recipes in
separate B=1 simulator processes. Each case retains a report, failure reason,
actual actions and states when available, call-level plan-variant arrays and a
typed action-to-plan trace. Effect-gate repetitions retain their plan index;
other runtime holds are explicitly marked as unassociated with a plan sample.
The reported retiming acceleration values for the example gripper are explicit
diagnostic assumptions, while arm values are constrained by the motion policy.

The legacy diagnostic flags do not submit a dataset or certify collisions.
`matrix.py --validate-sampled-collision --save-lerobot` can separately report
sampled geometry and receipt results, but those cases still use its diagnostic
motion-budget assumptions. The source-qualified `run-task` path above uses the
explicit profile budgets and independently validates each resulting episode.
