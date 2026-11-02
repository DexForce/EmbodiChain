# Expansion Engine

`embodichain.lab.expansion` owns offline cross-domain data production. Trajectory
algorithms stay in `sim.motion.expansion`; scene and workspace algorithms stay in
`sim.scene`. The old motion contract imports re-export the shared immutable types.

| Change | Owner |
|---|---|
| CLI job commands | `embodichain/cli/expansion.py` |
| Closed configuration and preflight | `embodichain/lab/expansion/configured.py` |
| Shared scheduling, admission and commit ports | `embodichain/lab/expansion/engine.py` |
| Job assembly, stage recovery and inspect | `embodichain/lab/expansion/workflow.py`, `stages.py` |
| World preparation and measured execution | `embodichain/lab/expansion/host.py` |
| Durable recipes, attempts and invocation budgets | `embodichain/lab/expansion/journal.py` |
| Task Program execution with native or prototype motion | `embodichain/lab/expansion/program.py`, `real2sim.py` |
| Source-pinned sampling and joint scene candidates | `embodichain/lab/expansion/recipes.py`, `scene.py` |
| Artifact staging, checks and dataset export | `embodichain/lab/expansion/persistence.py`, `export.py` |
| Native appearance and state rerender | `embodichain/lab/expansion/visual.py` |
| Bounded media blocks | `embodichain/data_pipeline/recording/media.py` |

`ExpansionEngine.from_job` resolves a deployment-owned backend and sensor suite.
`run` initializes one World with the requested capacity; `resume` reconciles
receipts and frozen stages before creating another attempt. `inspect` is read-only
and verifies content, source attachments and media. A recovered receipt grants
credit once; an interrupted physical execution never masquerades as a continuation.
Configured and custom hosts use the same engine admission/commit loop.
Task Program owns execution semantics; `motion_source` selects native planning or
its bound Real2Sim prototype, not a second task system. Immutable recipes declare
source/backend/phase/scope and shared-resource compatibility; the host checks them
before mutating the World.

Semantic IDs pin sampling inputs, while execution IDs describe mutable resource
budgets and postprocessing. Recipe IDs survive interrupted retries and B changes;
new attempt IDs distinguish physical executions, and persistent job-instance IDs
separate independently collected episodes. Frozen writes are recovered before
allocating new attempts. A hard-loss time budget is an explicit lower bound.

Confirmed parents may render/export before the collection target is met. Stage
failures remain independent; export-only recovery opens no World and needs no
source configuration. Version 2 manifests use contained relative hashed references;
export includes media, calibration and source evidence in a portable dataset bundle.
Legacy committed artifacts are readable in place and projected portably at export.
Legacy collection migration must verify unchanged resolved deployment evidence.

Cohorts have fixed robot/joint topology, backend, clock, observation schema and
World appearance. Native scene-atlas input retains the complete layout/installation
candidate. Assets with different topology require separate deployment Worlds.
All rows reset and settle together. Only the owner steps physics, and row generators
acknowledge commands only after successful steps. Early termination must drain the
Task Program safe-stop handshake; completed rows stop contributing frames.

Native numerical evidence has T submitted actions and T+1 measured states. It includes
all represented dynamic scene entities, source/action semantics, physical parameters
and call provenance. Position-velocity commands must use their ExpertActionSpec;
measured qpos is not a substitute for command actions. Required task and collision
checks reject unsupported geometry, source violations and parameter drift.

Media spools own copies, bound raw buffering and publish hashed manifests. Encoding
has bounded transient copies; no per-candidate dense full-episode image array is held.
Frame readers stream selected frames from stored blocks; full artifact audits still
validate bounded blocks. LeRobot export streams confirmed parents and visual children through the existing
recorder and checks the exact final journal identity set. ODS remains unchanged.

Pure visual replay requires complete scene state and an explicit observation-independent
expert contract. Physics is disabled across the entire World; render-state publication
preserves measured joint excursions while restoring physical limits. Each child owns
new images/calibration but retains the original actions, times and parent ancestry.
It increases observation-variant counts, not physical-success counts.

Unsupported parameter, sensor and planner combinations are errors, not silent B=1
fallbacks. The native physical operators admit measured mass/inertia, friction and CoM
changes, plus explicit robot joint stiffness/damping scales. Single-body Newton damping is not advertised as effective.
See the runnable jobs and qualification instructions in `examples/expansion/engine/`.

Geometry, actual physics, visual and source coverage remain separate diagnostic
channels. Native calibration/state readout is batched once per frame; per-row checks
use the captured evidence. Engine and host timing phases are nested, not additive.

Focused coverage lives in `tests/lab/expansion/`, `tests/data_pipeline/test_media_spool.py`,
`tests/sim/scene/` and the existing Task Program/Real2Sim suites. Native qualification
runs in isolated processes. Benchmark accepted-and-confirmed throughput with
`scripts/benchmark/expansion/offline_collection.py`; report setup and media costs.
