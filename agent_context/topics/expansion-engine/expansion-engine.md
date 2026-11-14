# Expansion Engine

`embodichain.lab.expansion` owns offline cross-domain data production. Trajectory
algorithms stay in `sim.motion.expansion`; scene and workspace algorithms stay in
`sim.scene`. The old motion contract and session imports re-export the exact canonical types.

| Change | Owner |
|---|---|
| CLI job commands | `embodichain/cli/expansion.py` |
| Closed configuration and preflight | `embodichain/lab/expansion/configured.py` |
| Canonical in-memory session, lifecycle limits and coverage port | `embodichain/lab/expansion/session.py`, `session_cfg.py`, `coverage.py` |
| Shared scheduling, admission and commit ports | `embodichain/lab/expansion/engine.py` |
| Rolling workload × B execution in one caller-owned World | `embodichain/lab/expansion/workloads.py` |
| Job assembly, stage recovery and inspect | `embodichain/lab/expansion/workflow.py`, `stages.py` |
| Native World ownership and compatibility facade | `embodichain/lab/expansion/host.py` |
| Native recipes, cohort preparation, step observation and episode assembly | `embodichain/lab/expansion/_gym_recipe.py`, `_gym_prepare.py`, `_gym_execution.py`, `_gym_frames.py`, `_gym_episode.py` |
| Durable recipes, attempts and invocation budgets | `embodichain/lab/expansion/journal.py` |
| Task Program execution with native or prototype motion | `embodichain/lab/expansion/program.py`, `real2sim.py` |
| Source-pinned sampling and joint scene candidates | `embodichain/lab/expansion/recipes.py`, `scene.py` |
| Artifact staging, checks and dataset export | `embodichain/lab/expansion/persistence.py`, `export.py` |
| Portable one-episode dataset export | `embodichain/lab/expansion/episode_export.py` |
| Native appearance and state rerender | `embodichain/lab/expansion/visual.py` |
| Bounded media blocks | `embodichain/data_pipeline/recording/media.py` |

`ExpansionEngine.from_job` resolves a deployment-owned backend and sensor suite.
`run` initializes one World with the requested capacity; `resume` reconciles
receipts and frozen stages before creating another attempt. `inspect` is read-only
and verifies content, source attachments and media. A recovered receipt grants
credit once; an interrupted physical execution never masquerades as a continuation.
Configured and custom hosts use the same engine admission/commit loop.
Task Program owns execution semantics; `motion_source` selects native planning or
its bound Real2Sim prototype, not a second task system. Occurrence-bound DemoSkill
sources can mix with native calls and qualified standard prototypes. The row
source adapter owns preparation/qualification only; the World, scheduling,
receipts and persistence still use the shared host. `prototype_validation_profile`
retains the tabletop guard for the selected prototype occurrences, while
`demo_skill_duration_scale` requires explicit full-joint motion limits and
asset-owned time permissions. Spatial and mapped timing recipes use separate
configured jobs. `sampled_rigid_v1` collision profiles independently check all
physical box/plane scene bodies and task predicates from measured states. Their
robot-base transforms are rebound after each cohort's scene preparation.
State-only jobs retain independent and robot-mounted ContactSensors while
omitting cameras; contact acquisition does not require media capture.
Immutable recipes declare
source/backend/phase/scope and shared-resource compatibility; the host checks them
before mutating the World.

Semantic IDs pin sampling inputs, while execution IDs describe mutable resource
budgets and postprocessing. Recipe IDs survive interrupted retries and B changes;
new attempt IDs distinguish physical executions, and persistent job-instance IDs
separate independently collected episodes. Native scene-case identity includes the
logical recipe: independently prepared recipes must not share trajectory-only
coverage admission merely because their sampled physical values match. Frozen writes are recovered before
allocating new attempts. A hard-loss time budget is an explicit lower bound.

Confirmed parents may render/export before the collection target is met. Stage
failures remain independent; export-only recovery opens no World and needs no
source configuration. Version 2 manifests use contained relative hashed references;
export includes media, calibration and source evidence in a portable dataset bundle.
Legacy committed artifacts are readable in place and projected portably at export.
Legacy collection migration must verify unchanged resolved deployment evidence.

`ExpansionSession` imports no simulation or trajectory package. `ExpansionSessionCfg`
owns collection, ready-pool, validation and pending-write limits; the default
`EpisodeCoverage` counts confirmed episode identities. Legacy trajectory configs
project into this contract and supply motion-owned `TrajectorySessionCoverage`,
retaining phase geometry and manipulability admission. Existing imports name the
same Session class; domain coverage algorithms remain in `motion.expansion`.

The Gym facade is the sole World lifecycle owner. Its components share an explicit
native context and cannot independently close/reset/step competing Worlds. Recipe
sampling, whole-cohort preparation, active-row frame buffering and measured episode
assembly have separate owners. Preserve the host constructor, port methods and
existing readout properties used by benchmark/showcase instrumentation.

`TaskProgramDemoBridge.program_id` and `env_ids` expose binding identity without
runtime traversal. `EmbodiedEnv.task_program_bridge_scope()` restores the borrowed
bridge even after failure. `Real2SimTabletopSource` and explicit
`Real2SimTabletopEpisodeEvidence` share source qualification with the original B=1
owner; do not impersonate private Gym trajectory buffers. Offline export uses
`LeRobotRecorder.persist_episode_payload()` and retains the canonical recording
journal/duplicate-submission behavior.

Cohorts have fixed robot/joint topology, backend, clock, observation schema and
World appearance. Native scene-atlas input retains the complete layout/installation
candidate. Assets with different topology require separate deployment Worlds.
The configured cohort path resets and settles its rows together. Only the owner steps physics, and row generators
acknowledge commands only after successful steps. Early termination must drain the
Task Program safe-stop handshake; completed rows stop contributing frames.
Native vector rejection cancels the row across semantic lanes and masks retained
commands. Its safe hold uses the next ordinary step, preserving surviving rows'
post-policy observation clock. Expected Real2Sim acceptance rejection retains
measured checks and rejects only that row; source corruption and runtime errors
still stop collection.

`WorkloadProgramExecutor` is an optional rolling integration for custom native
sources. Its supplied environment must have exactly `workload_count * batch_size`
rows and be the sole live World in the process. Each workload owns quotas, local
candidate ordinals and admission budgets; its B-row lane is reused after that
group drains. Preparing rows settle through ordinary control steps while live
rows continue to record and validate every boundary. The source owns scene
proposals, bridge binding and all physical checks, using `ExpansionEngine.admit_row`,
`mark_row_started` and `complete_row` instead of editing private engine queues.
Only confirmed native receipts earn quota or permit reuse; duplicate identities
cannot earn credit twice. Clock spans must equal each trajectory's command count.
Compatible fixed-grid planning and pure solver requests can be cooperatively
merged on the owner thread. Scoped planning budgets retain per-row charges;
unsupported requests keep native execution. Export callbacks receive frozen
group evidence and never access the World. See [rolling execution](workloads.md).

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
`export_expansion_episodes` additionally publishes one portable LeRobot dataset
per confirmed parent. Cached evidence reads still hash every file byte and
reconstruct checked, independently owned numerical tensors; source/receipt/media
inspection remains required. Immutable JSON nodes can be shared internally, while
public mutable projections own separate containers.

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

The interactive acceptance example lives in `examples/expansion/engine/showcase/`.
It composes public Engine workflows and ordinary example helpers; it must not own
another task runtime or silently alter acceptance rules. Preview geometry uses
measured state plus explicitly labeled FK, and pure-visual children retain their
physical parent's clock. Qualification requires admitted operator evidence, not
just a requested profile or completed nominal episodes. Report completeness and
capability acceptance are separate, including diagnostic unsupported combinations.
