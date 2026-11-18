# Scene expansion

Scene expansion proposes layouts before host execution or environment reset. The core
is independent of Task Program, GenSim and Gym; injected host callbacks own the
actual scene preparation and measured task evidence. General composition,
layout and physical state services are owned by
[scene services](../scene-services/scene-services.md). Proposal contracts are
imported from `embodichain.lab.sim.scene.expansion`.

`SceneLayoutInput` owns scaled local meshes, authored poses, a support tree and
movable permissions. `prepare_scene_randomization` prepares geometry;
`generate_scene_variants` returns addressed CPU/CUDA candidates;
`apply_scene_variants` preflights selected rigid poses without stepping.
`SimulationSceneExpansionHost` prepares/restores ordered rows with independent
validation. Offline scheduling and confirmed persistence belong to the
[Expansion Engine](../expansion-engine/expansion-engine.md).

V4 JSON/packed packages own source, policy and prepared geometry snapshots.
Loading checks every candidate before exposing a lazy sequence and does not
repeat mesh extraction or support rasterization. GenSim keeps legacy v1-v3
readers and migration, while shared loading never imports GenSim. Imported
fixed-orientation group certificates preserve their original conservative proxy
rather than being reinterpreted by the hierarchical sampler. V3 migration retains
the validated legacy support grids and height rules without re-extraction.

Joint task-conditioned layout/installation search is owned by
[scene and robot feasibility](feasibility.md). It builds new deployments between
attempts. Its configured search adapter still evaluates one B=1 deployment per
attempt through the shared selected-row preparation/restoration contract.

The task-facing rolling collector also exposes selectable bounded versus
workspace-conditioned pose proposals with optional receipt-confirmed task-pose
coverage feedback; see [workload feedback](../expansion-engine/workloads.md#optional-task-pose-proposal-feedback).
This is distinct from offline atlas region validation and does not certify
continuous reachable volume.

## Entry points

| Request | Owner |
|---|---|
| Shared proposal contracts and fixed source | `embodichain/lab/sim/scene/expansion/` |
| Candidate limits and movable entity permissions | `cfg.py` in that package |
| Arena poses and proposal identity | `contracts.py` in that package |
| Selected-row Task Program attempt and result metadata | `embodichain/lab/task_program/integrations/scene_expansion.py` |
| Physical preparation and owned state restore | `embodichain/lab/task_program/integrations/simulation/scene_expansion.py` |
| Offline batched scene/robot preparation and Task Programs | [Expansion Engine](../expansion-engine/expansion-engine.md) |
| Reusable object/workspace proposals and pose IK | `embodichain/lab/sim/scene/workspace.py` |
| Preparation hook before final observations | `embodichain/lab/gym/envs/base_env.py` |
| Reset-time persistence eligibility | `embodichain/lab/gym/envs/embodied_env.py` |
| Action consumption and recording lifecycle | `embodichain/lab/gym/envs/demo.py` |

## Resolution and ownership

```text
SceneExpansionCfg + authored pose changes
  -> FixedSceneCandidateSource -> row-to-SceneVariant mapping
  -> injected prepare_scene -> row-to-ScenePreparationResult mapping
  -> canonical Task Program bridge -> execute_demo_episode -> env.step
  -> measured_validator(row) -> row-to-SceneVariantEpisodeResult mapping
  -> caller-owned reset commit/discard
```

`FixedSceneCandidateSource` snapshots only its budgeted prefix and owns no
execution lifecycle. Core contracts describe changes to existing physical
entities, not a replacement SceneGraph or runtime registry.
`execute_scene_variants` requires a Task Program and explicit physical row keys;
preparation returns exactly those rows. One attempt owns the World. Initially valid rows
execute, and each result uses that row's completion and measured checks rather
than aggregate batch success. Explicit canonical bridge creation prevents
legacy segment overrides from changing the selected source.

`SimulationSceneExpansionHost` preflights every selected row before reset, applies
existing-rigid-object changes together, resets selected sensor/observation
history, refreshes current contacts without stepping, captures actual states and
runs `initial_validator(row)`. Any settling runs once for the complete World;
partial selections reject a settling policy before mutation. Its trusted reset
hook passes `reset_ids` and runs after ordinary reset events and
before physical objectives, final observations and recording seeding. A new
Task Program bridge creates fresh providers and motion generators, preventing
reuse of the old episode's same-clock scene cache or collision bindings.

## Physical restore and workspace boundary

`SimulationSceneInitialState` retains one row's host provenance and Gym control
period around the shared scene physical-state representation. The host exposes
immutable `initial_states`, `capture_initial_states(parent_scene_ids=...)` and
`restore_initial_states(...)` mappings. Snapshots retain their original row
assignment while content identities exclude slots. Numeric capture,
restore, rigid-pose application and state comparison delegate to
[scene services](../scene-services/scene-services.md). Restoration requires the owning host, unchanged bindings,
topology, assets/physical parameters, backend and control period. The host measures
mass properties, rigid friction/damping and articulation drives/limits before and
after reset events and rejects parameter drift. It restores
without settling again, reruns initial validation and records a mandatory
numeric round-trip check with quaternion-sign equivalence. Only measured qpos
receives the explicit `1e-4` limit excursion allowance; targets remain strict.
Empty initial validation remains unavailable despite passing auxiliary checks. Actual hashes can
differ within tolerance. Solver history/external forces are cleared; this is
episode restoration, not a complete backend checkpoint. Rigid groups,
deformables and explicit rigid constraints are rejected.

The shared `sim.scene.workspace.RobotSceneWorkspace` service selects one control part explicitly, consumes valid-only
current-arena TCP samples, and derives object poses by inverting the object-local
grasp transform. Optional bounds apply to object origins. It returns scene check
reports; consumers explicitly supply their execution validation evidence.
Sampling supplies
joint seeds and scores; actual pose IK also checks targets absent from the
cache and verifies joint limits and full-pose FK residuals. Recheck settled or modified poses. Support, collision planning between
phases and measured success remain separate gates.

An `EpisodeContext.check("scene_expansion/task_program", env_ids)` scope spans
preparation, execution and measurement. It suppresses
automatic reset and blocks persistence during explicit callback resets, even
when those resets replace episode annotations. Rejected recording buffers must
be discarded by reset before retrying; beginning another demo or replacing its
metadata does not create a clean recording boundary. Acceptance and metadata are
published per selected row; a rejected row cannot discard a successful peer's
eligibility. Physics still advances the whole World during execution.

## Invariants

- A changed pose is a proper 4x4 entity-to-local-arena Z-up transform, excluding
  replication offsets. Numeric inputs become immutable tuples.
- Proposal identity includes parent scene, seed, ordinal, geometry and
  provenance. It is independent of physical slots and distinct from actual
  settled initial-state and execution-attempt identities.
  Ordinal is part of the hash: reordering consumption can change `variant_id`
  for the same saved layout. File-backed consumers retain original layout
  identity and proposal ordinal in metadata for cross-run provenance.
- Only declared movable entity UIDs may change. Candidate and metadata
  containers are owned; duplicate entity IDs and invalid configuration fail
  explicitly rather than becoming ordinary geometric rejections.
- Initial checks, program completion and measured acceptance remain separate.
  Program success requires normal completion and the acceptance mask of the
  selected canonical bridge, alongside successful demo execution. Generic Gym
  success cannot replace that evidence. Empty or unavailable required checks
  cannot accept data; projected effects do not replace physical evidence.
- Failed or unverified recordings block reset-time dataset, camera and
  trajectory saves, including explicit commit rows. Rejection stays attached
  to the buffers until reset discards them; a successful retry without reset
  cannot make those buffers eligible. Acceptance is not a persistence receipt.

Related owners: [robot workspace](../robot-workspace/robot-workspace.md),
[motion planning](../motion-planning/motion-planning.md) and
[Task Programs](../task-programs/task-programs.md). None alone certifies a new layout.

## Change sites and validation

Keep bounded candidate contracts under `lab/sim/scene/expansion`; shared
layout policies and physical services belong to its parent scene package, with
reusable tensor geometry kernels under `compute`. Keep task-specific adaptation in Task Program
integrations and GenSim asset/model dependencies in GenSim. Extend the host
boundary when adding physical preparation, preserving Gym lifecycle ownership.

```bash
pytest -q tests/sim/scene/expansion
pytest -q tests/gym/envs/task_program/test_scene_expansion.py
pytest -q tests/sim/scene/test_workspace.py tests/gym/envs/task_program/test_simulation_scene_expansion.py
pytest -q tests/gym/envs/task_program/test_simulation_environment.py
python docs/scripts/check_api_docs.py
```

The focused tests exercise immutable geometry and identities, bounded source
consumption, real demo dispatch, reset ordering, full state restoration, workspace
frames/masks/IK, fresh bridge observations and recording gates using CPU ports.
They do not establish physical success across robot models or scene layouts.

Layout snapshots read `RigidObject.get_semantic_mesh()` so Mesh/Gaussian
appearance replacement preserves placement geometry. Source surfaces are retained
by the simulation visual binding; an explicit `semantic_shape` can provide the
proxy. Gaussian radiance data is not a replacement for this geometry contract.
