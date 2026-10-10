# Scene expansion

Scene expansion proposes changes before a new Task Program execution. The core
is independent of Task Program, GenSim and Gym; injected host callbacks own the
actual scene preparation and measured task evidence.

## Entry points

| Request | Owner |
|---|---|
| Shared proposal contracts and fixed source | `embodichain/lab/sim/scene_expansion/` |
| Candidate limits and movable entity permissions | `scene_expansion/cfg.py` in that package |
| Arena poses and proposal identity | `scene_expansion/contracts.py` in that package |
| One Task Program attempt and result metadata | `embodichain/lab/task_program/integrations/scene_expansion.py` |
| Physical preparation and owned state restore | `embodichain/lab/task_program/integrations/simulation/scene_expansion.py` |
| Object affordance/workspace sampling and pose IK | `embodichain/lab/task_program/integrations/simulation/workspace.py` |
| Preparation hook before final observations | `embodichain/lab/gym/envs/base_env.py` |
| Reset-time persistence eligibility | `embodichain/lab/gym/envs/embodied_env.py` |
| Action consumption and recording lifecycle | `embodichain/lab/gym/envs/demo.py` |

## Resolution and ownership

```text
SceneExpansionCfg + authored pose changes
  -> FixedSceneCandidateSource -> SceneVariant
  -> injected prepare_scene -> ScenePreparationResult
  -> canonical Task Program bridge -> execute_demo_episode -> env.step
  -> measured_validator -> SceneVariantEpisodeResult
  -> caller-owned reset commit/discard
```

`FixedSceneCandidateSource` snapshots only its budgeted prefix and owns no
execution lifecycle. Core contracts describe changes to existing physical
entities, not a replacement SceneGraph or runtime registry. Current execution
is B=1 and requires a Task Program; explicit canonical bridge creation prevents
legacy segment overrides from changing the selected source.

`SimulationSceneExpansionHost` implements the preparation callback for a fixed
B=1 scene: preflight topology and capture, apply existing-rigid-object changes,
settle, reset sensor/observation history, capture the actual state and run explicit
initial checks. Its trusted reset hook runs after ordinary reset events and
before physical objectives, final observations and recording seeding. A new
Task Program bridge creates fresh providers and motion generators, preventing
reuse of the old episode's same-clock scene cache or collision bindings.

The measured callback supplies task-specific state and event evidence. Gym
remains the owner of reset, stepping and recording; no new generation session or
persistence writer is introduced.

## Physical restore and workspace boundary

`SimulationSceneInitialState` owns numeric poses, velocities, full joint state,
controller targets and joint forces for supported rigid objects, robots and
articulations. Restoration requires the owning host, unchanged bindings,
topology, assets/physical parameters, backend and control period. It restores
without settling again, reruns initial validation and records a mandatory
numeric round-trip check with quaternion-sign equivalence. Actual hashes can
differ within tolerance. Solver history/external forces are cleared; this is
episode restoration, not a complete backend checkpoint. Rigid groups,
deformables and explicit rigid constraints are rejected.

`RobotSceneWorkspace` selects one control part explicitly, consumes valid-only
current-arena TCP samples, and derives object poses by inverting the object-local
grasp transform. Optional bounds apply to object origins. Sampling supplies
joint seeds and scores; actual pose IK also checks targets absent from the
cache. Recheck settled or modified poses. Support, collision planning between
phases and measured success remain separate gates.

## Invariants

- A changed pose is a proper 4x4 entity-to-local-arena Z-up transform, excluding
  replication offsets. Numeric inputs become immutable tuples.
- Proposal identity includes parent scene, seed, ordinal, geometry and
  provenance. It is independent of physical slots and distinct from actual
  settled initial-state and execution-attempt identities.
- Only declared movable entity UIDs may change. Candidate and metadata
  containers are owned; duplicate entity IDs and invalid configuration fail
  explicitly rather than becoming ordinary geometric rejections.
- Initial checks, program completion and measured acceptance remain separate.
  Empty or unavailable required checks cannot accept data; projected effects
  do not replace physical evidence.
- Failed/unverified scene metadata blocks reset-time dataset, camera and
  trajectory saves, including explicit commit rows. Reset consumes that
  metadata. Acceptance is not a receipt proving persistence.

Use [robot workspace](../robot-workspace/robot-workspace.md) for reachability
cache and frame contracts, [motion planning](../motion-planning/motion-planning.md)
for trajectories/collision worlds, and [Task Programs](../task-programs/task-programs.md)
for compilation, live grounding and runtime evidence. Those capabilities do
not by themselves provide scene restoration or certify a new layout.

## Change sites and validation

Keep layout/proposal algorithms under `lab/sim/scene_expansion`; put
task-specific adaptation under Task Program integrations. Keep GenSim scene
materialization in its own backend instead of introducing imports into the
core. Extend the host boundary when adding physical preparation, and
preserve Gym lifecycle ownership.

```bash
pytest -q tests/sim/scene_expansion
pytest -q tests/gym/envs/task_program/test_scene_expansion.py
pytest -q tests/gym/envs/task_program/test_scene_workspace.py tests/gym/envs/task_program/test_simulation_scene_expansion.py
pytest -q tests/gym/envs/task_program/test_simulation_environment.py
python docs/scripts/check_api_docs.py
```

The focused tests exercise immutable geometry and identities, bounded source
consumption, real demo dispatch, reset ordering, full state restoration, workspace
frames/masks/IK, fresh bridge observations and recording gates using CPU ports.
They do not establish physical success across robot models or scene layouts.
