# Scene expansion

Scene expansion proposes changes before a new Task Program execution. The core
is independent of Task Program, GenSim and Gym; injected host callbacks own the
actual scene preparation and measured task evidence. General composition,
layout and physical state services are owned by
[scene services](../scene-services/scene-services.md); legacy
`sim.scene_expansion` imports re-export the canonical `sim.scene.expansion`
contracts.

## Entry points

| Request | Owner |
|---|---|
| Shared proposal contracts and fixed source | `embodichain/lab/sim/scene/expansion/` |
| Candidate limits and movable entity permissions | `cfg.py` in that package |
| Arena poses and proposal identity | `contracts.py` in that package |
| One Task Program attempt and result metadata | `embodichain/lab/task_program/integrations/scene_expansion.py` |
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

The preparation callback is responsible for restore/application, physical
settling, actual initial-state checks, saved-state identity, binding refresh and
recording observation refresh. Generic execution does not provide those
physical operations. Callbacks can use the shared scene state, operations and
validation services. The concrete Gym reset host and robot-workspace adapter
are supplied separately by [PR #750](https://github.com/DexForce/EmbodiChain/pull/750).
The measured callback supplies task-specific state and
event evidence. Gym remains the owner of reset, stepping and recording; no new
generation session or persistence writer is introduced.

The attempt guard spans preparation, execution and measurement. It suppresses
automatic reset and blocks persistence during explicit callback resets, even
when those resets replace episode annotations. Rejected recording buffers must
be discarded by reset before retrying; beginning another demo or replacing its
metadata does not create a clean recording boundary.

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

Use [robot workspace](../robot-workspace/robot-workspace.md) for reachability
cache and frame contracts, [motion planning](../motion-planning/motion-planning.md)
for trajectories/collision worlds, and [Task Programs](../task-programs/task-programs.md)
for compilation, live grounding and runtime evidence. Those capabilities do
not by themselves provide scene restoration or certify a new layout.

## Change sites and validation

Keep bounded candidate contracts under `lab/sim/scene/expansion`; shared
layout policies and physical services belong to its parent scene package, with
reusable tensor geometry kernels under `compute`. Keep task-specific adaptation in Task Program
integrations and GenSim asset/model dependencies in GenSim. Extend the host
boundary when adding physical preparation, preserving Gym lifecycle ownership.

```bash
pytest -q tests/sim/scene_expansion
pytest -q tests/gym/envs/task_program/test_scene_expansion.py
python docs/scripts/check_api_docs.py
```

The focused tests exercise immutable geometry and identities, bounded source
consumption, real demo dispatch and reset-time recording gates using CPU ports.
They do not establish physical success across robot models or scene layouts.
