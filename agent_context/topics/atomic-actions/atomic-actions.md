# Atomic Skills (`atomic_actions`)

## Scope

Atomic Skill is the capability term; `AtomicAction` is its implementation
base type. The typed planning and execution framework lives in
`embodichain/lab/sim/atomic_actions/`. It owns action goals, resource binding,
planning, transport-neutral commands, execution sessions, recovery, tracking,
and effect-verification requests.

There are two supported caller paths:

1. Atomic Skills: direct callers construct `ActionInvocation` values.
2. Task Program: task-level semantic programs are validated, grounded, and
   lowered internally to Atomic Skills.

`embodichain.lab.task_program.semantics` is declarative only. It owns calls, scene
and robot-profile contracts, effects, and evidence declarations. It is not an
execution facade.

ActionBank is a separate Gym subsystem and is outside this topic.

## Ownership graph

```text
Direct Python caller                    Task Program
ActionInvocation                       semantic program
          |                                  |
          |                       validate + ground + lower
          +------------------+---------------+
                             |
                             v
                    AtomicActionEngine
                  resolve -> plan -> start
                             |
                             v
                    ExecutionSession
          context revisions + row-local recovery
                             |
                             v
                    ExecutionRunner
        observe -> schedule -> dispatch -> verify -> stop
                             |
                             v
        EndpointCommandRouter + transports + clock
```

Planning never steps simulation and never treats command completion as proof of
a physical effect.

## Package map

| Concern | Source of truth |
|---|---|
| Core action and goal contracts | `atomic_actions/core.py`, `goals.py`, `affordance.py` |
| Articulation affordance geometry adapter | `atomic_actions/articulation_geometry.py` |
| Invocation, binding, and policies | `invocation.py`, `bindings.py`, `policies.py`, `control.py` |
| Robot/task/scene state | `state.py`, `scene.py` |
| Plans and runtime commands | `plans.py`, `runtime_commands.py` |
| Candidate values and selected PickUp planning | `candidates.py`, `primitives/pick_up.py` |
| Session state machine | `execution.py` |
| Verification request/result values | `verification.py` |
| Runner and transports | `runner.py`, `transports.py`, `tracking.py` |
| Engine and built-in registration | `engine.py`, `primitives/`, `__init__.py` |
| Simulation ports | `sim_adapter.py` |
| Task Program semantic declarations | `embodichain/lab/task_program/semantics/` |
| Task Program lowering/execution | `embodichain/lab/task_program/compiler/`, `runtime/` |
| Gym lifecycle bridge | `embodichain/lab/gym/envs/task_program/bridge.py` |

## Read on demand

- [Planning and execution](execution.md): invocation resolution, control grid, row-local recovery and verification.
- [Articulation geometry](articulation-geometry.md): topology/mesh ownership and directional affordance contracts.

## Semantic integration boundary

`semantics` contains:

- `calls.py`: Pick, Place, HandOver, registered declarative calls, and catalog;
- `scene.py`: canonical refs, registry, affordances, collision roles;
- `profiles.py`: robot resources, endpoints, presets, `EffectAssurance`;
- `effects.py` / `evidence.py`: effect and measured-evidence contracts; and
- `integration.py`: provider-free manifests, binding, diagnostics.

It does not contain compiler, executor, parallel scheduler, or application
facade modules. Task Program owns the internal semantic lowering lifecycle.

Every `SkillPolicyPreset` selects explicit effect authority:

- `EffectAssurance.VERIFIED`: curated Pick/Place/HandOver calls require exact
  monitor mappings and measured evidence.
- `EffectAssurance.PROJECTED`: monitor mappings are forbidden; expected
  symbolic effects are projected after command completion.

Projected assurance is not physical task success.

Use `/add-atomic-action` to extend goals, planners, registration and focused tests.
Keep planning side-effect-free; actions do not own task sequencing or simulator stepping.

## Recommended change sites

| Change | Owning location |
|---|---|
| Goal/options or action behavior | matching file under `atomic_actions/primitives/` plus `goals.py` when shared |
| Articulation-link sampling or affordance geometry metadata | `atomic_actions/articulation_geometry.py`; keep `objects/articulation.py` limited to meshes, FK, and topology |
| Engine catalog or lifecycle | `engine.py` |
| Session transitions or recovery | `execution.py` |
| Verification value contracts | `verification.py` |
| Scheduling, dispatch, and safe stop | `runner.py` |
| Runtime payload/transport | `runtime_commands.py`, `transports.py` |
| Scene observation | `scene.py`, `sim_adapter.py` |
| Robot resources/presets | `semantics/profiles.py` |
| Semantic call/effect declarations | `semantics/calls.py`, `effects.py`, `evidence.py` |
| Task-level semantic sequencing | `embodichain/lab/task_program/` |

## Focused validation

```bash
pytest -q tests/sim/atomic_actions
pytest -q tests/lab/task_program/semantics
pytest -q tests/lab/task_program
python docs/scripts/check_api_docs.py
```

For public API changes also run the docs checker tests and Sphinx dummy build.
For simulator adapters add an environment-level test that exercises normal
`env.step()` consumption and safe cancellation.

## Offline PickUp export for fixed-scene collection

`lab/trajectory_generation/integrations/atomic.py::export_pickup_templates`
exports a successful MoveEndEffector → PickUp compilation into protected phase
qpos templates. It retains approach/close/lift boundaries, expands passive
mimic geometry and appends real hold commands. Only transit permits residuals.
`cube_pickup_collection.py` uses this source with full-state/contact validation
and confirmed LeRobot persistence across repeated full-batch restoration.

This is an offline atomic source: the qpos executor owns physical validation;
no projected `HeldObjectState` is committed as observed evidence. It does not
consume `initial_plan_provider` or run AtomicActionRuntime tracking/recovery.
The runtime adapter and contact-aware Gym source/host matrix remain separate
acceptance work. Keep the existing runtime and task-state contracts intact.

## Selected grasp planning

`AtomicActionEngine.enumerate_candidates()` and `plan_candidate()` retain the
normal request, scene-binding and endpoint-authorization boundaries. Optional
primitive hooks cannot override framework-owned public planning methods.
`AtomicCandidateBatch`/`AtomicCandidateSelection` bind evaluated data to the
resolved invocation and projected context; modified or stale certificates are
rejected. `compile(candidate_selections=..., eligible_mask=...)` calls selectors
at the current projected start and propagates a monotonic alive mask.

`PickUpCandidateBatch` retains all supplied grasps and both roll variants with
phase IK anchors and rejection reasons. Its selected `ik_interp` path includes
transit, Cartesian approach, close/settle and Cartesian lift; per-sample IK is
seeded from the preceding accepted solution and checked by FK, joint limits
and continuity. It does not certify collision freedom or physical grasp
success. The legacy winner path keeps its previous selection semantics and
shares pose normalization, not the new strict evaluator.

Scheduling logical branches over real physical replicas and compact batch
export belong to the [fixed-scene candidate source](../motion-planning/trajectory-generation.md#multi-grasp-atomic-candidate-source),
not to primitive planners or Task Program semantics.
