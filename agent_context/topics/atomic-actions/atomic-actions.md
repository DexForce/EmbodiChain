# Atomic Skills (`atomic_skills`)

## Scope

`embodichain.lab.sim.atomic_skills` owns typed goals, bindings, planning,
transport-neutral commands, execution, recovery, tracking and effect requests.
An Atomic Skill is an invocation/registration unit and may contain many phases.
Direct callers construct `SkillInvocation`; Task Program grounds and lowers
Semantic Calls to the same engine. Semantic declarations are not an execution
facade or a second registry.

## Ownership graph

```text
Direct invocation / grounded Semantic Call
  → AtomicSkillEngine: resolve, plan, compile or start
  → ExecutionSession: revisions, row-local state and recovery
  → ExecutionRunner: observe, schedule, dispatch, verify and stop
  → endpoint router, transports and clock
```

Planning never steps physics or treats command completion as physical evidence.
Hosts acquire versioned measurements; verification is the boundary that commits
or invalidates observed effects. Preserve required stop/hold boundaries on the
executed control grid instead of relying only on native planner timing.

## Package map

| Concern | Source of truth |
|---|---|
| Core action and goal contracts | `atomic_skills/core.py`, `goals.py`, `affordance.py` |
| Affordance sampling values and deterministic branch selection | `atomic_skills/affordance_sampling.py` |
| Articulation affordance geometry adapter | `atomic_skills/articulation_geometry.py` |
| Invocation, binding, and policies | `invocation.py`, `bindings.py`, `policies.py`, `control.py` |
| Robot/task/scene state | `state.py`, `scene.py` |
| Runtime physical evidence | `evidence.py`, `scene.py`, `runner.py` |
| Plans and runtime commands | `plans.py`, `runtime_commands.py` |
| Session state machine | `execution.py` |
| Verification request/result values | `verification.py` |
| Runner and transports | `runner.py`, `transports.py`, `tracking.py` |
| Engine and built-in registration | `engine.py`, `primitives/`, `__init__.py` |
| Simulation ports | `sim_adapter.py` |
| Task Program semantic declarations | `embodichain/lab/task_program/semantics/` |
| Task Program lowering/execution | `embodichain/lab/task_program/compiler/`, `runtime/` |
| Gym lifecycle bridge | `embodichain/lab/gym/envs/task_program/bridge.py` |

## Read on demand

- [Planning and execution](execution.md): immutable requests, control grid,
  row-local recovery, guarded effects, command velocity and adoption lifecycle.
- [Affordance sampling](affordance-sampling.md): branch identity, geometry and
  physical feasibility limits.
- [Articulation geometry](articulation-geometry.md): link sampling and topology.
- [Real2Sim](../data-pipeline/real2sim.md): optional source motion, protected
  replay and evidence receipts.

## Semantic integration boundary

Task Program owns call catalogs, scene/profile/effect declarations and lowering.
Bindings expose explicit endpoints/resources/capabilities; names infer none.
Every policy preset selects effect authority: VERIFIED needs exact monitor maps
and measured evidence; PROJECTED forbids monitors and projects symbolic effects
after completion. Projected completion is not measured task success.

`SkillPlanningServices` resolves explicitly supplied hand calibration without
capturing on declaration. Consumers preserve pad/link order and use current
resolved commands, including overrides. Scene captures native geometry;
GraspKit owns geometric qualification; the opt-in query context closes isolated
resources on success, failure or cancellation. Undeclared calibration rejects
before capture. Observation scope and query lifetime remain explicit. See
[Robot capture](../robot-system/robot-system.md#sampled-hand-geometry).

`source_native_contact` retains explicit historical policy/provenance over the
same shared algorithms. It supplies no alternate skill implementation and does
not alter ordinary PickUp defaults. `GraspOpeningTarget` remains one value class
re-exported through the existing imports.

## Direct-simulation example

`embodichain run-example sim/atomic_skill --skill ...` selects a sample scene;
`run-atomic-skill` is a compatibility alias. The ordinary `examples/` tree ships
as data. Its `__main__.py` selects typed configuration, scene modules own
physical setup/calibration, and `host.py` owns compilation, expansion and output.
Core code imports no examples. See the
[example guide](../../../examples/sim/atomic_skill/README.md) for commands and
calibrated embodiments, and the
[switching walkthrough](../../../scripts/tutorials/atomic_skill/README.md).

The shared Atomic plan adapter rebuilds trajectories, velocities and phase
metadata through Motion expansion. Built-in permissions can only narrow;
contact, hand, endpoint, callback and zero-time boundaries stay exact. Logical
variant identity is independent of physical slots; measured replanning remains
row-local. Execution clocks, settling and exports remain host-owned.
[Motion expansion](../motion-planning/trajectory-expansion.md) owns operators and
[Expansion Engine](../expansion-engine/expansion-engine.md) owns collection.

Model/source mesh frames, TCPs, gripper preload, mass scaling and contact options
are calibrated scene inputs. Preserve their selected owners and fingerprints;
never infer capabilities or general physical success from a sample name. The
host's sampled footprint/support checks remain finite measured acceptance,
separate from geometric proposal qualification.

## Recommended change sites

Modify the package-map owner for the changed contract. Add skill goals,
registration and focused coverage with `/add-atomic-skill`; expose a semantic
instruction through `/add-semantic-call` and Task Program. Keep execution,
environment reset and task orchestration out of primitive planning.

## Focused validation

Use `tests/sim/atomic_skills/` for primitive/engine/session behavior;
`tests/lab/task_program/semantics/` for declarative integration; and Gym bridge
cases for actual `env.step()` consumption and cancellation. Public export changes
also run `python docs/scripts/check_api_docs.py`. Simulator qualification remains
separate from offline decode/planning coverage.
