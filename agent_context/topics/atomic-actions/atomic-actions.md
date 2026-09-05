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
| Invocation, binding, and policies | `invocation.py`, `bindings.py`, `policies.py`, `control.py` |
| Robot/task/scene state | `state.py`, `scene.py` |
| Plans and runtime commands | `plans.py`, `runtime_commands.py` |
| Session state machine | `execution.py` |
| Verification request/result values | `verification.py` |
| Runner and transports | `runner.py`, `transports.py`, `tracking.py` |
| Engine and built-in registration | `engine.py`, `primitives/`, `__init__.py` |
| Simulation ports | `sim_adapter.py` |
| Task Program semantic declarations | `embodichain/lab/task_program/semantics/` |
| Task Program lowering/execution | `embodichain/lab/task_program/compiler/`, `runtime/` |
| Gym lifecycle bridge | `embodichain/lab/gym/envs/task_program/bridge.py` |

## Direct resolution path

### 1. Construct the engine

`AtomicActionEngine` owns one `ActionPlanningServices` snapshot: robot, motion
generator, device, planner backend, trajectory builder, and command profiles.
Built-ins are installed through the engine catalog. An engine is not a global
process registry.

The standard simulation composition root is
`create_simulation_atomic_action_engine()` in `sim_adapter.py`.

### 2. Bind resources

Each action publishes a `SkillBindingContract` with participant slots and
endpoint requirements. Direct callers may use engine binding helpers. Task
Program binds a declarative `RobotSkillProfile` to the same engine.

Bindings are immutable endpoint snapshots keyed by `(slot_id, endpoint_id)`.
They contain capabilities, semantic commands, resource claims, and runtime
targets. They do not contain live planners or mutable controller state.

### 3. Build an invocation

`ActionInvocation` contains:

- exact action `skill_id`;
- an action-owned frozen goal;
- resolved binding;
- typed action options;
- motion, tracking, and recovery policy; and
- a monotonic revision when replacing a compatible in-flight request.

The engine resolves the invocation into `ResolvedActionRequest`. The action's
framework-owned `plan()` validates exact goal/options types and then calls the
implementation `_plan(request, context)` hook.

### 4. Choose planning or execution

- `engine.plan(invocation, context)` returns one `ActionPlan`.
- `engine.compile(invocations, context)` performs a fixed offline projection.
- `engine.start(invocations, context)` creates an observed `ExecutionSession`.
- `ExecutionRunner` drives that session against observation, command, and clock
  ports without blocking `step()`.

## PlanningContext invariants

`PlanningContext` carries robot observation, scene snapshot, symbolic
`TaskState`, and authoritative `control_dt`.

When session code updates only symbolic state, use:

```python
replace(context, task=new_task_state)
```

Do not reconstruct `PlanningContext` from only robot/scene/task fields. Doing so
loses a caller-selected `control_dt` and changes trajectory timing during held
object guards, phase gates, retries, or later calls.

Focused regression coverage:

- `tests/sim/atomic_actions/test_engine_per_env.py`
- `tests/sim/atomic_actions/test_endpoint_runtime_e2e.py`

## Execution and verification

`execution.py` owns session events, attempts, ticks, state transitions, and
bounded recovery. `verification.py` owns immutable request/result values:

- `EffectVerificationRequest` / `EffectVerificationResult`;
- `HeldObjectGuardRequest` / `HeldObjectGuardResult`;
- `PhaseEffectGateRequest` / `PhaseEffectGateResult`; and
- `EffectExpectationResult`.

The runner correlates every result with the current request ID. A retry or row
mask change may replace a request, so delayed results for older IDs must not be
applied.

Held-object guards and phase-effect gates are observational:

- guards can remove an action-authorized held relation before dependent motion;
- gates can hold a named plan segment until evidence proves a transition; and
- neither mechanism creates constraints, freezes objects, or overwrites poses.

Pick gates attachment before lift. Place gates detachment before retract.
HandOver owns independent source/destination transfer boundaries.

## Row-local state

Vector environments share a synchronized call and command cursor, but success,
failure, retry budgets, held relations, and eligible masks are per row.
Successful peer rows are not reactivated by a retry on another row.

Use `runner.deactivate_rows()` instead of mutating session masks directly; it
also refreshes cached verification requests.

## Scene dependencies and recovery

Scene-relative goals declare the exact entity poses they consume. The session
compares dependency revisions against fresh snapshots and replans only within
the selected `RecoveryPolicy`.

`PlanningContext.control_dt` is the authoritative control grid. Every emitted
trajectory or endpoint command must align to it; integrations must not silently
resample fractional durations.

Tracking recovery is separate from task-level semantic recovery:

- Atomic Actions owns planning failure, target/collision revision, transport
  acknowledgement, tracking error, timeout, retry, and safe stop.
- Task Program may perform bounded workflow recovery after the action reaches
  a semantic effect boundary.

The `scripts/tutorials/atomic_action/place.py` tutorial configures matching
Newton contact stiffness and damping on the cube and gripper collision links
before `SimulationManager.prepare()`. MuJoCo-Warp's default response is too
compliant for this force-closure replay and otherwise lets the cube slip near
its pickup pose instead of reaching the place target.

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

## Adding or changing an Atomic Action

Use `.agents/skills/add-atomic-action/SKILL.md`. The main change sites are:

1. frozen goal/options/affordance contracts;
2. one `AtomicAction` subclass with exact `skill_id` and binding contract;
3. side-effect-free `_plan(request, context)` implementation;
4. built-in registration and exports;
5. semantic profile/catalog exposure only when the action should be
   agent-visible; and
6. focused planning, execution, recovery, registration, docs, and API tests.

Do not add task sequencing, simulator stepping, or global mutable registries to
an action implementation.

## Recommended change sites

| Change | Owning location |
|---|---|
| Goal/options or action behavior | matching file under `atomic_actions/primitives/` plus `goals.py` when shared |
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
pytest -q tests/lab/semantics
pytest -q tests/lab/task_program
python docs/scripts/check_api_docs.py
```

For public API changes also run the docs checker tests and Sphinx dummy build.
For simulator adapters add an environment-level test that exercises normal
`env.step()` consumption and safe cancellation.
