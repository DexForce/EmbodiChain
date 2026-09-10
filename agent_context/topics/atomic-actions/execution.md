# Atomic Action planning and execution

Read this when the request needs these details. [Topic overview](atomic-actions.md).

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

For selected trajectory candidates, `engine.start(...,
initial_plan_provider=provider)` materializes the first invocation's initial
`ActionPlan` from the newly resolved request and current `PlanningContext`.
The framework still binds collision options, authorizes command destinations,
checks plan identities and scene revisions, and installs tracking and phase
gates. The provider is consumed only during session construction; subsequent
invocations and recovery use the registered skill planner. Rebuild the plan,
binding-dependent effects, and session after reset instead of retaining runtime
state from a previous rollout.

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

PickUp's ragged grasp sampling uses an explicit candidate mask. Empty rows and
padding carry a safe current FK pose and cannot win selection; an entirely
empty batch fails without candidate IK. Failed or non-finite IK outputs retain
the preceding valid seed before later pickup stages are screened.

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
