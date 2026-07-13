# Newton Runtime Contracts and Differentiable Execution Design

**Status:** Approved in design discussion; awaiting written-spec review

**Date:** 2026-07-13

**EmbodiChain branch:** `feature/newton-physics-backend`

**DexSim implementation branch:** `feature/embodichain-newton-contracts`

**Target DexSim package version:** `0.4.4`

## 1. Purpose and supersession

This specification replaces the multi-environment and differentiable-runtime
portions of:

- `docs/superpowers/specs/2026-06-21-newton-backend-pr-design.md`
- `docs/superpowers/plans/2026-06-22-newton-backend-pr.md`
- the corresponding Target 4 and Target 5 status claims in
  `design/newton-backend-design.md`

The previous documents describe mutually incompatible clone-at-finalize and
spawn-time clone designs, treat rebuild-time runtime IDs as permanent, and
conflate a forward-kinematics demonstration with differentiable Newton
dynamics. They remain historical records but are not implementation sources
after this specification is accepted.

The work is delivered in two sequential stages:

1. A coordinated DexSim and EmbodiChain refactor covering the Newton public
   integration contract, lifecycle, multi-world isolation, runtime topology
   mutation, rigid bodies, and articulations.
2. A differentiable execution layer built on the resulting authoritative
   state, generation, binding, and lifecycle contracts. It supports both real
   solver dynamics and pure kinematics.

Stage 2 starts only after Stage 1 correctness and lifecycle tests pass.

## 2. Goals

### 2.1 Stage 1 goals

- Make DexSim the sole authority for Newton model, state, control, contacts,
  entity metadata, runtime mappings, and model generation.
- Remove EmbodiChain use of private DexSim registry, registration, and
  `dexsim_meta` details.
- Preserve the existing public `SimulationManager`, `RigidObject`, and
  `Articulation` call surfaces.
- Support initial build and post-finalize add/remove for rigid bodies and
  articulations.
- Preserve surviving rigid and articulation state across rebuilds.
- Rebind all runtime IDs after every successful model rebuild.
- Correctly isolate global and child-arena Newton worlds.
- Support two or more simultaneous `SimulationManager`/DexSim `World`
  instances in one process, including same-GPU operation and deterministic
  teardown.
- Make arena-local and world-frame pose semantics explicit and consistent.
- Support articulation topologies whose position and velocity widths differ,
  including spherical and free joints at the binding-contract level.
- Preserve the default physics backend behavior.

### 2.2 Stage 2 goals

- Make the default differentiable path execute
  `DifferentiableStepper` and the configured Newton solver.
- Match normal simulation time and substep semantics exactly.
- Support pure-kinematics environment steps through `newton.eval_fk` without
  misrepresenting them as dynamics.
- Support non-zero action gradients, finite-difference validation, and
  continuous multi-step differentiation.
- Preserve the normal environment lifecycle while making only a minimal,
  explicit differentiable-output addition to the functor surface.
- Make state-buffer ownership safe across forward, backward, reset, rebuild,
  and multiple worlds.

## 3. Non-goals

- A general functor-system rewrite.
- Runtime topology mutation for soft bodies or cloth. Such attempts fail
  explicitly in this iteration.
- Replacing EmbodiChain's environment framework with IsaacLab's architecture.
- Copying IsaacLab's class-level physics singleton, USD/Fabric coupling, or
  backend discovery by class-name convention.
- Broad renderer, sensor, or solver performance optimization unrelated to the
  new contracts.
- Heterogeneous articulations inside one `Articulation` batch. Separate
  batched assets may have different topologies, while instances within one
  batch retain the same topology.

## 4. Architectural boundary

The ownership rule is:

> DexSim owns physical truth; EmbodiChain owns environment semantics.

```text
SimulationManager
  -> PhysicsBackend
      -> BackendSceneContext
          -> DexSim World / NewtonManager
          -> arena transforms
          -> model generation
          -> entity and view registries
```

DexSim owns:

- `ModelBuilder`, `Model`, both runtime `State` buffers, `Control`, contacts,
  collision pipeline, solver, and CUDA graph;
- stable entity references and canonical replay descriptors;
- world assignment, runtime body/shape/articulation/link/joint mappings;
- build, rebuild, snapshot, restore, commit, and generation transitions;
- the public tensor and binding contracts used by integrations.

EmbodiChain owns:

- backend selection and environment orchestration;
- public object APIs and backend-independent views;
- arena-local frame semantics and arena transform tables;
- pending initialization of newly added objects;
- environment reset, step count, observations, rewards, hooks, and datasets;
- differentiable environment policy and task-specific action/output kernels.

Core object and utility code must not resolve its owner through
`dexsim.default_world()` or the default `SimulationManager` instance.

## 5. Lifecycle and generation

### 5.1 Lifecycle phases

The integration exposes the following conceptual phases:

```text
BUILDING
  -> MODEL_FINALIZED
  -> VIEWS_BOUND
  -> SOLVER_READY
  -> RUNNING
  -> STALE
  -> rebuild
  -> VIEWS_BOUND
```

DexSim may retain its internal state enum, but public results must distinguish
successful model finalization, successful binding readiness, solver readiness,
staleness, and closure. `READY` alone must not ambiguously mean both “model
exists” and “all external consumers are rebound.”

### 5.2 Model generation

Each `NewtonManager` has a public, read-only `model_generation`:

- it starts at `0` before the first finalized model;
- the first successful finalize commits generation `1`;
- every successful model-replacing rebuild increments it once;
- live writes that do not replace or re-index model arrays do not increment it;
- failed candidate builds do not increment it;
- separate worlds maintain independent generations.

All runtime bindings carry the generation against which they were resolved.
Using a stale binding raises an explicit generation error or causes the owning
view to rebind before access; it never silently uses old IDs.

### 5.3 Prepare result and rebuild events

DexSim exposes a public prepare result equivalent to:

```python
@dataclass(frozen=True)
class NewtonPrepareResult:
    generation: int
    did_build: bool
    did_rebuild: bool
    added_entities: tuple[NewtonEntityRef, ...]
    removed_entities: tuple[NewtonEntityRef, ...]
```

After an atomic runtime commit, DexSim publishes a `MODEL_REBUILT` event with
the old and new generations and the topology delta. Failed builds publish a
failure result but never a success event. EmbodiChain subscribes when its
backend activates and unsubscribes during close.

`PhysicsBackend.prepare()` returns an EmbodiChain-level result carrying the
same generation and rebuild facts. It establishes a ready-to-step runtime but
does not itself advance simulation time.

## 6. DexSim public integration contract

### 6.1 API version

DexSim exports:

```python
NEWTON_INTEGRATION_API_VERSION = 2
```

The package patch version advances to `0.4.4`. EmbodiChain pins the exact
package version and validates the integration API version at backend
activation. This prevents two materially different Newton integrations from
sharing an indistinguishable dependency version.

### 6.2 Stable entity references

Registration returns an opaque, stable `NewtonEntityRef` containing enough
identity to reject cross-world use. Runtime integer IDs are not exposed as
stable handles.

The identity is derived from the owning world and DexSim entity, not from a
model body index. Removal invalidates the reference for new bindings while
allowing rebuild snapshots to identify that the entity should be omitted.

### 6.3 Public rigid-body attachment

DexSim provides a supported attachment API equivalent to:

```python
attach_rigid_body(
    entity,
    *,
    actor_type,
    shape_type,
    physical_attr=None,
    body_desc=None,
    shape_desc=None,
    geometry_desc=None,
) -> NewtonEntityRef
```

This is the only integration entry point EmbodiChain uses for Newton rigid
bodies. It:

- resolves the owning manager from the entity's actual arena/world;
- derives global world `-1` or the correct child-world index;
- captures mesh, box, sphere, and other supported geometry parameters;
- stores a canonical descriptor that can be replayed during rebuild;
- supports the legacy `PhysicalAttr` projection and Newton-native body/shape
  descriptors;
- makes descriptor ownership per entity so clone mutation cannot alias the
  prototype;
- marks a finalized runtime stale when topology changes.

Clone operations recompute target world metadata from the target arena rather
than copying the prototype's world index.

### 6.4 Public generation-aware bindings

DexSim provides binding operations equivalent to:

```python
bind_rigid_entities(refs) -> RigidEntityBinding
bind_articulations(refs) -> ArticulationBinding
```

`RigidEntityBinding` includes generation, body IDs where applicable, shape
IDs, and world IDs. Static entities may have shape IDs without body IDs.

`ArticulationBinding` includes generation, articulation and link body IDs,
world IDs, and explicit per-active-joint spans for:

- current q position;
- target q position;
- q velocity;
- target q velocity;
- generalized force/control.

It separately reports `qpos_width` and `dof`/`qvel_width`. An active-joint
index is never interpreted as a flattened DOF index.

Bindings use `int32` indices and declare device, dtype, shape, ownership,
mutability, and lifetime. Public state data uses `float32`; public quaternions
use `xyzw`.

## 7. Transactional rebuild and state restoration

### 7.1 Rebuild sequence

Runtime topology mutation follows:

```text
add/remove entity
  -> manager STALE
  -> snapshot by stable entity reference
  -> build candidate builder/model/state/control/solver
  -> restore surviving entities into candidate runtime
  -> validate candidate mappings and resources
  -> atomically commit candidate runtime
  -> generation + 1
  -> publish MODEL_REBUILT
  -> EmbodiChain rebinds views
  -> initialize only newly added entities
  -> FK and required DexSim visual synchronization
```

The old runtime is not cleared before the candidate is validated. Candidate
construction may temporarily use additional memory; correctness and rollback
take precedence over rebuild-time peak memory.

If candidate construction or restoration fails:

- the manager remains `STALE` and stepping is prohibited;
- the previous runtime remains available for diagnostics but is not presented
  as current physical truth for the mutated scene;
- generation does not change;
- no rebuilt event is emitted;
- callers may correct the scene/configuration and retry prepare.

### 7.2 Snapshot coverage

Snapshots are keyed by stable entity reference rather than runtime body ID.

Rigid state coverage:

- pose;
- linear and angular velocity;
- linear and angular acceleration;
- pending external force and torque;
- both ping-pong state buffers where fields exist.

Articulation coverage:

- root pose and velocity;
- current and target q position;
- current and target q velocity;
- generalized forces and active controls;
- relevant drive/control state;
- both ping-pong state buffers.

Contacts are regenerated and are not restored. Removed entities are omitted.
New entities receive descriptor/default state and are reported in the prepare
result for owner-side initialization.

### 7.3 Differentiable model leases

A live differentiable session holds a lease on its model generation. A
topology-changing rebuild is rejected while an outstanding tape depends on
that generation. The user or environment must finish backward or explicitly
close/detach the session before rebuilding. This prevents model arrays from
being freed while Warp autograd still references them.

## 8. EmbodiChain backend and scene context

### 8.1 BackendSceneContext

Every simulated object receives its owner explicitly through a context that
contains:

- `SimulationManager` identity;
- DexSim `World` and physics scene;
- active `PhysicsBackend`;
- arena list and full world transforms;
- current backend/model generation;
- entity/view registration helpers.

Objects no longer call `dexsim.default_world()`, global
`get_physics_scene()`, or default-instance arena utilities in core paths.

### 8.2 View rebinding

Rigid and articulation views store a binding rather than permanent IDs. Before
each batch operation they perform an O(1) generation comparison. A mismatch
causes one binding refresh for the complete batch, invalidating dependent
sorted-ID and arena-transform caches.

READY steady state does not re-resolve IDs, allocate bindings, or loop over
entities in Python.

### 8.3 Pending initialization

EmbodiChain records newly added objects in `pending_initialization`. After a
successful prepare and view rebind, only those objects receive their initial
state/reset. Existing objects retain the state restored by DexSim.

Base entity constructors do not call overridable `reset()` methods. Object
initialization is an explicit manager/lifecycle phase.

### 8.4 Frame contract

The public API distinguishes world and arena-local frames. Existing
`set_local_pose()` and `get_local_pose()` remain arena-local before and after
finalization.

Conversion uses the complete arena rigid transform, including rotation, not
only XY translation. Root and link pose data returned by DexSim global APIs is
converted in the view. Quaternion convention is consistently `xyzw`.

Velocity and wrench APIs retain their documented frames; any API whose frame
is currently ambiguous is documented and validated as part of this refactor
rather than inferred from method names.

### 8.5 Articulation data contract

EmbodiChain stores separate q-position and velocity/force widths and delegates
active-joint span resolution to `ArticulationBinding`. Current all-1-DOF robot
calls remain source compatible. Spherical and free joints use their actual q
and qd widths.

Writing current q position triggers required FK invalidation/evaluation before
link pose or visual state is reported. Unsupported data such as articulation
q acceleration is represented through capabilities and raises an explicit
unsupported-operation error rather than returning plausible zeros.

## 9. Multi-world isolation and cleanup

Registries, generation counters, entity mappings, solvers, state buffers,
CUDA graphs, and callbacks are keyed by owning world. A reference or binding
from one world cannot be used with another.

Same-GPU CUDA capture may use a device-level coordinator for capture safety,
but that coordinator does not own simulation state and stores only weak
manager references. Capture timeout and peer diagnostics use existing public
or implemented helpers and cannot wait indefinitely by default.

EmbodiChain adds an idempotent `SimulationManager.close()` that never exits the
process. It releases backend subscriptions, bindings, DexSim world resources,
CUDA graphs, and instance registry entries.

Existing cleanup surfaces remain compatible:

- `destroy()` remains available and preserves its documented exit-process
  compatibility behavior;
- `SimulationManager.reset(instance_id)` closes the selected live instance
  before removing it, so a new instance cannot inherit its world state;
- repeated close/reset is safe.

## 10. Capabilities and validation

Backend capabilities become structured and cover operations in addition to
asset categories. The Newton capability description includes at least:

- supported asset kinds;
- supported solver and gradient combinations;
- CUDA graph support and invalidation rules;
- partial reset and FK support;
- runtime topology mutation by asset kind;
- heterogeneous q/qd span support;
- runtime collision-filter support;
- contact sensor and acceleration-field support;
- multi-world support.

Configuration validates positive dt/substeps, device normalization, solver
parameters, gradient requirements, collision pipeline compatibility, and CUDA
graph combinations before finalization. Unconsumed solver parameters are
errors, not silently ignored fields.

Unsupported operations fail at configuration or API boundaries. In
particular, this iteration rejects runtime topology mutation for soft bodies
and cloth, and reports upstream Newton limitations instead of returning fake
data.

## 11. Differentiable execution architecture

### 11.1 Functional core and stateful environment

The differentiable layer has two levels:

1. `DifferentiableSession`, a generation-bound functional rollout owner.
2. `DifferentiableEmbodiedEnv`, a stateful Gym/EmbodiChain wrapper.

The session owns independent state, control, contact, and tape buffers. It
never records directly into buffers that a later environment step will
overwrite before backward. Each forward retains its required buffers in the
autograd context until backward or explicit release.

The environment maintains a functional session state across steps and mirrors
the resulting state into the normal runtime for non-differentiable consumers,
rendering, and existing object APIs. Mirror writes do not replace tape-owned
buffers.

Any generation change invalidates the session. Reset detaches the reset
environments from prior episode history.

### 11.2 Explicit execution modes

Configuration selects:

```python
DifferentiableStepCfg(
    mode="dynamics" | "kinematics",
    bptt_horizon_steps=None,
)
```

The existing `truncate_backward_at` input remains accepted as a deprecated
alias for `bptt_horizon_steps`. Its former ambiguous solver-substep meaning is
not retained. Truncation occurs only at environment-step boundaries.

### 11.3 Dynamics mode

Dynamics mode must execute `DifferentiableStepper` and the configured solver.
For one environment step:

```text
write action/control
  -> repeat sim_steps_per_control physics steps
      -> repeat Newton num_substeps solver steps
          -> clear forces
          -> apply pending external forces
          -> collide
          -> DifferentiableStepper.step
          -> swap state
  -> clear one-shot external inputs
```

The total solver step count is:

```text
sim_steps_per_control * NewtonPhysicsCfg.num_substeps
```

The solver dt is `physics_dt / num_substeps`. Control remains applied with the
same cadence as normal simulation. State ownership and final-buffer selection
are independent of odd/even substep count.

No FK-only fallback is permitted when a task is configured for dynamics. A
zero gradient caused by an unsupported control path fails validation/tests
and must be corrected at the control/solver contract.

### 11.4 Kinematics mode

Kinematics mode executes:

```text
action
  -> task-defined q-position update
  -> newton.eval_fk
  -> body/link state
  -> differentiable observations and reward
```

It does not run collision or a solver and does not advance physical simulation
time. It does advance the environment episode step. Runtime q position and
DexSim visual state are synchronized after the functional result when enabled
by the environment.

Kinematics is a first-class, explicitly named mode, not evidence that dynamics
differentiation works.

### 11.5 Autograd output contract

The PyTorch/Warp bridge uses explicit outputs equivalent to:

```python
DifferentiableOutput(
    name="reward",
    tensor=reward_torch,
    source=reward_warp_array,
    requires_grad=True,
)
```

Every differentiable output has a Warp source whose gradient is seeded by the
custom backward. Observation and reward are handled independently, allowing
both `loss(obs)` and `loss(reward)` to propagate to action. Shape, dtype,
device, contiguity, and finite-value checks occur at the bridge boundary.

Terminated, truncated, info, and other non-differentiable outputs explicitly
declare that they do not receive a gradient.

## 12. Environment and minimal functor integration

`DifferentiableEmbodiedEnv.step()` preserves the normal lifecycle:

```text
action preprocessing
  -> differentiable action mapping
  -> dynamics or kinematics execution
  -> differentiable observation/reward output
  -> ordinary info and termination
  -> episode counters
  -> hooks and dataset handling
  -> reset completed environments
```

This iteration does not redesign functors. Tasks provide thin differentiable
action and output adapters, normally implemented with Warp kernels. Existing
ordinary functors continue to run and are detached unless explicitly backed by
a `DifferentiableOutput` source.

If configuration claims an observation or reward term is differentiable but
no valid Warp source is supplied, construction or the first validated step
raises an error. The system does not silently sever the graph.

Non-differentiable side effects such as logging, dataset recording, and most
hooks remain outside the tape.

## 13. Franka reference environments

The Franka reach task exposes two explicit configurations:

- **Dynamics:** action maps to a differentiable Newton effort/control path and
  must pass through `DifferentiableStepper` and the semi-implicit solver.
- **Kinematics:** action updates joint q position and uses `newton.eval_fk`.

Both modes use the same documented frame convention. Arena-local targets are
converted consistently against world-frame Newton body state, or body state is
converted to arena-local before reward evaluation.

The reference task registers through the normal task import path, uses a
deterministic local/fixture asset for required tests, closes its environment in
all test outcomes, and does not depend on a network download for required CI.

The dynamics acceptance path uses a control mode that is expected to produce a
real action-to-state gradient. The kinematics task remains useful as a faster
smoke test but is reported separately.

## 14. Error handling

Errors identify the owning world, entity reference, operation, and expected
versus actual generation where relevant. The design distinguishes:

- invalid configuration;
- unsupported backend capability;
- stale/removed entity binding;
- closed world/session;
- candidate rebuild failure;
- active differentiable lease blocking rebuild;
- cross-world reference use;
- tensor contract mismatch.

Runtime collision-filter changes and other setup-only fields either trigger a
documented rebuild or fail explicitly; an API must not report success after
updating metadata that the live model does not consume.

## 15. Testing and acceptance

### 15.1 DexSim tests

- Public rigid attachment for mesh, box, and sphere descriptors.
- Correct global and child-world IDs for one, two, and eight child arenas.
- Clone descriptors are independent and use the target arena's world ID.
- Initial finalize increments generation once.
- Add/remove rebuild increments generation and produces new runtime mappings.
- Rigid pose/velocity/acceleration/external-wrench state survives rebuild.
- Articulation root/current-target q/qd/qf/control state survives rebuild.
- Both state buffers remain valid after restore.
- Candidate build/restore failure leaves a diagnosable non-half-initialized
  manager and does not increment generation.
- Revolute, spherical, and free-joint q/qd spans bind correctly.
- Two same-GPU worlds can build, step, rebuild, capture where enabled, and
  close independently.
- Repeated close and failed-construction cleanup are safe.

### 15.2 EmbodiChain tests

- A view's cached binding changes from the old to the new generation after
  rebuild and addresses the correct entity in every environment.
- Adding an entity preserves old-object state and initializes only the new
  entity.
- Removing an entity invalidates its binding without changing surviving
  object identity or state.
- Rigid, articulation root, and link local poses are correct in every arena,
  including non-zero rotations.
- No core object/view lookup resolves through `default_world()`.
- Two `SimulationManager` instances do not share world, scene, arena,
  generation, bindings, or cleanup state.
- Existing public object and manager calls remain source compatible.
- Default-backend behavior and tests remain unchanged.
- Capability errors replace silent q-acceleration, collision-filter, or sensor
  no-ops covered by the new surface.

### 15.3 Differentiable tests

- Dynamics uses the real `DifferentiableStepper` and expected solver-step
  count.
- Action gradient is finite, non-zero, and has the expected shape.
- Central finite difference agrees in direction and reasonable tolerance with
  autograd for a deterministic small scene.
- At least three consecutive environment steps advance state and backpropagate
  safely.
- Odd and even Newton substep counts select the correct final state.
- `loss(obs)` and `loss(reward)` independently propagate to action.
- Kinematics FK pose and gradient match a direct `eval_fk` reference.
- Runtime mirror updates do not corrupt a still-live backward pass.
- BPTT truncation detaches at the requested environment-step boundary.
- Reset prevents cross-episode gradient leakage.
- A generation change invalidates an old session; a live model lease blocks
  rebuild until released.
- Dynamics and kinematics handle arena-local targets consistently in multiple
  environments.

### 15.4 Verification order

The merge gate runs in this order:

```text
DexSim unit tests
  -> EmbodiChain CPU/headless contract tests
  -> serial GPU Newton integration tests
  -> multi-world lifecycle tests
  -> differentiable finite-difference tests
  -> complete EmbodiChain regression suite
  -> formatting and project pre-commit checks
```

GPU and external-simulation tests use the repository's registered markers and
deterministic teardown. Required tests do not silently skip because an asset
download failed.

## 16. Performance constraints

The correctness refactor must preserve an efficient steady state:

- generation checks are O(1);
- READY views do not rebind without a generation change;
- binding refresh is batched;
- per-step pose/state access does not loop over entities in Python;
- arena transform tables are device-resident and rebuilt only when their
  owning context changes;
- differentiable rollout buffers are deliberately owned and reused only when
  doing so cannot invalidate an outstanding tape.

Broad solver/render benchmarking is outside this specification. Focused
benchmarks may be added to demonstrate that generation-aware binding does not
regress steady-state batch access.

## 17. Implementation and repository sequencing

After this written specification is approved:

1. Create DexSim branch `feature/embodichain-newton-contracts` from its current
   `dev` branch.
2. Write a Stage 1 implementation plan spanning DexSim and EmbodiChain, with
   tests preceding implementation changes.
3. Implement and verify the DexSim public contract and lifecycle first.
4. Update EmbodiChain to consume that contract and complete multi-env,
   rebuild, articulation, and multi-manager parity.
5. Run the Stage 1 merge gate.
6. Write the dependent Stage 2 implementation plan.
7. Implement dynamics and kinematics sessions, bridge, environment, and
   reference tasks.
8. Run the complete differentiable and repository merge gates.

Stage 1 and Stage 2 remain reviewable as separate commit series even though
the first two previously proposed PR scopes are now one coordinated refactor.

## 18. Accepted trade-offs

- Transactional rebuild temporarily consumes more memory than clearing the
  old runtime first.
- Explicit bindings and generation checks add types and lifecycle plumbing but
  remove unsafe permanent IDs.
- Runtime articulation mutation requires upstream snapshot work rather than an
  EmbodiChain-only workaround.
- Differentiable tape-owned buffers use more memory than mutating manager state
  in place; this is required for correct backward ownership.
- Functor integration remains deliberately narrow in this iteration.
- Soft-body and cloth runtime mutation is explicitly postponed rather than
  approximated with incomplete state preservation.
