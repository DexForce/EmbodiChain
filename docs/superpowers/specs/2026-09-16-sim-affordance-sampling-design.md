# Simulation-Level Affordance Sampling Design

## Objective

Introduce reproducible, batched Affordance sampling as a simulation Atomic
Action capability. The first delivery is exercised by direct Atomic Action
callers and a parallel PickUp tutorial. Task Program execution, Gym lifecycle,
episode retry, and dataset recording are deliberately deferred.

## Scope

This change owns only geometric freedom declared by an Affordance:

- grasp candidates produced for antipodal affordances;
- interaction-point selection;
- assembly rotational symmetry;
- press contact-frame roll; and
- explicitly declared twist grasp-roll symmetry.

Action or goal parameter ranges are out of scope. In particular, Slide travel,
Twist angle, and OpenDoor opening-fraction sampling remain exact caller-owned
values.

The change must not modify or import Task Program, Gym, task deployment,
dataset, recorder, or episode collection code.

## Public Contracts

`AffordanceSamplingContext` identifies a deterministic sampling stream with
`count`, `seed`, `episode_id`, and `attempt_id`. It is immutable and never held
as mutable state by an Affordance.

`AffordancePoseCandidates` stores batched candidate poses, costs, and an
explicit validity mask. Padding is never promoted to a valid candidate.

`AffordanceSample` is the single sampling result:

```python
@dataclass(frozen=True, slots=True)
class AffordanceSample:
    success: torch.Tensor
    poses: torch.Tensor
    metadata: dict[str, object]
```

The public candidate-selection entry point belongs to `Affordance`:

```python
def sample_candidates(
    self,
    candidates: AffordancePoseCandidates,
    *,
    sampling: AffordanceSamplingContext | None,
    env_ids: torch.Tensor,
    key: str,
    reference_poses: torch.Tensor | None = None,
) -> AffordanceSample:
    ...
```

`AffordancePoseCandidates` does not expose a competing public `select()` API.
Existing tensor-returning helpers such as `get_press_pose()`,
`get_grasp_pose()`, and `get_assemble_object_pose()` remain deterministic and
compatible. New sampling helpers return `AffordanceSample`.

## Sampling Semantics

Branch zero retains the nominal or minimum-cost pose. Other rows select
quality-ranked, geometrically distinct poses where possible. Candidate
diversity is computed in the supplied reference frame, duplicate poses are
collapsed, and candidates are reused only after unique choices are exhausted.

The first simulation-only host uses one stable environment ID per logical
branch. This one-to-one mapping is a host constraint, not a Task Program
contract. A future program scheduler may map logical branches to simulator
slots differently.

Sampling uses a private CPU generator derived from stable stream identity and
does not mutate PyTorch's global random state. The same context, key, and
environment IDs produce the same result independent of row order. Incrementing
`attempt_id` creates a new stream.

## Planning and Execution Boundary

`PlanningContext` carries an optional `AffordanceSamplingContext`, and
`AtomicActionEngine.initial_context()` accepts it for direct callers. Projection
to following offline actions preserves it.

Affordances select only within legal geometry. Atomic Actions retain ownership
of IK, reachability, full-trajectory feasibility, failed-row hold behavior, and
effect declarations. An action combines `AffordanceSample.success` with its
planner success and copies namespaced sample metadata into
`PlannerDiagnostics.metadata`.

No generic `AtomicAction.plan()` hook manufactures Affordance diagnostics.
Only actions that actually consume a sample report it.

## Action Coverage

- PickUp, AxisAlign, Slide, and OpenDoor sample antipodal candidate sets through
  their Affordance rather than selecting candidates directly.
- Press and Twist use specialized Affordance sampling helpers for contact-frame
  roll while their existing exact pose helpers remain unchanged.
- Place uses the assembly symmetry sampler for `AssembleGoal`.
- InteractionPoints exposes a sampler that retains success and metadata.

Explicit caller-provided grasp poses remain exact and do not become sampled.

## Direct Simulation Tutorial

The PickUp tutorial adds Affordance branch, seed, and attempt arguments. When
branch count is greater than one, it creates the same number of simulation
environments, passes an `AffordanceSamplingContext` to
`AtomicActionEngine.initial_context()`, prints per-row sampling diagnostics, and
replays the batched trajectory.

The tutorial is a consumer and demonstration of the sim-layer API. Sampling
logic does not live in the tutorial.

## Error Handling

All public sampling values validate batch shape, dtype, device, finite values,
and non-empty identity. Rows with no eligible candidate return `success=False`,
an identity placeholder pose, and candidate ID `-1`. The action planner must
hold failed rows through the existing `build_plan()` contract.

## Validation

Pure CPU tests cover deterministic streams, row-order independence, candidate
diversity, padding and empty rows, metadata ownership, compatibility of exact
pose helpers, geometric symmetry sampling, PlanningContext propagation, action
success intersection, diagnostics propagation, and tutorial argument wiring.

The focused validation set is:

```bash
conda run -n open pytest -q tests/sim/atomic_actions
black --check --diff --color ./
python docs/scripts/check_api_docs.py
python -m sphinx -b dummy docs/source docs/build/api-docs-check
```
