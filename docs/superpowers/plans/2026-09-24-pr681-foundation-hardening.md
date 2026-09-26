# PR 681 Generation Foundation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #681's source-neutral generation foundation internally consistent, failure-safe, and correctly layered without implementing the later multi-slot, coverage scheduler, or observation-fan-out roadmap.

**Architecture:** Keep trajectory expansion and `GenerationSession` value-only, move B=1 physical orchestration into `motion.execution`, and let the session own generation streams and atomic candidate admission. Harden Atomic plan reconstruction and source conversion at their typed boundaries, then align public docs and examples with the resulting APIs.

**Tech Stack:** Python 3.11+, PyTorch, `dataclasses`, EmbodiChain `@configclass`, pytest, Sphinx, Black 26.3.1.

**Spec:** `docs/superpowers/specs/2026-09-24-pr681-foundation-hardening-design.md`

## Global Constraints

- Do not implement issue #670 T6-T10: multi-slot execution, cache/writer pipelines, coverage-per-cost selection, and observation fan-out remain excluded.
- Environment configuration remains the sole owner of backend, `num_envs`, control period, robot, sensors, and recorder construction.
- `motion.expansion` must not restore, step, execute, or persist physical episodes.
- Candidate identity and random streams remain independent of physical environment slots.
- Task Program syntax and `program.yaml` remain unchanged.
- New Python APIs require the 2021-2026 Apache header, `from __future__ import annotations`, full annotations, Google-style docstrings, and `__all__` coverage.
- Use `/root/miniconda3/envs/py311/bin/python` for focused CPU validation and `black==26.3.1` through that interpreter.

## Review Focus

- A transform that mutates nested tensors in place must be rejected by Task 1's constructor-level snapshot test.
- A `PlanResult` containing a positive first interval, removable zero-duration padding, or zero-duration motion must be covered by Task 2's normalization tests.
- Repeated source refills and different Affordance branches sharing a unit ID must remain distinct under Task 4's candidate-stream tests.
- Exceptions before measured admission, during admission, and after an ambiguous sink submission must leave the exact session state asserted by Task 5.
- Empty, singleton, mismatched, and device-mismatched Affordance provenance rows must be covered by Task 6.

---

### Task 1: Harden Atomic Plan Transform and Same-Grid Rebuilding

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/engine.py:315-417`
- Modify: `embodichain/lab/sim/atomic_actions/engine.py:648-707`
- Test: `tests/sim/atomic_actions/test_engine.py`

**Interfaces:**
- Consumes: `ActionPlan.snapshot() -> ActionPlan`, `AtomicActionEngine._validate_plan(plan, context, request)`.
- Produces: `plan_request(request, context=None, *, plan_transform=None) -> ActionPlan`; a same-grid `rebuild_plan_from_trajectory()` that validates the source plan and exact `dt`.

- [ ] **Step 1: Write failing transform-isolation and public-entry tests**

```python
def test_plan_transform_revalidates_mutated_tensor_invariants() -> None:
    engine = _engine()
    engine.register(StubAction())

    def transform(request, context, plan):
        plan.plan_success.resize_(1)
        return plan

    with pytest.raises(ValueError, match="plan_success batch"):
        engine.plan(
            _invocation(engine, torch.ones(2, 3)),
            plan_transform=transform,
        )


def test_plan_request_exposes_the_same_call_scoped_transform() -> None:
    engine = _engine()
    action = StubAction()
    engine.register(action)
    invocation = _invocation(engine, torch.ones(2, 3))
    request = action.resolve_request(invocation)
    calls = []

    result = engine.plan_request(
        request,
        plan_transform=lambda resolved, context, plan: calls.append(resolved) or plan,
    )

    assert result.success_all
    assert calls == [request]
```

- [ ] **Step 2: Run the two tests and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py::test_plan_transform_revalidates_mutated_tensor_invariants \
  tests/sim/atomic_actions/test_engine.py::test_plan_request_exposes_the_same_call_scoped_transform
```

Expected: the mutation test returns an invalid plan instead of raising, and `plan_request()` rejects the keyword argument.

- [ ] **Step 3: Implement owned transform snapshots**

```python
PlanTransform = Callable[
    [ResolvedActionRequest, PlanningContext, ActionPlan],
    ActionPlan,
]

def plan_request(
    self,
    request: ResolvedActionRequest,
    context: PlanningContext | None = None,
    *,
    plan_transform: PlanTransform | None = None,
) -> ActionPlan:
    return self._plan_request(request, context, plan_transform=plan_transform)
```

Call the callback with `plan.snapshot()`, require an `ActionPlan`, call
`transformed.snapshot()` to execute every constructor invariant, and then run
`_validate_plan()` on that owned value.

- [ ] **Step 4: Write failing same-grid and source-plan binding tests**

```python
with pytest.raises(ValueError, match="same-grid.*dt"):
    engine.rebuild_plan_from_trajectory(
        request,
        context,
        plan,
        replace(plan.joint_trajectory, dt=torch.tensor([[0.0, 0.01, 0.03]])),
    )

with pytest.raises(ValueError, match="invocation_id"):
    engine.rebuild_plan_from_trajectory(
        request,
        context,
        replace(plan, invocation_id="different"),
        plan.joint_trajectory,
    )
```

- [ ] **Step 5: Run the new rebuild tests and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py::test_rebuild_plan_rejects_changed_same_grid_dt \
  tests/sim/atomic_actions/test_engine.py::test_rebuild_plan_validates_source_plan_identity
```

Expected: both calls succeed before the fix.

- [ ] **Step 6: Validate the original plan and exact grid before rebuilding**

```python
self._validate_context(context)
self._validate_plan(plan.snapshot(), context, request)
if trajectory.dt.shape != plan.joint_trajectory.dt.shape or not torch.equal(
    trajectory.dt,
    plan.joint_trajectory.dt,
):
    raise ValueError("same-grid plan rebuilding requires identical dt.")
```

Retain batch, frame, environment ID, device, and robot-DoF checks, and reject a
source plan without `joint_trajectory` before reading its grid.

- [ ] **Step 7: Run Atomic engine tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py \
  tests/sim/atomic_actions/test_engine_per_env.py
```

- [ ] **Step 8: Commit Task 1**

```bash
git add embodichain/lab/sim/atomic_actions/engine.py \
  tests/sim/atomic_actions/test_engine.py
git commit -m "fix(atomic): validate call-scoped plan transforms"
```

### Task 2: Normalize PlanResult Sources Without Losing Permissions

**Files:**
- Modify: `embodichain/lab/sim/motion/expansion/source.py:108-154`
- Test: `tests/sim/motion/expansion/test_contracts.py`

**Interfaces:**
- Consumes: successful single-row `PlanResult`, `TrajectoryPhase` half-open ranges.
- Produces: `PlanResultSourceAdapter(joint_names, phases=(), validator_id="default", allowed_operators=None)` and a strict `TrajectoryTemplate` with preserved duration and remapped phases.

- [ ] **Step 1: Add failing timing and permission tests**

```python
def test_plan_result_source_normalizes_representable_arrival_intervals() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [0.0], [1.0], [1.0]]]),
        dt=torch.tensor([[0.2, 0.0, 0.3, 0.0]]),
    )
    phase = TrajectoryPhase("free", 0, 4, "free", ("joint_residual",))
    template = PlanResultSourceAdapter(
        ("joint",), phases=(phase,)
    ).export_template(result, context=_source_context())

    assert template.dt.tolist() == pytest.approx([0.0, 0.2, 0.3])
    assert template.positions[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert template.phases == (
        TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),
    )
    assert template.allowed_operators == ("joint_residual",)


def test_plan_result_source_rejects_zero_duration_motion() -> None:
    result = PlanResult(
        success=True,
        positions=torch.tensor([[[0.0], [1.0]]]),
        dt=torch.tensor([[0.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="zero-duration position change"):
        PlanResultSourceAdapter(("joint",)).export_template(
            result, context=_source_context()
        )
```

- [ ] **Step 2: Run the source tests and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_contracts.py::test_plan_result_source_normalizes_representable_arrival_intervals \
  tests/sim/motion/expansion/test_contracts.py::test_plan_result_source_rejects_zero_duration_motion
```

Expected: strict timing construction fails and zero-duration motion lacks its
owning error.

- [ ] **Step 3: Implement normalization and phase remapping**

```python
def _normalized_plan_result_row(
    positions: torch.Tensor,
    dt: torch.Tensor,
    phases: tuple[TrajectoryPhase, ...],
) -> tuple[torch.Tensor, torch.Tensor, tuple[TrajectoryPhase, ...]]:
    prepend = bool(dt[0] > 0)
    retained = [0]
    for index in range(1, positions.shape[0]):
        if dt[index] == 0:
            if not torch.equal(positions[index], positions[index - 1]):
                raise ValueError("PlanResult contains a zero-duration position change")
            continue
        retained.append(index)
    normalized_positions = positions[retained]
    normalized_dt = dt[retained].clone()
    if prepend:
        normalized_positions = torch.cat(
            (normalized_positions[:1], normalized_positions), dim=0
        )
        normalized_dt = torch.cat((normalized_dt.new_zeros(1), normalized_dt))
    boundary = {
        old: int(prepend) + sum(index < old for index in retained)
        for old in range(positions.shape[0] + 1)
    }
    normalized_phases = []
    for phase_index, phase in enumerate(phases):
        start = boundary[phase.start]
        if prepend and phase_index == 0 and phase.start == 0:
            start = 0
        stop = boundary[phase.stop]
        if stop <= start:
            raise ValueError(f"timing normalization erases phase {phase.name!r}")
        normalized_phases.append(replace(phase, start=start, stop=stop))
    return normalized_positions, normalized_dt, tuple(normalized_phases)

def _phase_permission_union(
    phases: tuple[TrajectoryPhase, ...],
) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        operator
        for phase in phases
        for operator in phase.allowed_operators
    ))
```

Prepend a held first sample for positive `dt[0]`; drop only zero-interval rows
equal to the previous retained position; construct a boundary-index map for
phases; reject a phase whose remapped stop is not greater than its start.

- [ ] **Step 4: Add explicit permission override coverage**

Assert `allowed_operators=("via_points",)` overrides the inferred union and
duplicate names are rejected by `TrajectoryTemplate`.

- [ ] **Step 5: Run expansion contract and operator tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_contracts.py \
  tests/sim/motion/expansion/test_operators.py
```

- [ ] **Step 6: Commit Task 2**

```bash
git add embodichain/lab/sim/motion/expansion/source.py \
  tests/sim/motion/expansion/test_contracts.py
git commit -m "fix(motion): normalize plan result source templates"
```

### Task 3: Add Session-Owned Generation Streams and Atomic Proposals

**Files:**
- Modify: `embodichain/lab/sim/motion/expansion/contracts.py`
- Modify: `embodichain/lab/sim/motion/expansion/session.py:104-318`
- Modify: `embodichain/lab/sim/motion/expansion/__init__.py`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst`
- Test: `tests/sim/motion/expansion/test_session.py`
- Test: `tests/sim/motion/expansion/test_contracts.py`

**Interfaces:**
- Produces: immutable `ProposalRequest`; `GenerationSession.generation_generator(case_id, initial_state_id, *, source_id, source_revision, template_id, operation_id) -> torch.Generator`; `GenerationSession.propose_many(requests: Sequence[ProposalRequest]) -> tuple[tuple[CandidateIdentity, torch.Generator], ...]`.
- Preserves: `GenerationSession.propose(case_id, initial_state_id, *, source_id, source_revision, template_id, operator_id, geometry_family_id=None, parent_id=None) -> tuple[CandidateIdentity, torch.Generator]` by delegation.

- [ ] **Step 1: Define failing stream-independence tests**

Create two equal sessions and assert their first stream draws match, while two
successive generators from one session produce different draws. Use the full
case/source/revision/template/operation key in every call.

```python
def draw(session: GenerationSession) -> torch.Tensor:
    generator = session.generation_generator(
        "case",
        "initial",
        source_id="source",
        source_revision="revision",
        template_id="template",
        operation_id="trajectory_expansion",
    )
    return torch.rand(4, generator=generator)

left = _session()
right = _session()
left_first = draw(left)
left_second = draw(left)
right_first = draw(right)
assert torch.equal(left_first, right_first)
assert not torch.equal(left_first, left_second)
```

- [ ] **Step 2: Add a failing atomic batch-proposal test**

With `collection.max_proposals=1`, submit two valid `ProposalRequest` values and
assert `RuntimeError` leaves `snapshot()["counts"]["proposed"] == 0` and no
candidate audit entries.

```python
requests = (
    ProposalRequest("case", "initial", "source", "r1", "template", "nominal"),
    ProposalRequest("case", "initial", "source", "r1", "template", "residual"),
)
with pytest.raises(RuntimeError, match="Proposal budget"):
    session.propose_many(requests)
snapshot = session.snapshot()
assert snapshot["counts"]["proposed"] == 0
assert snapshot["audit"] == ()
```

- [ ] **Step 3: Run the new session tests and verify missing APIs**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_session.py::test_generation_streams_advance_and_replay_across_sessions \
  tests/sim/motion/expansion/test_session.py::test_propose_many_is_atomic_when_budget_is_insufficient
```

Expected: import or attribute failures.

- [ ] **Step 4: Implement `ProposalRequest` validation**

```python
@dataclass(frozen=True)
class ProposalRequest:
    scene_case_id: str
    initial_state_id: str
    source_id: str
    source_revision: str
    template_id: str
    operator_id: str
    geometry_family_id: str | None = None
    parent_id: str | None = None
```

Validate every non-optional ID with `_text()`; parent ownership remains a
session check.

- [ ] **Step 5: Implement generation streams and transactional proposal batches**

Add `_generation_ordinals` beside `_ordinals`. `generation_generator()` hashes
the augmentation seed, complete source/case stream, and current ordinal; it
advances only `_generation_ordinals`.

`propose_many()` tuple-copies and validates all requests, registered cases,
parents, stop state, total budget, duplicate IDs, and locally computed ordinals
before mutating `_ordinals`, `_attempts`, or proposal counts.

- [ ] **Step 6: Delegate `propose()` and export the request type**

Construct one `ProposalRequest`, call `propose_many((request,))`, and return its
only result. Add `ProposalRequest` to contracts and package `__all__`, plus its
autosummary and autoclass entries in the expansion API page.

- [ ] **Step 7: Run session and contract tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_session.py \
  tests/sim/motion/expansion/test_contracts.py
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 8: Commit Task 3**

```bash
git add embodichain/lab/sim/motion/expansion/contracts.py \
  embodichain/lab/sim/motion/expansion/session.py \
  embodichain/lab/sim/motion/expansion/__init__.py \
  docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst \
  tests/sim/motion/expansion/test_session.py \
  tests/sim/motion/expansion/test_contracts.py
git commit -m "fix(motion): own proposal streams in generation sessions"
```

### Task 4: Integrate Novel Streams into the Candidate Coordinator

**Files:**
- Modify: `embodichain/lab/sim/motion/expansion/variants.py:355-364`
- Modify: `embodichain/lab/sim/motion/expansion/variants.py:501-680`
- Modify: `embodichain/lab/sim/motion/expansion/coordinator.py`
- Modify: `embodichain/lab/sim/motion/expansion/cfg.py:399-422`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst`
- Test: `tests/sim/motion/expansion/test_contracts.py`
- Test: `tests/sim/motion/expansion/test_cfg.py`
- Test: `tests/sim/motion/expansion/test_variants.py`

**Interfaces:**
- Consumes: Task 3 `generation_generator()` and `propose_many()`.
- Produces: `expand_trajectory_variants(template, *, case, count, cfg, joint_limits, control_dt, generator=None, ...)`; duplicate-free repeated coordinator refills; FIFO-only accepted scheduling configuration.

- [ ] **Step 1: Write failing repeated-refill and lineage tests**

```python
first = coordinator.enqueue_source(template, count=2)
second = coordinator.enqueue_source(template, count=2)
assert not torch.equal(first[1].template.positions, second[0].template.positions)
assert len(second) == 1  # repeated nominal is filtered before proposal
```

Create two templates with different middle qpos and different
`affordance_selection`; assert candidate and geometry-family IDs differ.

- [ ] **Step 2: Write a failing coordinator atomicity test**

Configure one remaining proposal, call `enqueue_source(template, count=2)`, and
assert the exception leaves both `pending_count` and proposed count at zero.

- [ ] **Step 3: Write the FIFO-only configuration test**

```python
with pytest.raises(ValueError, match="coverage_per_cost.*not implemented"):
    TrajectoryGenerationJobCfg.from_mapping(
        {"scheduling": {"policy": "coverage_per_cost"}}
    )
```

- [ ] **Step 4: Run the new tests and verify red state**

Expect identical refills, family aliasing, partial proposal state, and accepted
inert scheduling configuration.

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_contracts.py::test_candidate_coordinator_refills_with_novel_variants \
  tests/sim/motion/expansion/test_contracts.py::test_candidate_coordinator_namespaces_affordance_geometry \
  tests/sim/motion/expansion/test_contracts.py::test_candidate_coordinator_enqueue_is_atomic \
  tests/sim/motion/expansion/test_cfg.py::test_generation_profile_rejects_unimplemented_scheduler
```

- [ ] **Step 5: Allow an injected expansion generator**

Add `generator: torch.Generator | None = None` to
`expand_trajectory_variants()`. Pass an injected private generator through
sequential `apply_trajectory_variant()` calls; preserve standalone digest-based
generation when it is omitted.

- [ ] **Step 6: Add exact candidate fingerprinting**

Compute SHA-256 over CPU-contiguous position and `dt` bytes plus joint names and
phase values. Store accepted fingerprints in a set bounded by the session
proposal budget. Do not allocate identities for known rows.

- [ ] **Step 7: Use the session stream and transactional identity admission**

Build an operation ID from stable JSON Affordance lineage, obtain one session
generator, expand, filter exact duplicates, then call `propose_many()` once.
Do not pass provisional geometry-family IDs. Append work items and update seen
fingerprints only after proposal admission succeeds. Return `()` if every row
is already known.

- [ ] **Step 8: Reject unimplemented scheduling policy**

Keep the schema field but accept only `fifo`; raise
`ValueError("scheduling.policy coverage_per_cost is not implemented")` for the
future mode.

Update the expansion configuration prose to state that only FIFO is accepted
in this foundation and that coverage-per-cost remains a future policy.

- [ ] **Step 9: Run expansion tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q tests/sim/motion/expansion
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 10: Commit Task 4**

```bash
git add embodichain/lab/sim/motion/expansion/{variants.py,coordinator.py,cfg.py} \
  docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst \
  tests/sim/motion/expansion/{test_contracts.py,test_cfg.py,test_variants.py}
git commit -m "fix(motion): generate novel coordinator candidates"
```

### Task 5: Move and Harden the Single-Slot Host Integration

**Files:**
- Modify: `embodichain/lab/sim/motion/execution.py`
- Delete: `embodichain/lab/sim/motion/expansion/single_slot.py`
- Modify: `embodichain/lab/sim/motion/expansion/__init__.py`
- Modify: `embodichain/lab/sim/motion/expansion/session.py:398-452`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.execution.rst`
- Create: `tests/sim/motion/test_generation_execution.py`
- Delete: `tests/sim/motion/expansion/test_single_slot.py`
- Modify: `tests/sim/motion/test_motion_imports.py`

**Interfaces:**
- Consumes: pure `CandidateCoordinator`, `GenerationSession`, and episode/receipt contracts.
- Produces from `embodichain.lab.sim.motion.execution`: `InitialStatePort`, `MeasuredExecutor`, `EpisodeSink`, `FixedSceneInitialStatePort`, `SingleSlotOutcome`, `SingleSlotRunner`.
- Changes `MeasuredExecutor.execute(item, prepared, *, episode_id, commit_id, on_rollout_started) -> ExpertEpisode`.

- [ ] **Step 1: Move existing happy-path tests to the execution owner**

Create `test_generation_execution.py` with the Apache header, copy the current
fixed-scene and committed-receipt tests, and import from
`embodichain.lab.sim.motion.execution`. Add an import test asserting the six
host symbols are absent from `motion.expansion`.

- [ ] **Step 2: Add failing lifecycle cleanup tests**

Add executors for validation failure, restore failure, execution failure after
calling `on_rollout_started()`, malformed measured `joint_positions`, and a
successful execution whose sink raises. Assert the first four leave
`pending_reserved_episodes == 0`. Assert the sink exception returns
`status == "write_pending"`, includes episode/commit/submission IDs, and leaves
exactly one `pending_write` reservation.

- [ ] **Step 3: Add a failing candidate-selection test**

Pre-admit another ready row for the same case, execute the coordinator's current
item, and assert the runner reserves that item's identity rather than the first
unrelated ready row.

- [ ] **Step 4: Run new host tests and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/test_generation_execution.py \
  tests/sim/motion/test_motion_imports.py
```

Expected: execution imports are missing and lifecycle tests expose retained
reservations or wrong-row assignment.

- [ ] **Step 5: Move protocols and runner into `motion.execution`**

Append the existing protocols and adapter to `execution.py`, update `__all__`,
and import expansion values from defining submodules to avoid package cycles.
Remove `single_slot.py` and all expansion re-exports.

Remove the six host values from the expansion API page and document them with
autosummary and autoclass directives in the motion execution API page.

- [ ] **Step 6: Reserve the requested ready candidate**

Add optional `candidate_id: str | None = None` to
`GenerationSession.take_ready()`. When supplied, select only that ready attempt;
preserve FIFO when omitted. The runner passes its current ID and verifies
`ready.identities == (item.spec.identity,)`.

- [ ] **Step 7: Make rollout start a one-shot executor callback**

```python
started = False

def on_rollout_started() -> None:
    nonlocal started
    if started:
        raise RuntimeError("rollout start callback may run only once")
    session.mark_rollout_started(item.spec.identity)
    started = True
```

Pass it to `MeasuredExecutor.execute()`. If execution returns without invoking
it, release the assigned candidate and raise
`RuntimeError("executor returned before starting the rollout")`.

- [ ] **Step 8: Make measured admission and persistence outcomes explicit**

Wrap `accept_episode()` so any exception releases a still-running candidate
with reason `measured_admission_failed` before re-raising. Keep `False` as the
measured rejection outcome.

Extend `SingleSlotOutcome` with optional `episode_id`, `commit_id`, and
`submission_id`. Catch raw sink exceptions only after `accept_episode()` and
return `write_pending` with those IDs while retaining `pending_write`. Apply all
returned receipts before returning `committed` or `write_failed`.

- [ ] **Step 9: Run execution, session, and expansion tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/test_execution.py \
  tests/sim/motion/test_generation_execution.py \
  tests/sim/motion/test_motion_imports.py \
  tests/sim/motion/expansion
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 10: Commit Task 5**

```bash
git add -A embodichain/lab/sim/motion/execution.py \
  embodichain/lab/sim/motion/expansion \
  tests/sim/motion/test_generation_execution.py \
  tests/sim/motion/test_motion_imports.py \
  tests/sim/motion/expansion/test_single_slot.py \
  docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst \
  docs/source/api_reference/embodichain/embodichain.lab.sim.motion.execution.rst
git commit -m "refactor(motion): move generation lifecycle to execution"
```

### Task 6: Validate Affordance Provenance Alignment

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/affordance_sampling.py:379-394`
- Test: `tests/sim/atomic_actions/test_affordance_sampling.py`

**Interfaces:**
- Produces: `_pose_sampling_metadata()` that validates row count, dtype/device, and optional reference-pose alignment before serialization.

- [ ] **Step 1: Add failing nominal-path alignment tests**

Call Twist and Press sampling with `sampling=None`, two target poses, and one
environment ID; assert `ValueError` contains `matching rows`. Parameterize a
reference-pose row mismatch and environment IDs on a different device when CUDA
is available.

- [ ] **Step 2: Run the tests and verify malformed metadata is returned**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_affordance_sampling.py::test_nominal_sampling_rejects_misaligned_provenance_rows
```

Expected: no exception for the CPU mismatch.

- [ ] **Step 3: Implement centralized provenance validation**

```python
if poses.shape[:1] != success.shape or success.shape != env_ids.shape:
    raise ValueError("poses, success, and env_ids must have matching rows")
if poses.device != success.device or success.device != env_ids.device:
    raise ValueError("poses, success, and env_ids must share a device")
if reference_poses is not None and (
    reference_poses.shape != poses.shape
    or reference_poses.device != poses.device
):
    raise ValueError("reference_poses must match selected pose rows and device")
```

Retain the existing world-frame JSON representation.

- [ ] **Step 4: Run Affordance tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_affordance_sampling.py \
  tests/sim/atomic_actions/test_actions.py
```

- [ ] **Step 5: Commit Task 6**

```bash
git add embodichain/lab/sim/atomic_actions/affordance_sampling.py \
  tests/sim/atomic_actions/test_affordance_sampling.py
git commit -m "fix(atomic): validate affordance provenance rows"
```

### Task 7: Route Showcase Recording and Repair Public API Docs

**Files:**
- Modify: `scripts/tutorials/atomic_action/tutorial_utils.py`
- Modify: `examples/sim/motion/task_environment_augmentation_showcase.py`
- Modify: `examples/sim/motion/README_task_environment_augmentation.md`
- Modify: `tests/sim/atomic_actions/test_tutorial_utils.py`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst`

**Interfaces:**
- Produces: parser option `--auto_play_video_path`; `start_auto_play_recording()` forwards it as `SimulationManager.start_window_record(save_path=args.auto_play_video_path)`.
- Documents: the Atomic adapter autosummary without changing the module scope of engine APIs.

- [ ] **Step 1: Add a failing recording-path helper test**

```python
args = parser.parse_args([
    "--auto_play",
    "--auto_play_video_path",
    "/tmp/showcase/place_auto_play.mp4",
])
assert start_auto_play_recording(sim, args, "place_auto_play")
sim.start_window_record.assert_called_once_with(
    fps=AUTO_PLAY_RECORD_FPS,
    max_memory=AUTO_PLAY_RECORD_MAX_MEMORY,
    save_path="/tmp/showcase/place_auto_play.mp4",
    video_prefix="place_auto_play",
    look_at=DEFAULT_AUTO_PLAY_LOOK_AT,
    use_sim_time=True,
)
```

- [ ] **Step 2: Run the helper test and verify the option is unknown**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_tutorial_utils.py::test_auto_play_recording_uses_explicit_video_path
```

Expected: argparse rejects `--auto_play_video_path`.

- [ ] **Step 3: Add recording path and showcase forwarding**

Add `--auto_play_video_path` to the shared parser with default `None`. Pass
`save_path=getattr(args, "auto_play_video_path", None)` to
`start_window_record()`. In the showcase, append `--auto_play` and
`--auto_play_video_path <output-dir>/place_auto_play.mp4` unless callers already
provided them. Update README and module docstring with the exact file.

- [ ] **Step 4: Fix Sphinx module scopes and relocated API ownership**

After the adapter autosummary in the Atomic page, restore:

```rst
.. currentmodule:: embodichain.lab.sim.atomic_actions
```

The host runner documentation was relocated with Task 5, so this task only
repairs the Atomic page's module scope.

- [ ] **Step 5: Run helper, API coverage, and syntax validation**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_tutorial_utils.py
/root/miniconda3/envs/py311/bin/python -m compileall -q \
  examples/sim/motion/task_environment_augmentation_showcase.py \
  scripts/tutorials/atomic_action/tutorial_utils.py
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 6: Run targeted Sphinx resolution validation**

```bash
output_dir=$(mktemp -d /tmp/embodichain-sphinx-XXXXXX)
log_path=$(mktemp /tmp/embodichain-sphinx-log-XXXXXX)
/root/miniconda3/envs/py311/bin/python -m sphinx -b dummy --keep-going \
  docs/source "$output_dir" >"$log_path" 2>&1 || true
if rg -n "failed to import.*(AtomicAction|SingleSlotRunner|InitialStatePort|EpisodeSink)" "$log_path"; then
  exit 1
fi
```

Record unrelated baseline warnings separately.

- [ ] **Step 7: Commit Task 7**

```bash
git add scripts/tutorials/atomic_action/tutorial_utils.py \
  examples/sim/motion/task_environment_augmentation_showcase.py \
  examples/sim/motion/README_task_environment_augmentation.md \
  tests/sim/atomic_actions/test_tutorial_utils.py \
  docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst
git commit -m "fix(examples): write augmentation replay to output directory"
```

### Task 8: Reconcile Context and Run Proportional Final Verification

**Files:**
- Modify if required: `agent_context/topics/motion-planning/motion-planning.md`
- Modify if required: `agent_context/MAP.yaml`
- Verify: all files changed by Tasks 1-7

**Interfaces:**
- Consumes: completed implementation and project context lifecycle rules.
- Produces: accurate owner guidance and a clean, proportionally verified branch.

- [ ] **Step 1: Compute affected project-context topics**

```bash
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/project-dev-context/scripts/context.py affected \
  --base origin/main --explain
```

Expected: `motion-planning`, `atomic-actions`, and the test-derived
`simulation-system` hint.

- [ ] **Step 2: Apply the context lifecycle decision**

If the motion overview already says host integrations own restore, execution,
and persistence, keep that guidance and add only the concrete
`motion/execution.py` owner if its navigation table lacks it. Do not append a PR
summary. Update `MAP.yaml` only if its source-of-truth paths do not already cover
`motion/execution.py`.

- [ ] **Step 3: Validate context when context files changed**

```bash
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/project-dev-context/scripts/context.py check
/root/miniconda3/envs/py311/bin/python -m pytest -q -c /dev/null --noconftest \
  tests/test_agent_context_map.py tests/test_agent_context_tools.py
```

- [ ] **Step 4: Run Black before the final code commit**

```bash
/root/miniconda3/envs/py311/bin/python -m black .
```

Review `git diff --stat`; only task-owned files may change.

- [ ] **Step 5: Run the full affected CPU test surface**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion \
  tests/sim/motion/test_execution.py \
  tests/sim/motion/test_generation_execution.py \
  tests/sim/motion/test_motion_imports.py \
  tests/sim/atomic_actions
```

Expected: zero failures; record exact pass count and warnings.

- [ ] **Step 6: Run documentation and diff gates**

```bash
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
git diff --check "$(git merge-base origin/main HEAD)"..HEAD
git status --short --branch
```

Expected: exports documented, no whitespace errors, and no unrelated changes.

- [ ] **Step 7: Apply the project pre-commit checklist**

Use `.agents/skills/pre-commit-check/SKILL.md` against the final changed-file
set. If context changed, commit it separately:

```bash
git add agent_context/MAP.yaml agent_context/topics/motion-planning/motion-planning.md
git commit -m "docs(context): locate generation host integration"
```

Skip that commit when the context lifecycle decision is no-update.

- [ ] **Step 8: Re-run verification after the last commit**

Repeat Steps 4-6 against committed HEAD. Confirm a clean worktree. Report live
simulator/GPU validation as not run unless those resources were exercised.
