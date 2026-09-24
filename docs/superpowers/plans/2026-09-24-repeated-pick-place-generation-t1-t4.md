# Repeated Pick/Place Configured Generation T1-T4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete issue #670's T1-T4 contracts and run a separate Generation Profile against the configured `repeated_pick_place` Task Program to produce and execute one nominal plus two deterministic same-grid trajectory candidates.

**Architecture:** First execute the existing PR #681 foundation-hardening plan so Atomic plans, session streams, proposal admission, FIFO scheduling, and B=1 host ownership are safe. Then add a strict profile loader, make Atomic and Task Program sources implement the common adapter protocol, thread a call-scoped transform through `ExecutionSession` and `SemanticCallExecutor`, and compose `CandidateCoordinator` inside a Task Program integration controller. The existing `TaskProgramDemoBridge` remains the only Gym execution path; the demo passes the profile at episode scope and does not implement T5 persistence.

**Tech Stack:** Python 3.11+, PyTorch, EmbodiChain `@configclass`, YAML through `embodichain.utils.utility.load_config`, Gymnasium, pytest, Black 26.3.1.

**Spec:** `docs/superpowers/specs/2026-09-24-repeated-pick-place-generation-t1-t4-design.md`

**Prerequisite plan:** `docs/superpowers/plans/2026-09-24-pr681-foundation-hardening.md`

## Global Constraints

- Complete and verify the prerequisite hardening plan before Task 1 below; do not reimplement its fixes in this plan.
- Keep `program.yaml`, `integration.yaml`, `env.yaml`, embodiment components, and execution-policy components unchanged.
- The Generation Profile must not accept backend, `num_envs`, `control_dt`, robot, sensor, simulator, or recorder construction fields.
- `motion.expansion` must not import Task Program, restore/reset a simulator, step an environment, or persist an episode.
- Generation transforms are scoped to one Atomic execution session and are never stored globally on `AtomicActionEngine`.
- The configured demo is B=1, FIFO, `max_inflight=1`, Affordance-disabled, and timing-disabled.
- Only Place `retract` may use `joint_residual` or `via_points`; `approach` and `release` remain exact.
- Candidate identity and RNG must remain independent of physical environment IDs.
- New Python files use the 2021-2026 DexForce Apache 2.0 header, `from __future__ import annotations`, complete public annotations, Google-style docstrings, and `__all__`.
- Do not add tests whose primary subject is the example script; validate production contracts and use compile/import checks for the example.
- Use `/root/miniconda3/envs/py311/bin/python` and Black 26.3.1 for validation.

## Prerequisite Gate

Before Task 1, execute every unchecked step in
`2026-09-24-pr681-foundation-hardening.md` in order. The required outputs are:

- `PlanTransform` isolation and exact same-grid rebuilding;
- normalized `PlanResultSourceAdapter` timing and permissions;
- `ProposalRequest`, session-owned generation streams, and atomic
  `propose_many()`;
- novel, duplicate-filtered coordinator refills and FIFO-only decoding;
- B=1 host orchestration under `motion.execution` with explicit pending-write
  reconciliation;
- aligned Affordance provenance;
- corrected showcase recording and Sphinx module scope; and
- a clean proportional hardening verification run.

Record the prerequisite plan's final commit SHA in this plan's execution notes.
If any prerequisite test fails, stop at that owning task rather than working
around the failure in the configured generation integration.

## Review Focus

- A profile containing environment-owned or unknown keys must fail during Task 1 decoding before an environment is created.
- A normal Task Program episode with no profile must take the identical runtime path, pinned by Task 4 and Task 6 tests.
- A configured transform must affect only `place`, preserve all `dt` and phase endpoints, and leave `pick` untouched, pinned by Task 5 tests.
- Candidate ordinal `0`, `1`, and `2` must be stable across equal sessions while distinct within one reference, pinned by Task 5 tests.
- Failure during candidate generation or an out-of-range ordinal must occur before `ExecutionSession` installs commands and leave no partial coordinator/session state, pinned by Task 5 and Task 6 tests.

---

### Task 1: Complete and Load the Generation Profile

**Files:**
- Modify: `embodichain/lab/sim/motion/expansion/cfg.py`
- Create: `embodichain/lab/sim/motion/expansion/profile.py`
- Modify: `embodichain/lab/sim/motion/expansion/__init__.py`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst`
- Test: `tests/sim/motion/expansion/test_cfg.py`
- Test: `tests/sim/motion/expansion/test_imports.py`

**Interfaces:**
- Consumes: `embodichain.utils.utility.load_config(path)` and `TrajectoryGenerationJobCfg.from_mapping(data)`.
- Produces: `load_generation_profile(path: str | Path) -> TrajectoryGenerationJobCfg`; profile fields `source_revision`, `unit_scope`, explicit phase maps, `max_variants_per_reference`, and observation metadata.

- [ ] **Step 1: Write failing closed-schema and loader tests**

Add these focused cases to `test_cfg.py`:

```python
def _task_program_profile_payload() -> dict[str, object]:
    return {
        "source": {
            "kind": "task_program",
            "source_id": "repeated_cube_pick_place",
            "source_revision": "config:repeated_pick_place_v1",
            "unit_scope": "action",
            "template_id": "place",
            "phase_permissions": {
                "approach": [],
                "release": [],
                "retract": ["joint_residual", "via_points"],
            },
            "phase_kinds": {
                "approach": "free",
                "release": "contact",
                "retract": "free",
            },
        },
        "augmentation": {"max_variants_per_reference": 3},
        "scheduling": {"candidate_budget": 3},
        "execution": {"max_inflight": 1},
        "observation": {"enabled": False, "profiles": []},
    }


def test_generation_profile_decodes_task_program_source_contract() -> None:
    cfg = TrajectoryGenerationJobCfg.from_mapping(_task_program_profile_payload())
    assert cfg.source.source_revision == "config:repeated_pick_place_v1"
    assert cfg.source.phase_permissions["retract"] == (
        "joint_residual",
        "via_points",
    )
    assert cfg.augmentation.max_variants_per_reference == 3
    assert cfg.observation.profiles == ()


@pytest.mark.parametrize("field", ["backend", "num_envs", "control_dt", "robot"])
def test_generation_profile_rejects_environment_owned_fields(field: str) -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        TrajectoryGenerationJobCfg.from_mapping({field: "forbidden"})


def test_generation_profile_rejects_implicit_or_inconsistent_phases() -> None:
    payload = _task_program_profile_payload()
    payload["source"]["phase_permissions"].pop("release")
    with pytest.raises(ValueError, match="phase_permissions.*phase_kinds"):
        TrajectoryGenerationJobCfg.from_mapping(payload)
```

Use `tmp_path` and `yaml.safe_dump()` only inside the loader test to prove a
real file reaches the strict decoder:

```python
def test_load_generation_profile_uses_strict_decoder(tmp_path: Path) -> None:
    path = tmp_path / "generation.yaml"
    path.write_text(yaml.safe_dump(_task_program_profile_payload()))
    cfg = load_generation_profile(path)
    assert cfg.source.kind == "task_program"
```

- [ ] **Step 2: Run the new tests and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_cfg.py::test_generation_profile_decodes_task_program_source_contract \
  tests/sim/motion/expansion/test_cfg.py::test_generation_profile_rejects_environment_owned_fields \
  tests/sim/motion/expansion/test_cfg.py::test_generation_profile_rejects_implicit_or_inconsistent_phases \
  tests/sim/motion/expansion/test_cfg.py::test_load_generation_profile_uses_strict_decoder
```

Expected: missing fields/imports and unknown loader API.

- [ ] **Step 3: Extend the strict mapping decoder and typed profile fields**

Teach `_decode()` to handle only the mapping forms used by the profile:

```python
def _decode_tuple_value(
    value: object,
    *,
    member_type: type,
    name: str,
) -> tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence")
    if any(type(item) is not member_type for item in value):
        raise ValueError(f"{name} contains an invalid value type")
    return tuple(value)


elif get_origin(expected) is dict:
    key_type, value_type = get_args(expected)
    if key_type is not str or not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a string-keyed mapping")
    if get_origin(value_type) is tuple:
        member_type = get_args(value_type)[0]
        decoded[key] = {
            str(item_key): _decode_tuple_value(
                item_value,
                member_type=member_type,
                name=f"{name}.{item_key}",
            )
            for item_key, item_value in value.items()
        }
    elif value_type is str:
        if any(type(item_key) is not str or type(item_value) is not str for item_key, item_value in value.items()):
            raise ValueError(f"{name} must map strings to strings")
        decoded[key] = dict(value)
    else:
        raise TypeError(f"unsupported config mapping annotation: {expected}")
```

Add and validate the fields:

```python
@configclass
class _SourceCfg:
    kind: str = "handwritten"
    source_id: str = "handwritten_qpos"
    source_revision: str = "unversioned"
    unit_scope: str = "action"
    template_id: str = "reference_0"
    phase_permissions: dict[str, tuple[str, ...]] = {}
    phase_kinds: dict[str, str] = {}


@configclass
class _ObservationCfg:
    enabled: bool = False
    profiles: tuple[str, ...] = ()
```

Add `max_variants_per_reference: int = 1` to
`TrajectoryAugmentationCfg`. Remove the transitional `source.kind == "atomic"`
spelling. Require Atomic/Task Program `unit_scope == "action"`, exact equality
between the two phase-map key sets, legal phase kinds, unique legal operators,
unique observation profiles, non-empty profiles exactly when observation is
enabled, `candidate_budget >= max_variants_per_reference`, and
`execution.max_inflight == 1` for this T4 schema.

- [ ] **Step 4: Implement the profile loader and public exports**

Create `profile.py`:

```python
from pathlib import Path

from embodichain.utils.utility import load_config

from .cfg import TrajectoryGenerationJobCfg

__all__ = ["load_generation_profile"]


def load_generation_profile(path: str | Path) -> TrajectoryGenerationJobCfg:
    """Load one callable-free Generation Profile through strict decoding."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Generation Profile does not exist: {source}")
    data = load_config(source)
    try:
        return TrajectoryGenerationJobCfg.from_mapping(data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid Generation Profile {source}: {error}") from error
```

Export it from `motion.expansion` and add API docs entries. Add an import-only
test to `test_imports.py`.

- [ ] **Step 5: Run profile and import tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion/test_cfg.py \
  tests/sim/motion/expansion/test_imports.py
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 6: Commit Task 1**

```bash
git add embodichain/lab/sim/motion/expansion/{cfg.py,profile.py,__init__.py} \
  docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst \
  tests/sim/motion/expansion/{test_cfg.py,test_imports.py}
git commit -m "feat(motion): load source-neutral generation profiles"
```

### Task 2: Make the Atomic Adapter a SourceAdapter

**Files:**
- Modify: `embodichain/lab/sim/motion/expansion/source.py:64-80`
- Modify: `embodichain/lab/sim/atomic_actions/trajectory_adapter.py`
- Modify: `embodichain/lab/sim/atomic_actions/__init__.py`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst`
- Test: `tests/sim/atomic_actions/test_engine.py`

**Interfaces:**
- Consumes: hardening output `AtomicActionEngine.rebuild_plan_from_trajectory()` and `SourceContext`.
- Produces: `ActionPlanTemplateAdapter.export_template(source: ActionPlan, *, context: SourceContext) -> TrajectoryTemplate`, structurally compatible with `SourceAdapter[ActionPlan]`.

- [ ] **Step 1: Write the failing structural-adapter test**

Update the existing adapter test so source identity comes only from
`SourceContext`:

```python
adapter = ActionPlanTemplateAdapter(
    joint_names=("j0", "j1", "j2"),
    phase_permissions={"stub": ("joint_residual",)},
    phase_kinds={"stub": "free"},
)
context = SourceContext(
    source_id="atomic_stub",
    source_revision="test:r1",
    unit_id="stub:0",
    scene_case=SceneCase(
        "case",
        "initial",
        "signature",
        "task",
        "robot",
    ),
    control_dt=ACTION_DT,
)
assert isinstance(adapter, SourceAdapter)
template = adapter.export_template(plan, context=context)
assert (template.source_id, template.source_revision, template.template_id) == (
    "atomic_stub",
    "test:r1",
    "stub:0",
)
```

Add negative assertions for an extra phase declaration, a missing segment
declaration, B=2, and a failed plan.

- [ ] **Step 2: Run the adapter test and verify red state**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py::test_action_plan_template_adapter_implements_source_contract
```

Expected: constructor/signature mismatch and no runtime-checkable protocol match.

- [ ] **Step 3: Refactor the adapter around `SourceContext`**

Mark `SourceAdapter` with `@runtime_checkable`, then use this public Atomic
adapter shape:

```python
@dataclass(frozen=True)
class ActionPlanTemplateAdapter:
    joint_names: tuple[str, ...]
    phase_permissions: Mapping[str, Sequence[str]]
    phase_kinds: Mapping[str, str]
    controlled_joint_indices: tuple[int, ...] | None = None
    validator_id: str = "default"
    kind: ClassVar[str] = "atomic_action"

    def export_template(
        self,
        source: ActionPlan,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        if not isinstance(source, ActionPlan):
            raise TypeError("source must be an ActionPlan")
        trajectory = source.joint_trajectory
        if trajectory is None:
            raise ValueError("ActionPlan must retain a joint trajectory")
        if trajectory.batch_size != 1 or not source.success_all:
            raise ValueError("the initial Atomic adapter supports one successful row")
        declared = {segment.name for segment in source.segments}
        if declared != set(self.phase_permissions) or declared != set(self.phase_kinds):
            raise ValueError("every ActionPlan segment requires an exact phase declaration")
        phases = tuple(
            TrajectoryPhase(
                segment.name,
                segment.start,
                segment.stop,
                kind=self.phase_kinds[segment.name],
                allowed_operators=tuple(self.phase_permissions[segment.name]),
            )
            for segment in source.segments
        )
        return TrajectoryTemplate(
            source_id=context.source_id,
            source_revision=context.source_revision,
            template_id=context.unit_id,
            joint_names=self.joint_names,
            positions=trajectory.positions[0],
            dt=trajectory.dt[0],
            phases=phases,
            allowed_operators=tuple(
                dict.fromkeys(
                    operator
                    for phase in phases
                    for operator in phase.allowed_operators
                )
            ),
            validator_id=self.validator_id,
            controlled_joint_indices=self.controlled_joint_indices,
        )
```

Require the segment-name set to equal both phase-map key sets. Build identity
only from `context`. Remove the old identity-bearing `export(plan)` API and
update all PR-local callers/tests because the PR is unmerged.

- [ ] **Step 4: Re-run Atomic adapter and API checks**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py \
  tests/sim/atomic_actions/test_actions.py
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 5: Commit Task 2**

```bash
git add embodichain/lab/sim/motion/expansion/source.py \
  embodichain/lab/sim/atomic_actions/{trajectory_adapter.py,__init__.py} \
  docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst \
  tests/sim/atomic_actions/test_engine.py
git commit -m "feat(atomic): expose action plans as generation sources"
```

### Task 3: Thread Call-Scoped Transforms Through ExecutionSession

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/engine.py:552-580`
- Modify: `embodichain/lab/sim/atomic_actions/execution.py:321-391`
- Modify: `embodichain/lab/sim/atomic_actions/__init__.py`
- Test: `tests/sim/atomic_actions/test_engine.py`
- Test: `tests/sim/atomic_actions/test_engine_per_env.py`

**Interfaces:**
- Consumes: hardening output `PlanTransform` and `_plan_request(..., plan_transform=...)`.
- Produces: `AtomicActionEngine.start(..., plan_transform: PlanTransform | None = None) -> ExecutionSession`; session-scoped transform reuse for initial planning, replanning, and prepared revisions.

- [ ] **Step 1: Write failing session-transform tests**

In `test_execution.py`, install a transform that records request revisions and
returns a valid snapshot with diagnostic metadata:

```python
calls: list[int] = []

def transform(request, context, plan):
    calls.append(request.revision)
    diagnostics = replace(
        plan.diagnostics,
        metadata={**plan.diagnostics.metadata, "generation_test": True},
    )
    return replace(plan, diagnostics=diagnostics)

session = engine.start((invocation,), context, plan_transform=transform)
assert calls == [0]
    assert session.active_plan.diagnostics.metadata["generation_test"] is True
```

Add a recovery/revision test using the existing session helpers and assert the
same transform is called for each actual planning call. Add a baseline test
asserting `engine.start()` without the keyword produces the prior plan.

- [ ] **Step 2: Run the focused tests and verify keyword rejection**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine_per_env.py::test_execution_session_applies_call_scoped_plan_transform \
  tests/sim/atomic_actions/test_engine_per_env.py::test_execution_session_reuses_transform_for_replan \
  tests/sim/atomic_actions/test_engine_per_env.py::test_execution_session_without_transform_is_unchanged
```

- [ ] **Step 3: Store and apply the transform at every session planning site**

Extend `AtomicActionEngine.start()` and `ExecutionSession.__init__()` with the
keyword. Validate callable-or-None once, store `_plan_transform`, and change
both `_plan_current()` and `_install_prepared_revision()` to call:

```python
plan = self._engine._plan_request(
    request,
    context,
    plan_transform=self._plan_transform,
)
```

Export `PlanTransform` from the public Atomic package. Do not change
`compile()` or install state on the engine.

- [ ] **Step 4: Run Atomic engine/execution tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/atomic_actions/test_engine.py \
  tests/sim/atomic_actions/test_engine_per_env.py
```

- [ ] **Step 5: Commit Task 3**

```bash
git add embodichain/lab/sim/atomic_actions/{engine.py,execution.py,__init__.py} \
  tests/sim/atomic_actions/{test_engine.py,test_engine_per_env.py}
git commit -m "feat(atomic): scope plan transforms to execution sessions"
```

### Task 4: Add the Task Program Source and Transform Factory Protocol

**Files:**
- Create: `embodichain/lab/task_program/runtime/generation.py`
- Create: `embodichain/lab/task_program/integrations/generation.py`
- Modify: `embodichain/lab/task_program/runtime/__init__.py`
- Modify: `embodichain/lab/task_program/integrations/__init__.py`
- Modify: `embodichain/lab/task_program/runtime/executor.py:225-311`
- Test: `tests/lab/task_program/test_generation.py`
- Test: `tests/lab/task_program/test_semantic_executor.py`

**Interfaces:**
- Produces runtime values `TaskProgramPlanRequest` and `TaskProgramPlanTransformFactory`.
- Produces integration adapter `TaskProgramSourceAdapter(atomic_adapter)` with `kind == "task_program"`.
- Changes `SemanticCallExecutor(..., plan_transform_factory=None)` to request one transform after grounding and pass it to `engine.start()`.

- [ ] **Step 1: Write failing Task Program source-adapter tests**

Create `test_generation.py` with the project header. For this delegation-only
test, patch the real Atomic adapter method so no synthetic `ActionPlan` graph is
needed:

```python
atomic = ActionPlanTemplateAdapter(
    joint_names=("j0", "j1", "j2"),
    phase_permissions={"place": ("joint_residual",)},
    phase_kinds={"place": "free"},
)
adapter = TaskProgramSourceAdapter(atomic)
assert adapter.kind == "task_program"
assert isinstance(adapter, SourceAdapter)
expected = _trajectory_template()
with patch.object(
    ActionPlanTemplateAdapter,
    "export_template",
    return_value=expected,
) as export_template:
    actual = adapter.export_template(Mock(spec=ActionPlan), context=context)
assert actual is expected
export_template.assert_called_once()
```

- [ ] **Step 2: Write failing transform-factory runtime tests**

Define a fake runtime-checkable factory that records
`TaskProgramPlanRequest`. Start a one-call workflow and assert the captured
request contains workflow ID, workflow call index, analysis call index, exact
`SemanticCallSpec`, and grounded `ActionInvocation`; assert its returned
transform reaches `engine.start()`. Add a no-factory test that compares the
existing result/events.

- [ ] **Step 3: Run the new tests and verify missing APIs**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/lab/task_program/test_generation.py \
  tests/lab/task_program/test_semantic_executor.py::test_executor_requests_call_scoped_plan_transform \
  tests/lab/task_program/test_semantic_executor.py::test_executor_without_transform_factory_is_unchanged
```

- [ ] **Step 4: Implement provider-independent runtime contracts**

Create `runtime/generation.py`:

```python
@dataclass(frozen=True, slots=True)
class TaskProgramPlanRequest:
    workflow_id: str
    workflow_call_index: int
    analysis_call_index: int
    call: SemanticCallSpec
    invocation: ActionInvocation


@runtime_checkable
class TaskProgramPlanTransformFactory(Protocol):
    def create_plan_transform(
        self,
        request: TaskProgramPlanRequest,
        *,
        engine: AtomicActionEngine,
    ) -> PlanTransform | None:
        """Return one session-scoped transform or None for an unchanged call."""
```

Validate stable IDs/indices and exact value types in the dataclass.

- [ ] **Step 5: Implement the thin Task Program source adapter**

In `integrations/generation.py`:

```python
@dataclass(frozen=True)
class TaskProgramSourceAdapter:
    atomic_adapter: ActionPlanTemplateAdapter
    kind: ClassVar[str] = "task_program"

    def export_template(
        self,
        source: ActionPlan,
        *,
        context: SourceContext,
    ) -> TrajectoryTemplate:
        return self.atomic_adapter.export_template(source, context=context)
```

This file may import Atomic and motion contracts; motion core must not import it.

- [ ] **Step 6: Request the transform after grounding**

Add the optional factory to `SemanticCallExecutor.__init__()`, validate it with
the runtime-checkable protocol, and in `_start_call()` build the request after
`grounded` is validated:

```python
plan_transform = None
if self._plan_transform_factory is not None:
    plan_transform = self._plan_transform_factory.create_plan_transform(
        TaskProgramPlanRequest(
            workflow_id=self._workflow_id,
            workflow_call_index=workflow_call_index,
            analysis_call_index=analysis_call_index,
            call=call,
            invocation=invocation,
        ),
        engine=self._engine,
    )
session = self._engine.start(
    (invocation,),
    context,
    eligible_mask=active_mask,
    plan_transform=plan_transform,
)
```

Require a `PlanTransform`-compatible callable or `None` before starting the
session.

- [ ] **Step 7: Export and run Task Program tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/lab/task_program/test_generation.py \
  tests/lab/task_program/test_semantic_executor.py \
  tests/lab/task_program/test_program_compiler.py
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
```

- [ ] **Step 8: Commit Task 4**

```bash
git add embodichain/lab/task_program/runtime/{generation.py,__init__.py,executor.py} \
  embodichain/lab/task_program/integrations/{generation.py,__init__.py} \
  tests/lab/task_program/{test_generation.py,test_semantic_executor.py}
git commit -m "feat(task-program): expose atomic plans to generation adapters"
```

### Task 5: Compose CandidateCoordinator Into a Task Program Transform

**Files:**
- Modify: `embodichain/lab/task_program/integrations/generation.py`
- Modify: `embodichain/lab/sim/motion/expansion/coordinator.py`
- Test: `tests/lab/task_program/test_generation.py`
- Test: `tests/sim/motion/expansion/test_contracts.py`

**Interfaces:**
- Consumes: profile, `TaskProgramPlanRequest`, engine, hardening outputs `generation_generator()`/`propose_many()`, and the Task Program source adapter.
- Produces: `TaskProgramCandidatePlanTransformFactory`; immutable `TaskProgramGenerationRecord`; coordinator-populated observation profiles and strict compatibility keys.

- [ ] **Step 1: Write failing candidate-controller tests**

Define local `_ThreePhasePlaceAction`, `_engine()`, and `_invocation()` fixtures
inside `test_generation.py`; copy the concrete robot/motion-generator binding
setup from `tests/sim/atomic_actions/test_engine.py` rather than importing a test
module. `_ThreePhasePlaceAction.skill_id` is `"place"`; its `_plan()` uses
`TimedTrajectory.from_uniform_step()` with seven samples and calls
`build_plan(..., segment_lengths={"approach": 2, "release": 2, "retract": 3})`.
The first/last sample of every segment is distinct so endpoint preservation is
observable, while the retract segment has one editable interior sample.

Build that B=1 Place-like plan and assert for requested ordinals 0, 1, and 2:
Create the profile from Task 1 and assert for requested ordinals 0, 1, and 2:

```python
factory = TaskProgramCandidatePlanTransformFactory(
    profile,
    candidate_index=index,
    program_id="repeated_cube_pick_place",
    integration_id="repeated_pick_place_v1",
    robot_profile_id="franka_panda",
)
transform = factory.create_plan_transform(request, engine=engine)
rebuilt = transform(resolved_request, planning_context, plan)
assert torch.equal(rebuilt.joint_trajectory.dt, plan.joint_trajectory.dt)
assert torch.equal(
    rebuilt.joint_trajectory.positions[:, release.start : release.stop],
    plan.joint_trajectory.positions[:, release.start : release.stop],
)
assert torch.equal(rebuilt.joint_trajectory.positions[:, 0], plan.joint_trajectory.positions[:, 0])
assert torch.equal(rebuilt.joint_trajectory.positions[:, -1], plan.joint_trajectory.positions[:, -1])
```

Assert candidate 0 equals nominal, candidates 1/2 differ from nominal and each
other, and equal fresh factories reproduce the same tensors and IDs.

- [ ] **Step 2: Add selector and failure tests**

Assert Pick returns `None`, source/program mismatch raises before transform
creation, candidate index 3 raises before `engine.start()`, timing-enabled or
Affordance-enabled profiles are rejected for this controller, and a forced
coordinator proposal exception leaves no records or pending candidates.

- [ ] **Step 3: Run tests and verify missing controller**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/lab/task_program/test_generation.py::test_candidate_transform_builds_three_same_grid_place_plans \
  tests/lab/task_program/test_generation.py::test_candidate_transform_ignores_unselected_skills \
  tests/lab/task_program/test_generation.py::test_candidate_transform_rejects_out_of_range_ordinal_atomically
```

- [ ] **Step 4: Populate observation and compatibility metadata in the coordinator**

Remove any caller-controlled partial default key for the configured route. Add
private deterministic key construction over:

```python
(
    scene_case.scene_case_id,
    scene_case.initial_state_id,
    template.joint_names,
    tuple((phase.phase_id, phase.start_index, phase.stop_index, phase.kind) for phase in template.phases),
    source_context.control_dt,
    backend_id,
    template.validator_id,
    observation_profiles,
)
```

Extend `CandidateCoordinator.__init__()` with validated optional `backend_id`
and derive `CandidateSpec.observation_profiles` from the copied profile.
Preserve an explicit complete key only for host tests that supply every field.

- [ ] **Step 5: Implement records and controller validation**

Use immutable values:

```python
@dataclass(frozen=True, slots=True)
class TaskProgramGenerationRecord:
    workflow_id: str
    workflow_call_index: int
    candidate_index: int
    selected_candidate_id: str
    candidates: tuple[CandidateSpec, ...]
    templates: tuple[TrajectoryTemplate, ...]

    def to_metadata(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_call_index": self.workflow_call_index,
            "candidate_index": self.candidate_index,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_ids": [
                candidate.identity.candidate_id for candidate in self.candidates
            ],
            "geometry_family_ids": [
                candidate.identity.geometry_family_id for candidate in self.candidates
            ],
            "trajectory_variants": [
                dict(candidate.trajectory_variant) for candidate in self.candidates
            ],
        }
```

`to_metadata()` contains only JSON values and excludes tensor payloads.

Validate `candidate_index` as a non-negative integer, source kind/id/revision,
`max_variants_per_reference <= candidate_budget`, B=1, Affordance disabled,
timing disabled, and exact selected skill.

- [ ] **Step 6: Implement the transform data path**

Inside the transform closure:

1. derive a deterministic initial-state digest from owned CPU qpos bytes;
2. create `SceneCase` from program/integration/profile identities;
3. read complete engine robot joint names and `(D, 2)` limits;
4. create/register `GenerationSession`;
5. build `SourceContext` using workflow/call/invocation identity;
6. construct `ActionPlanTemplateAdapter` from the profile and wrap it in
   `TaskProgramSourceAdapter`;
7. enqueue exactly `max_variants_per_reference` candidates;
8. select `items[candidate_index]`;
9. create a full-joint `TimedTrajectory` with the original env IDs; and
10. call `engine.rebuild_plan_from_trajectory()`.

Copy `CandidateSpec` metadata into `PlannerDiagnostics.metadata["generation"]`
through typed `dataclasses.replace()`. Append a record only after rebuild and
full plan validation succeed.

- [ ] **Step 7: Run controller and coordinator tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/lab/task_program/test_generation.py \
  tests/sim/motion/expansion/test_contracts.py \
  tests/sim/motion/expansion/test_session.py \
  tests/sim/motion/expansion/test_variants.py
```

- [ ] **Step 8: Commit Task 5**

```bash
git add embodichain/lab/task_program/integrations/generation.py \
  embodichain/lab/sim/motion/expansion/coordinator.py \
  tests/lab/task_program/test_generation.py \
  tests/sim/motion/expansion/test_contracts.py
git commit -m "feat(task-program): coordinate configured trajectory candidates"
```

### Task 6: Thread the Configured Controller Through the Gym Bridge

**Files:**
- Modify: `embodichain/lab/task_program/integrations/environment.py:258-292`
- Modify: `embodichain/lab/task_program/integrations/environment.py:657-820`
- Modify: `embodichain/lab/gym/envs/task_program/bridge.py:951-1076`
- Modify: `embodichain/lab/gym/envs/embodied_env.py:2234-2325`
- Test: `tests/gym/envs/task_program/test_environment.py`
- Test: `tests/gym/envs/task_program/test_bridge.py`
- Test: `tests/gym/envs/task_program/test_task_vertical_slices.py`

**Interfaces:**
- Changes `TaskProgramEnvironmentAdapter.create_bridge(program, *, generation_profile=None, candidate_index=0)`.
- Changes `EmbodiedEnv.create_demo_segments(..., generation_profile=None, generation_candidate_index=0)`.
- Produces `TaskProgramDemoBridge.generation_records` and `EmbodiedEnv.task_program_generation_records` snapshots.

- [ ] **Step 1: Write failing no-profile compatibility tests**

Use existing fake factories to create two equal bridges, one through the old
call and one with explicit `generation_profile=None`. Exhaust both and compare
actions, runtime metadata, completion mask, and event order. Assert
`generation_records == ()` for both.

- [ ] **Step 2: Write failing profile-threading vertical slice**

Install a fake `TaskProgramPlanTransformFactory` that tags one Place plan. Call:

```python
result = execute_demo_episode(
    env,
    generation_profile=profile,
    generation_candidate_index=1,
)
assert result.completed
assert env.task_program_generation_records[0].candidate_index == 1
```

Assert Pick actions are unchanged and every generated action still passes
through the buffered Gym command sink.

- [ ] **Step 3: Run the focused tests and verify keyword/API failures**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/gym/envs/task_program/test_environment.py \
  tests/gym/envs/task_program/test_bridge.py \
  tests/gym/envs/task_program/test_task_vertical_slices.py
```

- [ ] **Step 4: Extend runtime assembly without changing default ownership**

Add optional `plan_transform_factory` and `generation_trace_provider` fields to
`TaskProgramRuntimeAssembly`. Change `_assemble_execution_runtime()` to accept
one validated optional factory and pass it into `SemanticCallExecutor`.

In `create_bridge()`, when a profile is supplied:

1. strictly require `TrajectoryGenerationJobCfg`;
2. create `TaskProgramCandidatePlanTransformFactory` after semantic assembly so
   it receives exact program/integration/profile IDs;
3. pass it to execution assembly; and
4. pass it as a read-only record provider to the bridge.

When the profile is `None`, do none of those operations.

- [ ] **Step 5: Expose owned record snapshots through bridge and environment**

Add:

```python
@property
def generation_records(self) -> tuple[TaskProgramGenerationRecord, ...]:
    return () if self._generation_trace_provider is None else self._generation_trace_provider.records
```

`EmbodiedEnv.task_program_generation_records` delegates only to its active
bridge and returns `()` before a generated episode. Never expose the mutable
factory list.

- [ ] **Step 6: Accept episode-level profile arguments**

Make `create_task_program_bridge()` and `create_demo_segments()` consume the
explicit profile and ordinal instead of forwarding them to the legacy planner.
Validate that a profile cannot be supplied without a configured Task Program.
Leave all existing positional/keyword behavior unchanged otherwise.

- [ ] **Step 7: Run Gym and Task Program integration tests**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/gym/envs/task_program/test_environment.py \
  tests/gym/envs/task_program/test_bridge.py \
  tests/gym/envs/task_program/test_task_vertical_slices.py \
  tests/lab/task_program/test_semantic_executor.py \
  tests/lab/task_program/test_generation.py
```

- [ ] **Step 8: Commit Task 6**

```bash
git add embodichain/lab/task_program/integrations/environment.py \
  embodichain/lab/gym/envs/task_program/bridge.py \
  embodichain/lab/gym/envs/embodied_env.py \
  tests/gym/envs/task_program/{test_environment.py,test_bridge.py,test_task_vertical_slices.py}
git commit -m "feat(gym): execute configured generation candidates"
```

### Task 7: Add the Task-Local Profile and Configured Showcase

**Files:**
- Create: `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.demo.yaml`
- Modify: `embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/README.md`
- Create: `examples/sim/motion/repeated_pick_place_generation_showcase.py`
- Modify: `tests/test_task_program_package_data.py`
- Test: `tests/sim/motion/expansion/test_cfg.py`

**Interfaces:**
- Consumes: `load_generation_profile()`, `build_env_cfg_from_args()`, `execute_demo_episode()`, and `EmbodiedEnv.task_program_generation_records`.
- Produces: a task-local profile, JSON provenance, and candidate comparison plots under `--output-dir`.

- [ ] **Step 1: Add the profile fixture and failing package-data test**

Create the YAML exactly as specified in the design, including explicit empty
permissions for `approach` and `release`. In `test_task_program_package_data.py`
assert the file exists in source/package enumeration and strict-load it:

```python
profile_path = task_root / "manipulation/repeated_pick_place/generation.demo.yaml"
profile = load_generation_profile(profile_path)
assert profile.source.kind == "task_program"
assert profile.augmentation.max_variants_per_reference == 3
```

- [ ] **Step 2: Run package/profile tests and verify missing file**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/test_task_program_package_data.py \
  tests/sim/motion/expansion/test_cfg.py
```

- [ ] **Step 3: Implement the showcase launcher**

The script parser defines:

```python
parser.add_argument("--task-config", type=Path, default=_DEFAULT_TASK_CONFIG)
parser.add_argument(
    "--generation-profile", type=Path, default=_DEFAULT_GENERATION_PROFILE
)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--candidate-indices", nargs="+", type=int, default=[0, 1, 2])
parser.add_argument("--save-video", action="store_true")
```

Reuse `add_env_launcher_args_to_parser()` for device/headless/render flags and
`build_env_cfg_from_args()` for the task. Construct the ordinary Gym environment
with the configured `task_program_adapter_factory` from `action_config`.

For every candidate index:

```python
env.reset(options={"save_data": False})
result = execute_demo_episode(
    env,
    episode_index=candidate_index,
    generation_profile=profile,
    generation_candidate_index=candidate_index,
)
records = env.unwrapped.task_program_generation_records
write_generation_json(records, result, output_dir / f"candidate_{candidate_index}.json")
save_generation_plot(records, output_dir / f"candidate_{candidate_index}.png")
```

Use a `try/finally` to discard pending recordings and close the environment.
Reject duplicate/negative/out-of-range indices before constructing it. JSON
contains profile/source/candidate/Task Program metadata, never tensor payloads.
Implement `_write_generation_json()` with `json.dumps(..., allow_nan=False)`
over `record.to_metadata()` and `result.to_metadata()`. Implement
`_save_generation_plot()` with a lazy Matplotlib import; for every record, plot
every template's joint columns against cumulative `dt` and draw vertical lines
at each phase boundary. Label selected candidates and close every figure.

When `--save-video` is set, call `env.unwrapped.sim.start_window_record()` with
`save_path=<output-dir>/candidate_<index>.mp4` before the episode, then call
`stop_window_record()` and `wait_window_record_saves()` in a `finally` block.
Raise if recording cannot start; omission of `--save-video` performs no renderer
recording calls.

- [ ] **Step 4: Update the task README with the exact command and claims**

Document:

```bash
python examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.demo.yaml \
  --output-dir /tmp/repeated-pick-place-generation \
  --device cuda --headless
```

State that projected completion is not measured physical success or confirmed
coverage.

- [ ] **Step 5: Run static example and configured deployment validation**

```bash
/root/miniconda3/envs/py311/bin/python -m compileall -q \
  examples/sim/motion/repeated_pick_place_generation_showcase.py
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
```

Expected: syntax success and both deployments pass static levels 1-4.

- [ ] **Step 6: Commit Task 7**

```bash
git add embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/{generation.demo.yaml,README.md} \
  examples/sim/motion/repeated_pick_place_generation_showcase.py \
  tests/test_task_program_package_data.py \
  tests/sim/motion/expansion/test_cfg.py
git commit -m "feat(examples): add configured pick-place generation showcase"
```

### Task 8: Document Public APIs and Reconcile Project Context

**Files:**
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst`
- Modify: `docs/source/api_reference/embodichain/embodichain.lab.task_program.rst`
- Review/modify if required: `agent_context/topics/motion-planning/motion-planning.md`
- Review/modify if required: `agent_context/topics/task-programs/task-programs.md`
- Review/modify if required: `agent_context/MAP.yaml`

**Interfaces:**
- Documents every new public export and the profile/runtime ownership chain.
- Produces an explicit context lifecycle decision for `motion-planning` and `task-programs`.

- [ ] **Step 1: Run affected-topic routing**

```bash
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/project-dev-context/scripts/context.py affected \
  --base origin/main --explain
```

- [ ] **Step 2: Apply the lifecycle decision**

Update the motion overview only if it lacks the Generation Profile loader,
source adapters, or coordinator owner. Update Task Program context with the
optional episode-scoped generation transform boundary and configured showcase
only if current guidance cannot route a maintainer to the new integration.
Keep one owning explanation and link from the other topic. Update `MAP.yaml`
source-of-truth paths only when the new files are outside listed directories.

- [ ] **Step 3: Add public API documentation**

Document and autosummarize:

- `load_generation_profile`;
- `ActionPlanTemplateAdapter.export_template`;
- `TaskProgramPlanRequest` and `TaskProgramPlanTransformFactory`;
- `TaskProgramSourceAdapter`, `TaskProgramCandidatePlanTransformFactory`, and
  `TaskProgramGenerationRecord`.

Do not add the example as a public API.

- [ ] **Step 4: Validate API docs and context**

```bash
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/project-dev-context/scripts/context.py check
/root/miniconda3/envs/py311/bin/python -m pytest -q -c /dev/null --noconftest \
  tests/test_agent_context_map.py tests/test_agent_context_tools.py
```

- [ ] **Step 5: Commit Task 8**

Stage only files changed by the lifecycle decision:

```bash
git add docs/source/api_reference/embodichain/embodichain.lab.sim.motion.expansion.rst \
  docs/source/api_reference/embodichain/embodichain.lab.sim.atomic_actions.rst \
  docs/source/api_reference/embodichain/embodichain.lab.task_program.rst \
  agent_context/MAP.yaml \
  agent_context/topics/motion-planning/motion-planning.md \
  agent_context/topics/task-programs/task-programs.md
git commit -m "docs: describe configured generation integration"
```

Omit unchanged context files from `git add`; skip the context portion of the
commit when the lifecycle decision is no-update.

### Task 9: Run Proportional Final Verification

**Files:**
- Verify: every file changed by the prerequisite hardening plan and Tasks 1-8.

**Interfaces:**
- Consumes: completed T1-T4 implementation.
- Produces: a clean worktree and evidence-backed completion report.

- [ ] **Step 1: Load and apply the project pre-commit checklist**

Read `.agents/skills/pre-commit-check/SKILL.md`, select its simulation,
Task Program, configured-task, docs, and project-context checks, and record the
selection before running broad commands.

- [ ] **Step 2: Run Black before any final commit**

```bash
/root/miniconda3/envs/py311/bin/python -m black .
```

Review `git diff --stat` and revert no user-owned changes. Commit only formatting
introduced in task-owned files if Black changed them.

- [ ] **Step 3: Run the complete affected CPU test surface**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/sim/motion/expansion \
  tests/sim/motion/test_execution.py \
  tests/sim/motion/test_generation_execution.py \
  tests/sim/motion/test_motion_imports.py \
  tests/sim/atomic_actions \
  tests/lab/task_program \
  tests/gym/envs/task_program \
  tests/gym/envs/test_embodied_env_task_program.py \
  tests/test_task_program_package_data.py
```

Record the exact pass/fail/skip counts and warnings.

- [ ] **Step 4: Run static deployment, documentation, and diff gates**

```bash
/root/miniconda3/envs/py311/bin/python \
  .agents/skills/add-task-program/scripts/inspect_deployment.py \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.ur5.yaml
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
git diff --check "$(git merge-base origin/main HEAD)"..HEAD
git status --short --branch
```

- [ ] **Step 5: Run a live showcase only when the simulator resource is available**

```bash
/root/miniconda3/envs/py311/bin/python \
  examples/sim/motion/repeated_pick_place_generation_showcase.py \
  --task-config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --generation-profile embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/generation.demo.yaml \
  --output-dir /tmp/repeated-pick-place-generation \
  --candidate-indices 0 1 2 \
  --device cuda --headless
```

If CUDA/simulator execution is unavailable, do not substitute a fake physical
claim; report this command as not run and rely only on the static/CPU evidence.

- [ ] **Step 6: Re-run the completion gates after the last commit**

Repeat Steps 2-4 against committed HEAD. Confirm no untracked task files remain
and no unrelated work was staged. Report the prerequisite hardening commit
range, T1-T4 commit range, validation evidence, and residual live-simulation
risk.
