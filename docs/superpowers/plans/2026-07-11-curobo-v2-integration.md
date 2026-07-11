# CuRobo V2 Motion-Planning Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NVIDIA cuRobo V2 as an optional collision-aware EmbodiChain planner, route it through `MotionGenerator` and supported atomic actions, verify it at unit/CUDA/DexSim levels, and ship a runnable Panda obstacle-avoidance demo.

**Architecture:** `CuroboPlanner(BasePlanner)` owns lazy V2 bindings and converts standard batched `PlanState` inputs into V2 `JointState` and `GoalToolPose` calls. `MotionGenerator` becomes capability-driven rather than special-casing `NeuralPlanner`; atomic actions keep their existing full-DoF output contract and select the CuRobo path only when configured. The backend returns collision-checked cuRobo samples without EmbodiChain IK pre-processing or unsafe post-resampling.

**Tech Stack:** Python 3.10+, PyTorch, EmbodiChain `@configclass`, DexSim, pytest, NVIDIA cuRobo V2 (tested against upstream `3fd54dc782a82e5500a771cfd47856ea499d5fef`), CUDA.

## Global Constraints

- Support cuRobo V2 only; never import, install, or fallback to V1 `curobo.wrap.reacher.motion_gen`.
- cuRobo is optional: `import embodichain.lab.sim.planners` must work without the `curobo` module; import V2 only inside a lazy helper used by `CuroboPlanner`.
- Core dependencies in `pyproject.toml` remain unchanged. Installation instructions use NVIDIA's matching `.[cu12]`, `.[cu13]`, `.[cu12-torch]`, or `.[cu13-torch]` extras.
- Use Apache 2.0 headers, `from __future__ import annotations`, full public type annotations, Google-style docstrings, `@configclass` for configuration classes, and `__all__` in every new public Python module.
- Preserve `PlanState` and `PlanResult` tensors: leading batch dimension `B`; positions/velocities/accelerations `(B, N, D)`; `dt` `(B, N)`; duration `(B,)`; success `(B,)`.
- The first release supports one configured control part per request. It rejects coordinated dual-arm CuRobo requests and does not add attachment/detachment collision modeling or ActionBank support.
- The scene boundary accepts only cuRobo-supported `cuboid`, `mesh`, and `voxel` collision geometry. The demo uses a cuboid. Create collision cache capacity before allowing world updates.
- Construct/cache a V2 `BatchMotionPlanner` with `max_batch_size == actual B`; do not reuse a larger graph-seeded batch planner for a smaller batch.
- `max_planning_time` is post-plan validation based on returned V2 timing; do not pass it to `MotionPlannerCfg.create`, which has no such argument.
- Do not linearly resample or interpolate a cuRobo collision-checked output after planning. Preserve the returned samples; action composition must accept their runtime length.

---

## File Structure

| File | Responsibility |
|---|---|
| `embodichain/lab/sim/planners/base_planner.py` | Add backend capabilities and generic motion-context hook. |
| `embodichain/lab/sim/planners/motion_generator.py` | Dispatch capability-aware interpolation and contextual options. |
| `embodichain/lab/sim/planners/neural_planner.py` | Migrate the existing Neural context behavior to the new hooks. |
| `embodichain/lab/sim/planners/toppra_planner.py` | Supply a correct default `ToppraPlanOptions`. |
| `embodichain/lab/sim/planners/curobo_planner.py` | Lazy V2 bindings, profile/world configs, named-joint conversion, scene updates, planning, and `PlanResult` conversion. |
| `embodichain/lab/sim/planners/__init__.py` | Re-export the CuRobo public API without importing cuRobo itself. |
| `embodichain/lab/sim/atomic_actions/core.py` | Validate `motion_source` / `planner_type` settings. |
| `embodichain/lab/sim/atomic_actions/trajectory.py` | Build CuRobo options, validate planner/result contracts, preserve collision-checked output, and add joint-motion dispatch. |
| `embodichain/lab/sim/atomic_actions/primitives/move_joints.py` | Opt in to collision-aware joint-space planning. |
| `embodichain/lab/sim/atomic_actions/primitives/{pick_up,place,press}.py` | Compose phase trajectories using actual returned lengths and avoid CuRobo pre-IK filtering. |
| `embodichain/lab/sim/atomic_actions/primitives/{coordinated_pickment,coordinated_placement}.py` | Fail explicitly if configured for the unsupported CuRobo coordinated path. |
| `embodichain/data/assets/curobo/collision_franka_demo.yml` | Static cuboid scene shared by the demo and optional integration test. |
| `tests/sim/planners/test_curobo_planner.py` | Dependency-free config, conversion, mapping, capability, and fake-backend tests. |
| `tests/sim/planners/test_curobo_integration.py` | CUDA/cuRobo V2 integration test, skipped when unavailable. |
| `tests/sim/atomic_actions/test_trajectory_motion_source.py` | CuRobo routing, no pre-IK, contract, and failure-hold tests. |
| `tests/sim/atomic_actions/test_actions.py` | `MoveJoints` CuRobo path and variable phase-length tests. |
| `tests/sim/atomic_actions/test_curobo_motion_source_e2e.py` | Optional DexSim endpoint test through `AtomicActionEngine`. |
| `examples/sim/planners/curobo_planner.py` | Runnable Panda + cuboid CuRobo V2 planner/action demo. |
| `docs/source/overview/sim/planners/{index.rst,curobo_planner.md,motion_generator.md}` | Public usage, installation, limitations, and planner overview. |
| `tests/docs/test_curobo_planner_docs.py` | Source-level documentation coverage check. |

### Task 1: Generalize the Planner-to-MotionGenerator Protocol

**Files:**
- Modify: `embodichain/lab/sim/planners/base_planner.py:31-195`
- Modify: `embodichain/lab/sim/planners/motion_generator.py:23-247`
- Modify: `embodichain/lab/sim/planners/neural_planner.py:194-350`
- Modify: `embodichain/lab/sim/planners/toppra_planner.py:237-399`
- Test: `tests/sim/planners/test_motion_generator_batched.py`
- Test: `tests/sim/planners/test_neural_planner.py`

**Interfaces:**
- Consumes: existing `MotionGenOptions(start_qpos, control_part, plan_opts, is_interpolate)` and `PlanOptions`.
- Produces: `BasePlanner.preinterpolate_targets`, `BasePlanner.preserve_plan_samples`, `BasePlanner.default_plan_options()`, and `BasePlanner.with_motion_context(...)`; later `CuroboPlanner` implements these hooks.

- [ ] **Step 1: Write failing capability-routing tests**

Add a small fake planner to `tests/sim/planners/test_motion_generator_batched.py` and assert the original Cartesian target reaches a capability-false planner unchanged:

```python
class _DirectCartesianPlanner:
    preinterpolate_targets = False
    preserve_plan_samples = True

    def default_plan_options(self):
        return PlanOptions()

    def with_motion_context(self, options, *, start_qpos, control_part):
        self.received = (start_qpos.clone(), control_part)
        return options

    def plan(self, target_states, options):
        self.target_states = target_states
        return PlanResult(
            success=torch.tensor([True]),
            positions=torch.zeros(1, 3, 2),
        )


def test_direct_cartesian_planner_skips_preinterpolation(monkeypatch):
    planner = _DirectCartesianPlanner()
    generator = object.__new__(MotionGenerator)
    generator.planner = planner
    generator.device = torch.device("cpu")
    start = torch.tensor([[0.1, -0.2]])
    goal = PlanState.from_xpos(torch.eye(4).unsqueeze(0))

    result = generator.generate(
        [goal],
        MotionGenOptions(
            start_qpos=start,
            control_part="arm",
            is_interpolate=True,
        ),
    )

    assert result.success.item()
    assert planner.target_states[0].move_type is MoveType.EEF_MOVE
    assert torch.equal(planner.target_states[0].xpos, goal.xpos)
    assert torch.equal(planner.received[0], start)
    assert planner.received[1] == "arm"
```

Add a regression test in `test_neural_planner.py` that calls `MotionGenerator.generate()` with `MotionGenOptions.start_qpos` and `control_part` and verifies that the neural rollout starts at that qpos.

- [ ] **Step 2: Run the new tests to verify the current implementation fails**

Run:

```bash
pytest tests/sim/planners/test_motion_generator_batched.py -q
pytest tests/sim/planners/test_neural_planner.py -q
```

Expected: the direct-planner test fails because `MotionGenerator` either performs interpolation or does not call the generic context hook.

- [ ] **Step 3: Add the protocol hooks and migrate current planners**

In `BasePlanner`, add the following complete default behavior directly below `__init__`:

```python
    preinterpolate_targets: bool = True
    """Whether MotionGenerator may pre-interpolate targets for this backend."""

    preserve_plan_samples: bool = False
    """Whether callers must retain planner-returned sample points exactly."""

    def default_plan_options(self) -> PlanOptions:
        """Return backend-default planning options."""
        return PlanOptions()

    def with_motion_context(
        self,
        options: PlanOptions,
        *,
        start_qpos: torch.Tensor | None,
        control_part: str | None,
    ) -> PlanOptions:
        """Attach MotionGenerator runtime context to backend options.

        The base planner has no context fields and therefore returns ``options``
        unchanged. Backends with contextual options override this method.
        """
        return options
```

Implement the Neural override without mutating a caller's object unexpectedly:

```python
class NeuralPlanner(BasePlanner):
    preinterpolate_targets = False

    def with_motion_context(
        self,
        options: PlanOptions,
        *,
        start_qpos: torch.Tensor | None,
        control_part: str | None,
    ) -> NeuralPlanOptions:
        if not isinstance(options, NeuralPlanOptions):
            raise TypeError("NeuralPlanner requires NeuralPlanOptions")
        if options.control_part is None:
            options.control_part = control_part
        if options.start_qpos is None:
            options.start_qpos = start_qpos
        return options
```

Add `ToppraPlanner.default_plan_options()` returning `ToppraPlanOptions()`. Remove the Neural-only propagation block from `MotionGenerator.generate()` and replace it with:

```python
        if options.is_interpolate and not self.planner.preinterpolate_targets:
            logger.log_warning(
                f"{type(self.planner).__name__} does not support MotionGenerator "
                "pre-interpolation; disabling it."
            )
            options.is_interpolate = False

        if options.plan_opts is None:
            options.plan_opts = self.planner.default_plan_options()
        options.plan_opts = self.planner.with_motion_context(
            options.plan_opts,
            start_qpos=options.start_qpos,
            control_part=options.control_part,
        )
```

Keep the existing `EEF_MOVE`/`JOINT_MOVE` interpolation path unchanged when `preinterpolate_targets` is true.

- [ ] **Step 4: Run focused tests and formatting**

Run:

```bash
black embodichain/lab/sim/planners/base_planner.py embodichain/lab/sim/planners/motion_generator.py embodichain/lab/sim/planners/neural_planner.py embodichain/lab/sim/planners/toppra_planner.py tests/sim/planners/test_motion_generator_batched.py tests/sim/planners/test_neural_planner.py
pytest tests/sim/planners/test_motion_generator_batched.py tests/sim/planners/test_neural_planner.py -q
```

Expected: PASS, with existing TOPPRA and Neural tests retaining their behavior.

- [ ] **Step 5: Commit the protocol change**

```bash
git add embodichain/lab/sim/planners/base_planner.py embodichain/lab/sim/planners/motion_generator.py embodichain/lab/sim/planners/neural_planner.py embodichain/lab/sim/planners/toppra_planner.py tests/sim/planners/test_motion_generator_batched.py tests/sim/planners/test_neural_planner.py
git commit -m "refactor(planner): add backend capability hooks"
```

### Task 2: Create the Optional CuRobo Configuration and Conversion Layer

**Files:**
- Create: `embodichain/lab/sim/planners/curobo_planner.py`
- Modify: `embodichain/lab/sim/planners/__init__.py:17-21`
- Test: `tests/sim/planners/test_curobo_planner.py`

**Interfaces:**
- Consumes: `BasePlannerCfg`, `PlanOptions`, `PlanState`, `PlanResult`, `configclass`, `quat_from_matrix`, and `SimulationManager` through `BasePlanner`.
- Produces: `CuroboRobotProfileCfg`, `CuroboWorldCfg`, `CuroboPlannerCfg`, `CuroboPlanOptions`, `CuroboPlanner`, plus dependency-free `_reorder_by_names`, `_matrix_to_position_quaternion`, and `_validate_dynamic_obstacles` helpers.

- [ ] **Step 1: Write failing pure tests before adding the module**

Create `tests/sim/planners/test_curobo_planner.py` with the project header and a fake `SimulationManager`. Add these representative cases:

```python
def test_reorder_by_names_preserves_batch_and_time_dimensions():
    values = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    result = _reorder_by_names(values, ["joint_b", "joint_a"], ["joint_a", "joint_b"])
    assert torch.equal(result, torch.tensor([[[20.0, 10.0], [40.0, 30.0]]]))


def test_matrix_to_position_quaternion_uses_wxyz():
    matrix = torch.eye(4).unsqueeze(0)
    position, quaternion = _matrix_to_position_quaternion(matrix)
    assert torch.equal(position, torch.zeros(1, 3))
    assert torch.equal(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))


def test_missing_curobo_is_actionable(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _raise_module_not_found)
    with pytest.raises(ImportError, match=r"cu12.*cu13"):
        _require_curobo()


def test_unknown_dynamic_obstacle_is_rejected():
    with pytest.raises(ValueError, match="unknown obstacle"):
        _validate_dynamic_obstacles({"unknown": torch.eye(4)}, ["known"])
```

Also assert that `from embodichain.lab.sim.planners import CuroboPlannerCfg` succeeds without the installed `curobo` package.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest tests/sim/planners/test_curobo_planner.py -q
```

Expected: collection fails because `curobo_planner.py` and its exported types do not exist.

- [ ] **Step 3: Implement lazy configs and pure helpers**

Create `curobo_planner.py` with the project header, future import, `TYPE_CHECKING` guard, and this public surface:

```python
__all__ = [
    "CuroboPlanOptions",
    "CuroboPlanner",
    "CuroboPlannerCfg",
    "CuroboRobotProfileCfg",
    "CuroboWorldCfg",
]


@configclass
class CuroboRobotProfileCfg:
    robot_config_path: str = MISSING
    sim_to_curobo_joint_names: dict[str, str] = MISSING
    active_joint_names: list[str] | None = None
    fixed_joint_positions: dict[str, float] = {}
    base_link_name: str | None = None
    tool_frame_name: str | None = None


@configclass
class CuroboWorldCfg:
    world_config_path: str | None = None
    collision_cache: dict[str, int] = {"cuboid": 8, "mesh": 2, "voxel": 1}
    dynamic_obstacle_names: list[str] = []
    multi_env: bool = False


@configclass
class CuroboPlannerCfg(BasePlannerCfg):
    planner_type: str = "curobo"
    robot_profiles: dict[str, CuroboRobotProfileCfg] = MISSING
    world: CuroboWorldCfg = CuroboWorldCfg()
    warmup: bool = True
    collision_activation_distance: float = 0.01
    max_attempts: int = 5
    max_planning_time: float | None = None
    use_cuda_graph: bool = True
    interpolation_dt: float = 0.025


@configclass
class CuroboPlanOptions(PlanOptions):
    start_qpos: torch.Tensor | None = None
    control_part: str | None = None
    dynamic_obstacle_poses: dict[str, torch.Tensor] | None = None
    max_attempts: int | None = None
```

Implement `_require_curobo()` with `importlib.import_module` calls for the V2 public facades only. It must return an internal binding object containing `MotionPlanner`, `MotionPlannerCfg`, `BatchMotionPlanner`, `JointState`, `Pose`, and `GoalToolPose`; its `ImportError` must state both supported extras and the NVIDIA URL. Implement named reordering by validating equal unique name sets, not positional guesses. Implement matrix conversion with `embodichain.utils.math.quat_from_matrix`, whose output is wxyz, and reject non-`(B, 4, 4)` tensors.

Append this lazy-safe export to `planners/__init__.py`:

```python
from .curobo_planner import *
```

No statement at module scope may import `curobo`.

- [ ] **Step 4: Pass the pure tests and check absence behavior**

Run:

```bash
black embodichain/lab/sim/planners/curobo_planner.py embodichain/lab/sim/planners/__init__.py tests/sim/planners/test_curobo_planner.py
python -c "from embodichain.lab.sim.planners import CuroboPlannerCfg; print(CuroboPlannerCfg.__name__)"
pytest tests/sim/planners/test_curobo_planner.py -q
```

Expected: imports print `CuroboPlannerCfg`; pure tests pass without cuRobo installed.

- [ ] **Step 5: Commit the optional surface**

```bash
git add embodichain/lab/sim/planners/curobo_planner.py embodichain/lab/sim/planners/__init__.py tests/sim/planners/test_curobo_planner.py
git commit -m "feat(planner): add optional curobo configuration"
```

### Task 3: Implement CuRobo V2 Backend Planning and World Synchronization

**Files:**
- Modify: `embodichain/lab/sim/planners/curobo_planner.py`
- Modify: `tests/sim/planners/test_curobo_planner.py`
- Create: `tests/sim/planners/test_curobo_integration.py`

**Interfaces:**
- Consumes: Task 1 planner hooks and Task 2 configs/helpers.
- Produces: `CuroboPlanner.plan(target_states, options) -> PlanResult`, `CuroboPlanner.update_dynamic_obstacles(poses)`, `CuroboPlanner.close()`, and a V2 backend cache keyed by `(control_part, batch_size, multi_env)`.

- [ ] **Step 1: Write fake-binding tests for pose, c-space, output mapping, and failures**

Extend `test_curobo_planner.py` with injected fake V2 bindings. Its fake result must model the high-level V2 shapes `[B, 1, T, D]` and named full-joint output:

```python
def test_plan_pose_maps_curobo_full_output_to_control_part(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim)
    result = planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.tensor([[0.2, -0.1]]), control_part="arm"),
    )
    assert result.success.tolist() == [True]
    assert result.positions.shape == (1, 3, 2)
    assert torch.equal(result.positions[0, -1], torch.tensor([2.0, 1.0]))
    assert result.dt.shape == (1, 3)
    assert result.duration.shape == (1,)


def test_failed_v2_result_holds_start_qpos(fake_curobo, fake_sim):
    fake_curobo.next_result.success = torch.tensor([[False]])
    planner = _make_planner(fake_curobo, fake_sim)
    start = torch.tensor([[0.3, -0.4]])
    result = planner.plan([PlanState.from_qpos(start)], CuroboPlanOptions(start_qpos=start, control_part="arm"))
    assert result.success.tolist() == [False]
    assert torch.equal(result.positions, start.unsqueeze(1))
```

Add a two-waypoint test that verifies the planner invokes V2 sequentially, begins the second segment at the first segment's final output, and concatenates the safe samples without a linear resample. Add tests for malformed V2 output names, unknown profile, unavailable CUDA, and a returned `total_time` greater than `max_planning_time` marking the affected result unsuccessful.

- [ ] **Step 2: Run the fake-binding tests to verify they fail**

Run:

```bash
pytest tests/sim/planners/test_curobo_planner.py -q
```

Expected: fake V2 planning tests fail because `CuroboPlanner.plan()` is not implemented.

- [ ] **Step 3: Implement backend construction and standard result conversion**

Implement the following behavior in `CuroboPlanner`:

```python
class CuroboPlanner(BasePlanner):
    preinterpolate_targets = False
    preserve_plan_samples = True

    def default_plan_options(self) -> CuroboPlanOptions:
        return CuroboPlanOptions()

    def with_motion_context(self, options, *, start_qpos, control_part):
        if not isinstance(options, CuroboPlanOptions):
            raise TypeError("CuroboPlanner requires CuroboPlanOptions")
        if options.start_qpos is None:
            options.start_qpos = start_qpos
        if options.control_part is None:
            options.control_part = control_part
        return options

    @validate_plan_options(options_cls=CuroboPlanOptions)
    def plan(self, target_states, options=CuroboPlanOptions()) -> PlanResult:
        control_part, profile = self._resolve_profile(options)
        start = self._resolve_start_qpos(options.start_qpos, control_part)
        backend = self._get_backend(profile, batch_size=start.shape[0])
        self.update_dynamic_obstacles(options.dynamic_obstacle_poses, backend)
        return self._plan_segments(target_states, start, control_part, backend, options)
```

`_get_backend` must reject a non-CUDA robot device; invoke V2 `MotionPlannerCfg.create` with the profile path, `scene_model=world_config_path`, non-`None` `collision_cache`, `max_batch_size=batch_size`, `multi_env=cfg.world.multi_env`, `optimizer_collision_activation_distance`, `use_cuda_graph`, and `interpolation_dt`. Instantiate `MotionPlanner` for `B == 1` and `BatchMotionPlanner` for `B > 1`; call V2 warmup once per cache entry when `cfg.warmup` is true. Cache only entries matching the actual batch size. When `profile.active_joint_names` is set, compare it with `backend.planner.joint_names` and raise on any missing, duplicate, or differently ordered name.

For each segment, construct V2 states and goals as follows:

```python
active_start = _to_curobo_active_joint_state(start, profile, backend.planner.joint_names)
current_state = bindings.JointState.from_position(
    active_start,
    joint_names=backend.planner.joint_names,
)
position, quaternion = _matrix_to_position_quaternion(target.xpos)
goal = bindings.GoalToolPose.from_poses(
    {backend.tool_frame: bindings.Pose(position=position, quaternion=quaternion)},
    ordered_tool_frames=[backend.tool_frame],
    num_goalset=1,
)
v2_result = backend.planner.plan_pose(goal, current_state, max_attempts=max_attempts)
```

Use `plan_cspace(goal_state, current_state, ...)` for `JOINT_MOVE`. Map inputs by configured simulator-to-CuRobo joint names. V2 non-controlled joints must be locked in the robot profile or expanded/reduced through `backend.planner.kinematics.get_full_js` and `get_active_js`; never pass a nonexistent per-call fixed-joint argument.

Extract `interpolated_trajectory` structurally, select seed zero from `[B, 1, T, D]`, trim each valid row using `interpolated_last_tstep`, map by `trajectory.joint_names` back to simulator control order, then concatenate sequential segment samples. To preserve a rectangular `PlanResult`, pad each batch row by repeating its own last valid qpos; set failed rows to `start_qpos.unsqueeze(1)`. Derive `dt` from V2 trajectory `dt` when it has time samples and otherwise from `cfg.interpolation_dt`; calculate `duration = dt.sum(dim=1)`. Do not expose private V2 `TrajOptSolverResult` in public annotations.

`update_dynamic_obstacles` must validate the configured names and `(B, 4, 4)` shapes, convert to V2 `Pose` wxyz, and call `backend.planner.scene_collision_checker.update_obstacle_pose(name, pose, env_idx=index)` for each environment. `close()` must destroy every cached planner exactly once; `__del__` calls `close()` defensively.

- [ ] **Step 4: Add and run the optional real V2 integration test**

Create `tests/sim/planners/test_curobo_integration.py` with module-level guards:

```python
curobo = pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)


@pytest.mark.slow
def test_curobo_v2_plans_around_a_static_cuboid(...):
    ...
```

The test must build a single Panda-compatible profile, call `MotionGenerator.generate()` for an EEF target around the cuboid, and assert `(B,)` success, finite `PlanResult` fields, correct first qpos, final FK position tolerance, output joint ordering, positive duration, and a reachable dynamic-obstacle update. Run:

```bash
pytest tests/sim/planners/test_curobo_planner.py -q
pytest tests/sim/planners/test_curobo_integration.py -q
```

Expected: pure tests PASS; integration test is SKIPPED without V2/CUDA and PASS after the official V2 install.

- [ ] **Step 5: Commit backend implementation and tests**

```bash
git add embodichain/lab/sim/planners/curobo_planner.py tests/sim/planners/test_curobo_planner.py tests/sim/planners/test_curobo_integration.py
git commit -m "feat(planner): integrate curobo v2 backend"
```

### Task 4: Route Atomic Actions Through the Collision-Aware Backend

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/core.py:294-315`
- Modify: `embodichain/lab/sim/atomic_actions/trajectory.py:293-475`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/move_joints.py:78-133`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/{pick_up,place,press}.py`
- Modify: `embodichain/lab/sim/atomic_actions/primitives/{coordinated_pickment,coordinated_placement}.py`
- Test: `tests/sim/atomic_actions/test_trajectory_motion_source.py`
- Test: `tests/sim/atomic_actions/test_actions.py`

**Interfaces:**
- Consumes: `CuroboPlanOptions`, Task 1 capabilities, `ActionCfg`, and the existing `(success, controlled_trajectory)` builder contract.
- Produces: safe single-arm CuRobo atomic routing; `MoveJoints` motion-generator path; validation that prevents mislabeled planner use or a silent IK fallback.

- [ ] **Step 1: Write failing atomic routing tests**

Extend `test_trajectory_motion_source.py` with a fake CuRobo generator whose planner has `cfg.planner_type == "curobo"`, `preinterpolate_targets is False`, and `preserve_plan_samples is True`. Test all of these contracts:

```python
def test_curobo_builder_preserves_cartesian_targets_and_samples():
    mg = _mock_curobo_motion_generator(result_positions=torch.zeros(2, 7, 6))
    builder = TrajectoryBuilder(mg)
    success, trajectory = builder.plan_arm_traj(
        _pose_targets_for_two_envs(),
        torch.zeros(2, 6),
        n_waypoints=20,
        control_part="arm",
        arm_dof=6,
        cfg=ActionCfg(motion_source="motion_gen", planner_type="curobo", control_part="arm"),
    )
    assert success.tolist() == [True, True]
    assert trajectory.shape == (2, 7, 6)
    assert mg.generate.call_args.kwargs["options"].is_interpolate is False
    assert mg.generate.call_args.args[0][0].move_type is MoveType.EEF_MOVE
```

Add tests that mismatched action/generator planner types, an invalid `motion_source`, malformed/NaN positions, and `None` positions raise `ValueError`. Add a failure-row test proving an unsuccessful row is held at its start qpos.

In `test_actions.py`, add `MoveJointsCfg(motion_source="motion_gen", planner_type="curobo")` cases for a one-waypoint and multi-waypoint target. Assert ordered `JOINT_MOVE` PlanStates, full-DoF preservation of hand joints, returned CuRobo length, and per-env failure hold.

- [ ] **Step 2: Run the atomic tests to verify they fail**

Run:

```bash
pytest tests/sim/atomic_actions/test_trajectory_motion_source.py tests/sim/atomic_actions/test_actions.py -q
```

Expected: CuRobo cases fail because the builder treats every non-Neural planner as pre-interpolated and `MoveJoints` always linearly interpolates.

- [ ] **Step 3: Implement strict action and builder dispatch**

Add `ActionCfg.__post_init__` validation:

```python
    def __post_init__(self) -> None:
        valid_sources = {"ik_interp", "motion_gen"}
        if self.motion_source not in valid_sources:
            raise ValueError(f"motion_source must be one of {sorted(valid_sources)}")
        if self.motion_source == "motion_gen" and self.planner_type is None:
            raise ValueError("planner_type is required when motion_source='motion_gen'")
        if self.motion_source == "ik_interp" and self.planner_type is not None:
            raise ValueError("planner_type is only valid with motion_source='motion_gen'")
```

At the top of `TrajectoryBuilder._plan_motion_gen`, verify:

```python
actual_type = self.motion_generator.planner.cfg.planner_type
requested_type = getattr(cfg, "planner_type", None)
if requested_type != actual_type:
    raise ValueError(
        f"Action requested planner_type={requested_type!r}, "
        f"but MotionGenerator owns {actual_type!r}."
    )
```

Replace the `planner_type != "neural"` check with `not self.motion_generator.planner.preinterpolate_targets`. Replace `_build_plan_opts` with an explicit three-way factory: `ToppraPlanOptions` for TOPPRA, `NeuralPlanOptions` for Neural, and `CuroboPlanOptions(max_attempts=...)` for CuRobo. Reject unknown registered types rather than returning Neural options.

Validate every motion-generator `PlanResult` before using it:

```python
if positions is None or positions.ndim != 3:
    raise ValueError("MotionGenerator returned no (B, N, controlled_dof) positions")
if positions.shape[0] != n_envs or positions.shape[2] != arm_dof:
    raise ValueError("MotionGenerator returned incompatible trajectory shape")
if positions.device != self.device or not torch.isfinite(positions).all():
    raise ValueError("MotionGenerator returned non-finite or wrong-device positions")
```

If `planner.preserve_plan_samples` is true, return its trajectory unresampled; otherwise keep the existing `interpolate_with_distance` normalization. Add `TrajectoryBuilder.plan_joint_motion(...) -> tuple[torch.Tensor, torch.Tensor]`, which constructs batched `JOINT_MOVE` states and delegates to the same motion-generator validation path for `motion_gen`, or returns all-success linear interpolation for `ik_interp`.

Modify `MoveJoints.execute` to call `plan_joint_motion` rather than directly calling `plan_joint_traj`:

```python
        success, joint_traj = self.builder.plan_joint_motion(
            start_qpos,
            target_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.joint_dof,
            cfg=self.cfg,
        )
```

Keep `_embed` and all non-controlled full qpos untouched.

For `PickUp`, `Place`, and `Press`, allocate output phase tensors from `approach_arm.shape[1]`, `down_arm.shape[1]`, `back_arm.shape[1]`, and `lift_arm.shape[1]`, not their requested sample counts. Use `plan_joint_motion` for the `Press` return phase. When the configured planner is CuRobo, `PickUp._resolve_grasp_pose` selects the lowest-cost affordance variant without calling its existing EmbodiChain IK prefilter; CuRobo itself validates reachability/collision during motion planning. Raise immediately from coordinated primitive construction when the configuration requests `planner_type="curobo"`.

- [ ] **Step 4: Run atomic unit tests and regression tests**

Run:

```bash
black embodichain/lab/sim/atomic_actions/core.py embodichain/lab/sim/atomic_actions/trajectory.py embodichain/lab/sim/atomic_actions/primitives/move_joints.py embodichain/lab/sim/atomic_actions/primitives/pick_up.py embodichain/lab/sim/atomic_actions/primitives/place.py embodichain/lab/sim/atomic_actions/primitives/press.py embodichain/lab/sim/atomic_actions/primitives/coordinated_pickment.py embodichain/lab/sim/atomic_actions/primitives/coordinated_placement.py tests/sim/atomic_actions/test_trajectory_motion_source.py tests/sim/atomic_actions/test_actions.py
pytest tests/sim/atomic_actions/test_trajectory_motion_source.py tests/sim/atomic_actions/test_actions.py tests/sim/atomic_actions/test_action_result_success.py -q
```

Expected: PASS; default `ik_interp` tests remain unchanged and CuRobo paths never invoke robot IK pre-interpolation.

- [ ] **Step 5: Commit atomic integration**

```bash
git add embodichain/lab/sim/atomic_actions/core.py embodichain/lab/sim/atomic_actions/trajectory.py embodichain/lab/sim/atomic_actions/primitives tests/sim/atomic_actions/test_trajectory_motion_source.py tests/sim/atomic_actions/test_actions.py
git commit -m "feat(actions): route atomic motions through curobo"
```

### Task 5: Add the Static-World Asset, Optional DexSim E2E Test, and Runnable Demo

**Files:**
- Create: `embodichain/data/assets/curobo/collision_franka_demo.yml`
- Create: `tests/sim/atomic_actions/test_curobo_motion_source_e2e.py`
- Create: `examples/sim/planners/curobo_planner.py`

**Interfaces:**
- Consumes: `FrankaPandaCfg`, `CuroboPlannerCfg`, `CuroboRobotProfileCfg`, `CuroboWorldCfg`, `MotionGenerator`, `AtomicActionEngine`, `MoveEndEffector`, and the Task 3 V2 backend.
- Produces: an executable one-environment cuboid-avoidance proof that exercises planner → motion generator → atomic action → DexSim playback.

- [ ] **Step 1: Create the shared cuboid world asset and failing E2E test**

Add `embodichain/data/assets/curobo/collision_franka_demo.yml`:

```yaml
cuboid:
  - name: demo_block
    dims: [0.18, 0.40, 0.36]
    pose: [0.45, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0]
```

Create the e2e test with guards before any cuRobo-only imports:

```python
curobo = pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)


@pytest.mark.requires_sim
@pytest.mark.slow
def test_atomic_move_end_effector_uses_curobo_v2():
    sim, robot, engine = _make_franka_curobo_engine()
    try:
        target = _reachable_target_beyond_demo_block(robot)
        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        assert success.shape == (1,)
        assert success.item()
        assert trajectory.shape[2] == robot.dof
        _play_trajectory(sim, robot, trajectory)
        assert _position_error(robot, target) < 0.02
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
```

- [ ] **Step 2: Run the E2E test to verify it fails before the fixture/demo exists**

Run:

```bash
pytest tests/sim/atomic_actions/test_curobo_motion_source_e2e.py -q
```

Expected: SKIPPED without cuRobo; after installation it initially fails because the shared engine fixture has not been written.

- [ ] **Step 3: Implement the Panda profile fixture and CLI demo**

Use `FrankaPandaCfg.from_dict({"uid": "curobo_franka"})`, not the legacy differently-named robot helper in `neural_planner.py`. Define the profile mapping explicitly so no index order is assumed:

```python
FRANKA_SIM_TO_CUROBO = {
    "fr3_joint1": "panda_joint1",
    "fr3_joint2": "panda_joint2",
    "fr3_joint3": "panda_joint3",
    "fr3_joint4": "panda_joint4",
    "fr3_joint5": "panda_joint5",
    "fr3_joint6": "panda_joint6",
    "fr3_joint7": "panda_joint7",
}

profile = CuroboRobotProfileCfg(
    robot_config_path="franka.yml",
    sim_to_curobo_joint_names=FRANKA_SIM_TO_CUROBO,
    fixed_joint_positions={"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04},
    base_link_name="panda_link0",
    tool_frame_name="panda_hand",
)
```

In both fixture and example, instantiate a DexSim `CubeCfg(size=[0.18, 0.40, 0.36])` at `[0.45, 0.0, 0.18]` with the same UID-independent geometry as the YAML. Construct `MotionGenerator(MotionGenCfg(planner_cfg=CuroboPlannerCfg(...)))`, register `MoveEndEffector` configured with `motion_source="motion_gen"`, `planner_type="curobo"`, and `control_part="arm"`, then issue a target that requires passing the cuboid rather than entering it.

The demo CLI must accept `--headless`, `--step-repeat`, `--hold-steps`, and `--no-warmup`; it must default to CUDA and raise a clear error when CUDA/cuRobo is absent. It prints success, trajectory shape, duration, and final Cartesian position error before replaying each full-DoF sample with `robot.set_qpos(qpos=trajectory[:, waypoint])` and `sim.update(step=step_repeat)`.

- [ ] **Step 4: Run the optional real verification and example**

After following NVIDIA's V2 installation command matching the local CUDA/PyTorch combination, run:

```bash
pytest tests/sim/planners/test_curobo_integration.py tests/sim/atomic_actions/test_curobo_motion_source_e2e.py -q
python examples/sim/planners/curobo_planner.py --headless --hold-steps 1 --step-repeat 1
```

Expected: both tests PASS and the example prints a successful `(1, N, robot.dof)` trajectory with final position error below `0.02` m.

- [ ] **Step 5: Commit the runnable proof**

```bash
git add embodichain/data/assets/curobo/collision_franka_demo.yml tests/sim/atomic_actions/test_curobo_motion_source_e2e.py examples/sim/planners/curobo_planner.py
git commit -m "feat(example): add curobo planner simulation demo"
```

### Task 6: Document Installation, Profiles, and Supported Limits

**Files:**
- Create: `docs/source/overview/sim/planners/curobo_planner.md`
- Modify: `docs/source/overview/sim/planners/index.rst`
- Modify: `docs/source/overview/sim/planners/motion_generator.md`
- Create: `tests/docs/test_curobo_planner_docs.py`

**Interfaces:**
- Consumes: final public configs, demo command, and the V2 limitations from the approved design.
- Produces: discoverable end-user instructions that do not imply core dependency installation or unsupported attachment/dual-arm behavior.

- [ ] **Step 1: Write the documentation assertions as a source-level test**

Create `tests/docs/test_curobo_planner_docs.py` with the project header and a source-level check that the planner index includes `curobo_planner` and that the page contains all required literal API names:

```python
def test_curobo_planner_docs_are_linked_and_scoped():
    index = Path("docs/source/overview/sim/planners/index.rst").read_text()
    page = Path("docs/source/overview/sim/planners/curobo_planner.md").read_text()
    assert "curobo_planner.md" in index
    assert "CuroboPlannerCfg" in page
    assert 'planner_type="curobo"' in page
    assert "cuRobo V2" in page
    assert "attached-object" in page
```

- [ ] **Step 2: Run the source-level test to verify it fails**

Run:

```bash
pytest tests/docs -q -k curobo
```

Expected: FAIL because the page and index entry do not exist.

- [ ] **Step 3: Add concise user-facing documentation**

Document all of the following in `curobo_planner.md`:

```markdown
## Install cuRobo V2

Install cuRobo using NVIDIA's CUDA-matched extras, then verify it with
`python -c "import curobo; print(curobo.__version__)"`. EmbodiChain does not
install cuRobo as a core dependency.

## Configure a control part

Use `CuroboRobotProfileCfg.sim_to_curobo_joint_names` to map simulator names to
the V2 robot profile. Generate new collision-sphere/self-collision profiles
with V2 `RobotBuilder`; do not use a plain URDF alone.

## Supported scope

Single-arm `MoveEndEffector` and opt-in `MoveJoints` are supported. Static
cuboid, mesh, and voxel worlds are supported. Attached objects, automatic
scene extraction, coordinated dual-arm planning, ActionBank, and CPU execution
are not supported by this release.
```

Update the index to include the page and revise `motion_generator.md` so it describes cuRobo as an available collision-aware backend rather than a future capability. Include the exact demo command and a link to NVIDIA's official installation documentation.

- [ ] **Step 4: Build docs and run all focused checks**

Run:

```bash
pytest tests/docs -q
cd docs && make html
```

Expected: PASS and `docs/build/html/index.html` contains the CuRobo planner page.

- [ ] **Step 5: Commit documentation**

```bash
git add docs/source/overview/sim/planners/curobo_planner.md docs/source/overview/sim/planners/index.rst docs/source/overview/sim/planners/motion_generator.md tests/docs/test_curobo_planner_docs.py
git commit -m "docs(planner): document curobo v2 integration"
```

### Task 7: Run the Whole-Change Quality Gate

**Files:**
- Verify: every file created or modified by Tasks 1-6.

**Interfaces:**
- Consumes: the complete implementation and existing planner/atomic regression suites.
- Produces: evidence that optional dependency behavior, default planner behavior, code style, and docs all remain valid.

- [ ] **Step 1: Run static and targeted regression checks**

Run:

```bash
black --check --diff --color ./
pytest tests/sim/planners/test_plan_state_batched.py tests/sim/planners/test_motion_generator_batched.py tests/sim/planners/test_neural_planner.py tests/sim/planners/test_curobo_planner.py -q
pytest tests/sim/atomic_actions/test_action_result_success.py tests/sim/atomic_actions/test_trajectory_motion_source.py tests/sim/atomic_actions/test_actions.py -q
python -c "import embodichain.lab.sim.planners"
```

Expected: PASS without cuRobo installed; V2-only modules remain lazily guarded.

- [ ] **Step 2: Run V2/CUDA checks when the dependency is available**

Run:

```bash
python -c "import curobo; print(curobo.__version__)"
pytest tests/sim/planners/test_curobo_integration.py tests/sim/atomic_actions/test_curobo_motion_source_e2e.py -q
python examples/sim/planners/curobo_planner.py --headless --hold-steps 1 --step-repeat 1
```

Expected: PASS and the demo prints a successful full-DoF trajectory. If cuRobo is intentionally absent, record the two tests as skipped and do not claim the CUDA path passed.

- [ ] **Step 3: Run the project pre-commit skill and inspect the final diff**

Run the `/pre-commit-check` workflow, then:

```bash
git diff --check
git status --short
git log --oneline --max-count=8
```

Expected: no whitespace errors; only scoped CuRobo implementation, tests, example, docs, and this approved plan are present.
