# Isaac-Lab-Style Action Manager Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the shared-`EnvAction` preprocessing pipeline with an ordered flat ActionManager whose terms process and apply disjoint robot resources directly.

**Architecture:** `ActionManager` owns a flat policy tensor, slices it by configured term order, and calls each term's `process_actions()` and `apply_actions()` lifecycle. Concrete terms resolve and own selected joints, publish typed descriptors, and apply only those joints; `ControllerAction` remains a separate expert/controller-ready path. The change is intentionally breaking and migrates every official task without aliases or legacy adapters.

**Tech Stack:** Python 3.11+, PyTorch, TensorDict, Gymnasium spaces, EmbodiChain `@configclass`, pytest, Black 26.3.1.

**Spec:** `docs/superpowers/specs/2026-09-25-action-manager-isaaclab-refactor-design.md`

## Global Constraints

- Do not commit, stage, push, or publish until the user explicitly requests it.
- Preserve the existing uncommitted dataset-contract fixes.
- Remove old action class names, `input_key`, `process_action()`, `ActionTermCfg.mode`, post terms, Dict policy actions, and combined arm/gripper actions without aliases.
- Policy actions are one ordered flat floating tensor on the environment device.
- `ControllerAction` bypasses ActionManager and retains the validated qpos/qvel/qf path.
- Terms apply only resolved non-mimic resource IDs; no term writes stale mimic followers.
- The control hot path must not convert CUDA scalars to Python values or transfer them to the host.
- Dataset policy schemas come from manager descriptors; expert schemas remain owned by `ExpertActionSpec`.
- Run Black 26.3.1 and proportional tests after each task; run the full suite at the final gate.

## Review Focus

- Wrong policy action rank, batch, or width must fail before term state changes; Task 2.
- Non-contiguous selections and mimic followers must preserve policy order and write independent joints only; Tasks 3-4.
- A configured ActionManager must not intercept `ControllerAction`; Task 5.
- Overlapping term ownership must fail with both term and joint names; Task 2.
- Sync, async, and fragment recorders must preserve identical flat actions and descriptor order; Task 7.

---

## File Structure

**Create:**

- `embodichain/lab/gym/envs/managers/action_types.py` — immutable action descriptors.
- `tests/gym/envs/managers/test_action_types.py` — descriptor validation.
- `tests/gym/envs/managers/action_test_utils.py` — shared fake robot/environment/action builders used by Tasks 1-4.

**Core modifications:**

- `embodichain/lab/gym/envs/managers/action_manager.py`
- `embodichain/lab/gym/envs/managers/actions.py`
- `embodichain/lab/gym/envs/managers/cfg.py`
- `embodichain/lab/gym/envs/managers/__init__.py`
- `embodichain/lab/gym/envs/embodied_env.py`
- `embodichain/lab/gym/envs/types.py`
- `embodichain/lab/gym/utils/gym_utils.py`
- `embodichain/lab/gym/envs/managers/{datasets,async_datasets,dataset_manager}.py`

**Task/config modifications:**

- CartPole `env.yaml`.
- Default/Newton YAML for ANYmal C, G1, Go1, Go2, H1_2, and MicroDuck.
- repeated-pick-place `env.rlinf.yaml`, `env.rlinf_joint.yaml`, and README.
- `embodichain_tasks/embodichain_tasks/locomotion/velocity/_embodichain.py`.

**Documentation/context modifications:**

- `docs/source/api_reference/public_api.rst`
- manager-functor, env-framework, data-pipeline, and robot-system context owners.

---

### Task 1: Add Action Descriptor Types and the New ActionTerm Protocol

**Files:**
- Create: `embodichain/lab/gym/envs/managers/action_types.py`
- Create: `tests/gym/envs/managers/test_action_types.py`
- Create: `tests/gym/envs/managers/action_test_utils.py`
- Modify: `embodichain/lab/gym/envs/managers/action_manager.py`
- Modify: `embodichain/lab/gym/envs/managers/cfg.py`

**Interfaces:**
- Produces `ActionTermDescriptor`, `ActionDescriptor`, and the new abstract `ActionTerm` properties/methods.

- [ ] **Step 0: Create shared action test builders**

Create `action_test_utils.py` with these concrete helpers:

```python
class FakeRobot:
    def __init__(self, num_envs: int, joint_names: tuple[str, ...], joint_ids_by_part: dict[str, tuple[int, ...]]):
        self.joint_names = list(joint_names)
        self._joint_ids_by_part = joint_ids_by_part
        self.body_data = SimpleNamespace(
            qpos_limits=torch.tensor([[[-1.0, 1.0]] * len(joint_names)]),
            qvel_limits=torch.ones(1, len(joint_names)),
            qf_limits=torch.ones(1, len(joint_names)),
        )
        self.set_qpos = MagicMock()
        self.set_qvel = MagicMock()
        self.set_qf = MagicMock()
        self.get_qpos = MagicMock(return_value=torch.zeros(num_envs, len(joint_names)))
        self.compute_ik = MagicMock()
        self.cfg = SimpleNamespace(init_qpos=[0.0] * len(joint_names))

    def get_joint_ids(self, name: str, remove_mimic: bool = False) -> list[int]:
        return list(self._joint_ids_by_part[name])


def make_action_env(
    *,
    num_envs: int = 2,
    joint_names: tuple[str, ...] = ("joint_0", "joint_1", "joint_2"),
    parts: dict[str, tuple[int, ...]] | None = None,
) -> SimpleNamespace:
    parts = {"arm": tuple(range(len(joint_names)))} if parts is None else parts
    return SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        robot=FakeRobot(num_envs, joint_names, parts),
        active_joint_ids=list(range(len(joint_names))),
    )


def make_cfg(func: type[ActionTerm], **params: object) -> ActionTermCfg:
    return ActionTermCfg(func=func, params=dict(params))
```

After Task 1 defines the new abstract protocol, add `FakeAction` to the same
file. It implements every abstract property, copies raw actions to processed
actions, increments `apply_count`, and accepts constructor params
`action_dim`, `low`, `high`, `joint_ids`, `joint_names`, and `command_type`.
Add `make_manager(*named_param_dicts, num_envs=...)` that builds an ordered
`ComponentCfg` of `ActionTermCfg(func=FakeAction, params=...)` and constructs
the real `ActionManager` with `make_action_env()`.

- [ ] **Step 1: Write failing descriptor tests**

```python
def test_action_term_descriptor_rejects_mismatched_names() -> None:
    with pytest.raises(ValueError, match="feature_names, units, and action_dim"):
        ActionTermDescriptor(
            representation="joint_position",
            action_dim=2,
            feature_names=("joint_a",),
            units=("rad", "rad"),
            normalization=None,
            joint_names=("joint_a", "joint_b"),
            metadata={},
        )


def test_action_descriptor_serializes_bound_slice() -> None:
    term = ActionTermDescriptor(
        representation="eef_pose",
        action_dim=6,
        feature_names=("x", "y", "z", "roll", "pitch", "yaw"),
        units=("m", "m", "m", "rad", "rad", "rad"),
        normalization=None,
        joint_names=("joint_1", "joint_2"),
        metadata={"frame": "arena"},
    )
    descriptor = ActionDescriptor("arm_action", 0, 6, term)
    assert descriptor.to_dict()["slice"] == [0, 6]
```

- [ ] **Step 2: Verify RED**

Run: `/root/miniconda3/envs/py311/bin/python -m pytest -q tests/gym/envs/managers/test_action_types.py`

Expected: import/collection failure because descriptor types do not exist.

- [ ] **Step 3: Implement immutable descriptors**

```python
JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class ActionTermDescriptor:
    representation: str
    action_dim: int
    feature_names: tuple[str, ...]
    units: tuple[str, ...]
    normalization: str | None
    joint_names: tuple[str, ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive.")
        if len(self.feature_names) != self.action_dim or len(self.units) != self.action_dim:
            raise ValueError("feature_names, units, and action_dim must agree.")
        owned = json_safe_copy(self.metadata, field_name="metadata")
        object.__setattr__(self, "metadata", MappingProxyType(owned))


@dataclass(frozen=True, slots=True)
class ActionDescriptor:
    name: str
    start: int
    stop: int
    term: ActionTermDescriptor

    def __post_init__(self) -> None:
        if not self.name or self.start < 0 or self.stop - self.start != self.term.action_dim:
            raise ValueError("descriptor name and slice must match the term.")
```

Use `_json.py` JSON-safe copy/types and provide `to_dict()` returning JSON lists for tuples and slices.

- [ ] **Step 4: Replace the base protocol**

Remove `SUPPORTED_TYPES`, `input_key`, `process_action()`, and `__call__()` from `ActionTerm`. Add abstract `action_dim`, `action_space`, `raw_actions`, `processed_actions`, `command_type`, `controlled_joint_ids`, `descriptor`, `process_actions()`, and `apply_actions()`.

Remove `ActionTermCfg.mode`. Add one minimal fake new-style term in tests so manager imports remain testable before built-ins migrate.

- [ ] **Step 5: Verify GREEN and checkpoint**

Run the descriptor test, `git diff --check`, and `git status --short`. Expected: PASS, no staged files.

---

### Task 2: Implement Flat ActionManager Slicing and Ownership

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/action_manager.py`
- Test: `tests/gym/envs/managers/test_action_manager.py`
- Test: `tests/gym/envs/managers/test_action_manager_reset.py`

**Interfaces:**
- Consumes Task 1 protocol.
- Produces `process_action(torch.Tensor) -> None`, `apply_action()`, flat `single_action_space`, action history, descriptors, and overlap validation.

- [ ] **Step 1: Write failing ordered-slice and history tests**

```python
def test_manager_slices_flat_action_in_config_order() -> None:
    manager = make_manager(
        ("arm", {"action_dim": 2, "low": -1.0, "high": 1.0, "joint_ids": (0, 1)}),
        ("gripper", {"action_dim": 1, "low": 0.0, "high": 1.0, "joint_ids": (2,)}),
        num_envs=2,
    )
    first = manager.get_term("arm")
    second = manager.get_term("gripper")
    action = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    manager.process_action(action)
    torch.testing.assert_close(first.raw_actions, action[:, :2])
    torch.testing.assert_close(second.raw_actions, action[:, 2:3])
    assert [(d.name, d.start, d.stop) for d in manager.descriptors] == [
        ("arm", 0, 2), ("gripper", 2, 3)
    ]
```

- [ ] **Step 2: Write Review Focus failures**

```python
@pytest.mark.parametrize("shape", [(2,), (2, 1), (2, 4)])
def test_manager_rejects_wrong_flat_shape(shape: tuple[int, ...]) -> None:
    manager = make_manager(
        ("arm", {"action_dim": 3, "low": -1.0, "high": 1.0, "joint_ids": (0, 1, 2)}),
        num_envs=2,
    )
    with pytest.raises(ValueError, match="expected action shape"):
        manager.process_action(torch.zeros(shape))


def test_manager_rejects_dict_action() -> None:
    manager = make_manager(
        ("arm", {"action_dim": 2, "low": -1.0, "high": 1.0, "joint_ids": (0, 1)}),
        num_envs=1,
    )
    with pytest.raises(TypeError, match="flat torch.Tensor"):
        manager.process_action({"arm": torch.zeros(1, 2)})


def test_manager_rejects_overlapping_qpos_terms() -> None:
    with pytest.raises(ValueError, match="first.*second.*joint_1"):
        make_manager(
            ("first", {"action_dim": 2, "low": -1.0, "high": 1.0, "joint_ids": (0, 1), "command_type": "qpos"}),
            ("second", {"action_dim": 1, "low": -1.0, "high": 1.0, "joint_ids": (1,), "command_type": "qpos"}),
            num_envs=1,
        )
```

- [ ] **Step 3: Verify RED**

Run both action-manager test files. Expected: current Dict/mode manager fails all new assertions.

- [ ] **Step 4: Implement ordered slices and flat state**

Build `_slices` in config order, allocate `[num_envs, total_action_dim]` current/previous tensors, concatenate term Box lows/highs, bind descriptors, validate floating shape/device before mutation, and call each term with its slice.

```python
def process_action(self, action: torch.Tensor) -> None:
    if not isinstance(action, torch.Tensor):
        raise TypeError("ActionManager expects one flat torch.Tensor.")
    expected = (self._env.num_envs, self.total_action_dim)
    if tuple(action.shape) != expected:
        raise ValueError(f"Expected action shape {expected}, got {tuple(action.shape)}.")
    if not action.is_floating_point() or action.device != torch.device(self._env.device):
        raise TypeError("Policy action must be floating and on the environment device.")
    self._previous_action.copy_(self._action)
    self._action.copy_(action)
    for name, term in self._terms.items():
        term.process_actions(self._action[:, self._slices[name]])

def apply_action(self) -> None:
    for term in self._terms.values():
        term.apply_actions()
```

- [ ] **Step 5: Implement overlap validation and selected-row reset**

Build `(command_type, joint_id) -> term_name` ownership. Reject duplicate keys
with both term and joint names. Different command types on the same joint remain
legal, matching the existing ability to submit qpos and qvel together. Reset
manager current/previous rows and every term using the same row selection.

- [ ] **Step 6: Verify GREEN and checkpoint**

Run both manager test files, Black on changed Python files, `git diff --check`, and `git status --short`.

---

### Task 3: Replace Built-In Joint Actions

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/actions.py`
- Modify: `embodichain/lab/gym/envs/managers/__init__.py`
- Test: `tests/gym/envs/managers/test_action_manager.py`
- Test: `tests/gym/envs/managers/test_action_manager_reset.py`
- Test: `tests/gym/envs/managers/test_default_joint_position.py`

**Interfaces:**
- Produces `JointPositionAction`, `JointPositionToLimitsAction`, `JointVelocityAction`, `JointEffortAction`, `RelativeJointPositionAction`, and `DefaultJointPositionAction`.

- [ ] **Step 1: Write failing joint selection, mapping, and reset tests**

```python
def test_joint_position_applies_selected_policy_order() -> None:
    env = make_action_env(
        joint_names=("joint_0", "joint_1", "joint_2", "joint_3"),
        parts={"arm": (3, 1)},
    )
    term = JointPositionAction(
        make_cfg(JointPositionAction, part_name="arm", preserve_order=True), env
    )
    term.process_actions(torch.tensor([[0.3, 0.1]]))
    term.apply_actions()
    kwargs = env.robot.set_qpos.call_args.kwargs
    assert kwargs["joint_ids"] == [3, 1]
    torch.testing.assert_close(kwargs["qpos"], torch.tensor([[0.3, 0.1]]))


def test_position_to_limits_maps_each_dimension() -> None:
    env = make_action_env(joint_names=("joint_0", "joint_1"))
    env.robot.body_data.qpos_limits = torch.tensor([[[-2.0, 2.0], [0.0, 4.0]]])
    term = JointPositionToLimitsAction(
        make_cfg(JointPositionToLimitsAction, part_name="arm"), env
    )
    term.process_actions(torch.tensor([[-1.0, 1.0]]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[-2.0, 4.0]]))


def test_default_joint_action_resets_selected_history_only() -> None:
    env = make_action_env(num_envs=2, joint_names=("joint_0", "joint_1"))
    env.robot.cfg = SimpleNamespace(init_qpos=[0.0, 0.0])
    term = DefaultJointPositionAction(
        make_cfg(
            DefaultJointPositionAction,
            joint_names=["joint_0", "joint_1"],
            offset=[0.0, 0.0],
            scale=1.0,
        ),
        env,
    )
    term.process_actions(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    term.process_actions(torch.tensor([[0.5, 0.6], [0.7, 0.8]]))
    term.reset(torch.tensor([0]))
    torch.testing.assert_close(term.raw_actions[0], torch.zeros(2))
    torch.testing.assert_close(term.raw_actions[1], torch.tensor([0.7, 0.8]))
```

- [ ] **Step 2: Verify RED**

Run the three test files. Expected: new imports and process/apply methods are missing.

- [ ] **Step 3: Implement private `_JointAction`**

Resolve exactly one of `part_name` or `joint_names`, preserve requested order, remove mimic IDs, cache names/IDs/limits, allocate buffers, construct the descriptor, and apply via the selected robot setter.

- [ ] **Step 4: Implement six public joint actions**

```python
class JointPositionAction(_JointAction):
    command_type = "qpos"

    def process_actions(self, actions: torch.Tensor) -> None:
        self._previous_raw_actions.copy_(self._raw_actions)
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(actions * self._scale + self._offset)

    def apply_actions(self) -> None:
        self._env.robot.set_qpos(self._processed_actions, joint_ids=self._joint_ids)
```

Implement velocity/effort with their setters, relative position from selected current qpos, limits mapping per selected joint, and default position with default offset, scale, and `position_bias`.

Define raw policy spaces explicitly: absolute position uses selected qpos
limits; limits-mapped and default locomotion actions use `[-1, 1]`; velocity
and effort use selected limits; relative position uses configured `clip` bounds
or an unbounded Box when no clip is configured.

- [ ] **Step 5: Delete old names and exports**

Remove all old classes listed in the spec from definitions, `__all__`, module docs, and tests. Do not add aliases.

- [ ] **Step 6: Verify GREEN and checkpoint**

Run the three test files, Black, `git diff --check`, and `git status --short`.

---

### Task 4: Implement EEF and Parallel-Gripper Actions

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/actions.py`
- Test: `tests/gym/envs/managers/test_action_manager.py`
- Test: `tests/gym/envs/tasks/test_repeated_pick_place.py`

**Interfaces:**
- Produces `EefPoseAction` and `ParallelGripperAction` with direct selected-joint application.

- [ ] **Step 1: Write failing EEF tests**

```python
def test_eef_action_applies_ik_only_to_selected_arm() -> None:
    env = make_action_env(
        num_envs=1,
        joint_names=("joint_0", "joint_1", "joint_2"),
        parts={"arm": (0, 2)},
    )
    env.robot.get_qpos.return_value = torch.tensor([[0.1, 9.0, 0.2]])
    env.robot.compute_ik.return_value = (torch.tensor([True]), torch.tensor([[0.4, 0.5]]))
    term = EefPoseAction(
        make_cfg(EefPoseAction, part_name="arm", pose_representation="xyz_rpy"), env
    )
    term.process_actions(torch.zeros(1, 6))
    term.apply_actions()
    kwargs = env.robot.set_qpos.call_args.kwargs
    assert kwargs["joint_ids"] == [0, 2]
    torch.testing.assert_close(kwargs["qpos"], torch.tensor([[0.4, 0.5]]))


def test_eef_action_holds_failed_ik_row() -> None:
    env = make_action_env(
        num_envs=2,
        joint_names=("joint_0", "joint_1"),
        parts={"arm": (0, 1)},
    )
    env.robot.compute_ik.return_value = (
        torch.tensor([True, False]), torch.tensor([[0.4, 0.5], [0.8, 0.9]])
    )
    term = EefPoseAction(
        make_cfg(EefPoseAction, part_name="arm", pose_representation="xyz_rpy"), env
    )
    term.process_actions(torch.zeros(2, 6))
    torch.testing.assert_close(term.processed_actions[1], env.robot.get_qpos()[1, [0, 1]])
```

- [ ] **Step 2: Write failing gripper tests**

```python
def test_continuous_parallel_gripper_maps_opposing_commands() -> None:
    env = make_action_env(
        num_envs=2,
        joint_names=("j0", "j1", "j2", "j3", "j4", "left", "j6", "right"),
        parts={"hand": (5, 7)},
    )
    term = ParallelGripperAction(
        make_cfg(
            ParallelGripperAction,
            part_name="hand",
            command_mode="continuous",
            lower_command={"left": 0.0, "right": 0.04},
            upper_command={"left": 0.04, "right": 0.0},
        ),
        env,
    )
    term.process_actions(torch.tensor([[-1.0], [1.0]]))
    torch.testing.assert_close(term.processed_actions, torch.tensor([[0.0, 0.04], [0.04, 0.0]]))


def test_binary_parallel_gripper_has_one_policy_dimension() -> None:
    env = make_action_env(
        num_envs=2,
        joint_names=("left", "right"),
        parts={"hand": (0, 1)},
    )
    term = ParallelGripperAction(
        make_cfg(
            ParallelGripperAction,
            part_name="hand",
            command_mode="binary",
            open_command={"left": 0.04, "right": 0.04},
            close_command={"left": 0.0, "right": 0.0},
        ),
        env,
    )
    term.process_actions(torch.tensor([[-0.1], [0.1]]))
    assert term.action_dim == 1
    torch.testing.assert_close(term.processed_actions[0], term.close_command)
    torch.testing.assert_close(term.processed_actions[1], term.open_command)
```

- [ ] **Step 3: Add dimension and mimic tests**

Assert EEF stays 6D for six- and seven-DoF arms, joint arm plus gripper totals 7D for UR and 8D for Franka, and mimic followers never appear in controlled IDs or setter calls.

- [ ] **Step 4: Verify RED**

Run action-manager EEF/gripper tests and repeated-pick-place tests. Expected: new classes are absent and combined classes fail composition assertions.

- [ ] **Step 5: Implement `EefPoseAction`**

Cache arm IDs/names; store six-dimensional `xyz_rpy`; build target matrices device-natively; call selected-part IK; hold current selected qpos on failed rows; apply only selected arm IDs. Publish `eef_pose` descriptor names/units and arena/RPY metadata.

- [ ] **Step 6: Implement `ParallelGripperAction`**

Resolve non-mimic gripper joints and complete command maps at initialization. Use `torch.lerp` in continuous mode and `torch.where` in binary mode. Clamp continuous execution device-natively; strict raw dataset validation belongs to Task 7.

- [ ] **Step 7: Delete combined classes and verify GREEN**

Remove `EefPoseGripperTerm` and `JointPositionGripperTerm`. Run Task 4 tests, Black, `git diff --check`, and inspect status.

---

### Task 5: Separate Policy and ControllerAction Environment Paths

**Files:**
- Modify: `embodichain/lab/gym/envs/embodied_env.py`
- Modify: `embodichain/lab/gym/envs/types.py`
- Test: `tests/gym/envs/test_demo.py`
- Test: `tests/gym/envs/test_embodied_env.py`

**Interfaces:**
- Consumes Task 2 manager lifecycle.
- Produces manager policy processing/application and preserved direct controller-ready application.

- [ ] **Step 1: Write failing lifecycle test**

```python
def test_policy_action_processes_and_applies_through_manager() -> None:
    env = _controller_action_env()
    env.action_manager.process_action = Mock()
    env.action_manager.apply_action = Mock()
    raw = torch.zeros(env.num_envs, 3)
    processed = env._preprocess_action(raw)
    returned = env._step_action(processed)
    env.action_manager.process_action.assert_called_once_with(raw)
    env.action_manager.apply_action.assert_called_once_with()
    torch.testing.assert_close(returned, raw)
```

- [ ] **Step 2: Write ControllerAction bypass tests**

```python
@pytest.mark.parametrize("key", ["qpos", "qvel"])
def test_controller_action_bypasses_manager(key: str) -> None:
    env = _controller_action_env()
    env.action_manager.process_action = Mock()
    env.action_manager.apply_action = Mock()
    value = TensorDict({key: torch.zeros(env.num_envs, 3)}, batch_size=[env.num_envs])
    prepared = env._preprocess_action(ControllerAction(value))
    env._step_action(prepared)
    env.action_manager.process_action.assert_not_called()
    env.action_manager.apply_action.assert_not_called()
    getattr(env.robot, f"set_{key}").assert_called_once()
```

- [ ] **Step 3: Write policy/expert rollout semantic tests**

Assert rewards and policy history receive the manager's flat raw tensor, while ControllerAction expert recording receives the validated controller value for `ExpertActionSpec` encoding.

- [ ] **Step 4: Verify RED**

Run `test_demo.py` and `test_embodied_env.py`. Expected: current returned `EnvAction` and postprocessing order fail.

- [ ] **Step 5: Implement policy routing**

For normal tensors, call manager `process_action()` during preprocessing and `apply_action()` in `_step_action()`, returning the flat raw tensor unchanged to reward/rollout consumers. Remove `_postprocess_action()` and all post-mode behavior.

- [ ] **Step 6: Preserve the ControllerAction envelope**

Validate an owned snapshot but keep it wrapped until `_step_action()`. Unwrap there, apply through existing selected qpos/qvel/qf setters, and return the controller value for expert consumers.

- [ ] **Step 7: Verify GREEN and checkpoint**

Run both environment tests, Black, `git diff --check`, and inspect status.

---

### Task 6: Migrate Official Tasks and Locomotion Consumers

**Files:**
- Modify: all 15 official action YAML files enumerated in the spec.
- Modify: `embodichain_tasks/embodichain_tasks/locomotion/velocity/_embodichain.py`
- Modify: `embodichain/lab/gym/utils/gym_utils.py`
- Test: `tests/gym/envs/test_official_task_layout.py`
- Test: `tests/gym/envs/tasks/test_repeated_pick_place.py`
- Test: `tests/gym/envs/task_program/test_configured_integration.py`
- Test: locomotion tests discovered with `rg -l "ANYmalCFlatEnv|UnitreeGo1FlatEnv|UnitreeGo2FlatEnv|UnitreeG1FlatEnv|UnitreeH12FlatEnv|MicroDuckFlatEnv" tests`.

**Interfaces:**
- Produces official configurations containing only new class names and flat layouts.

- [ ] **Step 1: Write strict migration test**

```python
REMOVED = {
    "QposTerm", "QposDenormalizedTerm", "QposNormalizedTerm", "QvelTerm",
    "QfTerm", "DeltaQposTerm", "DefaultJointPositionTerm", "EefPoseTerm",
    "EefPoseGripperTerm", "JointPositionGripperTerm",
}

def test_official_action_configs_use_new_classes_only() -> None:
    for path in sorted(TASK_CONFIG_ROOT.rglob("*.yaml")):
        config = load_config(path)
        if not isinstance(config, dict):
            continue
        for term in config.get("env", {}).get("actions", {}).values():
            assert term["func"] not in REMOVED, path
            assert "mode" not in term, path
```

- [ ] **Step 2: Write RLinf flat-layout tests**

Decode both RLinf configs through production composition. Assert ordered names `arm_action, gripper_action`, Box shapes `(7,)` and `(8,)`, and `control_parts == ["arm", "hand"]`.

- [ ] **Step 3: Write locomotion state binding test**

Assert locomotion binds current and previous actions to `raw_actions` and `previous_raw_actions`, retains `position_bias`, and preserves selected-row reset behavior.

- [ ] **Step 4: Verify RED**

Run official layout, repeated-pick-place, configured integration, and discovered locomotion tests. Expected: removed names and combined terms fail.

- [ ] **Step 5: Migrate YAML and Python consumers**

Use `RelativeJointPositionAction` for CartPole and `DefaultJointPositionAction` for all locomotion files. Split RLinf configs into ordered EEF/joint arm plus continuous parallel gripper terms and add `control_parts: [arm, hand]`. Update locomotion property reads.

- [ ] **Step 6: Make config decoding strict**

Remove mode decoding and raise a migration-oriented error when an action config contains `mode` or a removed action class name.

- [ ] **Step 7: Verify GREEN and checkpoint**

Run Task 6 tests, Black, `git diff --check`, and inspect status.

---

### Task 7: Drive Policy Dataset Schemas From Action Descriptors

**Files:**
- Modify: `embodichain/lab/gym/envs/managers/datasets.py`
- Modify: `embodichain/lab/gym/envs/managers/async_datasets.py`
- Modify: `embodichain/lab/gym/envs/managers/dataset_manager.py`
- Modify: `embodichain/lab/gym/envs/embodied_env.py`
- Test: `tests/gym/envs/managers/test_dataset_functors.py`
- Test: `tests/gym/envs/managers/test_async_dataset_functors.py`
- Test: `tests/gym/envs/managers/test_dataset_manager.py`
- Test: `tests/gym/envs/test_demo.py`

**Interfaces:**
- Consumes manager raw actions/descriptors and `ExpertActionSpec`.
- Produces descriptor-owned policy schemas and unchanged expert schemas.

- [ ] **Step 1: Write failing composed-feature test**

```python
def test_eef_gripper_contract_uses_descriptor_order() -> None:
    arm = ActionDescriptor(
        "arm_action",
        0,
        6,
        ActionTermDescriptor(
            "eef_pose", 6, ("x", "y", "z", "roll", "pitch", "yaw"),
            ("m", "m", "m", "rad", "rad", "rad"), None, (), {"frame": "arena"}
        ),
    )
    gripper = ActionDescriptor(
        "gripper_action",
        6,
        7,
        ActionTermDescriptor(
            "parallel_gripper", 1, ("gripper",), ("normalized",),
            "minus_one_to_one", ("finger_joint",), {}
        ),
    )
    env = MockEnvForDataset(num_joints=7, has_sensors=False)
    env.action_manager = SimpleNamespace(descriptors=(arm, gripper))
    recorder = LeRobotRecorder(
        MockFunctorCfg(
            params={
                "save_path": "/tmp/test_dataset",
                "action_contract": {
                    "version": 1,
                    "representation": "eef_pose_parallel_gripper",
                },
            }
        ),
        env,
    )
    feature = recorder._build_features()["action"]
    assert feature["shape"] == (7,)
    assert feature["names"] == ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    assert feature["info"]["embodichain.action_terms"][1]["slice"] == [6, 7]
```

- [ ] **Step 2: Write policy/expert separation tests**

Assert policy recording stores the exact manager flat input and descriptors. Assert ControllerAction experts still store active qpos or `[qpos, qvel]` through `ExpertActionSpec`. Assert no controller-qpos feature exists.

- [ ] **Step 3: Write sync/async/fragment equivalence tests**

Use one three-frame, two-term tensor. Verify all three persistence routes contain identical action rows and descriptor order after the caller mutates the source buffer.

- [ ] **Step 4: Write CPU persistence validation tests**

Reject non-finite EEF values, parallel-gripper values outside `[-1, 1]`, descriptor gaps/overlaps, and a representation whose expected term sequence differs from manager descriptors.

- [ ] **Step 5: Verify RED**

Run dataset, async dataset, manager, and demo tests. Expected: class-name/action-mode inference fails descriptor assertions.

- [ ] **Step 6: Implement descriptor-driven contract parsing**

Add `eef_pose_parallel_gripper` and `joint_position_parallel_gripper`. Validate descriptor sequence during recorder construction; derive width/names from descriptors; store descriptor dictionaries under `embodichain.action_terms` in feature metadata and sidecars.

- [ ] **Step 7: Snapshot manager raw policy actions**

Append validated flat actions to active per-environment CPU history after manager processing. Sync and async recorders consume that history before reset. ControllerAction paths do not append policy history.

- [ ] **Step 8: Preserve persistence transactions**

Keep fragment IDs, commit ordering, `_episode_payloads()`, and caller-side async cloning unchanged. Slice flat actions with the same episode/fragment ranges as observations and annotations.

- [ ] **Step 9: Verify GREEN and checkpoint**

Run Task 7 tests, Black, `git diff --check`, and inspect status.

---

### Task 8: Update Public API, Documentation, and Project Context

**Files:**
- Modify: `docs/source/api_reference/public_api.rst`
- Modify: repeated-pick-place README.
- Modify: manager-functor, env-framework overview/execution, data-pipeline overview/persistence, and robot-system context files.
- Test: `tests/test_agent_context_map.py`
- Test: `tests/test_agent_context_tools.py`

**Interfaces:**
- Consumes completed public names and behavior from Tasks 1-7.
- Produces synchronized API documentation and migration guidance.

- [ ] **Step 1: Add static old-name and new-export assertions**

Assert all new public actions/descriptors are documented and no removed class appears in public API RST, official configs, or owning contexts.

- [ ] **Step 2: Verify RED**

Run API docs checker, context checker, and context map/tool tests. Expected: old names or missing new exports fail.

- [ ] **Step 3: Update API docs and task migration guide**

Document new classes/descriptors, remove old names, and show composed term order, flat dimensions, dataset slices, removed Dict/post behavior, and ControllerAction expert path.

- [ ] **Step 4: Revise context owners**

Manager context owns process/apply, slicing, ownership, and descriptors. Environment context owns policy/controller routing. Data pipeline owns descriptor policy schemas and ExpertActionSpec schemas. Robot context owns control parts, mimic exclusion, and future tendon resources.

- [ ] **Step 5: Verify GREEN and checkpoint**

Run:

```bash
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
/root/miniconda3/envs/py311/bin/python .agents/skills/project-dev-context/scripts/context.py check
/root/miniconda3/envs/py311/bin/python .agents/skills/project-dev-context/scripts/context.py stats --base origin/main
/root/miniconda3/envs/py311/bin/python -m pytest -q -c /dev/null --noconftest tests/test_agent_context_map.py tests/test_agent_context_tools.py
```

Expected: all pass; then run Black, `git diff --check`, and inspect status.

---

### Task 9: Integration, Simulator Smoke, and Final Gates

**Files:**
- Modify only files required by failures demonstrated in this task.
- Test all affected manager, environment, task, learning, dataset, context, and API surfaces.

**Interfaces:**
- Consumes Tasks 1-8.
- Produces an unstaged, verified working tree for user review.

- [ ] **Step 1: Run focused non-simulator integration**

```bash
/root/miniconda3/envs/py311/bin/python -m pytest -q \
  tests/gym/envs/managers \
  tests/gym/envs/test_demo.py \
  tests/gym/envs/test_embodied_env.py \
  tests/gym/envs/test_official_task_layout.py \
  tests/gym/envs/tasks/test_repeated_pick_place.py \
  tests/gym/envs/task_program/test_configured_integration.py \
  tests/learning \
  tests/test_task_catalog.py \
  tests/test_task_program_package_data.py
```

Expected: PASS with only documented pre-existing warnings.

- [ ] **Step 2: Select and run low-resource simulator smoke tests**

Collect candidate nodes:

```bash
/root/miniconda3/envs/py311/bin/python -m pytest --collect-only -q \
  tests/gym/envs/tasks/test_repeated_pick_place.py tests/gym/envs/tasks \
  | rg "newton|default|action"
```

Run the smallest existing one-environment Default and Newton nodes serially. Existing fixtures must destroy the simulation and flush cleanup. Expected: each processes/applies one action without mimic or shape errors.

- [ ] **Step 3: Audit GPU hot paths**

```bash
rg -n "\.item\(\)|bool\(torch|\.cpu\(\)|\.numpy\(\)" \
  embodichain/lab/gym/envs/managers/action_manager.py \
  embodichain/lab/gym/envs/managers/actions.py
```

Initialization may move static Box limits to CPU. `process_action()`, every `process_actions()`, and every `apply_actions()` must contain no host transfer or Python scalar extraction.

- [ ] **Step 4: Run full repository gates**

```bash
/root/miniconda3/envs/py311/bin/python -m black --check --diff --color ./
/root/miniconda3/envs/py311/bin/python docs/scripts/check_api_docs.py
/root/miniconda3/envs/py311/bin/python -m compileall -q embodichain embodichain_tasks tests
/root/miniconda3/envs/py311/bin/python -m pytest -q
git diff --check
```

Expected: every available command exits zero. Report exact skipped/unavailable GPU, renderer, simulator, or external-service node IDs.

- [ ] **Step 5: Verify removed names are absent**

```bash
if rg -n "QposTerm|QposDenormalizedTerm|QposNormalizedTerm|QvelTerm|QfTerm|DeltaQposTerm|DefaultJointPositionTerm|EefPoseTerm|EefPoseGripperTerm|JointPositionGripperTerm" \
  embodichain embodichain_tasks agent_context docs/source tests; then
  exit 1
fi
```

Expected: no matches.

- [ ] **Step 6: Produce the final uncommitted snapshot**

```bash
git status --short
git diff --stat
git diff --check
git diff --name-only --cached
```

Expected: intended changes are unstaged, cached output is empty, and no commit or push occurred.
