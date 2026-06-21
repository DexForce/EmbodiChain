# Atomic Actions Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `embodichain/lab/sim/atomic_actions/` from a `Union`+kwargs+inheritance design to a typed-targets + `WorldState` + composition design, per `docs/superpowers/specs/2026-06-21-atomic-actions-redesign-design.md`.

**Architecture:** Five files — `core.py` (slim ABC + typed targets + `WorldState`/`ActionResult`), `affordance.py` (extracted `Affordance` types, no `__post_init__` aliasing), `trajectory.py` (new `TrajectoryBuilder` composition helper), `actions.py` (four sibling actions all inheriting `AtomicAction` directly), `engine.py` (name-keyed `run(steps, state)` API, no `_resolve_target`, no `SemanticAnalyzer`).

**Tech Stack:** Python 3.10+, PyTorch, `@configclass`, `dataclasses`, pytest, existing `MotionGenerator` / `Robot` / `BatchEntity` / `pose_inv` / `interpolate_with_distance` infrastructure.

---

## Pre-flight

- [ ] **Read the spec end-to-end before Task 1.** Path: `docs/superpowers/specs/2026-06-21-atomic-actions-redesign-design.md`.

- [ ] **Confirm clean working tree.**

  Run: `git status`
  Expected: `nothing to commit, working tree clean`

- [ ] **Confirm existing tests are green before any change.**

  Run: `pytest tests/sim/atomic_actions/ -x --no-header -q`
  Expected: all tests pass. If not, stop and surface the failure.

- [ ] **Verification commands used after each task:**

  - Tests: `pytest tests/sim/atomic_actions/ -x --no-header -q`
  - Format: `black embodichain/lab/sim/atomic_actions/ tests/sim/atomic_actions/ scripts/tutorials/`
  - Import smoke test after each module rewrite: `python -c "from embodichain.lab.sim.atomic_actions import AtomicActionEngine"`

---

## Task 1: Extract `Affordance` types to `affordance.py` (fix geometry aliasing)

**Files:**
- Create: `embodichain/lab/sim/atomic_actions/affordance.py`
- Modify: `embodichain/lab/sim/atomic_actions/core.py` (remove the extracted classes)
- Test:   `tests/sim/atomic_actions/test_affordance.py` (new) replaces affordance portion of `test_core.py`

The current `Affordance` aliases its `geometry` dict to `ObjectSemantics.geometry` via `ObjectSemantics.__post_init__`. We are killing that aliasing: `Affordance` no longer stores `geometry` at all; subclasses that need mesh data take it via explicit fields.

- [ ] **Step 1: Create the failing test file `tests/sim/atomic_actions/test_affordance.py`.**

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (see top of file).
# ----------------------------------------------------------------------------

"""Tests for atomic_actions.affordance (Affordance, AntipodalAffordance, InteractionPoints)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import (
    Affordance,
    AntipodalAffordance,
    InteractionPoints,
)


class TestAffordance:
    def test_default_object_label_is_empty(self):
        assert Affordance().object_label == ""

    def test_custom_config_get_set(self):
        aff = Affordance()
        aff.set_custom_config("k", 1)
        assert aff.get_custom_config("k") == 1
        assert aff.get_custom_config("missing") is None
        assert aff.get_custom_config("missing", "d") == "d"

    def test_base_get_batch_size_is_one(self):
        assert Affordance().get_batch_size() == 1


class TestAntipodalAffordance:
    def test_stores_mesh_fields_directly(self):
        v = torch.randn(8, 3)
        t = torch.randint(0, 8, (5, 3))
        aff = AntipodalAffordance(mesh_vertices=v, mesh_triangles=t)
        assert aff.mesh_vertices is v
        assert aff.mesh_triangles is t

    def test_no_geometry_alias_field(self):
        # The redesign removes the shared-geometry-dict footgun.
        aff = AntipodalAffordance()
        assert not hasattr(aff, "geometry") or getattr(aff, "geometry", None) is None


class TestInteractionPoints:
    def test_default_points_shape(self):
        assert InteractionPoints().points.shape == (1, 3)

    def test_get_batch_size_matches_points(self):
        ip = InteractionPoints(points=torch.randn(4, 3))
        assert ip.get_batch_size() == 4

    def test_get_points_by_type_returns_subset(self):
        pts = torch.tensor([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        ip = InteractionPoints(points=pts, point_types=["push", "poke", "push"])
        result = ip.get_points_by_type("push")
        assert result is not None and result.shape == (2, 3)
        assert torch.equal(result[0], pts[0])
        assert torch.equal(result[1], pts[2])

    def test_get_points_by_type_returns_none_for_missing(self):
        ip = InteractionPoints(points=torch.zeros(2, 3), point_types=["push", "push"])
        assert ip.get_points_by_type("poke") is None

    def test_approach_direction_inverts_normal(self):
        normals = torch.tensor([[0., 0, 1.], [1., 0, 0]])
        ip = InteractionPoints(points=torch.zeros(2, 3), normals=normals)
        assert torch.equal(ip.get_approach_direction(0), torch.tensor([0., 0, -1.]))
        assert torch.equal(ip.get_approach_direction(1), torch.tensor([-1., 0, 0]))

    def test_approach_direction_default_when_no_normals(self):
        ip = InteractionPoints(points=torch.zeros(1, 3))
        assert torch.equal(ip.get_approach_direction(0), torch.tensor([0., 0, 1.]))
```

- [ ] **Step 2: Run the test to verify it fails (import error).**

  Run: `pytest tests/sim/atomic_actions/test_affordance.py -x --no-header -q`
  Expected: `ModuleNotFoundError: No module named 'embodichain.lab.sim.atomic_actions.affordance'`.

- [ ] **Step 3: Create `embodichain/lab/sim/atomic_actions/affordance.py`.**

Copy the standard Apache 2.0 header from `embodichain/lab/sim/atomic_actions/core.py` lines 1-15. Then implement:

```python
from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List

from embodichain.toolkits.graspkit.pg_grasp import (
    GraspGenerator,
    GraspGeneratorCfg,
)
from embodichain.toolkits.graspkit.pg_grasp.gripper_collision_checker import (
    GripperCollisionCfg,
)
from embodichain.utils import logger


@dataclass
class Affordance:
    """Base class for affordance data.

    Affordance represents interaction possibilities for an object.
    Unlike the previous design, Affordance no longer stores a shared geometry
    dict aliased from ObjectSemantics. Subclasses take only the specific
    fields they need.
    """

    object_label: str = ""
    """Label of the object this affordance belongs to."""

    custom_config: Dict[str, Any] = field(default_factory=dict)
    """User-defined configuration payload."""

    def set_custom_config(self, key: str, value: Any) -> None:
        self.custom_config[key] = value

    def get_custom_config(self, key: str, default: Any = None) -> Any:
        return self.custom_config.get(key, default)

    def get_batch_size(self) -> int:
        return 1


@dataclass
class AntipodalAffordance(Affordance):
    """Antipodal grasp affordance for parallel-jaw grippers."""

    mesh_vertices: torch.Tensor | None = None
    """Object mesh vertices, shape [N, 3]."""

    mesh_triangles: torch.Tensor | None = None
    """Object mesh triangle indices, shape [M, 3]."""

    generator_cfg: GraspGeneratorCfg | None = None
    """Optional grasp-generator configuration."""

    gripper_collision_cfg: GripperCollisionCfg | None = None
    """Optional gripper-collision configuration."""

    force_reannotate: bool = False
    """If True, recompute the grasp annotation on each access."""

    is_draw_grasp_xpos: bool = False
    """If True, draw grasp poses in the simulator on each call."""

    _generator: GraspGenerator | None = field(default=None, init=False, repr=False)

    def _init_generator(self) -> None:
        if self.mesh_vertices is None or self.mesh_triangles is None:
            logger.log_error(
                "mesh_vertices and mesh_triangles must be provided to initialize "
                "AntipodalAffordance.",
                ValueError,
            )
        self._generator = GraspGenerator(
            vertices=self.mesh_vertices,
            triangles=self.mesh_triangles,
            cfg=self.generator_cfg,
            gripper_collision_cfg=self.gripper_collision_cfg,
        )
        if self.force_reannotate or self._generator._hit_point_pairs is None:
            self._generator.annotate()

    def get_valid_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self._generator is None:
            self._init_generator()
        results = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_poses, _, costs = self._generator.get_valid_grasp_poses(
                obj_pose, approach_direction
            )
            if not is_success:
                logger.log_warning(
                    f"Failed to find valid grasp poses for {i}-th object."
                )
            results.append((grasp_poses, costs))
        return results

    def get_best_grasp_poses(
        self,
        obj_poses: torch.Tensor,
        approach_direction: torch.Tensor = torch.tensor(
            [0, 0, -1], dtype=torch.float32
        ),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._generator is None:
            self._init_generator()
        grasp_xpos_list: list[torch.Tensor] = []
        is_success_list: list[bool] = []
        open_length_list: list[float] = []
        for i, obj_pose in enumerate(obj_poses):
            is_success, grasp_xpos, open_length = self._generator.get_grasp_poses(
                obj_pose, approach_direction
            )
            if is_success:
                grasp_xpos_list.append(grasp_xpos.unsqueeze(0))
            else:
                logger.log_warning(f"No valid grasp pose found for {i}-th object.")
                grasp_xpos_list.append(
                    torch.eye(
                        4, dtype=torch.float32, device=self._generator.device
                    ).unsqueeze(0)
                )
            is_success_list.append(is_success)
            open_length_list.append(open_length)
        is_success_t = torch.tensor(
            is_success_list, dtype=torch.bool, device=self._generator.device
        )
        grasp_xpos = torch.concatenate(grasp_xpos_list, dim=0)
        open_length_t = torch.tensor(
            open_length_list, dtype=torch.float32, device=self._generator.device
        )
        if self.is_draw_grasp_xpos:
            self._draw_grasp_xpos(grasp_xpos, open_length_t)
        return is_success_t, grasp_xpos, open_length_t

    def _draw_grasp_xpos(
        self, grasp_xpos: torch.Tensor, open_length: torch.Tensor
    ) -> None:
        from embodichain.lab.sim.sim_manager import SimulationManager
        from embodichain.lab.sim.objects.gizmo import MarkerCfg

        sim = SimulationManager.get_instance()
        axis_xpos = [grasp_xpos[i].to("cpu").numpy() for i in range(grasp_xpos.shape[0])]
        sim.draw_marker(cfg=MarkerCfg(name="grasp_xpos", axis_xpos=axis_xpos, axis_len=0.05))


@dataclass
class InteractionPoints(Affordance):
    """Batch of 3D interaction points on an object surface."""

    points: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 3))
    normals: torch.Tensor | None = None
    point_types: List[str] = field(default_factory=list)

    def get_points_by_type(self, point_type: str) -> torch.Tensor | None:
        if point_type in self.point_types:
            indices = [i for i, t in enumerate(self.point_types) if t == point_type]
            return self.points[indices]
        return None

    def get_batch_size(self) -> int:
        return self.points.shape[0]

    def get_approach_direction(self, point_idx: int) -> torch.Tensor:
        if self.normals is not None:
            return -self.normals[point_idx]
        return torch.tensor(
            [0, 0, 1], dtype=self.points.dtype, device=self.points.device
        )


__all__ = ["Affordance", "AntipodalAffordance", "InteractionPoints"]
```

- [ ] **Step 4: Run the test to verify it passes.**

  Run: `pytest tests/sim/atomic_actions/test_affordance.py -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/affordance.py tests/sim/atomic_actions/test_affordance.py
git commit -m "Extract Affordance types into affordance.py (no geometry alias)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: Rewrite `core.py` — typed targets, `WorldState`, `ActionResult`, slim ABC

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/core.py`
- Replace: `tests/sim/atomic_actions/test_core.py`

After this task, `core.py` will export only: `ActionCfg`, `AtomicAction`, `HeldObjectState`, `ObjectSemantics`, `PoseTarget`, `GraspTarget`, `HeldObjectTarget`, `Target`, `WorldState`, `ActionResult`. The `Affordance`/`AntipodalAffordance`/`InteractionPoints`/`MoveObjectTarget` symbols are gone from `core.py` (the first three moved to `affordance.py` in Task 1; `MoveObjectTarget` is renamed to `HeldObjectTarget`).

- [ ] **Step 1: Replace `tests/sim/atomic_actions/test_core.py` entirely** with new tests for the new shape.

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (see top of file).
# ----------------------------------------------------------------------------

"""Tests for atomic_actions.core (typed targets, WorldState, ActionResult, ObjectSemantics)."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionCfg,
    ActionResult,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)


class TestTypedTargets:
    def test_pose_target_holds_tensor(self):
        x = torch.eye(4)
        assert PoseTarget(xpos=x).xpos is x

    def test_pose_target_is_frozen(self):
        t = PoseTarget(xpos=torch.eye(4))
        with pytest.raises(Exception):
            t.xpos = torch.zeros(4, 4)  # type: ignore[misc]

    def test_grasp_target_holds_semantics(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="mug")
        assert GraspTarget(semantics=sem).semantics is sem

    def test_held_object_target_holds_pose(self):
        x = torch.eye(4)
        assert HeldObjectTarget(object_target_pose=x).object_target_pose is x


class TestObjectSemantics:
    def test_does_not_mutate_affordance_geometry(self):
        # The redesign removes the __post_init__ aliasing footgun.
        aff = Affordance()
        geometry = {"bounding_box": [0.1, 0.1, 0.1]}
        ObjectSemantics(affordance=aff, geometry=geometry, label="mug")
        # affordance should not have a geometry attribute, or if it does it should
        # NOT be the same object as the semantics' geometry dict.
        assert getattr(aff, "geometry", None) is not geometry

    def test_sets_object_label_on_affordance(self):
        aff = Affordance()
        ObjectSemantics(affordance=aff, geometry={}, label="mug")
        assert aff.object_label == "mug"

    def test_default_optional_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        assert sem.label == "none"
        assert sem.properties == {}
        assert sem.entity is None


class TestHeldObjectState:
    def test_required_fields(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        s = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        assert s.semantics is sem
        assert s.object_to_eef.shape == (1, 4, 4)
        assert s.grasp_xpos.shape == (1, 4, 4)


class TestWorldState:
    def test_constructs_with_last_qpos_only(self):
        qpos = torch.zeros(2, 6)
        ws = WorldState(last_qpos=qpos)
        assert ws.last_qpos is qpos
        assert ws.held_object is None

    def test_carries_held_object(self):
        sem = ObjectSemantics(affordance=Affordance(), geometry={})
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0),
            grasp_xpos=torch.eye(4).unsqueeze(0),
        )
        ws = WorldState(last_qpos=torch.zeros(1, 6), held_object=held)
        assert ws.held_object is held


class TestActionResult:
    def test_shape_contract(self):
        traj = torch.zeros(2, 10, 8)
        ws = WorldState(last_qpos=torch.zeros(2, 8))
        res = ActionResult(success=True, trajectory=traj, next_state=ws)
        assert res.success is True
        assert res.trajectory.shape == (2, 10, 8)
        assert res.next_state is ws


class TestActionCfg:
    def test_defaults(self):
        cfg = ActionCfg()
        assert cfg.name == "default"
        assert cfg.control_part == "arm"
        assert cfg.interpolation_type == "linear"
        assert cfg.velocity_limit is None
        assert cfg.acceleration_limit is None
```

- [ ] **Step 2: Run the new tests; expect failures because `core.py` still has the old shape.**

  Run: `pytest tests/sim/atomic_actions/test_core.py -x --no-header -q`
  Expected: import errors for `PoseTarget`/`GraspTarget`/`HeldObjectTarget`/`WorldState`/`ActionResult`.

- [ ] **Step 3: Replace `embodichain/lab/sim/atomic_actions/core.py` entirely.**

Preserve the standard Apache header (current lines 1-15). Then replace the body with:

```python
from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, TYPE_CHECKING, Union

from embodichain.lab.sim.common import BatchEntity
from embodichain.utils import configclass

from .affordance import Affordance

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


# =============================================================================
# ObjectSemantics
# =============================================================================


@dataclass
class ObjectSemantics:
    """Semantic information about an interaction target."""

    affordance: Affordance
    """Affordance data describing how the object can be interacted with."""

    geometry: Dict[str, Any]
    """Geometric metadata (bounding box, mesh, etc.). Plain user data, NOT shared with affordance."""

    properties: Dict[str, Any] = field(default_factory=dict)
    """Physical properties: mass, friction, etc."""

    label: str = "none"
    """Object category label (e.g., 'mug', 'apple')."""

    entity: BatchEntity | None = None
    """Optional reference to the simulation entity for this object."""

    def __post_init__(self) -> None:
        # Only copy the label onto the affordance for convenience. DO NOT
        # alias the geometry dict — that was the bug fixed by this redesign.
        self.affordance.object_label = self.label


# =============================================================================
# Typed targets
# =============================================================================


@dataclass(frozen=True)
class PoseTarget:
    """End-effector pose target. Used by MoveAction and PlaceAction."""

    xpos: torch.Tensor
    """(4, 4) or (n_envs, 4, 4) homogeneous transform."""


@dataclass(frozen=True)
class GraspTarget:
    """Pickup target. The grasp pose is solved from the affordance + entity at execute time."""

    semantics: ObjectSemantics


@dataclass(frozen=True)
class HeldObjectTarget:
    """Move the currently-held object to a desired object pose."""

    object_target_pose: torch.Tensor
    """(4, 4) or (n_envs, 4, 4) target pose for the held object."""


Target = Union[PoseTarget, GraspTarget, HeldObjectTarget]


# =============================================================================
# World state passed between actions
# =============================================================================


@dataclass
class HeldObjectState:
    """State of an object currently held by the robot."""

    semantics: ObjectSemantics
    """Semantics of the held object."""

    object_to_eef: torch.Tensor
    """Batched transform from object frame to end-effector frame, shape [n_envs, 4, 4]."""

    grasp_xpos: torch.Tensor
    """Batched end-effector pose used to grasp the object, shape [n_envs, 4, 4]."""


@dataclass
class WorldState:
    """State the engine threads through a sequence of actions."""

    last_qpos: torch.Tensor
    """Robot joint positions at the start of the next action, shape [n_envs, robot.dof]."""

    held_object: HeldObjectState | None = None
    """Object currently held by the gripper, or None."""


@dataclass
class ActionResult:
    """Return value of every AtomicAction.execute call."""

    success: bool
    """Whether the action produced a valid full-DoF trajectory."""

    trajectory: torch.Tensor
    """Full-robot trajectory, shape (n_envs, n_waypoints, robot.dof)."""

    next_state: WorldState
    """World state to feed into the next action."""


# =============================================================================
# Configuration base
# =============================================================================


@configclass
class ActionCfg:
    """Configuration shared by all atomic actions."""

    name: str = "default"
    control_part: str = "arm"
    interpolation_type: str = "linear"
    velocity_limit: float | None = None
    acceleration_limit: float | None = None


# =============================================================================
# AtomicAction ABC (slim)
# =============================================================================


class AtomicAction(ABC):
    """Abstract base for atomic actions.

    Subclasses declare ``TargetType`` to advertise the concrete target dataclass
    they accept. ``execute`` is the only required method; ``validate`` has been
    dropped from the contract in this redesign.
    """

    TargetType: ClassVar[type]
    """Concrete target dataclass accepted by ``execute``."""

    def __init__(
        self,
        motion_generator: "MotionGenerator",
        cfg: ActionCfg | None = None,
    ) -> None:
        self.motion_generator = motion_generator
        self.cfg = cfg if cfg is not None else ActionCfg()
        self.robot = motion_generator.robot
        self.device = self.robot.device
        self.control_part = self.cfg.control_part

    @abstractmethod
    def execute(self, target: Target, state: WorldState) -> ActionResult:
        """Plan and return a full-DoF trajectory for this action.

        Args:
            target: Typed target dataclass; must be an instance of ``self.TargetType``.
            state: World state inherited from the previous action (or the engine seed).

        Returns:
            ActionResult with the planned trajectory and the successor world state.
        """


__all__ = [
    "ActionCfg",
    "ActionResult",
    "AtomicAction",
    "GraspTarget",
    "HeldObjectState",
    "HeldObjectTarget",
    "ObjectSemantics",
    "PoseTarget",
    "Target",
    "WorldState",
]
```

- [ ] **Step 4: Run the core tests.**

  Run: `pytest tests/sim/atomic_actions/test_core.py -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 5: Verify imports across the package still resolve (other modules still import the old `Affordance`/`AntipodalAffordance` from `core.py`; those will be fixed in Tasks 3-7 — the test suite as a whole will be broken during this transition).**

  Run: `python -c "from embodichain.lab.sim.atomic_actions.core import AtomicAction, WorldState, ActionResult, PoseTarget, GraspTarget, HeldObjectTarget"`
  Expected: no error.

- [ ] **Step 6: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/core.py tests/sim/atomic_actions/test_core.py
git commit -m "Rewrite atomic_actions.core: typed targets, WorldState, ActionResult, slim ABC

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: Create `trajectory.py` (`TrajectoryBuilder` composition helper)

**Files:**
- Create: `embodichain/lab/sim/atomic_actions/trajectory.py`
- Create: `tests/sim/atomic_actions/test_trajectory.py`

Extract every helper currently sitting on `MoveAction` (the de-facto utility bag) into a stateless collaborator that the four concrete actions hold by composition. Each action keeps its own `TrajectoryBuilder` instance (matching today's per-action helper lifetime — confirmed by user in design review).

The source helpers to relocate are in the current `embodichain/lab/sim/atomic_actions/actions.py`:

| Source method on `MoveAction` | New name in `TrajectoryBuilder` | Source line range |
|---|---|---|
| `_all_envs_success` | `all_envs_success` | 101–105 |
| `_resolve_pose_target` | `resolve_pose_target` | 107–135 |
| `_resolve_start_qpos` | `resolve_start_qpos` | 137–153 |
| `_compute_three_phase_waypoints` | `split_three_phase` | 155–183 |
| `_build_motion_gen_options` | `build_motion_gen_options` | 185–200 |
| `_plan_arm_trajectory` | `plan_arm_traj` | 202–244 |
| `_interpolate_hand_qpos` | `interpolate_hand_qpos` | 246–276 |

Plus, from the current `AtomicAction` in `core.py`:

| Source method on `AtomicAction` | New name in `TrajectoryBuilder` | Source line range |
|---|---|---|
| `_ik_solve` | `ik_solve` | 439–466 |
| `_fk_compute` | `fk_compute` | 468–486 |
| `_apply_offset` | `apply_local_offset` | 488–506 |

And from `_HandCloseAction` in `actions.py`:

| Source method on `_HandCloseAction` | New name in `TrajectoryBuilder` | Source line range |
|---|---|---|
| `_expand_hand_qpos` | `expand_hand_qpos` | 343–355 |
| `_repeat_hand_qpos` | `repeat_hand_qpos` | 357–361 |

The bodies are kept verbatim. Only differences in signature:

- Methods no longer reference `self.cfg`, `self.n_envs`, `self.dof`, or `self.arm_joint_ids` (those lived on `MoveAction`). Instead they take the value as an argument.
- `plan_arm_traj` and `resolve_start_qpos` previously read `self.cfg.control_part` and `self.dof` — these become explicit `control_part: str` and `arm_dof: int` kwargs.
- `build_motion_gen_options` previously read `self.cfg.control_part` — same fix.

- [ ] **Step 1: Write the failing test file `tests/sim/atomic_actions/test_trajectory.py`.**

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (see top of file).
# ----------------------------------------------------------------------------

"""Tests for atomic_actions.trajectory.TrajectoryBuilder."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.trajectory import TrajectoryBuilder


def _make_mock_motion_generator(num_envs: int = 2, arm_dof: int = 6) -> Mock:
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = arm_dof

    def get_qpos(name=None):
        return torch.zeros(num_envs, arm_dof)

    robot.get_qpos = get_qpos

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(num_envs, arm_dof)
        return torch.ones(num_envs, dtype=torch.bool), seed.clone()

    robot.compute_ik = compute_ik

    def compute_fk(qpos=None, name=None, to_matrix=True):
        n = qpos.shape[0] if qpos is not None else num_envs
        return torch.eye(4).unsqueeze(0).repeat(n, 1, 1)

    robot.compute_fk = compute_fk

    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    return mg


class TestAllEnvsSuccess:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_python_bool_true(self):
        assert self.builder.all_envs_success(True) is True

    def test_python_bool_false(self):
        assert self.builder.all_envs_success(False) is False

    def test_tensor_all_true(self):
        assert self.builder.all_envs_success(torch.tensor([True, True])) is True

    def test_tensor_any_false(self):
        assert self.builder.all_envs_success(torch.tensor([True, False])) is False


class TestResolvePoseTarget:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_unbatched_pose_broadcasts(self):
        pose = torch.eye(4)
        out = self.builder.resolve_pose_target(pose, n_envs=2)
        assert out.shape == (2, 4, 4)

    def test_batched_pose_passes_through(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        out = self.builder.resolve_pose_target(pose, n_envs=2)
        assert torch.equal(out, pose)

    def test_wrong_shape_raises(self):
        with pytest.raises(Exception):
            self.builder.resolve_pose_target(torch.eye(3), n_envs=2)


class TestSplitThreePhase:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_default_ratio(self):
        a, b, c = self.builder.split_three_phase(80, 5)
        assert b == 5
        assert a + b + c == 80
        # First-phase ratio is 0.6 of remaining waypoints
        assert a == int(round((80 - 5) * 0.6))

    def test_raises_when_first_phase_too_small(self):
        with pytest.raises(Exception):
            self.builder.split_three_phase(6, 5)


class TestApplyLocalOffset:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_offset_adds_to_translation(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([0.0, 0.0, 0.1])
        out = self.builder.apply_local_offset(pose, offset)
        assert torch.allclose(out[:, :3, 3], torch.tensor([0.0, 0.0, 0.1]).expand(2, 3))

    def test_batched_offset(self):
        pose = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
        offset = torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])
        out = self.builder.apply_local_offset(pose, offset)
        assert torch.allclose(out[0, :3, 3], torch.tensor([0.1, 0.0, 0.0]))
        assert torch.allclose(out[1, :3, 3], torch.tensor([0.0, 0.2, 0.0]))


class TestExpandHandQpos:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_unbatched_expanded(self):
        q = torch.tensor([0.1, 0.2])
        out = self.builder.expand_hand_qpos(q, n_envs=3, hand_dof=2)
        assert out.shape == (3, 2)
        assert torch.allclose(out[0], q)

    def test_batched_passes_through(self):
        q = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        out = self.builder.expand_hand_qpos(q, n_envs=2, hand_dof=2)
        assert torch.equal(out, q)


class TestInterpolateHandQpos:
    def setup_method(self):
        self.builder = TrajectoryBuilder(_make_mock_motion_generator())

    def test_endpoints_match(self):
        a = torch.tensor([[0.0, 0.0]])
        b = torch.tensor([[1.0, 1.0]])
        out = self.builder.interpolate_hand_qpos(a, b, n_waypoints=5)
        assert torch.allclose(out[:, 0], a)
        assert torch.allclose(out[:, -1], b)
```

- [ ] **Step 2: Run tests to verify failure (module does not exist).**

  Run: `pytest tests/sim/atomic_actions/test_trajectory.py -x --no-header -q`
  Expected: `ModuleNotFoundError: No module named 'embodichain.lab.sim.atomic_actions.trajectory'`.

- [ ] **Step 3: Create `embodichain/lab/sim/atomic_actions/trajectory.py`.**

Use the standard Apache header. The body is a straight relocation of the helpers listed in the table above. Use the current `embodichain/lab/sim/atomic_actions/actions.py` lines 101–276 and `core.py` lines 439–506 as the source of method bodies — adapt the signatures per the spec.

```python
from __future__ import annotations

import numpy as np
import torch
from typing import List, Optional, TYPE_CHECKING

from embodichain.lab.sim.planners import PlanState
from embodichain.lab.sim.planners.motion_generator import MotionGenOptions
from embodichain.lab.sim.planners.toppra_planner import ToppraPlanOptions
from embodichain.lab.sim.utility.action_utils import interpolate_with_distance
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.planners import MotionGenerator


class TrajectoryBuilder:
    """Stateless trajectory utilities shared by every atomic action.

    Holds a reference to the motion generator (and through it, the robot + device)
    so callers don't have to thread those through each helper call. All methods
    are pure: no per-call state is kept.
    """

    def __init__(self, motion_generator: "MotionGenerator") -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = self.robot.device

    # ------------------------------------------------------------------
    # Success / shape helpers
    # ------------------------------------------------------------------

    def all_envs_success(self, is_success: bool | torch.Tensor) -> bool:
        if isinstance(is_success, torch.Tensor):
            return bool(torch.all(is_success).item())
        return bool(is_success)

    def resolve_pose_target(
        self, target: torch.Tensor, n_envs: int
    ) -> torch.Tensor:
        if not isinstance(target, torch.Tensor):
            logger.log_error(
                f"target must be torch.Tensor of shape (4, 4) or ({n_envs}, 4, 4)",
                TypeError,
            )
        if target.shape == (4, 4):
            target = target.unsqueeze(0).repeat(n_envs, 1, 1)
        if target.shape != (n_envs, 4, 4):
            logger.log_error(
                f"target tensor must have shape (4, 4) or ({n_envs}, 4, 4), "
                f"but got {target.shape}",
                ValueError,
            )
        return target

    def resolve_start_qpos(
        self,
        start_qpos: Optional[torch.Tensor],
        *,
        n_envs: int,
        arm_dof: int,
        control_part: str,
    ) -> torch.Tensor:
        if start_qpos is None:
            start_qpos = self.robot.get_qpos(name=control_part)
        if start_qpos.shape == (arm_dof,):
            start_qpos = start_qpos.unsqueeze(0).repeat(n_envs, 1)
        if start_qpos.shape != (n_envs, arm_dof):
            logger.log_error(
                f"start_qpos must have shape ({n_envs}, {arm_dof}), "
                f"but got {start_qpos.shape}",
                ValueError,
            )
        return start_qpos

    # ------------------------------------------------------------------
    # Pose math
    # ------------------------------------------------------------------

    def apply_local_offset(
        self, pose: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        if not (pose.dim() == 3 and pose.shape[1:] == (4, 4)):
            logger.log_error("pose must have shape [N, 4, 4]", ValueError)
        if offset.dim() == 1:
            offset = offset.unsqueeze(0)
        if not (offset.dim() == 2 and offset.shape[1] == 3):
            logger.log_error("offset must have shape [N, 3] or [3]", ValueError)
        result = pose.clone()
        result[:, :3, 3] += offset
        return result

    # ------------------------------------------------------------------
    # IK / FK convenience
    # ------------------------------------------------------------------

    def ik_solve(
        self,
        target_pose: torch.Tensor,
        *,
        control_part: str,
        qpos_seed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if qpos_seed is None:
            qpos_seed = self.robot.get_qpos()
        success, qpos = self.robot.compute_ik(
            pose=target_pose.unsqueeze(0),
            qpos_seed=qpos_seed.unsqueeze(0),
            name=control_part,
        )
        if not success.all():
            raise RuntimeError(f"IK failed for target pose: {target_pose}")
        return qpos.squeeze(0)

    def fk_compute(self, qpos: torch.Tensor, *, control_part: str) -> torch.Tensor:
        if qpos.dim() == 1:
            qpos = qpos.unsqueeze(0)
        xpos = self.robot.compute_fk(qpos=qpos, name=control_part, to_matrix=True)
        return xpos.squeeze(0) if xpos.shape[0] == 1 else xpos

    # ------------------------------------------------------------------
    # Waypoint splitting
    # ------------------------------------------------------------------

    def split_three_phase(
        self,
        sample_interval: int,
        hand_interp_steps: int,
        *,
        first_phase_ratio: float = 0.6,
        first_phase_name: str = "first",
        third_phase_name: str = "third",
    ) -> tuple[int, int, int]:
        first = int(np.round(sample_interval - hand_interp_steps) * first_phase_ratio)
        if first < 2:
            logger.log_error(
                f"Not enough waypoints for {first_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        second = hand_interp_steps
        third = sample_interval - first - second
        if third < 2:
            logger.log_error(
                f"Not enough waypoints for {third_phase_name} trajectory. "
                "Increase sample_interval or decrease hand_interp_steps.",
                ValueError,
            )
        return first, second, third

    # ------------------------------------------------------------------
    # MotionGen options
    # ------------------------------------------------------------------

    def build_motion_gen_options(
        self,
        start_qpos: torch.Tensor,
        *,
        sample_interval: int,
        control_part: str,
    ) -> MotionGenOptions:
        return MotionGenOptions(
            start_qpos=start_qpos[0],
            control_part=control_part,
            is_interpolate=True,
            is_linear=False,
            interpolate_position_step=0.001,
            plan_opts=ToppraPlanOptions(sample_interval=sample_interval),
        )

    # ------------------------------------------------------------------
    # Arm trajectory planning
    # ------------------------------------------------------------------

    def plan_arm_traj(
        self,
        target_states_list: list[list[PlanState]],
        start_qpos: torch.Tensor,
        n_waypoints: int,
        *,
        control_part: str,
        arm_dof: int,
    ) -> tuple[bool, torch.Tensor]:
        n_envs = start_qpos.shape[0]
        n_state = len(target_states_list[0])
        xpos_traj = torch.zeros(
            (n_envs, n_state, 4, 4), dtype=torch.float32, device=self.device
        )
        for i, target_states in enumerate(target_states_list):
            for j, target_state in enumerate(target_states):
                xpos_traj[i, j] = target_state.xpos

        trajectory = torch.zeros(
            (n_envs, n_state, arm_dof), dtype=torch.float32, device=self.device
        )
        qpos_seed = start_qpos
        for j in range(n_state):
            is_success, qpos = self.robot.compute_ik(
                pose=xpos_traj[:, j], name=control_part, joint_seed=qpos_seed
            )
            if not self.all_envs_success(is_success):
                logger.log_warning(
                    f"Failed to compute IK for target state {j} in some environments."
                )
                return False, trajectory
            trajectory[:, j] = qpos
            qpos_seed = qpos
        trajectory = torch.concatenate([start_qpos.unsqueeze(1), trajectory], dim=1)
        interp = interpolate_with_distance(
            trajectory=trajectory, interp_num=n_waypoints, device=self.device
        )
        return True, interp

    # ------------------------------------------------------------------
    # Hand qpos helpers
    # ------------------------------------------------------------------

    def expand_hand_qpos(
        self, hand_qpos: torch.Tensor, *, n_envs: int, hand_dof: int
    ) -> torch.Tensor:
        hand_qpos = hand_qpos.to(device=self.device, dtype=torch.float32)
        if hand_qpos.shape == (hand_dof,):
            return hand_qpos.unsqueeze(0).repeat(n_envs, 1)
        if hand_qpos.shape == (n_envs, hand_dof):
            return hand_qpos
        logger.log_error(
            f"hand_qpos must have shape ({hand_dof},) or ({n_envs}, {hand_dof}), "
            f"but got {hand_qpos.shape}",
            ValueError,
        )

    def repeat_hand_qpos(
        self,
        hand_qpos: torch.Tensor,
        *,
        n_envs: int,
        hand_dof: int,
        n_waypoints: int,
    ) -> torch.Tensor:
        return (
            self.expand_hand_qpos(hand_qpos, n_envs=n_envs, hand_dof=hand_dof)
            .unsqueeze(1)
            .repeat(1, n_waypoints, 1)
        )

    def interpolate_hand_qpos(
        self,
        start_hand_qpos: torch.Tensor,
        end_hand_qpos: torch.Tensor,
        n_waypoints: int,
    ) -> torch.Tensor:
        is_unbatched = start_hand_qpos.dim() == 1 and end_hand_qpos.dim() == 1
        start_hand_qpos = start_hand_qpos.to(self.device)
        end_hand_qpos = end_hand_qpos.to(self.device)
        if start_hand_qpos.dim() == 1:
            start_hand_qpos = start_hand_qpos.unsqueeze(0)
        if end_hand_qpos.dim() == 1:
            end_hand_qpos = end_hand_qpos.unsqueeze(0)
        weights = torch.linspace(
            0, 1, steps=n_waypoints, device=self.device, dtype=start_hand_qpos.dtype
        )
        result = torch.lerp(
            start_hand_qpos.unsqueeze(1),
            end_hand_qpos.unsqueeze(1),
            weights[None, :, None],
        )
        if is_unbatched:
            return result.squeeze(0)
        return result


__all__ = ["TrajectoryBuilder"]
```

- [ ] **Step 4: Run tests.**

  Run: `pytest tests/sim/atomic_actions/test_trajectory.py -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 5: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/trajectory.py tests/sim/atomic_actions/test_trajectory.py
git commit -m "Add TrajectoryBuilder composition helper

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: Rewrite `actions.py` — four sibling actions

**Files:**
- Replace: `embodichain/lab/sim/atomic_actions/actions.py`
- Replace: `tests/sim/atomic_actions/test_actions.py`

This is the biggest single task. After it, `actions.py` contains exactly four classes (`MoveAction`, `PickUpAction`, `MoveObjectAction`, `PlaceAction`), each inheriting `AtomicAction` directly, each holding a `TrajectoryBuilder` as `self.builder`. `_HandCloseAction` is gone. The `*ActionCfg` classes lose their inheritance chain — fields that used to be shared via `GraspActionCfg`/`HandCloseActionCfg` are duplicated explicitly into the concrete cfg classes (simpler than mixins; matches `@configclass` conventions). The "Open Questions" item §9.1 in the spec is hereby resolved in favour of explicit field repetition.

The new return shape: every `execute` returns `ActionResult(success, trajectory, next_state)` where `trajectory` is full-robot DoF shaped `(n_envs, n_waypoints, robot.dof)`. Each action assembles its arm-DoF slice via TrajectoryBuilder and pads with hand qpos into the full-DoF columns using `robot.get_joint_ids("arm")` / `robot.get_joint_ids("hand")`.

- [ ] **Step 1: Replace `tests/sim/atomic_actions/test_actions.py` entirely** with new tests. Use the helper layout from the current `tests/sim/atomic_actions/test_actions.py` lines 47–100 as a starting point for the mock robot factory — the redesign keeps the same dimensions (NUM_ENVS=2, ARM_DOF=6, HAND_DOF=2, TOTAL_DOF=8).

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (see top of file).
# ----------------------------------------------------------------------------

"""Tests for the four concrete atomic action classes."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock, patch

from embodichain.lab.sim.atomic_actions.affordance import (
    AntipodalAffordance,
)
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.actions import (
    MoveAction,
    MoveActionCfg,
    MoveObjectAction,
    MoveObjectActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)


NUM_ENVS = 2
ARM_DOF = 6
HAND_DOF = 2
TOTAL_DOF = ARM_DOF + HAND_DOF


def _make_mock_robot():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = TOTAL_DOF

    def get_qpos(name=None):
        if name == "arm":
            return torch.zeros(NUM_ENVS, ARM_DOF)
        if name == "hand":
            return torch.zeros(NUM_ENVS, HAND_DOF)
        return torch.zeros(NUM_ENVS, TOTAL_DOF)

    robot.get_qpos = get_qpos

    def get_joint_ids(name=None):
        if name == "arm":
            return list(range(ARM_DOF))
        if name == "hand":
            return list(range(ARM_DOF, TOTAL_DOF))
        return list(range(TOTAL_DOF))

    robot.get_joint_ids = get_joint_ids

    def compute_ik(pose=None, qpos_seed=None, name=None, joint_seed=None):
        seed = joint_seed if joint_seed is not None else qpos_seed
        if seed is None:
            seed = torch.zeros(NUM_ENVS, ARM_DOF)
        return torch.ones(NUM_ENVS, dtype=torch.bool), seed.clone()

    robot.compute_ik = compute_ik

    def compute_batch_ik(pose=None, name=None, joint_seed=None):
        if joint_seed is not None:
            return (
                torch.ones(joint_seed.shape[:2], dtype=torch.bool),
                joint_seed.clone(),
            )
        return torch.ones(NUM_ENVS, dtype=torch.bool), torch.zeros(NUM_ENVS, ARM_DOF)

    robot.compute_batch_ik = compute_batch_ik

    def compute_fk(qpos=None, name=None, to_matrix=True):
        n = qpos.shape[0] if qpos is not None else NUM_ENVS
        return torch.eye(4).unsqueeze(0).repeat(n, 1, 1)

    robot.compute_fk = compute_fk
    return robot


def _make_mock_motion_generator():
    mg = Mock()
    mg.robot = _make_mock_robot()
    mg.device = torch.device("cpu")
    return mg


def _hand_open():
    return torch.zeros(HAND_DOF, dtype=torch.float32)


def _hand_close():
    return torch.full((HAND_DOF,), 0.025, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MoveAction
# ---------------------------------------------------------------------------


class TestMoveAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert MoveAction.TargetType is PoseTarget

    def test_execute_returns_full_dof_trajectory(self):
        action = MoveAction(self.mg, MoveActionCfg(sample_interval=10))
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
        ):
            state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
            result = action.execute(PoseTarget(xpos=torch.eye(4)), state)
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        # Move doesn't touch held_object
        assert result.next_state.held_object is None


# ---------------------------------------------------------------------------
# PickUpAction
# ---------------------------------------------------------------------------


class TestPickUpAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_grasp_target(self):
        assert PickUpAction.TargetType is GraspTarget

    def test_execute_populates_held_object_state(self):
        cfg = PickUpActionCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = PickUpAction(self.mg, cfg)

        # Fake affordance that returns a single identity grasp pose.
        affordance = AntipodalAffordance()
        affordance.get_valid_grasp_poses = Mock(
            return_value=[
                (torch.eye(4).unsqueeze(0), torch.tensor([0.5]))
                for _ in range(NUM_ENVS)
            ]
        )

        entity = Mock()
        entity.get_local_pose = Mock(
            return_value=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1)
        )

        sem = ObjectSemantics(
            affordance=affordance,
            geometry={},
            label="mug",
            entity=entity,
        )

        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=lambda trajectory, interp_num, device: torch.zeros(
                NUM_ENVS, interp_num, ARM_DOF
            ),
        ):
            state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
            result = action.execute(GraspTarget(semantics=sem), state)
        assert result.success is True
        assert result.trajectory.shape[0] == NUM_ENVS
        assert result.trajectory.shape[2] == TOTAL_DOF
        assert isinstance(result.next_state.held_object, HeldObjectState)
        assert result.next_state.held_object.semantics is sem


# ---------------------------------------------------------------------------
# MoveObjectAction
# ---------------------------------------------------------------------------


class TestMoveObjectAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_held_object_target(self):
        assert MoveObjectAction.TargetType is HeldObjectTarget

    def test_requires_held_object_in_state(self):
        cfg = MoveObjectActionCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveObjectAction(self.mg, cfg)
        state = WorldState(last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF))
        with pytest.raises(Exception):
            action.execute(HeldObjectTarget(object_target_pose=torch.eye(4)), state)

    def test_preserves_held_object(self):
        cfg = MoveObjectActionCfg(
            hand_close_qpos=_hand_close(),
            sample_interval=10,
        )
        action = MoveObjectAction(self.mg, cfg)
        sem = ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="mug")
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )
        state = WorldState(
            last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF), held_object=held
        )
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            return_value=torch.zeros(NUM_ENVS, 10, ARM_DOF),
        ):
            result = action.execute(
                HeldObjectTarget(object_target_pose=torch.eye(4)), state
            )
        assert result.success is True
        assert result.trajectory.shape == (NUM_ENVS, 10, TOTAL_DOF)
        assert result.next_state.held_object is held


# ---------------------------------------------------------------------------
# PlaceAction
# ---------------------------------------------------------------------------


class TestPlaceAction:
    def setup_method(self):
        self.mg = _make_mock_motion_generator()

    def test_target_type_is_pose_target(self):
        assert PlaceAction.TargetType is PoseTarget

    def test_execute_clears_held_object(self):
        cfg = PlaceActionCfg(
            hand_open_qpos=_hand_open(),
            hand_close_qpos=_hand_close(),
            sample_interval=20,
            hand_interp_steps=4,
        )
        action = PlaceAction(self.mg, cfg)
        sem = ObjectSemantics(affordance=AntipodalAffordance(), geometry={}, label="mug")
        held = HeldObjectState(
            semantics=sem,
            object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
        )
        state = WorldState(
            last_qpos=torch.zeros(NUM_ENVS, TOTAL_DOF), held_object=held
        )
        with patch(
            "embodichain.lab.sim.atomic_actions.trajectory.interpolate_with_distance",
            side_effect=lambda trajectory, interp_num, device: torch.zeros(
                NUM_ENVS, interp_num, ARM_DOF
            ),
        ):
            result = action.execute(PoseTarget(xpos=torch.eye(4)), state)
        assert result.success is True
        assert result.trajectory.shape[2] == TOTAL_DOF
        assert result.next_state.held_object is None
```

- [ ] **Step 2: Run tests; expect import errors / failures.**

  Run: `pytest tests/sim/atomic_actions/test_actions.py -x --no-header -q`
  Expected: failures because `actions.py` still has the old shape (the old `MoveAction.execute` returns a tuple, not `ActionResult`; cfgs still inherit `MoveActionCfg`, etc.).

- [ ] **Step 3: Replace `embodichain/lab/sim/atomic_actions/actions.py`.**

Apache header first. Then the body below. Method bodies that overlap with the old `actions.py` are kept logically identical — only the entry/exit ceremony changes:

- Old: `execute(target, start_qpos=None, **kwargs) -> (bool, Tensor, joint_ids)`
- New: `execute(target: TargetType, state: WorldState) -> ActionResult`
- Old: arm-only trajectory or arm+hand trajectory depending on action; caller re-indexes via `joint_ids`.
- New: every action builds a `(n_envs, n_waypoints, robot.dof)` tensor internally and returns it.

```python
from __future__ import annotations

import torch
from typing import ClassVar, Optional

from embodichain.lab.sim.planners import PlanState, MoveType
from embodichain.utils import configclass, logger
from embodichain.utils.math import pose_inv

from .affordance import AntipodalAffordance
from .core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)
from .trajectory import TrajectoryBuilder


# =============================================================================
# Cfg classes (flat — no inheritance among GraspActionCfg/HandCloseActionCfg)
# =============================================================================


@configclass
class MoveActionCfg(ActionCfg):
    name: str = "move"
    sample_interval: int = 50
    """Number of waypoints in the planned trajectory."""


@configclass
class PickUpActionCfg(ActionCfg):
    name: str = "pick_up"
    sample_interval: int = 80
    hand_interp_steps: int = 5
    hand_control_part: str = "hand"
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    lift_height: float = 0.1
    pre_grasp_distance: float = 0.15
    approach_direction: torch.Tensor = torch.tensor([0, 0, -1], dtype=torch.float32)


@configclass
class MoveObjectActionCfg(ActionCfg):
    name: str = "move_object"
    sample_interval: int = 50
    hand_control_part: str = "hand"
    hand_close_qpos: torch.Tensor | None = None


@configclass
class PlaceActionCfg(ActionCfg):
    name: str = "place"
    sample_interval: int = 80
    hand_interp_steps: int = 5
    hand_control_part: str = "hand"
    hand_open_qpos: torch.Tensor | None = None
    hand_close_qpos: torch.Tensor | None = None
    lift_height: float = 0.1


# =============================================================================
# Shared helpers private to this module
# =============================================================================


def _resolve_object_target(
    target: torch.Tensor, *, n_envs: int, device: torch.device
) -> torch.Tensor:
    target = target.to(device=device, dtype=torch.float32)
    if target.shape == (4, 4):
        target = target.unsqueeze(0).repeat(n_envs, 1, 1)
    if target.shape != (n_envs, 4, 4):
        logger.log_error(
            f"object_target_pose must be (4, 4) or ({n_envs}, 4, 4), but got {target.shape}",
            ValueError,
        )
    return target


# =============================================================================
# MoveAction
# =============================================================================


class MoveAction(AtomicAction):
    """Plan a free-space end-effector move to a target pose."""

    TargetType: ClassVar[type] = PoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveActionCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.robot_dof = self.robot.dof

    def execute(self, target: PoseTarget, state: WorldState) -> ActionResult:
        move_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        start_qpos = self.builder.resolve_start_qpos(
            state.last_qpos[:, self.arm_joint_ids]
            if state.last_qpos.shape[1] == self.robot_dof
            else state.last_qpos,
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        target_states_list = [
            [PlanState(xpos=move_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )
        full = self._embed_into_full_dof(arm_traj, state.last_qpos)
        next_state = WorldState(
            last_qpos=full[:, -1, :].clone(), held_object=state.held_object
        )
        return ActionResult(success=True, trajectory=full, next_state=next_state)

    def _embed_into_full_dof(
        self, arm_traj: torch.Tensor, last_full_qpos: torch.Tensor
    ) -> torch.Tensor:
        n_wp = arm_traj.shape[1]
        full = torch.empty(
            (self.n_envs, n_wp, self.robot_dof), dtype=torch.float32, device=self.device
        )
        # Pad every column from last_full_qpos, then overwrite arm columns.
        full[:, :, :] = last_full_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        return full
```

The next part of the file continues. Add it to the same file:

```python
# =============================================================================
# PickUpAction
# =============================================================================


class PickUpAction(AtomicAction):
    """Approach a grasp pose, close the gripper, lift."""

    TargetType: ClassVar[type] = GraspTarget

    def __init__(
        self,
        motion_generator,
        cfg: PickUpActionCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PickUpActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PickUpActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PickUpActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)
        self.approach_direction = self.cfg.approach_direction.to(self.device)

    def execute(self, target: GraspTarget, state: WorldState) -> ActionResult:
        sem = target.semantics
        if not isinstance(sem.affordance, AntipodalAffordance):
            logger.log_error(
                "PickUpAction requires an AntipodalAffordance on the target semantics."
            )
        if sem.entity is None:
            logger.log_error(
                "PickUpAction requires an entity on the target semantics."
            )

        is_success, grasp_xpos = self._resolve_grasp_pose(sem)
        if not self.builder.all_envs_success(is_success):
            logger.log_warning("PickUpAction failed to resolve a grasp pose.")
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )

        # Pre-grasp by offsetting backwards along grasp z.
        grasp_z = grasp_xpos[:, :3, 2]
        pre_grasp_xpos = self.builder.apply_local_offset(
            grasp_xpos, -grasp_z * self.cfg.pre_grasp_distance
        )

        # Start qpos comes from the threaded WorldState.
        if state.last_qpos.shape[1] == self.robot_dof:
            start_arm_qpos = state.last_qpos[:, self.arm_joint_ids]
        else:
            start_arm_qpos = state.last_qpos
        start_arm_qpos = self.builder.resolve_start_qpos(
            start_arm_qpos,
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        n_approach, n_close, n_lift = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="lift",
        )

        # Phase 1: approach
        target_states_list = [
            [
                PlanState(xpos=pre_grasp_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=grasp_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        ok, approach_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_approach,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUpAction failed to plan the approach trajectory.")
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )

        # Phase 3: lift (planned from grasp qpos)
        grasp_arm_qpos = approach_arm[:, -1, :]
        lift_xpos = self.builder.apply_local_offset(
            grasp_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, lift_arm = self.builder.plan_arm_traj(
            target_states_list,
            grasp_arm_qpos,
            n_lift,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("PickUpAction failed to plan the lift trajectory.")
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )

        # Phase 2: hand close (arm held at grasp qpos)
        hand_close_path = self.builder.interpolate_hand_qpos(
            self.hand_open_qpos, self.hand_close_qpos, n_close
        )

        # Assemble full-DoF trajectory
        full = torch.empty(
            (self.n_envs, n_approach + n_close + n_lift, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)

        # Approach: arm planned, hand open
        full[:, :n_approach, self.arm_joint_ids] = approach_arm
        full[:, :n_approach, self.hand_joint_ids] = self.hand_open_qpos
        # Close: arm held at grasp, hand interpolating
        full[:, n_approach : n_approach + n_close, self.arm_joint_ids] = (
            grasp_arm_qpos.unsqueeze(1)
        )
        full[:, n_approach : n_approach + n_close, self.hand_joint_ids] = (
            hand_close_path
        )
        # Lift: arm planned, hand closed
        full[:, n_approach + n_close :, self.arm_joint_ids] = lift_arm
        full[:, n_approach + n_close :, self.hand_joint_ids] = self.hand_close_qpos

        # Record held-object state
        obj_poses = sem.entity.get_local_pose(to_matrix=True)
        object_to_eef = torch.bmm(pose_inv(obj_poses), grasp_xpos)
        held = HeldObjectState(
            semantics=sem, object_to_eef=object_to_eef, grasp_xpos=grasp_xpos
        )
        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(), held_object=held
            ),
        )

    def _resolve_grasp_pose(
        self, semantics: ObjectSemantics
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Body kept verbatim from current actions.py:552-605. Logic is unchanged:
        # build per-env candidate-pose padding, batch-IK, select min-cost feasible.
        obj_poses = semantics.entity.get_local_pose(to_matrix=True)
        grasp_poses_result = semantics.affordance.get_valid_grasp_poses(
            obj_poses=obj_poses, approach_direction=self.approach_direction
        )
        n_envs = obj_poses.shape[0]
        init_qpos = self.robot.get_qpos(name=self.cfg.control_part)
        n_max_pose = max(r[0].shape[0] for r in grasp_poses_result)
        grasp_xpos_padding = torch.zeros(
            (n_envs, n_max_pose, 4, 4), dtype=torch.float32, device=self.device
        )
        grasp_cost_padding = torch.full(
            (n_envs, n_max_pose),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        for i in range(n_envs):
            n_pose = grasp_poses_result[i][0].shape[0]
            grasp_xpos_padding[i, :n_pose] = grasp_poses_result[i][0]
            grasp_cost_padding[i, :n_pose] = grasp_poses_result[i][1]
            grasp_xpos_padding[i, n_pose:] = grasp_poses_result[i][0][0]
            grasp_cost_padding[i, n_pose:] = grasp_poses_result[i][1][0]
        init_qpos_repeat = init_qpos[:, None, :].repeat(1, n_max_pose, 1)
        ik_success, _ = self.robot.compute_batch_ik(
            pose=grasp_xpos_padding,
            name=self.cfg.control_part,
            joint_seed=init_qpos_repeat,
        )
        grasp_cost_masked = torch.where(ik_success, grasp_cost_padding, 10000.0)
        best_cost, best_idx = grasp_cost_masked.min(dim=1)
        is_success = best_cost < 9999.0
        best_grasp_xpos = grasp_xpos_padding[
            torch.arange(n_envs, device=self.device), best_idx
        ]
        return is_success, best_grasp_xpos
```

Then `MoveObjectAction` and `PlaceAction`:

```python
# =============================================================================
# MoveObjectAction
# =============================================================================


class MoveObjectAction(AtomicAction):
    """Move the held object to a target object pose; keep the gripper closed."""

    TargetType: ClassVar[type] = HeldObjectTarget

    def __init__(
        self,
        motion_generator,
        cfg: MoveObjectActionCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or MoveObjectActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in MoveObjectActionCfg")
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def execute(self, target: HeldObjectTarget, state: WorldState) -> ActionResult:
        if state.held_object is None:
            logger.log_error(
                "MoveObjectAction requires WorldState.held_object — run PickUpAction first.",
                ValueError,
            )
        object_target_pose = _resolve_object_target(
            target.object_target_pose, n_envs=self.n_envs, device=self.device
        )
        object_to_eef = state.held_object.object_to_eef.to(
            device=self.device, dtype=torch.float32
        )
        if object_to_eef.shape == (4, 4):
            object_to_eef = object_to_eef.unsqueeze(0).repeat(self.n_envs, 1, 1)
        move_eef_xpos = torch.bmm(object_target_pose, object_to_eef)

        if state.last_qpos.shape[1] == self.robot_dof:
            start_arm_qpos = state.last_qpos[:, self.arm_joint_ids]
        else:
            start_arm_qpos = state.last_qpos
        start_arm_qpos = self.builder.resolve_start_qpos(
            start_arm_qpos,
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )

        target_states_list = [
            [PlanState(xpos=move_eef_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, arm_traj = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            self.cfg.sample_interval,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            logger.log_warning("MoveObjectAction failed to plan trajectory.")
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )

        full = torch.empty(
            (self.n_envs, self.cfg.sample_interval, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :, self.arm_joint_ids] = arm_traj
        full[:, :, self.hand_joint_ids] = self.hand_close_qpos

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(),
                held_object=state.held_object,
            ),
        )


# =============================================================================
# PlaceAction
# =============================================================================


class PlaceAction(AtomicAction):
    """Lower the held object to a place pose, open the gripper, retract."""

    TargetType: ClassVar[type] = PoseTarget

    def __init__(
        self,
        motion_generator,
        cfg: PlaceActionCfg | None = None,
    ) -> None:
        super().__init__(motion_generator, cfg or PlaceActionCfg())
        self.builder = TrajectoryBuilder(motion_generator)
        self.n_envs = self.robot.get_qpos().shape[0]
        self.arm_joint_ids = self.robot.get_joint_ids(name=self.cfg.control_part)
        self.hand_joint_ids = self.robot.get_joint_ids(name=self.cfg.hand_control_part)
        self.arm_dof = len(self.arm_joint_ids)
        self.hand_dof = len(self.hand_joint_ids)
        self.robot_dof = self.robot.dof

        if self.cfg.hand_open_qpos is None:
            logger.log_error("hand_open_qpos must be specified in PlaceActionCfg")
        if self.cfg.hand_close_qpos is None:
            logger.log_error("hand_close_qpos must be specified in PlaceActionCfg")
        self.hand_open_qpos = self.cfg.hand_open_qpos.to(self.device)
        self.hand_close_qpos = self.cfg.hand_close_qpos.to(self.device)

    def execute(self, target: PoseTarget, state: WorldState) -> ActionResult:
        place_xpos = self.builder.resolve_pose_target(target.xpos, n_envs=self.n_envs)
        if state.last_qpos.shape[1] == self.robot_dof:
            start_arm_qpos = state.last_qpos[:, self.arm_joint_ids]
        else:
            start_arm_qpos = state.last_qpos
        start_arm_qpos = self.builder.resolve_start_qpos(
            start_arm_qpos,
            n_envs=self.n_envs,
            arm_dof=self.arm_dof,
            control_part=self.cfg.control_part,
        )
        n_down, n_open, n_back = self.builder.split_three_phase(
            self.cfg.sample_interval,
            self.cfg.hand_interp_steps,
            first_phase_name="approach",
            third_phase_name="back",
        )

        lift_xpos = self.builder.apply_local_offset(
            place_xpos,
            torch.tensor([0, 0, 1], device=self.device) * self.cfg.lift_height,
        )

        # Phase 1: down (lift → place)
        target_states_list = [
            [
                PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE),
                PlanState(xpos=place_xpos[i], move_type=MoveType.EEF_MOVE),
            ]
            for i in range(self.n_envs)
        ]
        ok, down_arm = self.builder.plan_arm_traj(
            target_states_list,
            start_arm_qpos,
            n_down,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )
        reach_arm_qpos = down_arm[:, -1, :]

        # Phase 3: back (retract to lift)
        target_states_list = [
            [PlanState(xpos=lift_xpos[i], move_type=MoveType.EEF_MOVE)]
            for i in range(self.n_envs)
        ]
        ok, back_arm = self.builder.plan_arm_traj(
            target_states_list,
            reach_arm_qpos,
            n_back,
            control_part=self.cfg.control_part,
            arm_dof=self.arm_dof,
        )
        if not ok:
            return ActionResult(
                success=False,
                trajectory=torch.empty(
                    (self.n_envs, 0, self.robot_dof),
                    dtype=torch.float32,
                    device=self.device,
                ),
                next_state=state,
            )

        # Phase 2: hand open (arm held at reach qpos)
        hand_open_path = self.builder.interpolate_hand_qpos(
            self.hand_close_qpos, self.hand_open_qpos, n_open
        )

        full = torch.empty(
            (self.n_envs, n_down + n_open + n_back, self.robot_dof),
            dtype=torch.float32,
            device=self.device,
        )
        full[:, :, :] = state.last_qpos.unsqueeze(1)
        full[:, :n_down, self.arm_joint_ids] = down_arm
        full[:, :n_down, self.hand_joint_ids] = self.hand_close_qpos
        full[:, n_down : n_down + n_open, self.arm_joint_ids] = reach_arm_qpos.unsqueeze(1)
        full[:, n_down : n_down + n_open, self.hand_joint_ids] = hand_open_path
        full[:, n_down + n_open :, self.arm_joint_ids] = back_arm
        full[:, n_down + n_open :, self.hand_joint_ids] = self.hand_open_qpos

        return ActionResult(
            success=True,
            trajectory=full,
            next_state=WorldState(
                last_qpos=full[:, -1, :].clone(), held_object=None
            ),
        )


__all__ = [
    "MoveAction",
    "MoveActionCfg",
    "MoveObjectAction",
    "MoveObjectActionCfg",
    "PickUpAction",
    "PickUpActionCfg",
    "PlaceAction",
    "PlaceActionCfg",
]
```

- [ ] **Step 4: Run the action tests.**

  Run: `pytest tests/sim/atomic_actions/test_actions.py -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 5: Run the full atomic_actions test suite (the engine tests will still fail because Task 5 hasn't run yet — that's expected).**

  Run: `pytest tests/sim/atomic_actions/test_core.py tests/sim/atomic_actions/test_affordance.py tests/sim/atomic_actions/test_actions.py tests/sim/atomic_actions/test_trajectory.py -x --no-header -q`
  Expected: all four files pass.

- [ ] **Step 6: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/actions.py tests/sim/atomic_actions/test_actions.py
git commit -m "Rewrite atomic actions as siblings with composition

- MoveAction, PickUpAction, MoveObjectAction, PlaceAction all inherit
  AtomicAction directly. _HandCloseAction is removed.
- Each holds a TrajectoryBuilder for shared helpers.
- execute() takes a typed target + WorldState and returns ActionResult
  with a full-DoF trajectory.
- Cfg classes are flat (no inheritance among Grasp/HandClose variants).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: Rewrite `engine.py` — name-keyed `run(steps, state)`

**Files:**
- Replace: `embodichain/lab/sim/atomic_actions/engine.py`
- Replace: `tests/sim/atomic_actions/test_engine.py`

The new engine keeps the global module-level registry (`register_action`, `unregister_action`, `get_registered_actions`) but drops `SemanticAnalyzer`, `_resolve_target`, `_action_context`, the `actions_cfg_list` constructor arg, and `execute_static`. The new instance API is:

```python
engine = AtomicActionEngine(motion_generator)
engine.register(PickUpAction(motion_generator, cfg=...))
engine.register(MoveObjectAction(motion_generator, cfg=...))
engine.register(PlaceAction(motion_generator, cfg=...))
success, trajectory, final_state = engine.run([
    ("pick_up", GraspTarget(...)),
    ("move_object", HeldObjectTarget(...)),
    ("place", PoseTarget(...)),
])
```

- [ ] **Step 1: Replace `tests/sim/atomic_actions/test_engine.py` entirely.**

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (see top of file).
# ----------------------------------------------------------------------------

"""Tests for atomic_actions.engine."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.atomic_actions.affordance import Affordance
from embodichain.lab.sim.atomic_actions.core import (
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    WorldState,
)
from embodichain.lab.sim.atomic_actions.engine import (
    AtomicActionEngine,
    get_registered_actions,
    register_action,
    unregister_action,
)


# ---------------------------------------------------------------------------
# Global registry (kept from old design)
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    def teardown_method(self):
        unregister_action("_test_dummy")

    def test_register_and_retrieve(self):
        cls = Mock()
        register_action("_test_dummy", cls)
        assert get_registered_actions()["_test_dummy"] is cls

    def test_unregister(self):
        register_action("_test_dummy", Mock())
        unregister_action("_test_dummy")
        assert "_test_dummy" not in get_registered_actions()

    def test_unregister_nonexistent_is_noop(self):
        unregister_action("_does_not_exist")

    def test_get_registered_actions_returns_copy(self):
        out = get_registered_actions()
        out["_should_not_persist"] = Mock()
        assert "_should_not_persist" not in get_registered_actions()


# ---------------------------------------------------------------------------
# Engine run() semantics
# ---------------------------------------------------------------------------


NUM_ENVS = 2
TOTAL_DOF = 8


def _make_mg():
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.dof = TOTAL_DOF
    robot.get_qpos.return_value = torch.zeros(NUM_ENVS, TOTAL_DOF)

    mg = Mock()
    mg.robot = robot
    mg.device = torch.device("cpu")
    return mg


def _fake_action(name: str, target_type: type, *, sets_held=False, clears_held=False, fails=False):
    action = Mock(spec=AtomicAction)
    action.TargetType = target_type
    action.cfg = Mock()
    action.cfg.name = name

    def execute(target, state):
        if fails:
            return ActionResult(
                success=False,
                trajectory=torch.empty(NUM_ENVS, 0, TOTAL_DOF),
                next_state=state,
            )
        held = state.held_object
        if sets_held:
            sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="x")
            held = HeldObjectState(
                semantics=sem,
                object_to_eef=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
                grasp_xpos=torch.eye(4).unsqueeze(0).repeat(NUM_ENVS, 1, 1),
            )
        if clears_held:
            held = None
        traj = torch.zeros(NUM_ENVS, 5, TOTAL_DOF)
        return ActionResult(
            success=True,
            trajectory=traj,
            next_state=WorldState(
                last_qpos=traj[:, -1, :].clone(),
                held_object=held,
            ),
        )

    action.execute = Mock(side_effect=execute)
    return action


class TestEngineRun:
    def setup_method(self):
        self.mg = _make_mg()
        self.engine = AtomicActionEngine(self.mg)

    def test_register_and_lookup(self):
        action = _fake_action("pick_up", GraspTarget, sets_held=True)
        self.engine.register(action)
        assert "pick_up" in self.engine.actions

    def test_register_with_explicit_name_overrides_cfg(self):
        action = _fake_action("pick_up", GraspTarget, sets_held=True)
        self.engine.register(action, name="custom")
        assert "custom" in self.engine.actions

    def test_run_concatenates_trajectories(self):
        a = _fake_action("a", PoseTarget)
        b = _fake_action("b", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.register(b, name="b")
        ok, traj, _ = self.engine.run(
            [("a", PoseTarget(torch.eye(4))), ("b", PoseTarget(torch.eye(4)))]
        )
        assert ok is True
        assert traj.shape == (NUM_ENVS, 10, TOTAL_DOF)

    def test_run_threads_world_state(self):
        pick = _fake_action("pick", GraspTarget, sets_held=True)
        move = _fake_action("move", HeldObjectTarget)
        place = _fake_action("place", PoseTarget, clears_held=True)
        self.engine.register(pick, name="pick")
        self.engine.register(move, name="move")
        self.engine.register(place, name="place")
        sem = ObjectSemantics(affordance=Affordance(), geometry={}, label="x")
        ok, _, final_state = self.engine.run([
            ("pick", GraspTarget(sem)),
            ("move", HeldObjectTarget(torch.eye(4))),
            ("place", PoseTarget(torch.eye(4))),
        ])
        assert ok is True
        # The move action saw a non-None held_object (set by pick).
        move_state_arg = move.execute.call_args_list[0].args[1]
        assert move_state_arg.held_object is not None
        # Final state cleared by place.
        assert final_state.held_object is None

    def test_run_stops_on_first_failure(self):
        a = _fake_action("a", PoseTarget)
        b = _fake_action("b", PoseTarget, fails=True)
        c = _fake_action("c", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.register(b, name="b")
        self.engine.register(c, name="c")
        ok, traj, _ = self.engine.run(
            [
                ("a", PoseTarget(torch.eye(4))),
                ("b", PoseTarget(torch.eye(4))),
                ("c", PoseTarget(torch.eye(4))),
            ]
        )
        assert ok is False
        # `c` should not have been called.
        c.execute.assert_not_called()
        # We still get back the partial trajectory accumulated from `a`.
        assert traj.shape == (NUM_ENVS, 5, TOTAL_DOF)

    def test_run_raises_on_unknown_action_name(self):
        with pytest.raises(KeyError, match="ghost"):
            self.engine.run([("ghost", PoseTarget(torch.eye(4)))])

    def test_run_raises_on_target_type_mismatch(self):
        a = _fake_action("a", PoseTarget)
        self.engine.register(a, name="a")
        with pytest.raises(TypeError, match="target"):
            self.engine.run([("a", HeldObjectTarget(torch.eye(4)))])

    def test_run_seeds_state_from_robot_when_none_provided(self):
        a = _fake_action("a", PoseTarget)
        self.engine.register(a, name="a")
        self.engine.run([("a", PoseTarget(torch.eye(4)))])
        # First call's state argument
        state_arg = a.execute.call_args_list[0].args[1]
        assert state_arg.last_qpos.shape == (NUM_ENVS, TOTAL_DOF)
        assert state_arg.held_object is None
```

- [ ] **Step 2: Run the new engine tests; they will fail because the engine still has the old API.**

  Run: `pytest tests/sim/atomic_actions/test_engine.py -x --no-header -q`
  Expected: failures around `AtomicActionEngine.run` / missing constructor args / missing methods.

- [ ] **Step 3: Replace `embodichain/lab/sim/atomic_actions/engine.py` entirely.**

```python
from __future__ import annotations

import torch
from typing import Dict, Iterable, List, Optional, Tuple, Type

from embodichain.utils import logger

from .core import (
    ActionResult,
    AtomicAction,
    Target,
    WorldState,
)


# =============================================================================
# Global action registry (kept for third-party extensions)
# =============================================================================


_global_action_registry: Dict[str, Type[AtomicAction]] = {}


def register_action(name: str, action_class: Type[AtomicAction]) -> None:
    """Register a custom AtomicAction subclass globally under ``name``."""
    _global_action_registry[name] = action_class


def unregister_action(name: str) -> None:
    """Remove a previously-registered action class. No-op if absent."""
    _global_action_registry.pop(name, None)


def get_registered_actions() -> Dict[str, Type[AtomicAction]]:
    """Return a copy of the global action-class registry."""
    return _global_action_registry.copy()


# =============================================================================
# AtomicActionEngine
# =============================================================================


class AtomicActionEngine:
    """Sequences typed atomic actions while threading WorldState through them."""

    def __init__(self, motion_generator) -> None:
        self.motion_generator = motion_generator
        self.robot = motion_generator.robot
        self.device = motion_generator.device
        self._actions: Dict[str, AtomicAction] = {}

    @property
    def actions(self) -> Dict[str, AtomicAction]:
        """Dict of registered actions keyed by name (read-only view)."""
        return dict(self._actions)

    def register(self, action: AtomicAction, *, name: Optional[str] = None) -> None:
        """Register an action instance under ``name`` or its ``cfg.name``."""
        key = name if name is not None else action.cfg.name
        self._actions[key] = action

    def run(
        self,
        steps: Iterable[Tuple[str, Target]],
        state: Optional[WorldState] = None,
    ) -> Tuple[bool, torch.Tensor, WorldState]:
        """Run a sequence of named actions, threading WorldState through.

        Returns:
            (success, concatenated_full_dof_trajectory, final_state).

            On failure, ``success`` is False and the returned trajectory is the
            concatenation of all successful steps that completed before the
            failure. The failing step contributes no waypoints. ``final_state``
            is the state going INTO the failed step.
        """
        steps_list = list(steps)
        if state is None:
            state = WorldState(last_qpos=self.robot.get_qpos())

        full_traj = torch.empty(
            (state.last_qpos.shape[0], 0, self.robot.dof),
            dtype=torch.float32,
            device=self.device,
        )

        for name, target in steps_list:
            if name not in self._actions:
                raise KeyError(f"No action registered under name '{name}'")
            action = self._actions[name]
            if not isinstance(target, action.TargetType):
                raise TypeError(
                    f"Action '{name}' expects target of type "
                    f"{action.TargetType.__name__}, got {type(target).__name__}"
                )
            result: ActionResult = action.execute(target, state)
            if not result.success:
                return False, full_traj, state
            full_traj = torch.cat([full_traj, result.trajectory], dim=1)
            state = result.next_state

        return True, full_traj, state


__all__ = [
    "AtomicActionEngine",
    "get_registered_actions",
    "register_action",
    "unregister_action",
]
```

- [ ] **Step 4: Run the engine tests.**

  Run: `pytest tests/sim/atomic_actions/test_engine.py -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 5: Run the full atomic_actions test suite.**

  Run: `pytest tests/sim/atomic_actions/ -x --no-header -q`
  Expected: all four test files pass.

- [ ] **Step 6: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/engine.py tests/sim/atomic_actions/test_engine.py
git commit -m "Rewrite AtomicActionEngine with name-keyed run(steps, state)

Drops SemanticAnalyzer, _resolve_target, _action_context, execute_static.
Keeps the global register_action / unregister_action / get_registered_actions.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: Update `__init__.py` public exports

**Files:**
- Modify: `embodichain/lab/sim/atomic_actions/__init__.py`

The current `__init__.py` exports include `Affordance`/`AntipodalAffordance`/`InteractionPoints` from `core.py`, plus removed names (`MoveObjectTarget`). Update to:

- [ ] **Step 1: Replace the file contents (keep the Apache header at the top).**

```python
"""Atomic action abstraction layer for embodied AI motion generation.

This module provides a unified interface for atomic actions like reach,
pick up, move-object, and place, with support for typed targets and
threaded world state across sequenced actions.
"""

from .affordance import (
    Affordance,
    AntipodalAffordance,
    InteractionPoints,
)
from .core import (
    ActionCfg,
    ActionResult,
    AtomicAction,
    GraspTarget,
    HeldObjectState,
    HeldObjectTarget,
    ObjectSemantics,
    PoseTarget,
    Target,
    WorldState,
)
from .actions import (
    MoveAction,
    MoveActionCfg,
    MoveObjectAction,
    MoveObjectActionCfg,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
)
from .engine import (
    AtomicActionEngine,
    get_registered_actions,
    register_action,
    unregister_action,
)
from .trajectory import TrajectoryBuilder

__all__ = [
    # Affordances
    "Affordance",
    "AntipodalAffordance",
    "InteractionPoints",
    # Core primitives
    "ActionCfg",
    "ActionResult",
    "AtomicAction",
    "GraspTarget",
    "HeldObjectState",
    "HeldObjectTarget",
    "ObjectSemantics",
    "PoseTarget",
    "Target",
    "WorldState",
    # Actions + cfgs
    "MoveAction",
    "MoveActionCfg",
    "MoveObjectAction",
    "MoveObjectActionCfg",
    "PickUpAction",
    "PickUpActionCfg",
    "PlaceAction",
    "PlaceActionCfg",
    # Engine
    "AtomicActionEngine",
    "register_action",
    "unregister_action",
    "get_registered_actions",
    # Trajectory helper (for custom actions)
    "TrajectoryBuilder",
]
```

- [ ] **Step 2: Smoke test.**

  Run: `python -c "from embodichain.lab.sim.atomic_actions import AtomicActionEngine, PickUpAction, GraspTarget, HeldObjectTarget, PoseTarget, WorldState, ActionResult, TrajectoryBuilder, AntipodalAffordance"`
  Expected: no error.

- [ ] **Step 3: Run the full atomic_actions test suite.**

  Run: `pytest tests/sim/atomic_actions/ -x --no-header -q`
  Expected: all pass.

- [ ] **Step 4: Commit.**

```bash
git add embodichain/lab/sim/atomic_actions/__init__.py
git commit -m "Update atomic_actions public exports for redesign

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: Migrate `scripts/tutorials/sim/atomic_actions.py`

**Files:**
- Modify: `scripts/tutorials/sim/atomic_actions.py`

The tutorial currently constructs the engine with `actions_cfg_list=[pickup_cfg, place_cfg, move_cfg]` and calls `execute_static(target_list=[mug_semantics, place_xpos, rest_xpos])`. After this task it uses `engine.register(...)` per action and `engine.run([(name, typed_target), ...])`.

- [ ] **Step 1: Replace the import block (`scripts/tutorials/sim/atomic_actions.py` lines 63–71).**

```python
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    AntipodalAffordance,
    GraspTarget,
    MoveAction,
    MoveActionCfg,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
    PlaceAction,
    PlaceActionCfg,
    PoseTarget,
)
```

- [ ] **Step 2: Replace the AntipodalAffordance construction (lines 263–277).**

```python
mug_grasp_affordance = AntipodalAffordance(
    mesh_vertices=mug.get_vertices(env_ids=[0], scale=True)[0],
    mesh_triangles=mug.get_triangles(env_ids=[0])[0],
    gripper_collision_cfg=GripperCollisionCfg(
        max_open_length=0.088, finger_length=0.078, point_sample_dense=0.012
    ),
    generator_cfg=GraspGeneratorCfg(
        viser_port=11801,
        antipodal_sampler_cfg=AntipodalSamplerCfg(
            n_sample=20000, max_length=0.088, min_length=0.003
        ),
    ),
    force_reannotate=False,
)
```

- [ ] **Step 3: Replace the ObjectSemantics construction (lines 278–286).**

```python
mug_semantics = ObjectSemantics(
    affordance=mug_grasp_affordance,
    geometry={
        "mesh_vertices": mug.get_vertices(env_ids=[0], scale=True)[0],
        "mesh_triangles": mug.get_triangles(env_ids=[0])[0],
    },
    label="mug",
    entity=mug,
)
```

- [ ] **Step 4: Replace the engine construction + execution (lines 241–339).**

```python
# ------------------------------------------------------------------ #
# Step 4: Build the AtomicActionEngine and register actions          #
# ------------------------------------------------------------------ #
atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
atomic_engine.register(PickUpAction(motion_gen, cfg=pickup_cfg))
atomic_engine.register(PlaceAction(motion_gen, cfg=place_cfg))
atomic_engine.register(MoveAction(motion_gen, cfg=move_cfg))

sim.init_gpu_physics()
if not args.headless:
    sim.open_window()

# (semantics / poses set up earlier) ...

print("Planning pick → place → move trajectory...")
is_success, traj, _ = atomic_engine.run(
    steps=[
        ("pick_up", GraspTarget(semantics=mug_semantics)),
        ("place",   PoseTarget(xpos=place_xpos)),
        ("move",    PoseTarget(xpos=rest_xpos)),
    ],
)

if not is_success:
    print("Planning failed. Check that the target poses are reachable.")
    return

print(f"Success! Replaying {traj.shape[1]} waypoints...")
for i in range(traj.shape[1]):
    robot.set_qpos(traj[:, i])
    sim.update(step=4)
    time.sleep(1e-2)

input("Press Enter to exit...")
```

- [ ] **Step 5: Update the file-level docstring (lines 17–31) to mention `engine.run([(name, target), ...])` instead of `execute_static`.**

- [ ] **Step 6: Format with black.**

  Run: `black scripts/tutorials/sim/atomic_actions.py`
  Expected: file reformatted in place (or no change).

- [ ] **Step 7: Syntax / import smoke test.**

  Run: `python -c "import py_compile; py_compile.compile('scripts/tutorials/sim/atomic_actions.py', doraise=True)"`
  Expected: no output (success).

- [ ] **Step 8: Commit.**

```bash
git add scripts/tutorials/sim/atomic_actions.py
git commit -m "Migrate atomic_actions tutorial to typed-target engine API

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: Migrate `scripts/tutorials/atomic_action/pickup_atomic_actions.py`

**Files:**
- Modify: `scripts/tutorials/atomic_action/pickup_atomic_actions.py`

This is the pickup-only demo. After migration it constructs and registers a `PickUpAction`, then calls `engine.run([("pick_up", GraspTarget(...))])`.

- [ ] **Step 1: Read the current file to locate the engine usage.**

  Run: `grep -n "AtomicActionEngine\|execute_static\|ObjectSemantics\|AntipodalAffordance\|atomic_engine" scripts/tutorials/atomic_action/pickup_atomic_actions.py`
  Note the line numbers; you will replace these regions.

- [ ] **Step 2: Replace the import block at lines 35–44 (the block that begins `from embodichain.lab.sim.atomic_actions import (`).**

```python
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    GraspTarget,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
)
```

- [ ] **Step 3: Replace the engine instantiation around line 432.**

```python
atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
atomic_engine.register(PickUpAction(motion_gen, cfg=pickup_cfg))
```

- [ ] **Step 4: Replace the `AntipodalAffordance(... custom_config={...})` block.** The new constructor takes `mesh_vertices`/`mesh_triangles`/`gripper_collision_cfg`/`generator_cfg` directly.

  Find: the existing call `AntipodalAffordance(object_label=..., custom_config={"gripper_collision_cfg": ..., "generator_cfg": ...})`.
  Replace with the form shown in Task 7 Step 2 (adapt the variable names to match this script).

- [ ] **Step 5: Replace the `ObjectSemantics(...)` block** to remove `mesh_vertices`/`mesh_triangles` from the affordance's `custom_config` reliance — they now live directly on the affordance object. The `geometry` dict on `ObjectSemantics` can keep mesh entries for downstream metadata but no longer aliases anywhere.

- [ ] **Step 6: Replace any `execute_static(...)` call with `run(...)`.**

```python
is_success, traj, _ = atomic_engine.run(
    steps=[("pick_up", GraspTarget(semantics=mug_semantics))],
)
```

- [ ] **Step 7: Format with black.**

  Run: `black scripts/tutorials/atomic_action/pickup_atomic_actions.py`
  Expected: file reformatted in place (or no change).

- [ ] **Step 8: Compile-check.**

  Run: `python -c "import py_compile; py_compile.compile('scripts/tutorials/atomic_action/pickup_atomic_actions.py', doraise=True)"`
  Expected: no output (success).

- [ ] **Step 9: Commit.**

```bash
git add scripts/tutorials/atomic_action/pickup_atomic_actions.py
git commit -m "Migrate pickup tutorial to new atomic_actions API

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 9: Migrate `scripts/tutorials/atomic_action/move_object_atomic_actions.py`

**Files:**
- Modify: `scripts/tutorials/atomic_action/move_object_atomic_actions.py`

This is the pickup + move-object demo. After migration it registers `PickUpAction` and `MoveObjectAction` and runs them in sequence; the engine threads the `HeldObjectState` across.

- [ ] **Step 1: Replace the import block at lines 35–44.**

```python
from embodichain.lab.sim.atomic_actions import (
    AntipodalAffordance,
    AtomicActionEngine,
    GraspTarget,
    HeldObjectTarget,
    MoveObjectAction,
    MoveObjectActionCfg,
    ObjectSemantics,
    PickUpAction,
    PickUpActionCfg,
)
```

- [ ] **Step 2: Replace the engine instantiation around line 369.**

```python
atomic_engine = AtomicActionEngine(motion_generator=motion_gen)
atomic_engine.register(PickUpAction(motion_gen, cfg=pickup_cfg))
atomic_engine.register(MoveObjectAction(motion_gen, cfg=move_object_cfg))
```

- [ ] **Step 3: Replace the `MoveObjectTarget` construction with `HeldObjectTarget`.**

  Find: any `MoveObjectTarget(object_target_pose=...)` call.
  Replace with: `HeldObjectTarget(object_target_pose=...)`.

- [ ] **Step 4: Replace `execute_static(...)` with `engine.run(...)`.**

```python
is_success, traj, _ = atomic_engine.run(
    steps=[
        ("pick_up",     GraspTarget(semantics=mug_semantics)),
        ("move_object", HeldObjectTarget(object_target_pose=target_obj_pose)),
    ],
)
```

- [ ] **Step 5: Apply the same `AntipodalAffordance` / `ObjectSemantics` shape updates as in Task 8 Steps 4–5.**

- [ ] **Step 6: Format + compile check.**

  Run: `black scripts/tutorials/atomic_action/move_object_atomic_actions.py`
  Run: `python -c "import py_compile; py_compile.compile('scripts/tutorials/atomic_action/move_object_atomic_actions.py', doraise=True)"`
  Expected: no output (success).

- [ ] **Step 7: Commit.**

```bash
git add scripts/tutorials/atomic_action/move_object_atomic_actions.py
git commit -m "Migrate move_object tutorial to new atomic_actions API

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 10: Update Sphinx docs

**Files:**
- Modify: `docs/source/overview/sim/atomic_actions.md`
- Modify: `docs/source/tutorial/atomic_actions.rst`
- Modify: `docs/source/api/embodichain.lab.sim.atomic_actions.rst` (if it lists symbols)

These pages currently document the `ObjectSemantics(geometry={mesh_vertices, mesh_triangles})` + `execute_static(target_list=[...])` pattern. Update them to:

- Reflect typed targets (`PoseTarget`, `GraspTarget`, `HeldObjectTarget`).
- Reflect `engine.run([(name, target), ...])`.
- Reflect that `AntipodalAffordance` takes `mesh_vertices`/`mesh_triangles` directly.
- Drop references to `SemanticAnalyzer`, `MoveObjectTarget`, `execute_static`, `actions_cfg_list`.
- Add a short paragraph noting that `WorldState` is threaded through, and that `validate` was removed.

- [ ] **Step 1: Read the current overview page.**

  Run: `cat docs/source/overview/sim/atomic_actions.md`
  Note the sections that need rewording.

- [ ] **Step 2: Edit `docs/source/overview/sim/atomic_actions.md`.**
  - Replace any code snippet using `execute_static` with a snippet using `run(steps)`.
  - Replace `MoveObjectTarget` with `HeldObjectTarget` everywhere.
  - Replace `affordance.custom_config["gripper_collision_cfg"]` style with direct constructor fields.
  - Remove the paragraph (if present) about `SemanticAnalyzer` and string targets.

- [ ] **Step 3: Edit `docs/source/tutorial/atomic_actions.rst`.**
  - Mirror the tutorial-script changes from Task 7.

- [ ] **Step 4: Edit `docs/source/api/embodichain.lab.sim.atomic_actions.rst` (if present).**
  - Update any `automodule` / `autoclass` directives so that `SemanticAnalyzer`, `MoveObjectTarget`, `execute_static`, `_resolve_target` are not referenced.
  - Add directives for the new symbols: `PoseTarget`, `GraspTarget`, `HeldObjectTarget`, `WorldState`, `ActionResult`, `TrajectoryBuilder`.

- [ ] **Step 5: Build the docs locally to confirm they render.**

  Run: `export LC_ALL=C.UTF-8 LANG=C.UTF-8 && cd docs && make html`
  Expected: build completes with no Sphinx errors. Warnings about missing references are acceptable only if they predate this refactor.

- [ ] **Step 6: Commit.**

```bash
git add docs/source/overview/sim/atomic_actions.md docs/source/tutorial/atomic_actions.rst docs/source/api/embodichain.lab.sim.atomic_actions.rst
git commit -m "Update atomic_actions docs for redesign

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 11: Update the `add-atomic-action` skill scaffold

**Files:**
- Modify: `skills/add-atomic-action/SKILL.md`
- Modify: `.claude/skills/add-atomic-action/SKILL.md` (this is a symlink under `.claude/skills/` per the recent agent-context refactor — confirm with `readlink .claude/skills/add-atomic-action`)

The skill currently scaffolds new actions that inherit from `MoveAction`, override `execute(target, start_qpos=None, **kwargs) -> (bool, Tensor, joint_ids)`, and use `ClassVar updates_held_object_state`. Update the scaffold to:

- Inherit `AtomicAction` directly.
- Declare a `TargetType: ClassVar[type] = ...`.
- Implement `execute(self, target: TargetType, state: WorldState) -> ActionResult`.
- Use `self.builder = TrajectoryBuilder(motion_generator)` for helpers.
- Return a full-DoF trajectory.
- Build `next_state: WorldState` explicitly.
- Drop `validate`.

- [ ] **Step 1: Read the current SKILL.md.**

  Run: `cat skills/add-atomic-action/SKILL.md`
  Note the example template section.

- [ ] **Step 2: Replace the example template in SKILL.md.**

  Replace the existing template with one that matches the new shape. Use the spec §6 example as a guide. The new template should be ~30 lines, focused on:

  - Imports (`AtomicAction`, `WorldState`, `ActionResult`, the target type).
  - Cfg class inheriting `ActionCfg`.
  - Action class inheriting `AtomicAction` directly, declaring `TargetType`, holding `self.builder`.
  - `execute(target, state) -> ActionResult` returning a full-DoF trajectory.

- [ ] **Step 3: Update the skill's verification steps.** The current SKILL.md tells the user to verify by running `engine.execute_static(...)`; change this to `engine.run([(name, target)])`.

- [ ] **Step 4: Commit.**

```bash
git add skills/add-atomic-action/SKILL.md
git commit -m "Update add-atomic-action skill scaffold for redesign

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 12: Update agent-context topic (if applicable)

**Files:**
- Check: `agent_context/MAP.yaml`
- Possibly modify: `agent_context/topics/atomic-actions.md` or similar

The project's `agent_context/MAP.yaml` lists topics. If `atomic-actions` (or similar slug) is listed and points to a Markdown file, update that file to describe the new API.

- [ ] **Step 1: Check whether the topic exists.**

  Run: `grep -i atomic agent_context/MAP.yaml || echo "no topic"`
  - If `no topic` is printed, **skip the remainder of this task**.
  - Otherwise, note the filename from MAP.yaml.

- [ ] **Step 2: Edit the topic Markdown file** to describe the new API:
  - Typed targets (`PoseTarget`, `GraspTarget`, `HeldObjectTarget`).
  - `engine.run([(name, target), ...])`.
  - `WorldState` threading.
  - `TrajectoryBuilder` composition helper for new action authors.
  - The four built-in actions and what each one does to `state.held_object`.

- [ ] **Step 3: Commit.**

```bash
git add agent_context/
git commit -m "Update atomic-actions agent-context topic for redesign

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 13: Final verification — full repo pass

**Files:** none modified.

This task is a verification gate. Nothing new is implemented here; everything must already be in place.

- [ ] **Step 1: Run the full atomic_actions test suite.**

  Run: `pytest tests/sim/atomic_actions/ -x --no-header -q`
  Expected: all tests pass.

- [ ] **Step 2: Run a broader sweep to catch any cross-package import we missed.**

  Run: `pytest tests/sim/ -x --no-header -q -k "not slow"`
  Expected: no new failures. If something else in `tests/sim/` imported from the old API, fix it now.

- [ ] **Step 3: Format the entire touched surface.**

  Run: `black embodichain/lab/sim/atomic_actions/ tests/sim/atomic_actions/ scripts/tutorials/sim/atomic_actions.py scripts/tutorials/atomic_action/`
  Expected: either no changes, or formatting applied. If formatting applies, commit it.

  ```bash
  git add -u && git commit -m "Format atomic_actions redesign surface with black

Co-Authored-By: Claude <noreply@anthropic.com>"
  ```

- [ ] **Step 4: Run the project's pre-commit skill.**

  Run: `bash` (then) follow the `/pre-commit-check` skill instructions to lint headers / annotations / exports / docstrings.
  Expected: clean.

- [ ] **Step 5: Confirm nothing in the repo still imports removed names.**

  Run: `grep -rn "MoveObjectTarget\|execute_static\|SemanticAnalyzer\|_resolve_target\|updates_held_object_state\|_HandCloseAction" --include="*.py" --include="*.md" --include="*.rst" . | grep -v __pycache__ | grep -v docs/superpowers/`
  Expected: no matches (the spec / plan files under `docs/superpowers/` are allowed to mention these names as historical context).

- [ ] **Step 6: Confirm the package size dropped per the spec acceptance criteria.**

  Run: `wc -l embodichain/lab/sim/atomic_actions/*.py`
  Expected: `core.py` is around 150 lines, `affordance.py` around 200, `trajectory.py` around 250, `actions.py` around 500–600, `engine.py` around 80. Total roughly 1,100. If any single file is wildly larger or smaller than the spec target, surface the discrepancy in the PR description (don't silently let `actions.py` blow back to 900 lines).

- [ ] **Step 7: Confirm the tutorial scripts dropped at least 30%.**

  Run: `wc -l scripts/tutorials/atomic_action/*.py scripts/tutorials/sim/atomic_actions.py`
  Compare with pre-refactor baselines: `sim/atomic_actions.py` was 346 lines (expect ≤ ~240), `pickup_atomic_actions.py` was 499 lines (expect ≤ ~350), `move_object_atomic_actions.py` was 434 lines (expect ≤ ~300). If they shrank by less than 30%, surface it in the PR description — likely a residual leftover.

- [ ] **Step 8: Run an end-to-end smoke test on the tutorial.**

  Run: `python scripts/tutorials/sim/atomic_actions.py --num_envs 1 --headless --renderer hybrid`
  Expected: "Planning pick → place → move trajectory..." → "Success! Replaying N waypoints..." → exits cleanly.

  If this requires GPU and the runner has none, skip with a clear note in the PR description that headless GPU smoke-test must be run before merge.

- [ ] **Step 9: Final commit / branch state check.**

  Run: `git log --oneline main..HEAD | head -30`
  Confirm the commits tell a coherent story: extract affordance → rewrite core → trajectory → actions → engine → exports → 3 tutorials → docs → skill → context → final format. No "WIP" or "fix typo" commits should remain — squash them out if they exist.

---

## Self-Review Checklist (executor reads this AFTER finishing)

Before opening the PR:

- [ ] All four concrete action classes (`MoveAction`, `PickUpAction`, `MoveObjectAction`, `PlaceAction`) inherit `AtomicAction` directly. None inherit each other. No `_HandCloseAction` remains.
- [ ] `AtomicAction.execute` signature is `(self, target: Target, state: WorldState) -> ActionResult`. No `**kwargs`. No `start_qpos=None`. No `validate` on the ABC.
- [ ] No `updates_held_object_state` ClassVar exists anywhere.
- [ ] No `get_held_object_state` method exists on any action.
- [ ] `engine.run(steps, state)` exists. `execute_static` does not exist anywhere.
- [ ] `SemanticAnalyzer` does not exist anywhere.
- [ ] `_resolve_target` does not exist anywhere.
- [ ] `MoveObjectTarget` does not exist; `HeldObjectTarget` replaces it.
- [ ] `ObjectSemantics.__post_init__` does NOT mutate `affordance.geometry`.
- [ ] `AntipodalAffordance` takes `mesh_vertices` / `mesh_triangles` / `gripper_collision_cfg` / `generator_cfg` as explicit fields; no `custom_config` dict shuffling.
- [ ] Every action returns trajectories of shape `(n_envs, n_waypoints, robot.dof)`.
- [ ] The three tutorial scripts pass `python -c "import py_compile; py_compile.compile(...)"`.
- [ ] `pytest tests/sim/atomic_actions/ -x --no-header -q` is green.
- [ ] `black` produces no changes on the touched surface.
- [ ] The global registry (`register_action` / `unregister_action` / `get_registered_actions`) is still present and tested.

---

## Notes for the executor

- **TDD discipline:** every module rewrite goes test-first. If a step says "write failing test → confirm it fails → write code → confirm it passes" do not skip the "confirm it fails" step — it is the only signal that the test is actually exercising the new code.
- **DRY:** the `_embed_into_full_dof` pattern in `MoveAction` is repeated structurally in `PickUpAction`, `MoveObjectAction`, `PlaceAction`. After the first three actions, look at the repetition: if there is a clean single helper to extract into `TrajectoryBuilder` (e.g., `embed_arm_into_full(arm_traj, last_full_qpos, arm_joint_ids)`), extract it. The plan deliberately writes each one explicitly first so the shape is visible; a follow-up DRY pass is welcome.
- **YAGNI:** if a piece of code looks dead after the refactor (a helper not called by anyone), remove it. Examples: `_compute_lift_xpos` (already commented out today), `is_draw_grasp_xpos` if no caller exercises it.
- **Commit frequently:** the per-task commits in the plan are the minimum. If a task balloons mid-implementation, split it into a sub-commit before continuing.
- **Behavioural equivalence vs old API:** the redesign is not behaviour-changing. If a tutorial used to produce a specific trajectory shape (n_envs, n_waypoints, dof), the new tutorial should produce the same shape with the same numerical values modulo float-equivalence. If you find a numerical drift, stop and surface it before committing — it means a helper was relocated incorrectly.
