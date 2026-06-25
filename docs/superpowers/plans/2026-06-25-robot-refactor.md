# Robot Definition Refactor + `add-robot` Skill — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify `CobotMagicCfg` and `DexforceW1Cfg` under one shared `RobotCfg` protocol (a `_build_defaults` hook + a 3-line `from_dict` template + base-level serialization), fix 10 confirmed bugs, refresh the `add_robot.rst` docs (and create the missing tutorial), and add an `add-robot` skill.

**Architecture:** Hybrid — lift the `_build_defaults` no-op hook and generic `to_dict`/`to_string`/`save_to_file` serialization into the base `RobotCfg` (core `cfg.py`); each subclass keeps a thin `from_dict` that calls `_build_defaults` then `merge_robot_cfg`. Variant/version logic stays in subclasses. The base `RobotCfg.from_dict` stays as the generic dict→cfg constructor (it MUST NOT become the template — `merge_robot_cfg` calls it internally, so making it the template would infinite-recurse). The `build_pk_serial_chain` URDF-drift bug is fixed by routing each robot's PK URDF through one `_pk_urdf_path` source plus a DOF-count drift guard.

**Tech Stack:** Python 3, `@configclass` (custom dataclass-like decorator), `pytorch_kinematics`, Sphinx/RST docs, `black==26.3.1`.

**Spec:** `docs/superpowers/specs/2026-06-25-robot-refactor-design.md`

---

## File Structure

**Modified code:**
- `embodichain/lab/sim/cfg.py` — `RobotCfg`: add `_build_defaults` no-op + `to_dict`/`to_string`/`save_to_file`.
- `embodichain/lab/sim/robots/cobotmagic.py` — `_build_default_cfgs` → `_build_defaults`; reduce `from_dict`; add `__all__`; route PK via `_pk_urdf_path`.
- `embodichain/lab/sim/robots/dexforce_w1/cfg.py` — collapse builders → `_build_defaults`; reduce `from_dict`; delete `to_dict`/`to_string`/`save_to_file` (inherited); route PK via `_pk_urdf_path`.
- `embodichain/lab/sim/robots/dexforce_w1/utils.py` — stop setting `solver_cfg` in `build_dexforce_w1_cfg`; fix `__all__`; fix `root_link_name` tuple.
- `embodichain/lab/sim/robots/dexforce_w1/types.py` — `all` → `__all__`; add `DexforceW1HandBrand`.
- `embodichain/lab/sim/robots/dexforce_w1/params.py` — remove phantom `validate()` call.
- `embodichain/lab/sim/robots/__init__.py` — add `__all__`.

**New tests:**
- `tests/sim/objects/test_robot_cfg.py` — pure-logic cfg tests (no sim): base round-trip, DexforceW1 round-trip + solver-set-once, CobotMagic round-trip, drift guards.

**Modified docs:**
- `docs/source/guides/add_robot.rst` — rewrite checklist/params + Common Mistakes.
- `docs/source/tutorial/add_robot.rst` — **create** (Path A + Path B).
- `agent_context/topics/robot-system/robot-system.md` — align checklist + tables.

**New skill:**
- `.agents/skills/add-robot/SKILL.md` + `.agents/skills/add-robot/agents/openai.yaml`.
- `.claude/skills/add-robot/SKILL.md`.
- `.github/copilot/add-robot.md` + `.github/copilot/instructions.md` (index entry).

**Test layout note:** Existing `tests/sim/objects/test_robot.py` is sim-based (spins a `SimulationManager`, has FK golden values). All NEW tests are pure-logic (cfg construction + round-trip + PK DOF) and go in a new `test_robot_cfg.py` so they run fast without a sim. The existing `test_robot.py` FK golden test is the regression safety net for Task 2.

---

## Task 1: Lift `_build_defaults` hook + serialization into base `RobotCfg`

**Files:**
- Modify: `embodichain/lab/sim/cfg.py` (add to `RobotCfg`, ~line 1484 onward)
- Create test: `tests/sim/objects/test_robot_cfg.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sim/objects/test_robot_cfg.py` with the Apache header, `from __future__ import annotations`, a synthetic round-trip cfg, and the round-trip test:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# ... (full Apache 2.0 header — copy from tests/sim/objects/test_robot.py lines 1-15)
# ----------------------------------------------------------------------------
from __future__ import annotations

import enum

import numpy as np
import pytest

from embodichain.lab.sim.cfg import RobotCfg, JointDrivePropertiesCfg
from embodichain.utils import configclass
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg


class _RoundTripVariant(enum.Enum):
    A = "a"
    B = "b"


@configclass
class _RoundTripCfg(RobotCfg):
    """Synthetic cfg to exercise the base serialization + _build_defaults hook."""

    variant: _RoundTripVariant = _RoundTripVariant.A

    @classmethod
    def from_dict(cls, init_dict):
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict=None):
        init_dict = init_dict or {}
        self.uid = "roundtrip"
        self.variant = _RoundTripVariant(init_dict.get("variant", "a"))
        self.control_parts = {"arm": ["J1", "J2"]}
        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={"J[1-2]": 1e4}, damping={"J[1-2]": 1e3}
        )


def test_robotcfg_to_dict_roundtrip():
    cfg = _RoundTripCfg.from_dict({"variant": "b"})
    assert cfg.variant == _RoundTripVariant.B

    d = cfg.to_dict()
    assert d["uid"] == "roundtrip"
    assert d["variant"] == "b"

    cfg2 = _RoundTripCfg.from_dict(d)
    assert cfg2.uid == "roundtrip"
    assert cfg2.variant == _RoundTripVariant.B
    assert cfg2.control_parts == {"arm": ["J1", "J2"]}
    assert cfg2.drive_pros.stiffness == {"J[1-2]": 1e4}


def test_robotcfg_save_to_file(tmp_path):
    cfg = _RoundTripCfg.from_dict({"variant": "b"})
    fp = tmp_path / "cfg.json"
    cfg.save_to_file(str(fp))
    import json

    loaded = json.loads(fp.read_text())
    assert loaded["variant"] == "b"
    assert loaded["uid"] == "roundtrip"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: FAIL — `AttributeError: 'RobotCfg' object has no attribute 'to_dict'` (and `_build_defaults` may be absent too).

- [ ] **Step 3: Add `_build_defaults` no-op + serialization to `RobotCfg`**

In `embodichain/lab/sim/cfg.py`, inside `class RobotCfg(ArticulationCfg):` (after the `from_dict` method, before `build_pk_serial_chain`, ~line 1566), add:

```python
    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Populate default config fields from ``init_dict``.

        Subclasses override this to read variant/version fields from
        ``init_dict``, set them on ``self``, and populate ``urdf_cfg``,
        ``control_parts``, ``solver_cfg``, ``drive_pros`` and ``attrs``.
        The base implementation is a no-op.

        .. attention::
            Do NOT call :func:`merge_robot_cfg` from here — the subclass
            ``from_dict`` calls this hook first, then ``merge_robot_cfg``.
            Calling ``merge_robot_cfg`` here would recurse, because
            ``merge_robot_cfg`` itself calls ``RobotCfg.from_dict``.

        Args:
            init_dict: The raw override dict passed to ``from_dict``.
        """
        return None

    def to_dict(self):
        """Serialize config to a plain dict (enums, numpy, nested configclass)."""

        def serialize(obj, _visited=None):
            if _visited is None:
                _visited = set()
            if isinstance(obj, (dict, object)) and not isinstance(
                obj, (str, int, float, bool, type(None))
            ):
                obj_id = id(obj)
                if obj_id in _visited:
                    return None
                _visited.add(obj_id)

            import enum

            if isinstance(obj, enum.Enum):
                return obj.value
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {str(k): serialize(v, _visited) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [serialize(v, _visited) for v in obj]
            if hasattr(obj, "to_dict") and obj is not self:
                return serialize(obj.to_dict(), _visited)
            if hasattr(obj, "__dict__"):
                return {k: serialize(v, _visited) for k, v in obj.__dict__.items()}
            return obj

        return serialize(self)

    def to_string(self):
        """Return config as a JSON string."""
        import json

        return json.dumps(self.to_dict(), indent=2)

    def save_to_file(self, filepath):
        """Save config to a local file as JSON."""
        with open(filepath, "w") as f:
            f.write(self.to_string())
```

Ensure `numpy` is imported at the top of `cfg.py` (it already is — verify).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/cfg.py tests/sim/objects/test_robot_cfg.py
git commit -m "refactor(robot-cfg): lift _build_defaults hook + serialization into base RobotCfg"
```

---

## Task 2: Conform `DexforceW1Cfg` + fix bugs

**Files:**
- Modify: `embodichain/lab/sim/robots/dexforce_w1/cfg.py`
- Modify: `embodichain/lab/sim/robots/dexforce_w1/utils.py`
- Modify: `embodichain/lab/sim/robots/dexforce_w1/types.py`
- Modify: `embodichain/lab/sim/robots/dexforce_w1/params.py`
- Modify: `tests/sim/objects/test_robot_cfg.py`

- [ ] **Step 1: Write characterization tests (run on current code to capture behavior)**

Append to `tests/sim/objects/test_robot_cfg.py`:

```python
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1ArmKind,
    DexforceW1Version,
)
from embodichain.lab.sim.solvers import SRSSolverCfg


def test_dexforce_w1_roundtrip():
    cfg = DexforceW1Cfg.from_dict(
        {"uid": "dexforce_w1", "version": "v021", "arm_kind": "anthropomorphic"}
    )
    d = cfg.to_dict()
    assert d["uid"] == "dexforce_w1"
    assert d["arm_kind"] == "anthropomorphic"
    cfg2 = DexforceW1Cfg.from_dict(d)
    assert cfg2.uid == "dexforce_w1"
    assert cfg2.arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC
    assert cfg2.version == DexforceW1Version.V021


def test_dexforce_w1_solver_cfg_is_srs_and_set_once():
    cfg = DexforceW1Cfg.from_dict({"arm_kind": "industrial"})
    assert isinstance(cfg.solver_cfg["left_arm"], SRSSolverCfg)
    assert isinstance(cfg.solver_cfg["right_arm"], SRSSolverCfg)
```

- [ ] **Step 2: Run tests on current (unrefactored) code**

Run: `pytest tests/sim/objects/test_robot_cfg.py::test_dexforce_w1_roundtrip tests/sim/objects/test_robot_cfg.py::test_dexforce_w1_solver_cfg_is_srs_and_set_once -v`
Expected: PASS — these capture current behavior (`to_dict` exists on DexforceW1 today; final solver is SRSSolverCfg). These are regression guards.

- [ ] **Step 3: Fix `types.py` — `all` → `__all__`, add `DexforceW1HandBrand`**

In `embodichain/lab/sim/robots/dexforce_w1/types.py`, replace lines 19-24:

```python
all = [
    "DexforceW1Version",
    "DexforceW1ArmKind",
    "DexforceW1ArmSide",
    "DexforceW1Type",
]
```

with:

```python
__all__ = [
    "DexforceW1Version",
    "DexforceW1ArmKind",
    "DexforceW1ArmSide",
    "DexforceW1Type",
    "DexforceW1HandBrand",
]
```

- [ ] **Step 4: Fix `utils.py` `__all__`**

In `embodichain/lab/sim/robots/dexforce_w1/utils.py` line 31, change `all = [` to `__all__ = [`.

- [ ] **Step 5: Fix `HandManager.get_config` `root_link_name` tuple bug**

In `embodichain/lab/sim/robots/dexforce_w1/utils.py`, find (in `HandManager.get_config`, ~line 222) the line returning a 1-tuple:

```python
root_link_name = (f"{prefix.lower()}_finger2_link",)
```

Change to:

```python
root_link_name = f"{prefix.lower()}_finger2_link"
```

(Locate it with `grep -n "finger2_link" embodichain/lab/sim/robots/dexforce_w1/utils.py` if the line moved.)

- [ ] **Step 6: Stop setting `solver_cfg` in `build_dexforce_w1_cfg`**

In `embodichain/lab/sim/robots/dexforce_w1/utils.py`, replace lines 736-744:

```python
    if solver_cfg is not None:
        cfg.solver_cfg = solver_cfg
    else:
        cfg.solver_cfg = build_dexforce_w1_solver_cfg(
            arm_kind=arm_kind,
            arm_sides=arm_sides,
            component_versions=component_versions,
            urdf_cfg=urdf_cfg,
        )

    return cfg
```

with:

```python
    if solver_cfg is not None:
        cfg.solver_cfg = solver_cfg

    return cfg
```

The cfg class's `_build_defaults` now owns solver construction. Leave `build_dexforce_w1_solver_cfg` in place (still reachable by direct call); note in the commit that the `else` branch (the PytorchSolver default) was dead code — `_build_default_solver_cfg` always overwrote it with `SRSSolverCfg`.

- [ ] **Step 7: Remove phantom `validate()` call in `params.py`**

In `embodichain/lab/sim/robots/dexforce_w1/params.py`, find (in `W1ArmKineParams.from_dict`, ~line 256) `inst.validate()` and remove that single call. Use `grep -n "validate" embodichain/lab/sim/robots/dexforce_w1/params.py` to locate; delete the line `inst.validate()` (keep surrounding logic intact). If the surrounding block was `inst = cls(...); inst.validate(); return inst`, it becomes `inst = cls(...); return inst`.

- [ ] **Step 8: Add `_build_defaults` to `DexforceW1Cfg`, reduce `from_dict`, delete serialization overrides**

In `embodichain/lab/sim/robots/dexforce_w1/cfg.py`, replace the `from_dict` method (lines 58-96) with the 3-line template:

```python
    @classmethod
    def from_dict(cls, init_dict: Dict[str, str | float | tuple | dict]) -> DexforceW1Cfg:
        """Initialize DexforceW1Cfg from a dictionary.

        Args:
            init_dict: Dictionary of configuration parameters.

        Returns:
            A DexforceW1Cfg instance. Defaults are built via
            :meth:`_build_defaults`, then ``init_dict`` overrides are merged.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)
```

Then replace the three builders (`_build_default_solver_cfg`, `_build_default_physics_cfgs`, `_build_default_cfg` — lines 98-299) with `_build_defaults` calling two retained private helpers:

```python
    def _build_defaults(self, init_dict: dict | None = None) -> None:
        """Build default urdf/control/solver/physics from variant fields.

        Reads ``version``/``arm_kind``/``with_default_eef`` from ``init_dict``,
        sets them on ``self``, then populates ``urdf_cfg``, ``control_parts``,
        ``solver_cfg``, ``drive_pros`` and ``attrs``.
        """
        init_dict = init_dict or {}
        version = init_dict.get("version", DexforceW1Version.V021)
        arm_kind = init_dict.get("arm_kind", DexforceW1ArmKind.INDUSTRIAL)
        with_default_eef = init_dict.get("with_default_eef", True)

        self.version = DexforceW1Version(version) if isinstance(version, str) else version
        self.arm_kind = (
            DexforceW1ArmKind(arm_kind) if isinstance(arm_kind, str) else arm_kind
        )
        self.with_default_eef = with_default_eef

        # urdf_cfg + control_parts (build_dexforce_w1_cfg no longer sets solver_cfg)
        if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.DH_PGC_GRIPPER_M,
            }
        else:
            hand_types = {
                DexforceW1ArmSide.LEFT: DexforceW1HandBrand.BRAINCO_HAND,
                DexforceW1ArmSide.RIGHT: DexforceW1HandBrand.BRAINCO_HAND,
            }
        hand_versions = {
            DexforceW1ArmSide.LEFT: self.version,
            DexforceW1ArmSide.RIGHT: self.version,
        }
        base_cfg = build_dexforce_w1_cfg(
            arm_kind=self.arm_kind,
            hand_types=hand_types,
            hand_versions=hand_versions,
            include_hand=with_default_eef,
        )
        self.urdf_cfg = base_cfg.urdf_cfg
        self.control_parts = base_cfg.control_parts

        # physics
        physics = self._build_default_physics_cfgs(
            arm_kind=self.arm_kind, with_default_eef=with_default_eef
        )
        for key, value in physics.items():
            setattr(self, key, value)

        # solver (set exactly once — was previously double-set)
        self.solver_cfg = self._build_default_solver_cfg(arm_kind=self.arm_kind)
```

Keep `_build_default_solver_cfg` and `_build_default_physics_cfgs` as-is BUT convert them from `@staticmethod` to regular instance methods (remove the `@staticmethod` decorator; they take `self` as first arg, though they don't use it). Concretely:

- `_build_default_solver_cfg(self, arm_kind: DexforceW1ArmKind)` — remove `@staticmethod`, change signature first param to `self`, and replace the `is_industrial: bool` param with `arm_kind: DexforceW1ArmKind`. Replace `if is_industrial:` with `if arm_kind == DexforceW1ArmKind.INDUSTRIAL:`. Replace the `W1ArmKineParams(...)` calls' `arm_kind=` to use `arm_kind` directly (already enum).
- `_build_default_physics_cfgs(self, arm_kind, with_default_eef=True)` — remove `@staticmethod`, add `self` first param. Body unchanged (the `arm_kind == "anthropomorphic"` string compare still works because callers may pass the enum; update the compare to `arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC` for type-safety, and the `else` branch is industrial).

Delete `to_dict`, `to_string`, `save_to_file` (lines 301-340) — now inherited from `RobotCfg`.

Keep `build_pk_serial_chain` (Task 4 will edit it).

- [ ] **Step 9: Run characterization tests + round-trip**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: PASS (base round-trip, save_to_file, DexforceW1 round-trip, solver-set-once).

- [ ] **Step 10: Run the FK golden regression test (sim)**

Run: `pytest tests/sim/objects/test_robot.py::TestRobotCPU -v -x`
Expected: PASS — FK golden values (`test_compute_fk`), joint IDs, mimic, etc. unchanged. This is the critical regression check: the refactor must not change kinematic output.

If the FK test fails, the solver construction in `_build_default_solver_cfg` diverged from the old path — re-check that `_build_default_solver_cfg` produces byte-identical `SRSSolverCfg` objects (same TCP, dh_params, limits).

- [ ] **Step 11: Commit**

```bash
git add embodichain/lab/sim/robots/dexforce_w1/ tests/sim/objects/test_robot_cfg.py
git commit -m "refactor(dexforce_w1): conform to _build_defaults hook, fix __all__/tuple/validate/dead-solver-set"
```

---

## Task 3: Conform `CobotMagicCfg`

**Files:**
- Modify: `embodichain/lab/sim/robots/cobotmagic.py`
- Modify: `tests/sim/objects/test_robot_cfg.py`

- [ ] **Step 1: Write characterization test**

Append to `tests/sim/objects/test_robot_cfg.py`:

```python
from embodichain.lab.sim.robots.cobotmagic import CobotMagicCfg
from embodichain.lab.sim.solvers import OPWSolverCfg


def test_cobotmagic_from_dict_and_roundtrip():
    cfg = CobotMagicCfg.from_dict({})
    assert cfg.uid == "CobotMagic"
    assert set(cfg.control_parts.keys()) == {
        "left_arm",
        "left_eef",
        "right_arm",
        "right_eef",
    }
    assert isinstance(cfg.solver_cfg["left_arm"], OPWSolverCfg)
    assert isinstance(cfg.solver_cfg["right_arm"], OPWSolverCfg)

    d = cfg.to_dict()
    assert d["uid"] == "CobotMagic"
    cfg2 = CobotMagicCfg.from_dict(d)
    assert cfg2.uid == "CobotMagic"
    assert cfg2.control_parts == cfg.control_parts
    assert isinstance(cfg2.solver_cfg["left_arm"], OPWSolverCfg)
```

- [ ] **Step 2: Run test on current code**

Run: `pytest tests/sim/objects/test_robot_cfg.py::test_cobotmagic_from_dict_and_roundtrip -v`
Expected: PASS (current `from_dict` + the now-inherited `to_dict` already cover this).

- [ ] **Step 3: Convert `_build_default_cfgs` → `_build_defaults`, reduce `from_dict`, add `__all__`**

In `embodichain/lab/sim/robots/cobotmagic.py`:

Replace the `from_dict` method (lines 46-56) with:

```python
    @classmethod
    def from_dict(cls, init_dict: Dict[str, Union[str, float, int]]) -> CobotMagicCfg:
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)
```

Replace the `_build_default_cfgs` static method (lines 58-163) with an instance `_build_defaults` that sets fields on `self` (move the body verbatim, but `setattr` onto `self` instead of returning a dict):

```python
    def _build_defaults(self, init_dict: dict | None = None) -> None:
        init_dict = init_dict or {}
        arm_urdf = get_data_path("CobotMagicArm/CobotMagicWithGripperV100.urdf")
        left_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, 0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        right_arm_xpos = np.array(
            [
                [1.0, 0.0, 0.0, 0.233],
                [0.0, 1.0, 0.0, -0.300],
                [0.0, 0.0, 1.0, 0.000],
                [0.0, 0.0, 0.0, 1.000],
            ]
        )
        self.uid = "CobotMagic"
        self.urdf_cfg = URDFCfg(
            components=[
                {"component_type": "left_arm", "urdf_path": arm_urdf, "transform": left_arm_xpos},
                {"component_type": "right_arm", "urdf_path": arm_urdf, "transform": right_arm_xpos},
            ]
        )
        self.control_parts = {
            "left_arm": ["LEFT_JOINT1", "LEFT_JOINT2", "LEFT_JOINT3", "LEFT_JOINT4", "LEFT_JOINT5", "LEFT_JOINT6"],
            "left_eef": ["LEFT_JOINT7", "LEFT_JOINT8"],
            "right_arm": ["RIGHT_JOINT1", "RIGHT_JOINT2", "RIGHT_JOINT3", "RIGHT_JOINT4", "RIGHT_JOINT5", "RIGHT_JOINT6"],
            "right_eef": ["RIGHT_JOINT7", "RIGHT_JOINT8"],
        }
        self.solver_cfg = {
            "left_arm": OPWSolverCfg(
                end_link_name="left_link6",
                root_link_name="left_arm_base",
                tcp=np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]),
            ),
            "right_arm": OPWSolverCfg(
                end_link_name="right_link6",
                root_link_name="right_arm_base",
                tcp=np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]]),
            ),
        }
        self.min_position_iters = 8
        self.min_velocity_iters = 2
        self.drive_pros = JointDrivePropertiesCfg(
            stiffness={
                "LEFT_JOINT[1-6]": 7e4, "RIGHT_JOINT[1-6]": 7e4,
                "LEFT_JOINT[7-8]": 3e2, "RIGHT_JOINT[7-8]": 3e2,
            },
            damping={
                "LEFT_JOINT[1-6]": 1e3, "RIGHT_JOINT[1-6]": 1e3,
                "LEFT_JOINT[7-8]": 3e1, "RIGHT_JOINT[7-8]": 3e1,
            },
            max_effort={
                "LEFT_JOINT[1-6]": 3e6, "RIGHT_JOINT[1-6]": 3e6,
                "LEFT_JOINT[7-8]": 3e3, "RIGHT_JOINT[7-8]": 3e3,
            },
        )
        self.attrs = RigidBodyAttributesCfg(
            mass=0.1, static_friction=0.95, dynamic_friction=0.9,
            linear_damping=0.7, angular_damping=0.7, contact_offset=0.001,
            rest_offset=0.001, restitution=0.01, max_depenetration_velocity=1e1,
        )
```

Add `__all__` near the top of the module (after imports, before the class):

```python
__all__ = ["CobotMagicCfg"]
```

Remove the now-unused `Any` import only if nothing else uses it (it is used in the old `_build_default_cfgs` return type — after removal, check `from typing import ... Any` is still needed; if not, drop it). Keep `Dict`, `List`, `Union`.

- [ ] **Step 4: Run tests**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: PASS (all cfg tests including CobotMagic).

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/robots/cobotmagic.py tests/sim/objects/test_robot_cfg.py
git commit -m "refactor(cobotmagic): conform to _build_defaults hook, add __all__"
```

---

## Task 4: Fix `build_pk_serial_chain` URDF drift via `_pk_urdf_path` + DOF drift guard

**Files:**
- Modify: `embodichain/lab/sim/robots/cobotmagic.py` (add `_pk_urdf_path`, edit `build_pk_serial_chain`)
- Modify: `embodichain/lab/sim/robots/dexforce_w1/cfg.py` (add `_pk_urdf_path`, edit `build_pk_serial_chain`)
- Modify: `tests/sim/objects/test_robot_cfg.py` (drift-guard tests)

- [ ] **Step 1: Write the drift-guard tests**

Append to `tests/sim/objects/test_robot_cfg.py`:

```python
def _dof_of_pk_chain(chain) -> int:
    return len(chain.get_joint_names())


def test_dexforce_w1_pk_dof_matches_control_parts():
    pk = pytest.importorskip("pytorch_kinematics")
    cfg = DexforceW1Cfg.from_dict({"arm_kind": "anthropomorphic"})
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for arm in ("left_arm", "right_arm"):
        assert _dof_of_pk_chain(chains[arm]) == len(cfg.control_parts[arm]), (
            f"{arm}: PK chain DOF drifted from control_parts"
        )


def test_cobotmagic_pk_dof_matches_control_parts():
    pk = pytest.importorskip("pytorch_kinematics")
    cfg = CobotMagicCfg.from_dict({})
    try:
        chains = cfg.build_pk_serial_chain()
    except Exception as exc:
        pytest.skip(f"PK URDF asset unavailable: {exc}")
    for arm in ("left_arm", "right_arm"):
        assert _dof_of_pk_chain(chains[arm]) == len(cfg.control_parts[arm]), (
            f"{arm}: PK chain DOF drifted from control_parts"
        )
```

- [ ] **Step 2: Run tests — expect PASS (no drift today) or informative SKIP**

Run: `pytest tests/sim/objects/test_robot_cfg.py -k pk_dof -v`
Expected: PASS (CobotMagic arm = 6 joints, 6 DOF; DexforceW1 arm = 7 joints, 7 DOF) or SKIP if the PK URDF assets aren't installed in the env.

- [ ] **Step 3: Route CobotMagic PK through `_pk_urdf_path`**

In `embodichain/lab/sim/robots/cobotmagic.py`, add a property and edit `build_pk_serial_chain`:

```python
    @property
    def _pk_urdf_path(self) -> str:
        """URDF used for FK/IK serial chains (arm-only, gripper-stripped).

        .. attention::
            The root_link→end_link kinematics here must match the arm in the
            simulation URDF. A DOF drift guard in the tests checks this.
        """
        return get_data_path("CobotMagicArm/CobotMagicNoGripper.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        urdf_path = self._pk_urdf_path
        left_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path, device=device,
            end_link_name="link6", root_link_name="base_link",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path, device=device,
            end_link_name="link6", root_link_name="base_link",
        )
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}
```

- [ ] **Step 4: Route DexforceW1 PK through `_pk_urdf_path`**

In `embodichain/lab/sim/robots/dexforce_w1/cfg.py`, replace the URDF selection in `build_pk_serial_chain` (the `if/elif` at lines 349-352) with a method call:

```python
    def _pk_urdf_path(self) -> str:
        """URDF used for FK/IK serial chains, by arm kind.

        .. attention::
            The root_link→end_link kinematics here must match the arms in the
            simulation (assembled) URDF. A DOF drift guard in the tests checks this.
        """
        if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            return get_data_path("DexforceW1V021/DexforceW1_v02_2.urdf")
        return get_data_path("DexforceW1V021/DexforceW1_v02_1.urdf")

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs
    ) -> Dict[str, "pk.SerialChain"]:
        from embodichain.lab.sim.utility.solver_utils import (
            create_pk_serial_chain,
        )

        urdf_path = self._pk_urdf_path()
        left_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path, device=device,
            end_link_name="left_ee", root_link_name="left_arm_base",
        )
        right_arm_chain = create_pk_serial_chain(
            urdf_path=urdf_path, device=device,
            end_link_name="right_ee", root_link_name="right_arm_base",
        )
        return {"left_arm": left_arm_chain, "right_arm": right_arm_chain}
```

- [ ] **Step 5: Run drift-guard tests + FK golden**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: PASS.

Run: `pytest tests/sim/objects/test_robot.py::TestRobotCPU::test_compute_fk tests/sim/objects/test_robot.py::TestRobotCPU::test_fk -v -x`
Expected: PASS (FK unchanged — `_pk_urdf_path` returns the same URDFs as before, just routed through one source).

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/robots/cobotmagic.py embodichain/lab/sim/robots/dexforce_w1/cfg.py tests/sim/objects/test_robot_cfg.py
git commit -m "fix(robot): route build_pk_serial_chain through _pk_urdf_path + add DOF drift guard"
```

---

## Task 5: Register `__all__` in `robots/__init__.py`

**Files:**
- Modify: `embodichain/lab/sim/robots/__init__.py`

- [ ] **Step 1: Add `__all__`**

In `embodichain/lab/sim/robots/__init__.py`, after the existing imports (line 18), add:

```python
__all__ = ["DexforceW1Cfg", "CobotMagicCfg"]
```

- [ ] **Step 2: Verify imports work**

Run: `python -c "from embodichain.lab.sim.robots import DexforceW1Cfg, CobotMagicCfg; print('ok', DexforceW1Cfg, CobotMagicCfg)`
Expected: `ok <class ...DexforceW1Cfg'> <class ...CobotMagicCfg'>` with no import errors (this also confirms the `types.py`/`utils.py` `__all__` fixes didn't break `from .dexforce_w1 import *`).

- [ ] **Step 3: Commit**

```bash
git add embodichain/lab/sim/robots/__init__.py
git commit -m "refactor(robots): add __all__ to robots registry"
```

---

## Task 6: Rewrite `guides/add_robot.rst`

**Files:**
- Modify: `docs/source/guides/add_robot.rst`

- [ ] **Step 1: Rewrite the quick-reference guide**

Replace the entire contents of `docs/source/guides/add_robot.rst` with:

```rst
.. _guide_add_robot:

Adding a New Robot — Quick Reference
=====================================

This guide is a checklist + reference for adding a new robot to EmbodiChain. For
the full step-by-step walkthrough with code examples, see :doc:`/tutorial/add_robot`.

The protocol
------------

Every robot config subclasses :class:`~embodichain.lab.sim.cfg.RobotCfg` and
overrides two hooks:

- ``_build_defaults(self, init_dict=None)`` — populate ``urdf_cfg``,
  ``control_parts``, ``solver_cfg``, ``drive_pros`` and ``attrs`` from variant
  fields read out of ``init_dict``.
- ``build_pk_serial_chain(self, device=...)`` — return a
  ``{control_part: pk.SerialChain}`` mapping, reading the PK URDF from a single
  ``_pk_urdf_path`` source (so it cannot drift from the sim URDF).

The ``from_dict`` is a 3-line template (do not reimplement it)::

    cfg = cls()
    cfg._build_defaults(init_dict)
    return merge_robot_cfg(cfg, init_dict)

Serialization (``to_dict`` / ``to_string`` / ``save_to_file``) is inherited from
``RobotCfg`` and round-trips: ``RobotCfg.from_dict(cfg.to_dict())`` reproduces the cfg.

Checklist
---------

1. **Prepare the URDF** — place the URDF (+ meshes) in the assets directory.
2. **Override** ``_build_defaults(self, init_dict=None)`` — set variant fields from
   ``init_dict``, then populate ``urdf_cfg`` / ``control_parts`` / ``solver_cfg`` /
   ``drive_pros`` / ``attrs``.
3. **Define control parts** — group joints into logical sets (e.g. ``arm``, ``gripper``).
4. **Configure the IK solver** — ``OPWSolverCfg`` (6-DOF), ``SRSSolverCfg`` (7-DOF),
   or a generic ``SolverCfg``.
5. **Set drive properties** — stiffness/damping/max_effort per joint group.
6. **Implement** ``build_pk_serial_chain`` reading from ``_pk_urdf_path``.
7. **Keep** ``from_dict`` as the 3-line template — do not reimplement.
8. **Register** in ``embodichain/lab/sim/robots/__init__.py`` (and set ``__all__``).
9. **Add documentation** — ``docs/source/resources/robot/<name>.md`` + update
   ``resources/robot/index.rst``.
10. **Test** — a ``__main__`` smoke test + the DOF drift guard + ``preview-asset`` CLI.

Approaches
----------

- **Single-file** (variant-less robots): one ``my_robot.py`` with everything.
- **Package** (robots with variants — versions/arm-kinds/hand-brands): a directory
  with ``types.py`` (enums), ``cfg.py`` (a variant-aware ``_build_defaults``),
  optional ``params.py`` / ``utils.py`` helpers, and ``__init__.py``.

Key parameters
--------------

+---------------------+----------------------------------+----------------------------------+
| Parameter           | Type                             | Description                      |
+=====================+==================================+==================================+
| ``uid``             | str                              | Unique robot identifier          |
+---------------------+----------------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                          | URDF file and components         |
+---------------------+----------------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]             | Joint groups for control         |
+---------------------+----------------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]             | IK solver configurations         |
+---------------------+----------------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg          | Joint stiffness, damping, force  |
+---------------------+----------------------------------+----------------------------------+
| ``attrs``           | RigidBodyAttributesCfg           | Rigid-body physics attributes    |
+---------------------+----------------------------------+----------------------------------+
| variant fields      | enum / str / bool                | Optional subclass fields         |
|                     |                                  | (e.g. ``version``, ``arm_kind``) |
+---------------------+----------------------------------+----------------------------------+
| ``_pk_urdf_path``   | property or method → str         | URDF for the FK/IK serial chain  |
+---------------------+----------------------------------+----------------------------------+

Common mistakes
---------------

+-----------------------------------+----------------------------------------------------------+
| Mistake                           | Fix                                                      |
+===================================+==========================================================+
| ``all`` instead of ``__all__``    | Use ``__all__`` — lowercase ``all`` breaks ``import *``. |
+-----------------------------------+----------------------------------------------------------+
| ``solver_cfg`` set twice          | Set it once in ``_build_defaults`` only.                  |
+-----------------------------------+----------------------------------------------------------+
| PK URDF drifts from sim URDF      | Route PK through ``_pk_urdf_path``; keep the DOF guard.  |
+-----------------------------------+----------------------------------------------------------+
| Reimplementing ``from_dict``      | Keep the 3-line template; put logic in ``_build_defaults``.|
+-----------------------------------+----------------------------------------------------------+
| ``root_link_name`` as a tuple     | It must be a ``str``.                                     |
+-----------------------------------+----------------------------------------------------------+
| Calling a nonexistent ``validate``| Don't call methods that don't exist.                      |
+-----------------------------------+----------------------------------------------------------+

See Also
--------

- :doc:`/tutorial/add_robot` — Full step-by-step tutorial
- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/guides/add_robot.rst
git commit -m "docs(robot): rewrite add_robot quick-reference for the unified protocol"
```

---

## Task 7: Create `tutorial/add_robot.rst`

**Files:**
- Create: `docs/source/tutorial/add_robot.rst`

- [ ] **Step 1: Create the full tutorial**

Create `docs/source/tutorial/add_robot.rst`:

```rst
.. _tutorial_add_robot:

Adding a New Robot
==================

This tutorial walks through adding a new robot config to EmbodiChain. For the
quick-reference checklist, see :doc:`/guides/add_robot`.

A robot config subclasses :class:`~embodichain.lab.sim.cfg.RobotCfg`, which itself
extends ``ArticulationCfg`` → ``ObjectBaseCfg``. You override two hooks and keep
``from_dict`` as a 3-line template. Serialization is free.

Two paths: **single-file** (variant-less robots) and **package with variants**
(robots with versions / arm-kinds / hand-brands).

Path A — single-file robot
--------------------------

A variant-less robot lives in one file, e.g. ``embodichain/lab/sim/robots/my_robot.py``::

    from __future__ import annotations

    import numpy as np
    import torch
    from typing import TYPE_CHECKING, Dict

    from embodichain.lab.sim.cfg import (
        RobotCfg, URDFCfg, JointDrivePropertiesCfg, RigidBodyAttributesCfg,
    )
    from embodichain.lab.sim.solvers import OPWSolverCfg
    from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
    from embodichain.data import get_data_path
    from embodichain.utils import configclass

    if TYPE_CHECKING:
        import pytorch_kinematics as pk

    __all__ = ["MyRobotCfg"]


    @configclass
    class MyRobotCfg(RobotCfg):

        @classmethod
        def from_dict(cls, init_dict):
            cfg = cls()
            cfg._build_defaults(init_dict)
            return merge_robot_cfg(cfg, init_dict)

        def _build_defaults(self, init_dict=None):
            init_dict = init_dict or {}
            self.uid = "MyRobot"
            self.urdf_cfg = URDFCfg(
                components=[
                    {
                        "component_type": "arm",
                        "urdf_path": get_data_path("MyRobot/arm.urdf"),
                        "transform": np.eye(4),
                    }
                ]
            )
            self.control_parts = {"arm": ["JOINT[1-6]"]}
            self.solver_cfg = {
                "arm": OPWSolverCfg(
                    end_link_name="link6",
                    root_link_name="base_link",
                    tcp=np.eye(4),
                )
            }
            self.drive_pros = JointDrivePropertiesCfg(stiffness={"JOINT[1-6]": 1e4})

        @property
        def _pk_urdf_path(self) -> str:
            return get_data_path("MyRobot/arm.urdf")

        def build_pk_serial_chain(
            self, device: torch.device = torch.device("cpu"), **kwargs
        ) -> Dict[str, "pk.SerialChain"]:
            from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

            chain = create_pk_serial_chain(
                urdf_path=self._pk_urdf_path, device=device,
                end_link_name="link6", root_link_name="base_link",
            )
            return {"arm": chain}


    if __name__ == "__main__":
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg

        sim = SimulationManager(SimulationManagerCfg(headless=True, num_envs=1))
        cfg = MyRobotCfg.from_dict({})
        robot = sim.add_robot(cfg=cfg)
        print("MyRobot added:", cfg.to_dict()["uid"])

Path B — package robot with variants
-------------------------------------

When a robot has variants (versions, arm kinds, hand brands), use a package
directory ``embodichain/lab/sim/robots/my_robot/``:

- ``types.py`` — enums (``MyRobotVersion``, ``MyRobotArmKind``) with a proper
  ``__all__``.
- ``cfg.py`` — ``MyRobotCfg`` with a variant-aware ``_build_defaults`` that reads
  variant fields from ``init_dict``::

      def _build_defaults(self, init_dict=None):
          init_dict = init_dict or {}
          self.version = MyRobotVersion(init_dict.get("version", "v1"))
          self.arm_kind = MyRobotArmKind(init_dict.get("arm_kind", "default"))
          ...  # urdf_cfg / control_parts / solver_cfg / drive_pros / attrs
          self.solver_cfg = self._build_default_solver(arm_kind=self.arm_kind)

- ``params.py`` / ``utils.py`` — **optional** helpers (kinematic params, URDF
  assembly builders). They are not part of the protocol; factor them out only when
  the cfg file would otherwise be unwieldy.
- ``__init__.py`` — ``from .cfg import MyRobotCfg``.

``_pk_urdf_path`` for a variant robot is a method (the path depends on the variant)::

    def _pk_urdf_path(self) -> str:
        if self.arm_kind == MyRobotArmKind.KIND_A:
            return get_data_path("MyRobot/arm_a.urdf")
        return get_data_path("MyRobot/arm_b.urdf")

Registering the robot
---------------------

In ``embodichain/lab/sim/robots/__init__.py``::

    from .my_robot import MyRobotCfg

    __all__ = ["MyRobotCfg"]   # add to the existing __all__

Testing
-------

Add a ``__main__`` smoke test (round-trip + add to sim) and the DOF drift guard
(see ``tests/sim/objects/test_robot_cfg.py`` for examples)::

    chains = cfg.build_pk_serial_chain()
    for part, chain in chains.items():
        assert len(chain.get_joint_names()) == len(cfg.control_parts[part])

Use the ``preview-asset`` CLI to visually verify the URDF loads.

Serialization
------------

``to_dict`` / ``to_string`` / ``save_to_file`` are inherited from ``RobotCfg``::

    cfg.save_to_file("my_robot.json")
    cfg2 = MyRobotCfg.from_dict(cfg.to_dict())   # round-trips

Use this to snapshot tuned configs or persist calibrated parameters.

See Also
--------

- :doc:`/guides/add_robot` — Quick-reference checklist
- :doc:`/tutorial/robot` — Using robots in simulation
- :doc:`/overview/sim/solvers/index` — IK solver reference
- :doc:`/resources/robot/index` — Existing robot documentation
```

- [ ] **Step 2: Verify the broken link now resolves**

Run: `grep -rn "tutorial/add_robot" docs/source/`
Expected: references in `guides/add_robot.rst` and `agent_context/` now point to an existing file.

- [ ] **Step 3: Commit**

```bash
git add docs/source/tutorial/add_robot.rst
git commit -m "docs(robot): create the missing add_robot tutorial (Path A + Path B)"
```

---

## Task 8: Align `agent_context/topics/robot-system/robot-system.md`

**Files:**
- Modify: `agent_context/topics/robot-system/robot-system.md`

- [ ] **Step 1: Align the "Adding a New Robot" checklist + tables**

In `agent_context/topics/robot-system/robot-system.md`, update the "Adding a New Robot" section (lines ~84-97) so its steps match Task 6's checklist: override `_build_defaults` (not `from_dict` + `_build_default_cfgs`), keep `from_dict` as the 3-line template, route `build_pk_serial_chain` through `_pk_urdf_path`, mention inherited serialization. Replace the entry-points reference to the add-robot tutorial/guide with `docs/source/guides/add_robot.rst` and `docs/source/tutorial/add_robot.rst`. Ensure the Key Parameters mention `_pk_urdf_path` and variant fields, matching the guide's table verbatim.

- [ ] **Step 2: Commit**

```bash
git add agent_context/topics/robot-system/robot-system.md
git commit -m "docs(robot-context): align robot-system checklist with unified protocol"
```

---

## Task 9: Create the canonical `add-robot` skill

**Files:**
- Create: `.agents/skills/add-robot/SKILL.md`
- Create: `.agents/skills/add-robot/agents/openai.yaml`

- [ ] **Step 1: Create `SKILL.md`**

Create `.agents/skills/add-robot/SKILL.md`:

```markdown
---
name: add-robot
description: Use when adding a new robot to EmbodiChain — scaffolds a RobotCfg subclass (single-file or package layout) with the _build_defaults hook, build_pk_serial_chain, registration, docs page, and test stub.
---

# Add Robot

## When to Use

- Adding a new robot to EmbodiChain.
- Adding a variant to an existing robot (a new version / arm kind / hand brand).
- Scaffolding a `RobotCfg` subclass.

## The RobotCfg Protocol

Every robot config subclasses `RobotCfg` and overrides two hooks:

- `_build_defaults(self, init_dict=None)` — read variant fields from `init_dict`,
  set them on `self`, then populate `urdf_cfg` / `control_parts` / `solver_cfg` /
  `drive_pros` / `attrs`.
- `build_pk_serial_chain(self, device=...)` — return `{control_part: pk.SerialChain}`,
  reading the PK URDF from a single `_pk_urdf_path` source.

`from_dict` is a 3-line template (do not reimplement):

```python
cfg = cls()
cfg._build_defaults(init_dict)
return merge_robot_cfg(cfg, init_dict)
```

`to_dict` / `to_string` / `save_to_file` are inherited from `RobotCfg` and round-trip.

## Two Layouts

| Layout | When | Files |
|--------|------|-------|
| Single-file | Variant-less robot | `my_robot.py` |
| Package | Robot with variants (versions / arm kinds / hand brands) | `types.py`, `cfg.py`, optional `params.py`/`utils.py`, `__init__.py` |

## The Contract (read first)

A cfg's `_build_defaults` must populate:

- `uid` (str)
- `urdf_cfg` (URDFCfg) or `fpath`
- `control_parts` (Dict[str, List[str]]; joint names support regex)
- `solver_cfg` (Dict[str, SolverCfg]; keys match `control_parts`)
- `drive_pros` (JointDrivePropertiesCfg)
- `attrs` (RigidBodyAttributesCfg)

`build_pk_serial_chain` must read from `_pk_urdf_path` (a property for
constant-path robots, a method for variant-dependent paths). The PK chain's DOF
must match the matching `control_parts` entry (the test stub asserts this).

## Steps

1. **Pick a layout** using the table above. Single-file for variant-less robots;
   package for robots with variants.
2. **Create the cfg file(s).** Subclass `RobotCfg`. Declare variant fields (enums)
   if using the package layout.
3. **Implement `_build_defaults(self, init_dict=None)`.** Set variant fields from
   `init_dict`, then populate the Contract fields. Single-file template:

   ```python
   def _build_defaults(self, init_dict=None):
       init_dict = init_dict or {}
       self.uid = "MyRobot"
       self.urdf_cfg = URDFCfg(components=[...])
       self.control_parts = {"arm": ["JOINT[1-6]"]}
       self.solver_cfg = {"arm": OPWSolverCfg(end_link_name="link6", root_link_name="base_link")}
       self.drive_pros = JointDrivePropertiesCfg(stiffness={"JOINT[1-6]": 1e4})
   ```

   Variant-aware template (reads version / arm_kind):

   ```python
   def _build_defaults(self, init_dict=None):
       init_dict = init_dict or {}
       self.version = MyRobotVersion(init_dict.get("version", "v1"))
       self.arm_kind = MyRobotArmKind(init_dict.get("arm_kind", "default"))
       ...  # then urdf_cfg / control_parts / solver_cfg / drive_pros / attrs
   ```

4. **Implement `build_pk_serial_chain`** reading from `_pk_urdf_path`:

   ```python
   @property
   def _pk_urdf_path(self) -> str:
       return get_data_path("MyRobot/arm.urdf")

   def build_pk_serial_chain(self, device=torch.device("cpu"), **kwargs):
       chain = create_pk_serial_chain(
           urdf_path=self._pk_urdf_path, device=device,
           end_link_name="link6", root_link_name="base_link",
       )
       return {"arm": chain}
   ```

5. **Keep `from_dict` as the 3-line template** — do not reimplement it.
6. **Add `__all__` and register** in `embodichain/lab/sim/robots/__init__.py`:

   ```python
   from .my_robot import MyRobotCfg
   __all__ = ["MyRobotCfg"]
   ```

7. **Add documentation:** create `docs/source/resources/robot/<name>.md` and add
   it to `docs/source/resources/robot/index.rst`.
8. **Add a test stub** with a `__main__` smoke test + the DOF drift guard. Use
   `/add-test` for full test scaffolding; the guard snippet is:

   ```python
   chains = cfg.build_pk_serial_chain()
   for part, chain in chains.items():
       assert len(chain.get_joint_names()) == len(cfg.control_parts[part])
   ```

9. **Verify:** `preview-asset` CLI + `RobotCfg.from_dict(cfg.to_dict())` round-trip.

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| `all` instead of `__all__` | Use `__all__` — lowercase `all` breaks `import *`. |
| `solver_cfg` set twice | Set it once in `_build_defaults` only. |
| PK URDF drifts from sim URDF | Route PK through `_pk_urdf_path`; keep the DOF guard. |
| Reimplementing `from_dict` | Keep the 3-line template; put logic in `_build_defaults`. |
| `root_link_name` as a tuple | It must be a `str`. |
| Calling a nonexistent `validate` | Don't call methods that don't exist. |

## Quick Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `uid` | str | Unique robot identifier |
| `urdf_cfg` | URDFCfg | URDF file and components |
| `control_parts` | Dict[str, List[str]] | Joint groups for control |
| `solver_cfg` | Dict[str, SolverCfg] | IK solver configurations |
| `drive_pros` | JointDrivePropertiesCfg | Joint stiffness, damping, force |
| `attrs` | RigidBodyAttributesCfg | Rigid-body physics attributes |
| variant fields | enum / str / bool | Optional subclass fields |
| `_pk_urdf_path` | property or method → str | URDF for the FK/IK serial chain |

**File locations:**

- Config: `embodichain/lab/sim/robots/<name>.py` or `embodichain/lab/sim/robots/<name>/`
- Registry: `embodichain/lab/sim/robots/__init__.py`
- Docs: `docs/source/resources/robot/<name>.md`
- Tests: `tests/sim/objects/test_robot_cfg.py`
- Base class: `embodichain/lab/sim/cfg.py` (`RobotCfg`)
- Guide: `docs/source/guides/add_robot.rst` · Tutorial: `docs/source/tutorial/add_robot.rst`
```

- [ ] **Step 2: Create `agents/openai.yaml`**

Create `.agents/skills/add-robot/agents/openai.yaml`:

```yaml
name: add-robot
canonical_skill: .agents/skills/add-robot/SKILL.md
project: EmbodiChain
```

- [ ] **Step 3: Commit**

```bash
git add .agents/skills/add-robot/
git commit -m "feat(skill): add canonical add-robot skill"
```

---

## Task 10: Create the Claude + Copilot adapters

**Files:**
- Create: `.claude/skills/add-robot/SKILL.md`
- Create: `.github/copilot/add-robot.md`
- Modify: `.github/copilot/instructions.md`

- [ ] **Step 1: Create the Claude adapter**

Create `.claude/skills/add-robot/SKILL.md`:

```markdown
---
name: add-robot
description: Claude adapter for the canonical EmbodiChain add-robot skill.
---

# Add Robot - Claude Adapter

Canonical source: `.agents/skills/add-robot/`

## When to use

- Adding a new robot to EmbodiChain.
- Adding a variant to an existing robot.
- Scaffolding a `RobotCfg` subclass.

## Start here

1. Use this adapter when adding or extending a robot config.
2. Then follow `.agents/skills/add-robot/SKILL.md`.

The canonical skill covers the `RobotCfg` protocol (`_build_defaults` hook +
`build_pk_serial_chain` + inherited serialization), the single-file vs package
layouts, the 9-step scaffold, common mistakes, and a quick reference.
```

- [ ] **Step 2: Create the Copilot adapter**

Create `.github/copilot/add-robot.md`:

```markdown
# Add Robot for GitHub Copilot

Canonical source: `.agents/skills/add-robot/`

Use this adapter when adding a new robot to EmbodiChain or adding a variant to an
existing robot (a new version / arm kind / hand brand). Then follow
`.agents/skills/add-robot/SKILL.md` for the `RobotCfg` protocol (`_build_defaults`
hook + `build_pk_serial_chain` + inherited serialization), the single-file vs
package layouts, and the scaffold steps.
```

- [ ] **Step 3: Add the index entry**

In `.github/copilot/instructions.md`, add a bullet to the adapter list (match the
existing entries' format):

```markdown
- Add robots: .github/copilot/add-robot.md
```

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/add-robot/ .github/copilot/
git commit -m "feat(skill): add Claude + Copilot adapters for add-robot"
```

---

## Task 11: Format, pre-commit check, PR

**Files:** none (verification + PR)

- [ ] **Step 1: Run black**

Run: `black .`
Expected: reformats any files; review the diff.

- [ ] **Step 2: Run the full cfg test suite**

Run: `pytest tests/sim/objects/test_robot_cfg.py -v`
Expected: PASS (all pure-logic cfg tests).

- [ ] **Step 3: Run the sim regression suite (if assets/GPU available)**

Run: `pytest tests/sim/objects/test_robot.py::TestRobotCPU -v -x`
Expected: PASS — FK golden values, joint IDs, mimic, IK unchanged.

- [ ] **Step 4: Run the pre-commit check skill**

Invoke the `/pre-commit-check` skill (or `black . &&` the project's lint commands). Fix any reported violations.

- [ ] **Step 5: Commit formatting + open the PR**

```bash
git add -A
git commit -m "style: black formatting"
```

Then invoke the `/pr` skill to open the PR against `main`, summarizing: unified `RobotCfg` protocol, 10 bug fixes, refreshed `add_robot.rst` + new tutorial, and the `add-robot` skill.

---

## Self-Review

**Spec coverage:**
- §4.1 (`_build_defaults` hook + `from_dict` template + serialization in base, recursion note) → Task 1 + the explicit recursion guidance in Task 1/2. ✓
- §4.2 DexforceW1 conformance + dead-solver-set removal + `__all__`/tuple/validate fixes → Task 2. ✓
- §4.2 CobotMagic conformance + `__all__` → Task 3. ✓
- §4.2 `build_pk_serial_chain` `_pk_urdf_path` + drift guard → Task 4. ✓ (Drift guard refined to DOF-count — joint-name matching would false-positive because the PK URDF legitimately uses different names than the assembled sim URDF; DOF-count catches real kinematic drift. This is the implementation refinement the spec flagged as an open choice.)
- §4.3 guide rewrite + tutorial creation + `robot-system.md` alignment → Tasks 6, 7, 8. ✓
- §4.4 canonical skill + `openai.yaml` + Claude adapter + Copilot adapter + `instructions.md` → Tasks 9, 10. ✓
- §4.5 base round-trip test + regression + drift-guard test + backward-compatible `from_dict` signature + ordered rollout → Tasks 1-11 sequence. ✓
- §5 file list: all touched files appear in tasks. ✓
- §6 out-of-scope: `assemble_urdf()`-derived PK (not done — explicit-source approach used), `add-solver` missing `openai.yaml` (flagged in spec, not touched), no new runtime validation, no physics-backend changes. ✓

**Type/signature consistency:**
- `_build_defaults(self, init_dict=None)` — uniform across base, `_RoundTripCfg`, `CobotMagicCfg`, `DexforceW1Cfg`. ✓
- `_pk_urdf_path` — property on CobotMagic, method on DexforceW1 (matches the variant-dependent distinction). ✓
- `_build_default_solver_cfg(arm_kind=...)` and `_build_default_physics_cfgs(arm_kind=..., with_default_eef=...)` — Task 2 changes their signatures from static `(is_industrial: bool)` / `(arm_kind: str)` to instance `(arm_kind: DexforceW1ArmKind)`; the only caller is `_build_defaults`, updated in the same task. ✓
- `__all__` exports match the class names imported in `robots/__init__.py`. ✓

**Placeholder scan:** no TBD/TODO/"add appropriate X". Every code step shows complete code. ✓

One implementation choice documented above (DOF-count drift guard instead of joint-name) — a deliberate refinement, not a placeholder.
