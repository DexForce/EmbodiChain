# Robot Definition Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-robot boilerplate with a standard `RobotDef` protocol, generic `build_cfg`, and a registry for robot lookup by name.

**Architecture:** A `RobotDef` protocol defines the contract every robot must satisfy (urdf_cfg, control_parts, solver_cfg, drive_pros, attrs, build_pk_serial_chain). A `build_cfg()` default implementation converts any `RobotDef` into a `RobotCfg` for the spawner. A `@register_robot` decorator + `build_robot_cfg()` function provide name-based lookup. Existing `RobotCfg` subclasses become thin backward-compat wrappers.

**Tech Stack:** Python dataclasses, `@configclass`, existing `merge_robot_cfg`, `typing.Protocol`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `embodichain/lab/sim/robots/protocol.py` | `RobotDef` protocol + generic `build_cfg` base mixin |
| Create | `embodichain/lab/sim/robots/registry.py` | `@register_robot`, `get_robot_def`, `build_robot_cfg` |
| Create | `tests/sim/robots/test_protocol.py` | Unit tests for protocol, registry, build_cfg |
| Refactor | `embodichain/lab/sim/robots/cobotmagic.py` | `CobotMagicDef` replaces `CobotMagicCfg` |
| Refactor | `embodichain/lab/sim/robots/dexforce_w1/def.py` | New `DexforceW1Def` replaces `cfg.py` |
| Modify | `embodichain/lab/sim/robots/dexforce_w1/utils.py` | Extract helper functions, keep existing managers |
| Modify | `embodichain/lab/sim/robots/__init__.py` | Export new defs + backward-compat aliases |
| Modify | `embodichain/lab/sim/robots/dexforce_w1/__init__.py` | Export `DexforceW1Def` + backward-compat `DexforceW1Cfg` |
| Modify | `docs/source/guides/add_robot.rst` | Update to reflect new RobotDef protocol |
| Update | `tests/sim/objects/test_robot.py` | Verify existing tests still pass |

---

### Task 1: Create RobotDef Protocol and Base Mixin

**Files:**
- Create: `embodichain/lab/sim/robots/protocol.py`

- [ ] **Step 1: Write the test for RobotDef protocol**

```python
# tests/sim/robots/test_protocol.py

import numpy as np
import pytest
import torch

from embodichain.lab.sim.robots.protocol import RobotDef
from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg


class DummyRobotDef:
    """Minimal RobotDef implementation for testing."""

    name: str = "DummyRobot"
    urdf_cfg: URDFCfg = None
    control_parts: dict = {}
    solver_cfg: dict = {}
    drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg()
    attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()

    def build_pk_serial_chain(self, device):
        return {}


class TestRobotDefProtocol:

    def test_build_cfg_returns_robot_cfg(self):
        robot_def = DummyRobotDef()
        cfg = robot_def.build_cfg(uid="test_robot")
        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "test_robot"

    def test_build_cfg_applies_overrides(self):
        robot_def = DummyRobotDef()
        cfg = robot_def.build_cfg(uid="test", init_pos=(0.0, 0.0, 1.0))
        assert cfg.init_pos == (0.0, 0.0, 1.0)

    def test_build_cfg_preserves_control_parts(self):
        robot_def = DummyRobotDef()
        robot_def.control_parts = {"arm": ["J1", "J2"]}
        cfg = robot_def.build_cfg()
        assert cfg.control_parts == {"arm": ["J1", "J2"]}

    def test_build_cfg_merges_drive_pros(self):
        robot_def = DummyRobotDef()
        robot_def.drive_pros = JointDrivePropertiesCfg(
            stiffness={"J1": 1e4, "J2": 1e3},
            damping={"J1": 1e3, "J2": 1e2},
        )
        cfg = robot_def.build_cfg(
            drive_pros={"stiffness": {"J1": 5e4}}
        )
        assert cfg.drive_pros.stiffness["J1"] == 5e4
        assert cfg.drive_pros.stiffness["J2"] == 1e3

    def test_build_cfg_with_solver_cfg_override(self):
        robot_def = DummyRobotDef()
        robot_def.solver_cfg = {"arm": SolverCfg(end_link_name="link6")}
        cfg = robot_def.build_cfg(
            solver_cfg={"arm": {"tcp": np.eye(4)}}
        )
        assert np.allclose(cfg.solver_cfg["arm"].tcp, np.eye(4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py -v`
Expected: FAIL — `module 'embodichain.lab.sim.robots.protocol' not found`

- [ ] **Step 3: Create protocol.py with RobotDef and build_cfg**

```python
# embodichain/lab/sim/robots/protocol.py

from __future__ import annotations

import torch
import numpy as np

from typing import Dict, List, Union, runtime_checkable, Protocol

from embodichain.lab.sim.cfg import (
    RobotCfg,
    URDFCfg,
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
)
from embodichain.lab.sim.solvers import SolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import logger

__all__ = ["RobotDef"]


@runtime_checkable
class RobotDef(Protocol):
    """Standard protocol that every robot definition must satisfy.

    Simple robots declare data as class-level fields.
    Complex robots with variants use @property methods that compute
    values based on constructor parameters.
    """

    name: str

    @property
    def urdf_cfg(self) -> URDFCfg: ...

    @property
    def control_parts(self) -> Dict[str, List[str]]: ...

    @property
    def solver_cfg(self) -> Dict[str, SolverCfg]: ...

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg: ...

    @property
    def attrs(self) -> RigidBodyAttributesCfg: ...

    def build_pk_serial_chain(
        self, device: torch.device
    ) -> Dict[str, object]:
        ...

    def build_cfg(self, **overrides) -> RobotCfg:
        """Convert this definition into a RobotCfg for the spawner.

        Creates a RobotCfg populated with protocol properties,
        then applies user overrides via merge_robot_cfg.
        """
        cfg = RobotCfg()
        cfg.uid = overrides.pop("uid", self.name)

        if self.urdf_cfg is not None:
            cfg.urdf_cfg = self.urdf_cfg

        if self.control_parts is not None:
            cfg.control_parts = self.control_parts

        if self.solver_cfg is not None:
            cfg.solver_cfg = self.solver_cfg

        cfg.drive_pros = self.drive_pros
        cfg.attrs = self.attrs

        # Copy extra fields from the def that map to RobotCfg fields
        for attr_name in ("min_position_iters", "min_velocity_iters",
                          "fix_base", "disable_self_collision",
                          "build_pk_chain", "init_qpos", "body_scale"):
            val = getattr(self, attr_name, None)
            if val is not None and hasattr(cfg, attr_name):
                setattr(cfg, attr_name, val)

        # Override build_pk_serial_chain on the cfg instance
        if hasattr(self, "build_pk_serial_chain"):
            cfg.build_pk_serial_chain = self.build_pk_serial_chain

        # Apply user overrides (init_pos, drive_pros, solver_cfg, etc.)
        if overrides:
            cfg = merge_robot_cfg(cfg, overrides)

        return cfg
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestRobotDefProtocol -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/robots/protocol.py tests/sim/robots/test_protocol.py
git commit -m "feat: add RobotDef protocol with generic build_cfg"
```

---

### Task 2: Create Robot Registry

**Files:**
- Create: `embodichain/lab/sim/robots/registry.py`
- Modify: `tests/sim/robots/test_protocol.py` — add registry tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/sim/robots/test_protocol.py`:

```python
from embodichain.lab.sim.robots.registry import (
    register_robot,
    get_robot_def,
    build_robot_cfg,
    _ROBOT_REGISTRY,
)


class TestRobotRegistry:

    def test_register_and_lookup(self):
        @register_robot("TestDummy")
        class TestDef:
            name: str = "TestDummy"
            urdf_cfg = None
            control_parts = {}
            solver_cfg = {}
            drive_pros = JointDrivePropertiesCfg()
            attrs = RigidBodyAttributesCfg()

            def build_pk_serial_chain(self, device):
                return {}

            def build_cfg(self, **overrides):
                from embodichain.lab.sim.robots.protocol import RobotDef
                return RobotDef.build_cfg(self, **overrides)

        assert "TestDummy" in _ROBOT_REGISTRY
        robot_def = get_robot_def("TestDummy")
        assert robot_def.name == "TestDummy"

    def test_get_unknown_robot_raises(self):
        with pytest.raises(ValueError, match="Unknown robot"):
            get_robot_def("NonExistentRobot999")

    def test_build_robot_cfg_convenience(self):
        robot_def = get_robot_def("TestDummy")
        cfg = build_robot_cfg("TestDummy", overrides={"uid": "my_test"})
        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "my_test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestRobotRegistry -v`
Expected: FAIL — `module 'embodichain.lab.sim.robots.registry' not found`

- [ ] **Step 3: Create registry.py**

```python
# embodichain/lab/sim/robots/registry.py

from __future__ import annotations

from typing import Dict, Type

from embodichain.lab.sim.cfg import RobotCfg
from embodichain.utils import logger

__all__ = ["register_robot", "get_robot_def", "build_robot_cfg"]

_ROBOT_REGISTRY: Dict[str, Type] = {}


def register_robot(name: str):
    """Decorator to register a robot definition class.

    Args:
        name: Unique name for the robot (e.g., "CobotMagic", "DexforceW1").

    Returns:
        The class unchanged, after registering it.
    """
    def wrapper(cls):
        if name in _ROBOT_REGISTRY:
            logger.log_warning(
                f"Robot '{name}' already registered. Overwriting."
            )
        _ROBOT_REGISTRY[name] = cls
        return cls
    return wrapper


def get_robot_def(name: str, **variant_kwargs) -> object:
    """Look up a robot definition by name and instantiate it.

    Args:
        name: Registered robot name.
        **variant_kwargs: Constructor arguments for the robot def class
            (e.g., arm_kind, version, hand_types for DexforceW1).

    Returns:
        An instance of the robot definition class.

    Raises:
        ValueError: If the robot name is not registered.
    """
    if name not in _ROBOT_REGISTRY:
        raise ValueError(
            f"Unknown robot: '{name}'. "
            f"Available: {list(_ROBOT_REGISTRY.keys())}"
        )
    return _ROBOT_REGISTRY[name](**variant_kwargs)


def build_robot_cfg(name: str, **kwargs) -> RobotCfg:
    """Look up a robot by name and build a RobotCfg.

    Args:
        name: Registered robot name.
        **kwargs: Variant kwargs are passed to the constructor.
            Use ``overrides={...}`` dict for cfg-level overrides
            (uid, init_pos, drive_pros, solver_cfg, etc.).

    Returns:
        RobotCfg ready for SimulationManager.add_robot().
    """
    overrides = kwargs.pop("overrides", {})
    robot_def = get_robot_def(name, **kwargs)
    return robot_def.build_cfg(**overrides)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestRobotRegistry -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/robots/registry.py tests/sim/robots/test_protocol.py
git commit -m "feat: add robot registry with register_robot and build_robot_cfg"
```

---

### Task 3: Refactor CobotMagic to use RobotDef

**Files:**
- Modify: `embodichain/lab/sim/robots/cobotmagic.py`
- Modify: `tests/sim/robots/test_protocol.py` — add CobotMagic tests

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/robots/test_protocol.py`:

```python
from embodichain.lab.sim.robots.cobotmagic import CobotMagicDef


class TestCobotMagicDef:

    def test_cobotmagic_def_builds_valid_cfg(self):
        robot_def = CobotMagicDef()
        cfg = robot_def.build_cfg(uid="test_cobot")
        assert cfg.uid == "test_cobot"
        assert cfg.urdf_cfg is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts
        assert "left_arm" in cfg.solver_cfg
        assert "right_arm" in cfg.solver_cfg

    def test_cobotmagic_def_registry(self):
        from embodichain.lab.sim.robots.registry import get_robot_def
        robot_def = get_robot_def("CobotMagic")
        cfg = robot_def.build_cfg(uid="from_registry")
        assert cfg.uid == "from_registry"
        assert "left_arm" in cfg.control_parts

    def test_cobotmagic_backward_compat_from_dict(self):
        from embodichain.lab.sim.robots.cobotmagic import CobotMagicCfg
        cfg = CobotMagicCfg.from_dict({"uid": "compat_test"})
        assert cfg.uid == "compat_test"
        assert "left_arm" in cfg.control_parts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestCobotMagicDef -v`
Expected: FAIL — `ImportError: cannot import name 'CobotMagicDef'`

- [ ] **Step 3: Refactor cobotmagic.py**

Replace the file content with the new `CobotMagicDef` class and a backward-compatible `CobotMagicCfg` wrapper. The file should:

1. Keep the Apache 2.0 header.
2. Import `from embodichain.lab.sim.robots.protocol import RobotDef`.
3. Import `from embodichain.lab.sim.robots.registry import register_robot`.
4. Define `@register_robot("CobotMagic") class CobotMagicDef` with flat class-level declarations for:
   - `name = "CobotMagic"`
   - `urdf_cfg` — the same `URDFCfg(components=[...])` as the current `_build_default_cfgs` but as a class-level field
   - `control_parts` — same dict as current
   - `solver_cfg` — same dict with `OPWSolverCfg` as current
   - `drive_pros` — same `JointDrivePropertiesCfg` as current
   - `attrs` — same `RigidBodyAttributesCfg` as current
   - `min_position_iters = 8`, `min_velocity_iters = 2`
   - `build_pk_serial_chain` — same logic as current
   - `build_cfg` — inherited from `RobotDef` protocol (call `RobotDef.build_cfg(self, **overrides)`)
5. Define `CobotMagicCfg` as backward-compat wrapper:

```python
class CobotMagicCfg(RobotCfg):
    """Backward-compatible CobotMagic config. Delegates to CobotMagicDef."""

    @classmethod
    def from_dict(cls, init_dict):
        return CobotMagicDef().build_cfg(**init_dict)
```

6. Keep the `if __name__ == "__main__"` block but update to use `CobotMagicDef`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestCobotMagicDef -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/robots/cobotmagic.py tests/sim/robots/test_protocol.py
git commit -m "refactor: replace CobotMagicCfg with CobotMagicDef + backward-compat wrapper"
```

---

### Task 4: Refactor DexforceW1 to use RobotDef

**Files:**
- Create: `embodichain/lab/sim/robots/dexforce_w1/def.py`
- Modify: `embodichain/lab/sim/robots/dexforce_w1/__init__.py`
- Modify: `tests/sim/robots/test_protocol.py` — add DexforceW1 tests

This is the largest task. The existing `cfg.py` (390 LOC) gets replaced by `def.py` (~80 LOC) that delegates to helper functions extracted from the current static methods.

- [ ] **Step 1: Write the failing test**

Append to `tests/sim/robots/test_protocol.py`:

```python
from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Def, DexforceW1Cfg


class TestDexforceW1Def:

    def test_anthropomorphic_default(self):
        robot_def = DexforceW1Def(arm_kind="anthropomorphic")
        cfg = robot_def.build_cfg(uid="w1_test")
        assert cfg.uid == "w1_test"
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts
        assert "left_arm" in cfg.solver_cfg

    def test_industrial_default(self):
        robot_def = DexforceW1Def(arm_kind="industrial")
        cfg = robot_def.build_cfg(uid="w1_ind")
        assert cfg.uid == "w1_ind"
        assert "left_arm" in cfg.control_parts

    def test_registry_lookup(self):
        from embodichain.lab.sim.robots.registry import build_robot_cfg
        cfg = build_robot_cfg(
            "DexforceW1",
            arm_kind="anthropomorphic",
            overrides={"uid": "from_reg"},
        )
        assert cfg.uid == "from_reg"

    def test_backward_compat_from_dict(self):
        cfg = DexforceW1Cfg.from_dict({
            "uid": "compat_w1",
            "version": "v021",
            "arm_kind": "anthropomorphic",
        })
        assert cfg.uid == "compat_w1"
        assert "left_arm" in cfg.control_parts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestDexforceW1Def -v`
Expected: FAIL — `ImportError: cannot import name 'DexforceW1Def'`

- [ ] **Step 3: Create dexforce_w1/def.py**

Create the new `def.py` file with:

1. Apache 2.0 header.
2. Imports from `types.py`, `params.py`, `utils.py`, `protocol.py`, `registry.py`, and `cfg.py` (base classes).
3. `@register_robot("DexforceW1") class DexforceW1Def` with:
   - `name = "DexforceW1"`
   - Variant fields: `version`, `arm_kind`, `hand_types`, `include_chassis/torso/head/hand`
   - `__post_init__` that fills default `hand_types` based on `arm_kind`
   - `@property urdf_cfg` — delegates to `_build_urdf_cfg()` helper
   - `@property control_parts` — delegates to `_build_control_parts()` helper
   - `@property solver_cfg` — delegates to `_build_solver_cfg()` helper
   - `@property drive_pros` — delegates to `_build_drive_pros()` helper
   - `@property attrs` — returns the standard `RigidBodyAttributesCfg`
   - `min_position_iters = 32`, `min_velocity_iters = 8`
   - `build_pk_serial_chain` — same logic as current `DexforceW1Cfg.build_pk_serial_chain`
   - `build_cfg` — inherited from protocol

The helper functions (`_build_urdf_cfg`, `_build_control_parts`, `_build_solver_cfg`, `_build_drive_pros`) are extracted from the current `DexforceW1Cfg._build_default_cfg`, `_build_default_physics_cfgs`, `_build_default_solver_cfg` and the existing `utils.py` functions (`build_dexforce_w1_assembly_urdf_cfg`, `build_dexforce_w1_cfg`, `build_dexforce_w1_solver_cfg`). These helpers live in `def.py` as module-level functions.

- [ ] **Step 4: Update dexforce_w1/__init__.py**

```python
from .def import DexforceW1Def
from .cfg import DexforceW1Cfg
```

The old `cfg.py` file is kept as-is for backward compatibility, with `DexforceW1Cfg` now delegating to `DexforceW1Def`:

```python
class DexforceW1Cfg(RobotCfg):
    """Backward-compatible DexforceW1 config. Delegates to DexforceW1Def."""
    # ... keep existing fields for type-checking ...

    @classmethod
    def from_dict(cls, init_dict):
        from embodichain.lab.sim.robots.dexforce_w1.def import DexforceW1Def
        version = init_dict.get("version", "v021")
        arm_kind = init_dict.get("arm_kind", "anthropomorphic")
        with_default_eef = init_dict.get("with_default_eef", True)
        remaining = {k: v for k, v in init_dict.items()
                     if k not in ("version", "arm_kind", "with_default_eef")}
        return DexforceW1Def(
            version=version, arm_kind=arm_kind,
            include_hand=with_default_eef,
        ).build_cfg(**remaining)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py::TestDexforceW1Def -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/robots/dexforce_w1/def.py embodichain/lab/sim/robots/dexforce_w1/__init__.py embodichain/lab/sim/robots/dexforce_w1/cfg.py tests/sim/robots/test_protocol.py
git commit -m "refactor: add DexforceW1Def with RobotDef protocol + backward-compat DexforceW1Cfg"
```

---

### Task 5: Update __init__.py exports

**Files:**
- Modify: `embodichain/lab/sim/robots/__init__.py`

- [ ] **Step 1: Update the robots __init__.py**

```python
from .protocol import RobotDef
from .registry import register_robot, get_robot_def, build_robot_cfg
from .cobotmagic import CobotMagicDef, CobotMagicCfg
from .dexforce_w1 import DexforceW1Def, DexforceW1Cfg

__all__ = [
    "RobotDef",
    "register_robot",
    "get_robot_def",
    "build_robot_cfg",
    "CobotMagicDef",
    "CobotMagicCfg",
    "DexforceW1Def",
    "DexforceW1Cfg",
]
```

- [ ] **Step 2: Run all robot tests**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/ tests/sim/objects/test_robot.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add embodichain/lab/sim/robots/__init__.py
git commit -m "feat: export RobotDef, registry, and new robot defs from __init__"
```

---

### Task 6: Update add_robot.rst documentation

**Files:**
- Modify: `docs/source/guides/add_robot.rst`

- [ ] **Step 1: Rewrite add_robot.rst to reflect the new RobotDef protocol**

Replace the content with updated guide that:

1. Updates the checklist to reference `RobotDef` instead of `from_dict` and `_build_default_cfgs`
2. Shows two approaches using the new system:
   - **Simple robot** (single-file): `@register_robot` + flat `RobotDef` declarations
   - **Complex robot** (package): variant parameters + `@property` methods + extracted helpers
3. Provides complete code examples for both
4. Shows how to use `build_robot_cfg()` and `SimulationManager.add_robot()`
5. Documents backward compatibility with the old `*Cfg.from_dict()` API

Key content:

```rst
.. _guide_add_robot:

Adding a New Robot — Quick Reference
=====================================

This guide provides a checklist and key reference for adding a new robot
to EmbodiChain using the ``RobotDef`` protocol.

Checklist
---------

1. **Prepare the URDF** — Place your URDF file (and associated meshes)
   in the robot assets directory.
2. **Create the robot definition** — Implement the ``RobotDef`` protocol
   with ``@register_robot("MyRobot")``.
3. **Define control parts** — Group joints into logical sets
   (e.g., ``arm``, ``gripper``).
4. **Configure IK solver** — Choose ``OPWSolverCfg``, ``SRSSolverCfg``,
   or a generic ``SolverCfg`` per control part.
5. **Set drive properties** — Configure stiffness, damping, and max
   effort per joint group.
6. **Implement** ``build_pk_serial_chain`` — Required for
   PyTorch-Kinematics IK support.
7. **Register in** ``embodichain/lab/sim/robots/__init__.py``.
8. **Add documentation** — Create ``docs/source/resources/robot/my_robot.md``
   and update ``resources/robot/index.rst``.
9. **Test** — Add a ``__main__`` block or use the ``preview-asset`` CLI
   to verify.

Approaches
----------

Simple robot (single-file)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For robots without variants, declare all config as class-level fields:

.. code-block:: python

    from embodichain.lab.sim.robots.protocol import RobotDef
    from embodichain.lab.sim.robots.registry import register_robot

    @register_robot("MyRobot")
    class MyRobotDef:
        name: str = "MyRobot"

        urdf_cfg: URDFCfg = URDFCfg(components=[
            {"component_type": "arm",
             "urdf_path": get_data_path("MyRobot/arm.urdf")},
        ])

        control_parts: Dict[str, List[str]] = {
            "arm": ["JOINT[1-6]"],
            "gripper": ["FINGER[1-2]"],
        }

        solver_cfg: Dict[str, SolverCfg] = {
            "arm": OPWSolverCfg(
                end_link_name="link6",
                root_link_name="base_link",
            ),
        }

        drive_pros: JointDrivePropertiesCfg = JointDrivePropertiesCfg(
            stiffness={"JOINT[1-6]": 1e4, "FINGER[1-2]": 1e2},
            damping={"JOINT[1-6]": 1e3, "FINGER[1-2]": 1e1},
        )

        attrs: RigidBodyAttributesCfg = RigidBodyAttributesCfg()

        def build_pk_serial_chain(self, device):
            return {}

        def build_cfg(self, **overrides):
            return RobotDef.build_cfg(self, **overrides)

Usage::

    from embodichain.lab.sim.robots import build_robot_cfg
    cfg = build_robot_cfg("MyRobot", overrides={"uid": "my_robot"})
    robot = sim.add_robot(cfg=cfg)

Complex robot (package)
~~~~~~~~~~~~~~~~~~~~~~~

For robots with variants (arm types, hand brands, versions), use
``@property`` methods and variant parameters:

.. code-block:: python

    @register_robot("MyComplexRobot")
    class MyComplexRobotDef:
        name: str = "MyComplexRobot"

        # Variant parameters
        arm_kind: str = "standard"

        def __post_init__(self):
            # Fill defaults based on arm_kind
            ...

        @property
        def urdf_cfg(self) -> URDFCfg:
            return _build_urdf(self.arm_kind)

        @property
        def control_parts(self) -> Dict[str, List[str]]:
            return _build_control_parts(self.arm_kind)

        @property
        def solver_cfg(self) -> Dict[str, SolverCfg]:
            return _build_solver_cfg(self.arm_kind)

        @property
        def drive_pros(self) -> JointDrivePropertiesCfg:
            return _build_drive_pros(self.arm_kind)

        @property
        def attrs(self) -> RigidBodyAttributesCfg:
            return RigidBodyAttributesCfg()

        def build_pk_serial_chain(self, device):
            return {}

        def build_cfg(self, **overrides):
            return RobotDef.build_cfg(self, **overrides)

Key Parameters
--------------

+---------------------+----------------------------+----------------------------------+
| Parameter           | Type                       | Description                      |
+=====================+============================+==================================+
| ``name``            | str                        | Unique robot identifier          |
+---------------------+----------------------------+----------------------------------+
| ``urdf_cfg``        | URDFCfg                    | URDF file and components         |
+---------------------+----------------------------+----------------------------------+
| ``control_parts``   | Dict[str, List[str]]       | Joint groups for control         |
+---------------------+----------------------------+----------------------------------+
| ``solver_cfg``      | Dict[str, SolverCfg]       | IK solver configurations         |
+---------------------+----------------------------+----------------------------------+
| ``drive_pros``      | JointDrivePropertiesCfg    | Joint stiffness, damping, force  |
+---------------------+----------------------------+----------------------------------+
| ``attrs``           | RigidBodyAttributesCfg     | Physical attributes              |
+---------------------+----------------------------+----------------------------------+

Backward Compatibility
----------------------

The old ``*Cfg.from_dict()`` API continues to work. ``CobotMagicCfg``
and ``DexforceW1Cfg`` are thin wrappers that delegate to the new
``RobotDef`` implementations.

.. tip::

   See :doc:`/tutorial/robot` for using robots in simulation and
   :doc:`/resources/robot/index` for existing robot documentation.
```

- [ ] **Step 2: Commit**

```bash
git add docs/source/guides/add_robot.rst
git commit -m "docs: update add_robot guide for RobotDef protocol"
```

---

### Task 7: Run full test suite and verify

**Files:** None (verification only)

- [ ] **Step 1: Run protocol tests**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/robots/test_protocol.py -v`
Expected: All PASS

- [ ] **Step 2: Run existing robot tests**

Run: `cd /root/sources/EmbodiChain && python -m pytest tests/sim/objects/test_robot.py -v`
Expected: All PASS (backward compatibility verified)

- [ ] **Step 3: Run black formatter**

Run: `cd /root/sources/EmbodiChain && black embodichain/lab/sim/robots/ tests/sim/robots/test_protocol.py`
Expected: No errors

- [ ] **Step 4: Commit any formatting changes**

```bash
git add -u
git commit -m "style: apply black formatting to robot definition files"
```
