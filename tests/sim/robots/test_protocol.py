# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Tests for the RobotDef protocol, its build_cfg method, and the robot registry."""

from __future__ import annotations

import numpy as np
import pytest

from embodichain.lab.sim.cfg import (
    JointDrivePropertiesCfg,
    RigidBodyAttributesCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.robots.protocol import RobotDef
from embodichain.lab.sim.solvers import SolverCfg

# ---------------------------------------------------------------------------
# Minimal concrete implementation for testing
# ---------------------------------------------------------------------------


class _StubRobotDef:
    """A minimal concrete class that satisfies the RobotDef protocol."""

    def __init__(
        self,
        name: str = "test_robot",
        urdf_cfg: URDFCfg | None = None,
        control_parts: dict[str, list[str]] | None = None,
        solver_cfg: dict[str, SolverCfg] | SolverCfg | None = None,
        drive_pros: JointDrivePropertiesCfg | None = None,
        attrs: RigidBodyAttributesCfg | None = None,
        min_position_iters: int = 4,
        min_velocity_iters: int = 1,
        fix_base: bool = True,
        disable_self_collision: bool = True,
        build_pk_chain: bool = True,
        init_qpos: object | None = None,
        body_scale: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> None:
        self.name = name
        self.urdf_cfg = urdf_cfg
        self.control_parts = control_parts
        self.solver_cfg = solver_cfg
        self.drive_pros = drive_pros or JointDrivePropertiesCfg()
        self.attrs = attrs or RigidBodyAttributesCfg()
        self.min_position_iters = min_position_iters
        self.min_velocity_iters = min_velocity_iters
        self.fix_base = fix_base
        self.disable_self_collision = disable_self_collision
        self.build_pk_chain = build_pk_chain
        self.init_qpos = init_qpos
        self.body_scale = body_scale


def _make_stub(**kwargs) -> _StubRobotDef:
    """Create a stub with sensible defaults for testing."""
    defaults = dict(
        control_parts={"arm": ["joint1", "joint2"]},
        drive_pros=JointDrivePropertiesCfg(stiffness=5e3, damping=5e2),
        attrs=RigidBodyAttributesCfg(mass=2.0),
        min_position_iters=8,
        min_velocity_iters=2,
        fix_base=False,
        disable_self_collision=False,
        body_scale=(2.0, 2.0, 2.0),
    )
    defaults.update(kwargs)
    return _StubRobotDef(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRobotDefProtocol:
    """Verify the RobotDef protocol is satisfied and build_cfg works."""

    def test_build_cfg_returns_robot_cfg(self) -> None:
        """Verify build_cfg returns a RobotCfg instance with correct uid."""
        stub = _make_stub(name="my_bot")
        cfg = RobotDef.build_cfg(stub)

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "my_bot"

    def test_build_cfg_applies_overrides(self) -> None:
        """Verify init_pos override works."""
        stub = _make_stub(name="my_bot")
        cfg = RobotDef.build_cfg(stub, init_pos=(1.0, 2.0, 3.0))

        assert cfg.init_pos == (1.0, 2.0, 3.0)

    def test_build_cfg_preserves_control_parts(self) -> None:
        """Verify control_parts are copied into the resulting RobotCfg."""
        parts = {"arm": ["j1", "j2"], "hand": ["j3"]}
        stub = _make_stub(control_parts=parts)
        cfg = RobotDef.build_cfg(stub)

        assert cfg.control_parts == parts

    def test_build_cfg_merges_drive_pros(self) -> None:
        """Verify partial drive_pros merge via overrides dict."""
        stub = _make_stub()
        cfg = RobotDef.build_cfg(stub, drive_pros={"stiffness": 9e3})

        # The merge should update stiffness while preserving other fields
        assert cfg.drive_pros.stiffness == 9e3
        # damping should still be from the original stub drive_pros
        assert cfg.drive_pros.damping == 5e2

    def test_build_cfg_with_solver_cfg_override(self) -> None:
        """Verify solver_cfg TCP override via merge."""
        from embodichain.lab.sim.solvers import PytorchSolverCfg

        original_solver = {
            "arm": PytorchSolverCfg(
                end_link_name="ee",
                root_link_name="base",
                tcp=np.eye(4),
            )
        }
        stub = _make_stub(solver_cfg=original_solver)

        override_tcp = np.array(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        cfg = RobotDef.build_cfg(
            stub,
            solver_cfg={"arm": {"tcp": override_tcp}},
        )

        np.testing.assert_array_almost_equal(cfg.solver_cfg["arm"].tcp, override_tcp)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

from embodichain.lab.sim.robots.registry import (
    _ROBOT_REGISTRY,
    register_robot,
    get_robot_def,
    build_robot_cfg,
)


@register_robot("TestDummy")
class _TestDummyRobotDef:
    """Minimal robot def registered as 'TestDummy' for registry tests."""

    def __init__(self, name: str = "TestDummy", **kwargs: object) -> None:
        self.name = name
        self._kwargs = kwargs

    @property
    def urdf_cfg(self) -> URDFCfg | None:
        return None

    @property
    def control_parts(self) -> dict[str, list[str]] | None:
        return {"arm": ["j1"]}

    @property
    def solver_cfg(self) -> SolverCfg | None:
        return None

    @property
    def drive_pros(self) -> JointDrivePropertiesCfg:
        return JointDrivePropertiesCfg()

    @property
    def attrs(self) -> RigidBodyAttributesCfg:
        return RigidBodyAttributesCfg()

    def build_cfg(self, **overrides: object) -> RobotCfg:
        cfg = RobotCfg()
        cfg.uid = overrides.pop("uid", self.name)
        return cfg


class TestCobotMagicDef:
    """Tests for the CobotMagic robot definition and registry integration."""

    def test_cobotmagic_def_builds_valid_cfg(self) -> None:
        """Verify CobotMagicDef produces a valid RobotCfg with correct fields."""
        from embodichain.lab.sim.robots.cobotmagic import CobotMagicDef

        robot_def = CobotMagicDef()
        cfg = robot_def.build_cfg()

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "CobotMagic"
        assert cfg.urdf_cfg is not None
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts
        assert cfg.solver_cfg is not None
        assert "left_arm" in cfg.solver_cfg
        assert "right_arm" in cfg.solver_cfg

    def test_cobotmagic_def_registry(self) -> None:
        """Verify CobotMagicDef is registered and can be looked up by name."""
        from embodichain.lab.sim.robots.registry import get_robot_def

        robot_def = get_robot_def("CobotMagic")
        cfg = robot_def.build_cfg()

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "CobotMagic"
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts

    def test_cobotmagic_backward_compat_from_dict(self) -> None:
        """Verify CobotMagicCfg.from_dict still works as backward-compatible wrapper."""
        from embodichain.lab.sim.robots.cobotmagic import CobotMagicCfg

        cfg = CobotMagicCfg.from_dict({})

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "CobotMagic"
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts


class TestDexforceW1Def:
    """Tests for the DexforceW1 robot definition and registry integration."""

    def test_anthropomorphic_default(self) -> None:
        """Verify DexforceW1Def with anthropomorphic arms produces valid cfg."""
        from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Def

        robot_def = DexforceW1Def(arm_kind="anthropomorphic")
        cfg = robot_def.build_cfg()

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "DexforceW1"
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts
        assert cfg.solver_cfg is not None
        assert "left_arm" in cfg.solver_cfg
        assert "right_arm" in cfg.solver_cfg

    def test_industrial_default(self) -> None:
        """Verify DexforceW1Def with industrial arms produces valid cfg."""
        from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Def

        robot_def = DexforceW1Def(arm_kind="industrial")
        cfg = robot_def.build_cfg()

        assert isinstance(cfg, RobotCfg)
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts

    def test_registry_lookup(self) -> None:
        """Verify DexforceW1 can be looked up via the registry."""
        cfg = build_robot_cfg("DexforceW1", arm_kind="anthropomorphic")

        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "DexforceW1"
        assert cfg.control_parts is not None

    def test_backward_compat_from_dict(self) -> None:
        """Verify DexforceW1Cfg.from_dict still works as backward-compat wrapper."""
        from embodichain.lab.sim.robots.dexforce_w1 import DexforceW1Cfg

        cfg = DexforceW1Cfg.from_dict(
            {"version": "v021", "arm_kind": "anthropomorphic"}
        )

        assert isinstance(cfg, RobotCfg)
        assert cfg.control_parts is not None
        assert "left_arm" in cfg.control_parts
        assert "right_arm" in cfg.control_parts


class TestRobotRegistry:
    """Tests for the robot registry (register_robot, get_robot_def, build_robot_cfg)."""

    def test_register_and_lookup(self) -> None:
        """Verify that a registered robot can be looked up and instantiated."""
        instance = get_robot_def("TestDummy")
        assert instance.name == "TestDummy"

    def test_get_unknown_robot_raises(self) -> None:
        """Verify that looking up an unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown robot"):
            get_robot_def("NonExistentRobotXYZ")

    def test_build_robot_cfg_convenience(self) -> None:
        """Verify build_robot_cfg returns a RobotCfg with the correct uid."""
        cfg = build_robot_cfg("TestDummy")
        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "TestDummy"

    def test_build_robot_cfg_with_overrides(self) -> None:
        """Verify build_robot_cfg passes overrides through to build_cfg."""
        cfg = build_robot_cfg("TestDummy", overrides={"uid": "custom_uid"})
        assert isinstance(cfg, RobotCfg)
        assert cfg.uid == "custom_uid"
