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

"""Tests for backend-neutral URDF source mass-property handling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dexsim.spawn import ArticulationDesc, LinkDesc

from embodichain.lab.sim.spawn.source import (
    _apply_dexsim_source_overlay,
    _capture_dexsim_source_physics,
    _clear_invalid_source_com,
)

pytestmark = pytest.mark.no_sim


def _link(name: str) -> LinkDesc:
    return LinkDesc(name, "", np.eye(4, dtype=np.float32))


def _physical_attr(
    mass: float,
    inertia: tuple[float, float, float],
    com_position: tuple[float, float, float] = (0.1, 0.2, 0.3),
) -> SimpleNamespace:
    return SimpleNamespace(
        mass=mass,
        inertia=np.asarray(inertia, dtype=np.float32),
        com_position=np.asarray(com_position, dtype=np.float32),
        com_quaternion=np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32),
    )


def test_default_source_capture_keeps_valid_inertia_and_discards_invalid_com(
    tmp_path,
) -> None:
    urdf_path = tmp_path / "source.urdf"
    urdf_path.write_text(
        """
        <robot name="source">
          <link name="valid">
            <inertial>
              <mass value="2.0"/>
              <inertia ixx="1.0" ixy="0.0" ixz="0.0"
                       iyy="2.0" iyz="0.0" izz="3.0"/>
            </inertial>
          </link>
          <link name="invalid">
            <inertial>
              <mass value="0.5"/>
              <inertia ixx="0.0" ixy="0.0" ixz="0.0"
                       iyy="0.0" iyz="0.0" izz="0.0"/>
            </inertial>
          </link>
          <link name="unowned"/>
        </robot>
        """,
        encoding="utf-8",
    )
    links = [_link("valid"), _link("invalid"), _link("unowned")]
    desc = ArticulationDesc(
        name="source",
        links=links,
        urdf_path=str(urdf_path),
    )
    attrs = {
        "valid": _physical_attr(2.0, (1.0, 2.0, 3.0)),
        # The native loader exposes an epsilon tensor for the zero source
        # inertia. The source XML, not this fallback, controls provenance.
        "invalid": _physical_attr(0.5, (1.0e-6, 1.0e-6, 1.0e-6)),
        "unowned": _physical_attr(1.0, (0.2, 0.3, 0.4)),
    }
    handle = SimpleNamespace(get_physical_attr=lambda name: attrs[name])

    _capture_dexsim_source_physics(handle, desc)

    valid = desc.get_link_desc("valid")
    assert valid.rigid_body is not None
    assert valid.rigid_body.mass == pytest.approx(2.0)
    np.testing.assert_array_equal(valid.rigid_body.inertia, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(valid.rigid_body.com_position, (0.1, 0.2, 0.3))
    assert valid._inertia_from_source

    invalid = desc.get_link_desc("invalid")
    assert invalid.rigid_body is not None
    assert invalid.rigid_body.mass == pytest.approx(0.5)
    assert invalid.rigid_body.inertia is None
    assert invalid.rigid_body.com_position is None
    assert invalid.rigid_body.com_quaternion is None
    assert not invalid._inertia_from_source

    unowned = desc.get_link_desc("unowned")
    assert unowned.rigid_body is not None
    assert unowned.rigid_body.mass is None
    assert unowned.rigid_body.inertia is None
    assert not unowned._embodichain_source_inertia_valid


def test_invalid_source_com_normalization_does_not_touch_authored_inertia() -> None:
    source = _link("source")
    source.rigid_body = source_body = _physical_body(
        inertia=None,
        com_position=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
    )
    authored = _link("authored")
    authored.rigid_body = authored_body = _physical_body(
        inertia=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        com_position=np.asarray((4.0, 5.0, 6.0), dtype=np.float32),
    )
    authored._inertia_from_source = True
    desc = ArticulationDesc(name="robot", links=[source, authored])
    source._embodichain_source_inertia_valid = False
    source._embodichain_has_collision_geometry = True

    _clear_invalid_source_com(desc)

    assert source_body.com_position is None
    np.testing.assert_array_equal(authored_body.com_position, (4.0, 5.0, 6.0))


def test_default_overlay_writes_only_explicitly_marked_link_physics() -> None:
    skipped = _link("skipped")
    applied = _link("applied")
    for link in (skipped, applied):
        link.rigid_body = _physical_body(
            inertia=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
            com_position=np.asarray((0.1, 0.2, 0.3), dtype=np.float32),
        )
    skipped._embodichain_apply_physics = False
    applied._embodichain_apply_physics = True
    applied._embodichain_mass_override = True
    applied.rigid_body.mass = 2.0
    desc = ArticulationDesc(name="robot", links=[skipped, applied])

    raw = _RawBody()
    binding = SimpleNamespace(
        get_physical_body=lambda name: raw if name == "applied" else None,
        set_physical_attr=lambda _attr, name, _replace: raw.calls.append(
            ("attr", name)
        ),
    )
    handle = SimpleNamespace(
        _physics_binding=binding,
        articulation_desc=ArticulationDesc(name="robot", links=[]),
        _desc_shared=True,
    )

    _apply_dexsim_source_overlay(handle, desc)

    assert raw.calls[0] == ("attr", "applied")
    assert raw.mass == pytest.approx(2.0)
    np.testing.assert_array_equal(raw.inertia, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(raw.com_position, (0.1, 0.2, 0.3))
    assert handle.articulation_desc.get_link_desc("skipped").rigid_body is not None


class _RawBody:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.mass = 1.0
        self.inertia = np.ones(3, dtype=np.float32)
        self.com_position = np.zeros(3, dtype=np.float32)
        self.com_quaternion = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    def set_mass(self, mass: float) -> None:
        self.mass = mass

    def set_mass_space_inertia_tensor(self, inertia: np.ndarray) -> None:
        self.inertia = np.asarray(inertia, dtype=np.float32)

    def get_cmass_local_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.com_position, self.com_quaternion

    def set_cmass_local_pose(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> None:
        self.com_position = np.asarray(position, dtype=np.float32)
        self.com_quaternion = np.asarray(quaternion, dtype=np.float32)


def _physical_body(*, inertia, com_position):
    from dexsim.spawn import RigidBodyPhysicsDesc

    return RigidBodyPhysicsDesc.dynamic(
        mass=1.0,
        inertia=inertia,
        com_position=com_position,
    )
