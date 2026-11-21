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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    MimicComplianceCfg,
    RobotCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.objects import Articulation
from embodichain.lab.sim.shapes import CubeCfg


@pytest.mark.parametrize("cfg_type", [ArticulationCfg, RobotCfg])
def test_mimic_compliance_config_round_trips(cfg_type: type[ArticulationCfg]) -> None:
    settings = {"width": {"natural_frequency": 0.0, "damping_ratio": 0.0}}
    cfg = cfg_type.from_dict({"mimic_compliance": settings})
    assert cfg.mimic_compliance == {"width": MimicComplianceCfg(0.0, 0.0)}
    assert cfg.to_dict()["mimic_compliance"] == settings


def _make_articulation(backend: str = "dexsim") -> Articulation:
    art = Articulation(ArticulationCfg(uid="gripper"))
    art._spawn_result = SimpleNamespace(backend=backend, topology_revision=1)
    art._entities = [
        Mock(
            get_actived_joint_names=Mock(return_value=order),
            get_mimic_info=Mock(
                return_value=SimpleNamespace(mimic_parent=[order.index("width")])
            ),
            set_mimic_compliance=Mock(return_value=0),
        )
        for order in (["width", "slide", "finger"], ["slide", "finger", "width"])
    ]
    return art


@pytest.mark.no_sim
def test_set_mimic_compliance_uses_selected_instances_native_cache_order() -> None:
    art = _make_articulation()
    art.set_mimic_compliance(0.0, 0.0, ["width"], env_ids=torch.tensor([1]))
    art._entities[0].set_mimic_compliance.assert_not_called()
    art._entities[1].get_actived_joint_names.assert_called_once_with(
        in_cache_order=True
    )
    art._entities[1].set_mimic_compliance.assert_called_once_with(0.0, 0.0, [2])


@pytest.mark.no_sim
@pytest.mark.parametrize("joint", ["finger", "slide", "missing"])
def test_set_mimic_compliance_rejects_non_masters_before_writing(joint: str) -> None:
    art = _make_articulation()
    with pytest.raises(ValueError, match="not an active mimic master"):
        art.set_mimic_compliance(0.0, 0.0, ["width", joint])
    for entity in art._entities:
        entity.set_mimic_compliance.assert_not_called()


@pytest.mark.no_sim
def test_set_mimic_compliance_reports_sdk_status() -> None:
    art = _make_articulation()
    art._entities[1].set_mimic_compliance.return_value = 7
    with pytest.raises(RuntimeError, match=r"gripper.*env 1.*width.*status 7"):
        art.set_mimic_compliance(0.0, 0.0, "width")


@pytest.mark.no_sim
@pytest.mark.parametrize(
    ("frequency", "ratio", "env_ids", "error"),
    [
        (-1.0, 0.0, None, ValueError),
        (0.0, float("nan"), None, ValueError),
        (True, 0.0, None, TypeError),
        (0.0, 0.0, [0, -1], IndexError),
        (0.0, 0.0, [0, 2], IndexError),
        (0.0, 0.0, [1.5], TypeError),
    ],
)
def test_set_mimic_compliance_validates_parameters_before_writing(
    frequency: float,
    ratio: float,
    env_ids: list[int] | None,
    error: type[Exception],
) -> None:
    art = _make_articulation()
    with pytest.raises(error):
        art.set_mimic_compliance(frequency, ratio, "width", env_ids)
    for entity in art._entities:
        entity.set_mimic_compliance.assert_not_called()


@pytest.mark.no_sim
@pytest.mark.parametrize("backend", ["dexsim", "newton"])
def test_unconfigured_mimic_compliance_keeps_native_behavior(backend: str) -> None:
    art = _make_articulation(backend)
    art._prepare_spawn_runtime_config(art._spawn_result)
    for entity in art._entities:
        entity.set_mimic_compliance.assert_not_called()


@pytest.mark.no_sim
def test_newton_rejects_default_mimic_compliance() -> None:
    art = _make_articulation("newton")
    with pytest.raises(NotImplementedError, match="newton"):
        art.set_mimic_compliance(0.0, 0.0, "width")
    art.cfg.mimic_compliance = {"width": MimicComplianceCfg(0.0, 0.0)}
    with pytest.raises(NotImplementedError, match="newton"):
        art._prepare_spawn_runtime_config(art._spawn_result)
    for entity in art._entities:
        entity.set_mimic_compliance.assert_not_called()


@pytest.mark.no_sim
def test_configured_compliance_can_retry_an_incomplete_revision() -> None:
    art = _make_articulation()
    art.cfg.mimic_compliance = {"width": MimicComplianceCfg(0.0, 0.0)}
    setter = art._entities[1].set_mimic_compliance
    setter.side_effect = [-1, 0]

    with pytest.raises(RuntimeError, match="status -1"):
        art._prepare_spawn_runtime_config(art._spawn_result)

    art._prepare_spawn_runtime_config(art._spawn_result)
    art._prepare_spawn_runtime_config(art._spawn_result)
    assert setter.call_count == 2


def _write_gripper_urdf(path: Path) -> None:
    inertial = """<inertial><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>"""
    body = (
        inertial
        + '<collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision>'
    )
    path.write_text(
        f"""<robot name="gripper">
  <link name="base">{body}</link>
  <link name="driver">{body}</link>
  <link name="left">{body}</link>
  <link name="right">{body}</link>
  <joint name="width" type="prismatic">
    <parent link="base"/><child link="driver"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1000" velocity="10"/>
  </joint>
  <joint name="left_finger" type="prismatic">
    <parent link="base"/><child link="left"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1000" velocity="10"/>
    <mimic joint="width" multiplier="0.5"/>
  </joint>
  <joint name="right_finger" type="prismatic">
    <parent link="base"/><child link="right"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1000" velocity="10"/>
    <mimic joint="width" multiplier="-0.5"/>
  </joint>
</robot>""",
        encoding="utf-8",
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_configured_hard_mimic_survives_replication_and_topology_change(
    tmp_path: Path, device: str
) -> None:
    urdf_path = tmp_path / "gripper.urdf"
    _write_gripper_urdf(urdf_path)
    cfg = ArticulationCfg(
        uid="gripper",
        fpath=str(urdf_path),
        init_pos=(0.0, 0.0, 0.5),
        asset_physics_mode="overlay",
        joint_drive_props=JointDrivePropertiesCfg(drive_type="none"),
        mimic_compliance={"width": MimicComplianceCfg(0.0, 0.0)},
        root_props=ArticulationRootPropertiesCfg(
            min_position_iters=24, min_velocity_iters=8
        ),
        build_pk_chain=False,
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device=device,
            num_envs=2,
            physics_dt=0.002,
        )
    )
    art = None

    def measure_mimic_error(art: Articulation) -> torch.Tensor:
        joint = {name: index for index, name in enumerate(art.joint_names)}
        forces = torch.zeros((2, art.dof), device=sim.device)
        forces[:, joint["width"]] = 30.0
        for _ in range(40):
            art.set_qf(forces)
            sim.update(step=1)
        qpos = art.get_qpos()
        width = qpos[:, joint["width"]]
        assert torch.all((width > 0.005) & (width < 0.9)), width
        left_error = qpos[:, joint["left_finger"]] - 0.5 * width
        right_error = qpos[:, joint["right_finger"]] + 0.5 * width
        return torch.maximum(left_error.abs(), right_error.abs())

    try:
        art = sim.add_articulation(cfg)
        sim.prepare()
        torch.testing.assert_close(
            measure_mimic_error(art),
            torch.zeros(2, device=sim.device),
            atol=1e-4,
            rtol=0,
        )

        art.reset()
        art.set_mimic_compliance(10.0, 1.0, "width", env_ids=[1])
        hard_error, soft_error = measure_mimic_error(art).tolist()
        assert hard_error < 1e-4
        assert soft_error > 1e-3

        revision = sim.spawn_result.topology_revision
        sim.add_rigid_object(
            RigidObjectCfg(uid="new_cube", shape=CubeCfg(), init_pos=(2.0, 2.0, 2.0))
        )
        assert sim.spawn_result.topology_revision > revision
        art.reset()
        torch.testing.assert_close(
            measure_mimic_error(art),
            torch.zeros(2, device=sim.device),
            atol=1e-4,
            rtol=0,
        )

        assert sim.remove_asset(cfg.uid)
        art = sim.add_articulation(cfg)
        torch.testing.assert_close(
            measure_mimic_error(art),
            torch.zeros(2, device=sim.device),
            atol=1e-4,
            rtol=0,
        )
    finally:
        art = None
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
