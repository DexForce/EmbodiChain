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

"""SO101 defaults preserve the official motor reference and source chain frames."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain_robots import SO101FollowerCfg
from embodichain_robots import so101

pytestmark = pytest.mark.no_sim

_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


@pytest.fixture
def official_source(monkeypatch: pytest.MonkeyPatch) -> Path:
    # Unmodified official source at a758567c3978dfeefe282ede0500085a48fe8f78.
    path = Path(__file__).parent / "fixtures/so101/so101_new_calib.urdf"
    monkeypatch.setattr(so101, "get_data_path", lambda _: str(path))
    return path


def test_so101_official_dynamics_and_asset_owned_inertials(
    official_source: Path,
) -> None:
    cfg = SO101FollowerCfg.from_dict({})
    cfg.validate()
    assert cfg.control_parts == {"arm": _ARM_JOINTS, "eef": ["gripper"]}
    assert cfg.root_props.fixed_base is True
    assert cfg.root_props.self_collision_enabled is False
    drive = cfg.joint_drive_props
    # The Robot Studio's joints_properties.xml, not URDF's generic 10 Nm cap.
    assert (drive.stiffness, drive.damping, drive.friction, drive.armature) == (
        17.8,
        0.60,
        0.052,
        0.028,
    )
    assert drive.max_effort == 3.35
    assert drive.max_velocity == 10.0
    assert cfg.attrs.mass_props is None
    solver = cfg.solver_cfg["arm"]
    assert solver.class_type == "PytorchSolver"
    assert solver.is_only_position_constraint is True
    assert (solver.root_link_name, solver.end_link_name) == (
        "base_link",
        "gripper_frame_link",
    )


def test_so101_roundtrip_overrides_and_instance_isolation(
    official_source: Path,
) -> None:
    cfg = SO101FollowerCfg.from_dict(
        {"uid": "custom", "joint_drive_props": {"stiffness": 24.0}}
    )
    restored = SO101FollowerCfg.from_dict(cfg.to_dict())
    assert restored.to_dict() == cfg.to_dict()
    assert restored.joint_drive_props.stiffness == 24.0
    assert restored.joint_drive_props.damping == 0.60
    restored.control_parts["arm"].clear()
    assert cfg.control_parts["arm"] == _ARM_JOINTS
    assert SO101FollowerCfg.from_dict({}).joint_drive_props.stiffness == 17.8


def test_so101_chains_match_control_dofs_and_preserve_official_tcp(
    official_source: Path,
) -> None:
    cfg = SO101FollowerCfg.from_dict({})
    chains = cfg.build_pk_serial_chain()
    assert cfg._pk_urdf_path == cfg.urdf_cfg.components["arm"]["urdf_path"]
    assert set(chains) == set(cfg.control_parts)
    for part, chain in chains.items():
        assert chain.get_joint_parameter_names() == cfg.control_parts[part]
    arm = chains["arm"].forward_kinematics(torch.zeros(1, 5)).get_matrix()
    jaw = chains["eef"].forward_kinematics(torch.zeros(1, 1)).get_matrix()
    # Jaw's zero frame comes directly from the official gripper joint origin.
    torch.testing.assert_close(jaw[0, :3, 3], torch.tensor([0.0202, 0.0188, -0.0234]))
    # Independent float64 composition of the pinned CAD's six zero-pose origins.
    torch.testing.assert_close(
        arm[0, :3, 3],
        torch.tensor([0.3913614702, -0.0000092121, 0.2264697102]),
        atol=2e-6,
        rtol=0,
    )


def test_so101_position_ik_accepts_unconstrained_orientation(
    official_source: Path,
) -> None:
    cfg = SO101FollowerCfg.from_dict({})
    solver_cfg = cfg.solver_cfg["arm"]
    solver_cfg.urdf_path = cfg._pk_urdf_path
    solver_cfg.joint_names = list(_ARM_JOINTS)
    solver = solver_cfg.init_solver(device=torch.device("cpu"))
    chain = cfg.build_pk_serial_chain()["arm"]
    target = chain.forward_kinematics(
        torch.tensor([[0.2, -0.3, 0.4, -0.15, 0.1]])
    ).get_matrix()
    target[:, :3, :3] = torch.eye(3)
    success, solution = solver.get_ik(target, qpos_seed=torch.zeros(1, 5))
    assert success.all()
    actual = chain.forward_kinematics(solution.reshape(1, 5)).get_matrix()
    assert torch.linalg.vector_norm(actual[:, :3, 3] - target[:, :3, 3]) < 1e-3


if __name__ == "__main__":
    from embodichain.lab.scripts.preview_robot import cli

    cli(["--robot", "SO101Follower"])
