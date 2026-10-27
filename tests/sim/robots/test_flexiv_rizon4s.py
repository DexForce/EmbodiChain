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

"""Rizon 4S asset provenance, kinematic frames and published simulation dynamics."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch
import yaml

from embodichain.data.assets.robot_assets import FlexivRizon4S
from embodichain.lab.sim.cfg import RobotCfg
from embodichain_robots import FlexivRizon4SCfg


@pytest.mark.no_sim
def test_downloaded_asset_preserves_official_parameters() -> None:
    """The cached download contains complete meshes and source inertials."""
    root = Path(FlexivRizon4S().extract_dir)
    provenance = json.loads((root / "provenance.json").read_text())
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    source = ET.parse(root / "rizon4s.urdf").getroot()
    for mesh in source.findall(".//mesh"):
        assert (root / mesh.get("filename")).is_file()
    for mesh in (root / "meshes").rglob("*.obj"):
        for line in mesh.read_text().splitlines():
            if line.startswith("mtllib "):
                assert (mesh.parent / line.removeprefix("mtllib ")).is_file()
    # Native ray tracing requires triangles; keep the official ring vertices.
    ring = root / "meshes/Rizon4s/visual"
    original = (ring / "ring.obj").read_text().splitlines()
    derived = (ring / "ring_triangles.obj").read_text().splitlines()
    assert [line for line in original if line.startswith("v ")] == [
        line for line in derived if line.startswith("v ")
    ]
    assert all(len(line.split()) == 4 for line in derived if line.startswith("f "))

    inertials = yaml.safe_load(
        (
            root / "reference/flexiv_description/config/Rizon4s/inertia_parameters.yaml"
        ).read_text()
    )["inertia_parameters"]
    for name, expected in inertials.items():
        inertial = source.find(f"link[@name='{name}']/inertial")
        assert float(inertial.find("mass").get("value")) == expected["mass"]
        np.testing.assert_allclose(
            [float(x) for x in inertial.find("origin").get("xyz").split()],
            [expected["origin"][axis] for axis in ("x", "y", "z")],
        )
        for component, value in expected["inertia"].items():
            assert float(inertial.find("inertia").get(component)) == value


@pytest.mark.no_sim
def test_roundtrip_and_custom_asset_keep_kinematic_source(tmp_path: Path) -> None:
    """A custom URDF changes both simulation and PK, and overrides round-trip."""
    original = FlexivRizon4SCfg.from_dict({})
    source = ET.parse(original.fpath)
    origin = source.find("joint[@name='joint1']/origin")
    translation = [float(x) for x in origin.get("xyz").split()]
    shoulder_offset = 0.12
    translation[2] += shoulder_offset
    origin.set("xyz", " ".join(str(x) for x in translation))
    custom = tmp_path / "custom.urdf"
    source.write(custom)
    cfg = FlexivRizon4SCfg.from_dict(
        {
            "fpath": str(custom),
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "arm",
                        "urdf_path": str(custom),
                        "params": {"color": "#e0e0e0"},
                    }
                ]
            },
            "joint_drive_props": {"stiffness": {"joint[1-2]": 2500.0}},
            "solver_cfg": {"arm": {"num_samples": 4}},
        }
    )
    cfg.validate()
    assert (
        cfg._pk_urdf_path == cfg.fpath == cfg.solver_cfg["arm"].urdf_path == str(custom)
    )
    assert cfg.joint_drive_props.stiffness["joint[1-2]"] == 2500.0
    assert cfg.solver_cfg["arm"].num_samples == 4
    assert cfg.urdf_cfg.components["arm"]["params"] == {"color": "#e0e0e0"}
    assert original.joint_drive_props.stiffness["joint[1-2]"] == 4000.0
    snapshot = cfg.to_dict()
    assert FlexivRizon4SCfg.from_dict(snapshot).to_dict() == snapshot
    assert RobotCfg.from_dict(snapshot).to_dict() == snapshot
    qpos = torch.tensor([original.init_qpos])
    expected = (
        original.build_pk_serial_chain()["arm"].forward_kinematics(qpos).get_matrix()
    )
    actual = cfg.build_pk_serial_chain()["arm"].forward_kinematics(qpos).get_matrix()
    torch.testing.assert_close(
        actual[:, :3, 3] - expected[:, :3, 3], torch.tensor([[0, 0, shoulder_offset]])
    )


@pytest.mark.no_sim
def test_flange_chain_and_default_ik_agree() -> None:
    """The seven-joint chain and solver target the official bare flange."""
    cfg = FlexivRizon4SCfg.from_dict({})
    chain = cfg.build_pk_serial_chain()["arm"]
    assert chain.get_joint_parameter_names() == cfg.control_parts["arm"]
    assert len(chain.get_joint_parameter_names()) == 7
    home = chain.forward_kinematics(torch.tensor([cfg.init_qpos])).get_matrix()
    # Independent reference: expanded official Rizon4s kinematics at its ready pose.
    torch.testing.assert_close(
        home[0, :3, 3],
        torch.tensor([0.6870048, -0.1129999, 0.2835806]),
        atol=1e-6,
        rtol=0,
    )
    solver_cfg = cfg.solver_cfg["arm"]
    from embodichain.lab.sim.motion.solvers import SSIKSolverCfg

    assert isinstance(solver_cfg, SSIKSolverCfg)
    assert solver_cfg.backend == "warp" and solver_cfg.geometry_mode == "rr"
    assert solver_cfg.rr_search == "seed_local"
    solver_cfg.joint_names = cfg.control_parts["arm"]
    solver = solver_cfg.init_solver(device=torch.device("cpu"))
    qpos = torch.tensor([[0.1, -0.5, 0.2, 1.3, -0.3, 0.8, 0.1]])
    target = chain.forward_kinematics(qpos).get_matrix()
    torch.testing.assert_close(solver.get_fk(qpos), target)
    success, solved = solver.get_ik(target, qpos_seed=qpos + 0.01)
    assert bool(success.all())
    torch.testing.assert_close(
        solver.get_fk(solved.reshape_as(qpos)), target, atol=1e-5, rtol=0
    )


@pytest.mark.requires_sim
class TestFlexivRizon4SDynamics:
    def setup_method(self) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import RenderCfg

        self.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                physics_dt=0.01,
                robot_ik_gizmo=None,
                render_cfg=RenderCfg(renderer="auto"),
            )
        )

    def teardown_method(self) -> None:
        from embodichain.lab.sim import SimulationManager

        self.sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()

    def test_source_dynamics_and_tuned_drives_hold_and_move_under_gravity(self) -> None:
        """Native drives retain source limits/inertials and apply tuned damping."""
        from pxr import Usd

        cfg = FlexivRizon4SCfg.from_dict({})
        robot = self.sim.add_robot(cfg)
        self.sim.prepare()
        assert robot.active_joint_names == cfg.control_parts["arm"]
        root = Path(cfg.fpath).parent
        physics = Usd.Stage.Open(str(root / "reference/isaac_sim_ws/physx.usda"))
        source = ET.parse(cfg.fpath).getroot()
        expected_drives = [[] for _ in range(6)]
        for name in robot.joint_names:
            prim = physics.GetPrimAtPath(f"/rizon_4s/Physics/{name}")
            limits = source.find(f"joint[@name='{name}']/limit")
            values = (
                prim.GetAttribute("drive:angular:physics:stiffness").Get()
                * 180
                / math.pi,
                prim.GetAttribute("drive:angular:physics:damping").Get()
                * 180
                / math.pi
                * 3,
                float(limits.get("effort")),
                float(limits.get("velocity")),
                prim.GetAttribute("physxJoint:jointFriction").Get(),
                prim.GetAttribute("physxJoint:armature").Get(),
            )
            for expected, value in zip(expected_drives, values):
                expected.append(value)
        for actual, expected in zip(robot.get_joint_drive(), expected_drives):
            torch.testing.assert_close(
                actual, torch.tensor([expected]), rtol=1e-5, atol=1e-6
            )
        for link in source.findall("link"):
            mass = link.find("inertial/mass")
            if mass is not None:
                assert robot.get_mass(link.get("name")).item() == pytest.approx(
                    float(mass.get("value"))
                )
                tensor = link.find("inertial/inertia")
                ixx, iyy, izz, ixy, ixz, iyz = (
                    float(tensor.get(component))
                    for component in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                )
                principal = np.linalg.eigvalsh(
                    [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]]
                )
                np.testing.assert_allclose(
                    np.sort(robot.get_inertia(link.get("name")).cpu().numpy()[0, 0]),
                    principal,
                    rtol=1e-4,
                    atol=1e-7,
                )

        home = torch.tensor([cfg.init_qpos])
        for _ in range(300):
            self.sim.update()
        assert torch.isfinite(robot.get_qpos()).all()
        # Gravity produces finite PD tracking error; no ideal gravity compensation is assumed.
        torch.testing.assert_close(robot.get_qpos(), home, atol=0.03, rtol=0)
        target = home + torch.tensor([[0.12, 0.10, -0.08, -0.10, 0.08, -0.06, 0.08]])
        robot.set_qpos(target)
        for _ in range(300):
            self.sim.update()
        torch.testing.assert_close(robot.get_qpos(), target, atol=0.03, rtol=0)
        chain = cfg.build_pk_serial_chain()["arm"]
        torch.testing.assert_close(
            robot.get_link_pose("flange", to_matrix=True),
            chain.forward_kinematics(robot.get_qpos()).get_matrix(),
            atol=1e-4,
            rtol=0,
        )

    def test_reachable_ik_drag_settles_without_repeated_rebound(self) -> None:
        """A continuous native IK drag settles without the original drive oscillation."""
        from dexsim.kit.ik.pose import (
            local_pose_from_world,
            rotation_matrix_to_quat_xyzw,
        )

        from embodichain.lab.sim.objects.gizmo import GizmoCfg, _build_robot_ik

        robot = self.sim.add_robot(FlexivRizon4SCfg.from_dict({}))
        self.sim.prepare()
        for _ in range(400):
            self.sim.update(step=1)
        adapter, solver, _, _ = _build_robot_ik(robot, "arm", GizmoCfg())
        names = adapter.get_actived_joint_names()
        adapter.set_target_qpos(adapter.get_current_qpos())
        start = robot.get_link_pose("flange", to_matrix=True)[0].cpu().numpy().copy()
        hold_velocities = []
        # Move 8 cm along a reachable arc in 250 ms, then hold for one second.
        # Match the native gizmo's current-state seed and 24 IK iterations.
        for step in range(225):
            phase = np.clip((step - 100) / 25, 0.0, 1.0)
            target = start.copy()
            target[:3, 3] += [
                0.0,
                0.08 * np.sin(np.pi * phase),
                0.04 * (1.0 - np.cos(np.pi * phase)),
            ]
            local = local_pose_from_world(adapter.get_world_pose(), target)
            solver.set_target_pose(
                local[:3, 3], rotation_matrix_to_quat_xyzw(local[:3, :3])
            )
            if solver.target_state_changed():
                current = adapter.get_current_qpos()
                solver.solve(names, current, iterations=24)
                adapter.set_target_qpos(solver.qpos_for_joint_names(names, current))
                solver.reset_target_state_changed()
            self.sim.update(step=1)
            if step >= 125:
                hold_velocities.append(robot.get_qvel()[0].cpu().numpy().copy())

        velocities = np.stack(hold_velocities)
        assert np.isfinite(velocities).all()
        # Ignore near-zero numerical noise. The old damping reversed up to four
        # times; allow one correction, but reject repeated physical rebound.
        reversals = (velocities[1:] * velocities[:-1] < 0) & (
            np.abs(velocities[1:]) > 0.01
        )
        assert int(reversals.sum(axis=0).max()) <= 1
        end = robot.get_link_pose("flange", to_matrix=True)[0].cpu().numpy()
        # These force drives have static gravity sag without compensation.
        np.testing.assert_allclose(end[:3, 3], target[:3, 3], atol=0.015, rtol=0)
        assert np.linalg.norm(end[:3, 3] - start[:3, 3]) > 0.06


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "--run-gpu"]))
