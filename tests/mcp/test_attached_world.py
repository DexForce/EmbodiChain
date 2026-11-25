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

"""Live, read-only attachment acceptance for bare and assembled arms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from examples.sim.attached_world import fixture


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["dexsim", "newton"])
@pytest.mark.parametrize("assembled", [False, True], ids=["arm", "pgi"])
def test_attach_existing_world(tmp_path: Path, backend: str, assembled: bool) -> None:
    from dexsim.runtime import Session
    from dexsim.runtime.errors import SessionRemoteError

    asset = Path(fixture.robot_config(assembled).fpath)
    session = Session.create(
        directory=tmp_path,
        backend=backend,
        **({"solver": "mujoco_warp"} if backend == "newton" else {}),
        device="cuda:0",
        headless=True,
        asset_root=str(asset.parent),
        request_timeout=120,
    )

    def state() -> list[dict[str, Any]]:
        return [
            fixture.native(session, "scene.list_objects"),
            *(
                fixture.native(session, "articulation.get_state", object_ref=name)
                for name in ("robot_a", "robot_b")
            ),
            fixture.native(
                session, "simulation.get_rigid_body_state", object_ref="moving_cube"
            ),
        ]

    def bind(paths: dict[str, str]) -> int:
        return session.run_file(
            fixture.__file__, arguments={"assembled": assembled, "paths": paths}
        )["result"]

    try:
        fixture.populate(session, asset)
        original_revision = session.evaluate("scene.topology_revision")["result"]
        count = session.evaluate(
            "__import__('embodichain.lab.sim',fromlist=['SimulationManager']).SimulationManager.get_instance_num()"
        )["result"]
        before = state()
        with pytest.raises(SessionRemoteError):
            bind({**fixture.PATHS, "platform": "robot_a"})
        instance_id = bind(fixture.PATHS)
        assert (
            session.evaluate("state['attachment'].instance_id")["result"] == instance_id
        )
        assert session.evaluate(
            "SimulationManager = __import__('embodichain.lab.sim',fromlist=['SimulationManager']).SimulationManager; manager=state['attachment']; selected=manager.resolve_entity('manipulator'); selected is SimulationManager.get_instance(manager.instance_id).get_robot('UR5')"
        )["result"]
        assert session.evaluate(
            "manager.resolve_entity('cube') is manager.get_rigid_object('cube')"
        )["result"]
        revision = session.evaluate("scene.topology_revision")["result"]
        assert session.evaluate("manager.world_gravity")["result"] == pytest.approx(
            [0, 0, -9.81]
        )
        after = state()
        # DexSim's command counter includes script attempts; topology and all
        # measured fields remain invariant across attachment.
        for snap in (before, after):
            for entry in snap:
                entry.pop("scene_revision", None)
        assert revision == original_revision
        assert after == before
        assert session.evaluate(
            "state['attachment'].get_robot('UR5').get_qpos(name='arm').tolist()[0]"
        )["result"] == pytest.approx(before[2]["joint_position"][:6])
        positions = [0.1] * 6
        fixture.native(
            session,
            "articulation.set_state",
            object_ref="robot_b",
            joint_names=fixture.ARM,
            joint_position=positions,
        )
        assert session.evaluate(
            "state['attachment'].get_robot('UR5').get_qpos(name='arm').tolist()[0]"
        )["result"] == pytest.approx(positions)
        session.evaluate(
            "import weakref; retained = selected.body_data; saved_fetch = retained.articulation_view.fetch_qpos; batch = weakref.ref(selected._data)"
        )
        fixture.native(
            session,
            "scene.create_primitive",
            name="extra",
            primitive="cube",
            size=[0.03] * 3,
        )
        fixture.native(session, "scene.rebuild")
        with pytest.raises(SessionRemoteError, match="Stale"):
            session.evaluate("selected.get_qpos()")
        session.evaluate(
            "state['attachment'].detach(); state['attachment'].destroy(exit_process=False)"
        )
        assert (
            session.evaluate(
                "from embodichain.lab.sim import SimulationManager; SimulationManager.flush_cleanup_queue(); SimulationManager.get_instance_num()"
            )["result"]
            == count
        )
        for expression in ("retained.qpos", "saved_fetch(None)"):
            with pytest.raises(SessionRemoteError, match="Stale"):
                session.evaluate(expression)
        assert session.evaluate("import gc; gc.collect(); batch() is None")["result"]
        fixture.native(session, "simulation.step", dt=0.01, steps=1)
        assert (
            fixture.native(session, "articulation.get_state", object_ref="robot_b")[
                "object_ref"
            ]
            == "robot_b"
        )
    finally:
        session.shutdown()


@pytest.mark.parametrize("drift", ["axis", "fixed_mount"])
def test_reject_same_name_different_kinematics(tmp_path: Path, drift: str) -> None:
    """Identical DOF names do not qualify a different axis or fixed TCP transform."""
    from types import SimpleNamespace
    import numpy as np
    from dexsim.scene import JointDofLayout
    from embodichain.lab.sim.cfg import RobotCfg
    from embodichain.lab.sim.dexsim_session.attachment import _bind_robot_cfg

    path = tmp_path / "arm.urdf"
    path.write_text(
        '<robot name="arm"><link name="base"/><link name="wrist"/><link name="tcp"/>'
        '<joint name="axis" type="revolute"><parent link="base"/><child link="wrist"/>'
        '<axis xyz="0 0 1"/></joint><joint name="mount" type="fixed"><parent link="wrist"/>'
        '<child link="tcp"/></joint></robot>'
    )
    axis = SimpleNamespace(
        parent_link_name="base",
        child_link_name="wrist",
        origin_pose=np.eye(4),
        axis=[0, 0, 1],
        mimic_parent=None,
    )
    mount = SimpleNamespace(
        parent_link_name="wrist",
        child_link_name="tcp",
        joint_type=SimpleNamespace(name="FIXED"),
        origin_pose=np.eye(4),
    )
    handle = SimpleNamespace(
        path="arm",
        get_link_names=lambda: ["base", "wrist", "tcp"],
        joint_dof_layout=[JointDofLayout("axis", 0, 1, "angle")],
        get_joint_desc={"axis": axis, "mount": mount}.__getitem__,
    )
    cfg = RobotCfg(uid="arm", fpath=str(path), control_parts={"tool": ["axis"]})
    _bind_robot_cfg(cfg, handle)
    if drift == "axis":
        axis.axis = [1, 0, 0]
    else:
        mount.origin_pose[0, 3] = 0.01
    with pytest.raises(ValueError, match="axis|frame"):
        _bind_robot_cfg(cfg, handle)
