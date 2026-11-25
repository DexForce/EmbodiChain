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
            "manager.resolve_entity('payload') is manager.get_rigid_object('cube')"
        )["result"]
        revision = session.evaluate("scene.topology_revision")["result"]
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
        fixture.native(session, "simulation.step", dt=0.01, steps=1)
        assert (
            fixture.native(session, "articulation.get_state", object_ref="robot_b")[
                "object_ref"
            ]
            == "robot_b"
        )
    finally:
        session.shutdown()
