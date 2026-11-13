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

"""URDF name binding, physical video clocks and private viewer lifetime."""

from __future__ import annotations

from pathlib import Path
import socket
import threading
import time

import numpy as np
import pytest

from embodichain.toolkits.trajectory_annotator._robot_preview import RobotGhostPreview


def _preview(tmp_path: Path, *, missing_mesh: bool = False) -> dict:
    model = tmp_path / "robot.urdf"
    geometry = (
        '<mesh filename="absent.stl"/>' if missing_mesh else '<box size="0.1 0.1 0.5"/>'
    )
    model.write_text(f"""<robot name="oracle">
      <link name="base"/><link name="arm"><visual><geometry>{geometry}</geometry></visual></link>
      <link name="tool"/>
      <joint name="arm_joint" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
      <joint name="hand_joint" type="prismatic"><parent link="arm"/><child link="tool"/><axis xyz="1 0 0"/><limit lower="0" upper="0.1" effort="1" velocity="1"/></joint>
    </robot>""")
    original = np.array([[0.01, 0.0], [0.02, 0.3], [0.03, 0.6]])
    proposed = original + np.array([0.0, 0.1])
    return dict(
        report=dict(
            time_offset=10.0,
            joint_names=["hand_joint", "arm_joint"],
            accepted=False,
            robot_profile=dict(
                solvers={"arm": {"urdf_path": str(model), "root_link_name": "base"}}
            ),
        ),
        joint_names=["hand_joint", "arm_joint"],
        timestamps=[10.0, 11.0, 12.0],
        original_qpos=original,
        native_qpos=proposed,
        patched_qpos=original,
        proposed_qpos=proposed,
        original_links=np.zeros((3, 1, 2, 3)),
    )


def test_robot_ghost_uses_named_joints_latest_video_time_and_one_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import viser.extras
    from viser import _client_autobuild

    # This CPU-only case exercises scene/FK publication, not the JS build tool.
    monkeypatch.setattr(_client_autobuild, "ensure_client_is_built", lambda: None)
    updates = []
    original_type = viser.extras.ViserUrdf

    class ObservedUrdf(original_type):
        def __init__(self, *args, **kwargs):
            self.ghost = kwargs["root_node_name"].endswith("/ghost")
            super().__init__(*args, **kwargs)

        def update_cfg(self, configuration):
            if self.ghost:
                updates.append((configuration.copy(), threading.get_ident()))
            super().update_cfg(configuration)

    monkeypatch.setattr(viser.extras, "ViserUrdf", ObservedUrdf)
    viewer = RobotGhostPreview(_preview(tmp_path))
    try:
        for value in np.linspace(0.0, 2.0, 100):
            viewer.set_time(float(value))
        viewer.set_time(1.1)
        deadline = time.monotonic() + 15
        while (status := viewer.snapshot()).get("sample_index") != 1:
            assert status["phase"] != "error", status
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert status["sample_time_seconds"] == 11.0
        assert status["video_time_seconds"] == 1.0
        assert status["ghost_series"] == "proposed"
        # URDF order is arm/hand, while the input recording is hand/arm.
        np.testing.assert_allclose(updates[-1][0], [0.4, 0.02])
        assert len({thread for _, thread in updates}) == 1
        assert updates[-1][1] != threading.get_ident()
        port = status["port"]
    finally:
        viewer.close()
    assert viewer.snapshot()["phase"] == "closed"
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0


def test_missing_robot_visual_is_explicit_without_an_empty_viewer(
    tmp_path: Path,
) -> None:
    viewer = RobotGhostPreview(_preview(tmp_path, missing_mesh=True))
    try:
        deadline = time.monotonic() + 5
        while (status := viewer.snapshot())["phase"] == "loading":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert status["phase"] == "error"
        assert "absent.stl" in status["error"]
        assert status["port"] is None
    finally:
        viewer.close()
