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

"""Full-resolution, calibrated dual-model playback without a physics runtime."""

from __future__ import annotations

from pathlib import Path
import socket
import threading
from typing import Any

import numpy as np
import pytest

from embodichain.toolkits.trajectory_annotator._viewer import (
    TrajectoryViewer,
    _load_models,
)

pytestmark = pytest.mark.no_sim


def test_models_follow_full_clock_and_recorded_mimics_and_release_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect pose placement, independent models, styles, time sampling and ownership."""
    import viser.extras

    path = tmp_path / "robot.urdf"
    path.write_text("""<robot name="calibrated">
      <link name="world"/><link name="root"/><link name="arm">
        <visual><geometry><box size="0.1 0.1 0.3"/></geometry>
        <material name="orange"><color rgba="0.8 0.3 0.1 1"/></material></visual></link>
      <link name="finger1"/><link name="finger2"/>
      <joint name="mount" type="fixed"><parent link="world"/><child link="root"/>
        <origin xyz="0 0 1" rpy="0 0 1.5707963267948966"/></joint>
      <joint name="arm_joint" type="prismatic"><parent link="root"/><child link="arm"/>
        <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
      <joint name="finger1" type="prismatic"><parent link="arm"/><child link="finger1"/>
        <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/></joint>
      <joint name="finger2" type="prismatic"><parent link="arm"/><child link="finger2"/>
        <axis xyz="0 -1 0"/><limit lower="-1" upper="1" effort="10" velocity="2"/>
        <mimic joint="finger1" multiplier="1"/></joint>
    </robot>""")
    original = np.array([[0.02, 0, 0.02], [0.03, 0.1, 0.04], [0.11, 0.5, 0.06]])
    patched = original.copy()
    patched[1, 1] = 0.3
    names = ["finger2", "arm_joint", "finger1"]
    report = dict(
        joint_names=names,
        start=1,
        time_offset=10.0,
        robot_profile=dict(
            joint_names=names,
            solvers={
                "configured_part": dict(urdf_path=str(path), root_link_name="root")
            },
            base_poses={
                "configured_part": [
                    [0, -1, 0, 2],
                    [1, 0, 0, 3],
                    [0, 0, 1, 4],
                    [0, 0, 0, 1],
                ]
            },
        ),
    )
    _, _, world_from_model = _load_models(report)[0]
    np.testing.assert_allclose(world_from_model[:3, :3], np.eye(3), atol=1e-14)
    np.testing.assert_allclose(world_from_model[:3, 3], [2, 3, 3])

    records: dict[str, Any] = {}
    owners = set()
    actual_urdf = viser.extras.ViserUrdf
    condition = threading.Condition()

    class ObservedUrdf(actual_urdf):
        def __init__(self, server: Any, model: Any, **kwargs: Any) -> None:
            self.model = model
            self.kind = kwargs["root_node_name"].split("/")[1]
            self.color = kwargs["mesh_color_override"]
            super().__init__(server, model, **kwargs)

        def update_cfg(self, configuration: Any) -> None:
            super().update_cfg(configuration)
            with condition:
                owners.add(threading.get_ident())
                records[self.kind] = dict(
                    qpos=configuration,
                    color=self.color,
                    arm=self.model.get_transform("arm").copy(),
                    finger2=self.model.get_transform("finger2", "arm").copy(),
                )
                condition.notify_all()

    monkeypatch.setattr(viser.extras, "ViserUrdf", ObservedUrdf)
    viewer = TrajectoryViewer(
        dict(
            timestamps=np.array([10.0, 10.1, 10.4]),
            original_qpos=original,
            patched_qpos=patched,
        ),
        report,
    )
    try:
        state = viewer.start()
        assert state["phase"] == "ready", state
        viewer.set_time(0.25)
        with condition:
            assert condition.wait_for(
                lambda: abs(records["patched"]["qpos"]["arm_joint"] - 0.4) < 1e-10,
                timeout=5,
            )
            np.testing.assert_allclose(
                [records["original"]["qpos"][name] for name in names], [0.07, 0.3, 0.05]
            )
            np.testing.assert_allclose(
                [records["patched"]["qpos"][name] for name in names], [0.07, 0.4, 0.05]
            )
            # Rz90 maps the prismatic arm's local X into model Y.
            np.testing.assert_allclose(
                records["original"]["arm"][:3, 3], [0, 0.3, 1], atol=1e-14
            )
            np.testing.assert_allclose(
                records["patched"]["finger2"][:3, 3], [0, -0.07, 0], atol=1e-14
            )
            assert records["original"]["color"] is None
            assert records["patched"]["color"] == (0.16, 0.47, 1.0, 0.35)
            assert len(owners) == 1 and threading.get_ident() not in owners
        # Scrubbing before/after the joint recording clamps to its endpoints.
        viewer.set_time(-1)
        with condition:
            assert condition.wait_for(
                lambda: records["original"]["qpos"]["arm_joint"] == 0, timeout=5
            )
        for time_seconds in np.linspace(0, 1, 100):
            viewer.set_time(float(time_seconds))
        with condition:
            assert condition.wait_for(
                lambda: records["patched"]["qpos"]["arm_joint"] == 0.5, timeout=5
            )
    finally:
        viewer.close()
    assert viewer.snapshot()["phase"] == "closed"
    with socket.socket() as connection:
        connection.settimeout(1)
        assert connection.connect_ex(("127.0.0.1", state["port"])) != 0
