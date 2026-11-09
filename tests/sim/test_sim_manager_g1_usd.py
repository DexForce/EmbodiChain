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

"""Exercise the original #904 asset through the public whole-scene API.

Set EMBODICHAIN_G1_USD to the original G1_120s/robot.usda, with all referenced
layers and meshes present. The asset is supplied locally by the test operator.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

CONTROL_PARTS = {
    "left_eef": ["idx41_gripper_l_outer_joint1"],
    "right_eef": ["idx81_gripper_r_outer_joint1"],
}
CLOSURE_JOINTS = {
    "idx39_gripper_l_inner_joint2",
    "idx49_gripper_l_outer_joint2",
    "idx79_gripper_r_inner_joint2",
    "idx89_gripper_r_outer_joint2",
}
ANCHOR_TOLERANCE_M = 1e-4
MIN_OPENING_M = 1e-3
RETURN_TOLERANCE_M = 1e-3
TARGET_TOLERANCE_RAD = 0.02


@pytest.mark.subprocess_sim
def test_add_usd_preserves_original_g1_gripper_loops() -> None:
    path = os.environ.get("EMBODICHAIN_G1_USD")
    if not path:
        pytest.skip("Set EMBODICHAIN_G1_USD to the original G1_120s/robot.usda")
    # NoRender requires a fresh process because other tests initialize a
    # different process-wide renderer.
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            str(Path(path).resolve(strict=True)),
        ],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_g1(path: str) -> None:
    import dexsim
    import numpy as np
    import torch
    from dexsim.scene import SpawnedConstraint

    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg, RobotCfg
    from embodichain.lab.sim.objects import Robot

    def jaw_gaps(robot: Robot) -> torch.Tensor:
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    robot.get_link_pose(f"gripper_{side}_inner_link4")[0, :3]
                    - robot.get_link_pose(f"gripper_{side}_outer_link4")[0, :3]
                )
                for side in ("l", "r")
            ]
        )

    def check_anchors(closures: list[SpawnedConstraint]) -> None:
        for handle in closures:
            distance = np.linalg.norm(handle.get_relative_transform()[:3, 3])
            assert distance <= ANCHOR_TOLERANCE_M, (handle.get_name(), distance)

    # A second Manager exercises reimport after the first Scene was removed.
    for reimport in (False, True):
        sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                device="cpu",
                num_envs=1,
                physics_cfg=DefaultPhysicsCfg(device="cpu", physics_dt=0.001),
                render_cfg=RenderCfg(renderer="no-render"),
            )
        )
        robot = scene = None
        closures = []
        try:
            robot = sim.add_usd(
                "G1",
                path,
                robot_cfgs={"/G1": RobotCfg(uid="G1", control_parts=CONTROL_PARTS)},
            )["/G1"]
            assert isinstance(robot, Robot)
            sim.prepare()
            scene = sim.spawn_result
            assert scene is not None
            closures = [
                handle
                for handle in scene.handles.values()
                if isinstance(handle, SpawnedConstraint)
            ]
            assert len(closures) == len(CLOSURE_JOINTS)
            assert {handle.get_name() for handle in closures} == CLOSURE_JOINTS
            sim.get_world().enable_physics(True)

            if reimport:
                sim.update(step=10)
                check_anchors(closures)
            else:
                # Sample all anchors as both jaws open and return to zero.
                gaps = []
                for target in (0.0, 0.4, 0.0):
                    for part in CONTROL_PARTS:
                        robot.set_qpos(torch.full((1, 1), target), name=part)
                    for _ in range(10):
                        sim.update(step=100)
                        check_anchors(closures)
                    for part in CONTROL_PARTS:
                        assert robot.get_qpos(name=part).item() == pytest.approx(
                            target, abs=TARGET_TOLERANCE_RAD
                        )
                    gaps.append(jaw_gaps(robot))

                initial, opened, returned = gaps
                assert torch.all(opened - initial > MIN_OPENING_M), (initial, opened)
                torch.testing.assert_close(
                    returned, initial, atol=RETURN_TOLERANCE_M, rtol=0
                )
                assert sim.remove_asset(robot.uid)
                assert not any(handle.is_valid for handle in closures)
        finally:
            closures.clear()
            robot = scene = None
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()
        assert dexsim.get_world_num() == 0


if __name__ == "__main__":
    _check_g1(sys.argv[1])
