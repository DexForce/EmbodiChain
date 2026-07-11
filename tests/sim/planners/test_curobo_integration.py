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

"""Optional cuRobo V2 + CUDA integration test.

Skipped entirely when cuRobo or CUDA is unavailable. When both are present,
it builds a Panda profile + static cuboid world, plans a collision-aware EEF
move through the EmbodiChain ``MotionGenerator`` API, and verifies the
``PlanResult`` contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Module-level guards: skip the whole file without cuRobo or CUDA. These must
# run before any cuRobo-only import.
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)

from embodichain import data as _data  # noqa: E402
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg  # noqa: E402
from embodichain.lab.sim.objects import RigidObjectCfg  # noqa: E402
from embodichain.lab.sim.robots import FrankaPandaCfg  # noqa: E402
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg  # noqa: E402
from embodichain.lab.sim.shapes import CubeCfg  # noqa: E402
from embodichain.lab.sim.planners import (  # noqa: E402
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
    MoveType,
    PlanState,
)
from embodichain.lab.sim.planners.curobo_planner import (  # noqa: E402
    CuroboPlanOptions,
    CuroboPlannerCfg,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
)

ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.40, 0.36]
DEMO_BLOCK_POS = [0.45, 0.0, 0.18]


def _demo_world_path() -> str:
    return str(
        Path(_data.__file__).parent / "assets" / "curobo" / "collision_franka_demo.yml"
    )


def _franka_profile() -> CuroboRobotProfileCfg:
    """Explicit sim->cuRobo joint mapping; no index order is assumed."""
    sim_to_curobo = {f"fr3_joint{i}": f"panda_joint{i}" for i in range(1, 8)}
    return CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names=sim_to_curobo,
        fixed_joint_positions={
            "panda_finger_joint1": 0.04,
            "panda_finger_joint2": 0.04,
        },
        base_link_name="panda_link0",
        tool_frame_name="panda_hand",
    )


def _make_sim_robot():
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    # Mirror the cuRobo cuboid in DexSim so the planner and simulator agree.
    sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=DEMO_BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=DEMO_BLOCK_POS,
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    return sim, robot


@pytest.mark.slow
def test_curobo_v2_plans_around_a_static_cuboid():
    sim, robot = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            robot_profiles={CONTROL_PART: _franka_profile()},
            world=CuroboWorldCfg(world_config_path=_demo_world_path()),
        )
        mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))

        start_qpos = robot.get_qpos(name=CONTROL_PART)
        start_xpos = robot.compute_fk(
            qpos=start_qpos, name=CONTROL_PART, to_matrix=True
        )
        # Target beyond the cuboid so the planner must route around it.
        target_xpos = start_xpos.clone()
        target_xpos[0, :3, 3] = torch.tensor([0.55, 0.20, 0.30], device=robot.device)

        result = mg.generate(
            [PlanState.from_xpos(target_xpos)],
            MotionGenOptions(
                start_qpos=start_qpos,
                control_part=CONTROL_PART,
                plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
            ),
        )

        assert result.success.shape == (1,)
        assert bool(result.success.item())
        assert result.positions is not None
        assert torch.isfinite(result.positions).all()
        # Controlled joint count matches the arm (7).
        assert result.positions.shape[-1] == 7
        # Trajectory starts at the requested start qpos.
        assert torch.allclose(result.positions[0, 0], start_qpos[0], atol=1e-3)
        # Positive duration.
        assert float(result.duration[0]) > 0.0

        # Final FK position reaches the target within tolerance.
        final_q = result.positions[0, -1:].to(robot.device)
        final_xpos = robot.compute_fk(qpos=final_q, name=CONTROL_PART, to_matrix=True)
        err = torch.norm(final_xpos[0, :3, 3] - target_xpos[0, :3, 3])
        assert float(err) < 0.02
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
