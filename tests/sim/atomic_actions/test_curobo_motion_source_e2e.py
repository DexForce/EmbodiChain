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

"""Optional DexSim end-to-end test for cuRobo through AtomicActionEngine.

Skipped when cuRobo or CUDA is unavailable. When both are present, it builds a
single-arm Franka + static cuboid scene, executes ``MoveEndEffector`` with
``planner_type='curobo'`` through the engine, and asserts a full-DoF
collision-aware trajectory that reaches the target after playback.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# Module-level guards before any cuRobo-only import.
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)

from embodichain import data as _data  # noqa: E402
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg  # noqa: E402
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg  # noqa: E402
from embodichain.lab.sim.objects import RigidObjectCfg  # noqa: E402
from embodichain.lab.sim.robots import FrankaPandaCfg  # noqa: E402
from embodichain.lab.sim.shapes import CubeCfg  # noqa: E402
from embodichain.lab.sim.planners import (  # noqa: E402
    MotionGenCfg,
    MotionGenerator,
)
from embodichain.lab.sim.planners.curobo_planner import (  # noqa: E402
    CuroboPlannerCfg,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.atomic_actions import AtomicActionEngine  # noqa: E402
from embodichain.lab.sim.atomic_actions.actions import (  # noqa: E402
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.lab.sim.atomic_actions.core import EndEffectorPoseTarget  # noqa: E402

ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.40, 0.36]
DEMO_BLOCK_POS = [0.45, 0.0, 0.18]
POS_TOL = 0.02


def _demo_world_path() -> str:
    return str(
        Path(_data.__file__).parent / "assets" / "curobo" / "collision_franka_demo.yml"
    )


def _franka_profile() -> CuroboRobotProfileCfg:
    sim_to_curobo = {f"fr3_joint{i}": f"panda_joint{i}" for i in range(1, 8)}
    return CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names=sim_to_curobo,
        base_link_name="panda_link0",
        tool_frame_name="panda_hand",
        tool_frame_to_tcp=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.1034],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )


def _make_franka_curobo_engine():
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=1)
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
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
    mg = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=ROBOT_UID,
                robot_profiles={CONTROL_PART: _franka_profile()},
                world=CuroboWorldCfg(world_config_path=_demo_world_path()),
                warmup=False,
                use_cuda_graph=False,
            )
        )
    )
    engine = AtomicActionEngine(mg)
    engine.register(
        MoveEndEffector(
            mg,
            MoveEndEffectorCfg(
                motion_source="motion_gen",
                planner_type="curobo",
                control_part=CONTROL_PART,
                sample_interval=80,
            ),
        ),
        name="move_end_effector",
    )
    return sim, robot, engine


def _reachable_target_beyond_demo_block(robot) -> torch.Tensor:
    """A target beyond the cuboid so the planner must route around it."""
    qpos = robot.get_qpos(name=CONTROL_PART)
    fk = robot.compute_fk(qpos=qpos, name=CONTROL_PART, to_matrix=True)
    target = fk[0].clone()
    target[:3, 3] = torch.tensor([0.55, 0.30, 0.45], device=robot.device)
    return target


def _play_trajectory(sim, robot, trajectory: torch.Tensor, step_repeat: int = 1):
    """Replay every waypoint with matching state and drive targets.

    ``target=True`` alone only updates the articulation drive target.  The
    physics step can therefore still be catching up with the previous sample
    when the next one is supplied.  Set the current state first so the replay
    validates the planned configuration rather than the drive controller's
    tracking transient.
    """
    all_joint_ids = list(range(robot.dof))
    for w in range(trajectory.shape[1]):
        waypoint = trajectory[:, w]
        robot.set_qpos(qpos=waypoint, joint_ids=all_joint_ids, target=False)
        robot.set_qpos(qpos=waypoint, joint_ids=all_joint_ids, target=True)
        sim.update(step=step_repeat)


def _position_error(robot, target: torch.Tensor) -> float:
    qpos = robot.get_qpos(name=CONTROL_PART)
    fk = robot.compute_fk(qpos=qpos, name=CONTROL_PART, to_matrix=True)
    return float(torch.norm(fk[0, :3, 3] - target[:3, 3]))


@pytest.mark.requires_sim
@pytest.mark.slow
def test_atomic_move_end_effector_uses_curobo_v2():
    sim, robot, engine = _make_franka_curobo_engine()
    try:
        target = _reachable_target_beyond_demo_block(robot)
        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        assert success.shape == (1,)
        assert bool(success.item())
        assert trajectory.shape[2] == robot.dof
        _play_trajectory(sim, robot, trajectory)
        assert _position_error(robot, target) < POS_TOL
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
