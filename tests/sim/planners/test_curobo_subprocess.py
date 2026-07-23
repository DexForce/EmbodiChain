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
"""Smoke test for the subprocess-isolated cuRobo planner.

Verifies that planning through the side-process worker (own CUDA context, CUDA
graphs enabled) succeeds and is fast, without crashing DexSim's Vulkan/CUDA
semaphores - the failure that occurs when cuRobo captures graphs in-process.
Gated on CUDA + cuRobo availability; skips otherwise.
"""

from __future__ import annotations

import time

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    EndEffectorPoseTarget,
    MoveEndEffector,
    MoveEndEffectorCfg,
)
from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
from embodichain.lab.sim.objects import RigidObjectCfg, Robot
from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlannerCfg,
    CuroboWorldCfg,
)
from embodichain.lab.sim.robots import FrankaPandaCfg
from embodichain.lab.sim.shapes import CubeCfg

# Skip the whole module if cuRobo V2 is not installed.
pytest.importorskip("curobo", reason="cuRobo V2 not installed.")

ROBOT_UID = "curobo_franka_subprocess_test"
CONTROL_PART = "arm"
BLOCK_DIMS = [0.18, 0.40, 0.36]
BLOCK_POS = (0.45, 0.0, 0.18)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuRobo requires a CUDA device."
)


def _build_scene() -> tuple[SimulationManager, Robot, object]:
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True, sim_device="cuda", num_envs=1, arena_space=2.0
        )
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    assert robot is not None
    block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="block",
            shape=CubeCfg(size=BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=BLOCK_POS,
            init_rot=(0.0, 0.0, 0.0),
        )
    )
    return sim, robot, block


def _target_beyond_block(robot: Robot) -> torch.Tensor:
    qpos = robot.get_qpos(name=CONTROL_PART)
    target = robot.compute_fk(qpos=qpos, name=CONTROL_PART, to_matrix=True)[0].clone()
    target[:3, 3] = torch.tensor([0.55, 0.30, 0.45], device=robot.device)
    return target


def _make_engine(sim, robot, block):
    motion_generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=ROBOT_UID,
                world=CuroboWorldCfg(rigid_objects=[block]),
            )
        )
    )
    engine = AtomicActionEngine(motion_generator)
    engine.register(
        MoveEndEffector(
            motion_generator,
            MoveEndEffectorCfg(
                motion_source="motion_gen",
                planner_type="curobo",
                control_part=CONTROL_PART,
                sample_interval=80,
            ),
        ),
        name="move_end_effector",
    )
    return engine


def test_curobo_subprocess_plans_fast_without_crashing() -> None:
    """The isolated worker plans a collision-free trajectory in well under a second."""
    sim, robot, block = _build_scene()
    try:
        engine = _make_engine(sim, robot, block)
        target = _target_beyond_block(robot)

        # First plan pays the one-time worker spawn + graph-capturing warmup.
        success0, trajectory0, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        assert bool(success0.item()), "first (warmup) plan failed"
        assert trajectory0.shape[0] == 1

        # Second plan: worker is warm, so this should be ~0.02s. Allow headroom.
        start = time.perf_counter()
        success1, trajectory1, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        warm_plan_seconds = time.perf_counter() - start

        assert bool(success1.item()), "second plan failed"
        assert warm_plan_seconds < 1.0, (
            f"Isolated cuRobo plan took {warm_plan_seconds:.3f}s after warmup; "
            "expected <1s (CUDA graphs should be captured)."
        )
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
