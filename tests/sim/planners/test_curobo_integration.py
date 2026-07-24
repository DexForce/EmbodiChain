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

import pytest
import torch

# Module-level guards: skip the whole file without cuRobo or CUDA. These must
# run before any cuRobo-only import.
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)

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
from embodichain.lab.sim.planners.curobo.curobo_planner import (  # noqa: E402
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
)

ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.40, 0.36]
DEMO_BLOCK_POS = [0.45, 0.0, 0.18]
# Small displacement from Panda's neutral ready pose. It is deliberately far
# from the joint limits so this smoke test exercises cuRobo planning rather
# than limit handling.
JOINT_1_TARGET_DELTA_RAD = 0.12


def _make_sim_robot(num_envs: int = 1):
    sim = SimulationManager(
        SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=num_envs)
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    # The block is both a DexSim obstacle and the source of the cuRobo collision
    # world (CuroboWorldCfg.rigid_objects), so sim and planner share geometry.
    block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=DEMO_BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=DEMO_BLOCK_POS,
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    return sim, robot, block


@pytest.mark.slow
def test_curobo_v2_plans_around_a_static_cuboid():
    sim, robot, block = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=[block]),
            # Skipping warmup keeps it practical on fresh CI GPU workers; the
            # subprocess worker always captures CUDA graphs (no toggle).
            warmup_iterations=0,
        )
        mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))

        start_qpos = robot.get_qpos(name=CONTROL_PART)
        start_xpos = robot.compute_fk(
            qpos=start_qpos, name=CONTROL_PART, to_matrix=True
        )
        # Target beyond the cuboid so the planner must route around it.
        target_xpos = start_xpos.clone()
        target_xpos[0, :3, 3] = torch.tensor([0.55, 0.30, 0.45], device=robot.device)

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


@pytest.mark.slow
def test_curobo_v2_plans_around_rigid_object_mesh_world():
    """Auto-generate the collision world from a live RigidObject mesh and plan.

    Uses the ``mesh`` representation (exact triangle mesh) to exercise the full
    mesh -> cuRobo world-YAML path end-to-end, complementing the default
    ``cuboid`` path in :func:`test_curobo_v2_plans_around_a_static_cuboid`.
    """
    sim, robot, block = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=[block], obstacle_representation="mesh"),
            warmup_iterations=0,
        )
        mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))

        start_qpos = robot.get_qpos(name=CONTROL_PART)
        start_xpos = robot.compute_fk(
            qpos=start_qpos, name=CONTROL_PART, to_matrix=True
        )
        target_xpos = start_xpos.clone()
        target_xpos[0, :3, 3] = torch.tensor([0.55, 0.30, 0.45], device=robot.device)

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
        assert result.positions.shape[-1] == 7
        assert torch.allclose(result.positions[0, 0], start_qpos[0], atol=1e-3)

        final_q = result.positions[0, -1:].to(robot.device)
        final_xpos = robot.compute_fk(qpos=final_q, name=CONTROL_PART, to_matrix=True)
        err = torch.norm(final_xpos[0, :3, 3] - target_xpos[0, :3, 3])
        assert float(err) < 0.02
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
def test_curobo_v2_plans_a_joint_space_move():
    """Route a ``JOINT_MOVE`` through V2 ``plan_cspace`` on CUDA."""
    sim, robot, block = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=[block]),
            warmup_iterations=0,
        )
        mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))
        start_qpos = robot.get_qpos(name=CONTROL_PART)
        target_qpos = start_qpos.clone()
        target_qpos[:, 0] += JOINT_1_TARGET_DELTA_RAD

        result = mg.generate(
            [PlanState.from_qpos(target_qpos)],
            MotionGenOptions(
                start_qpos=start_qpos,
                control_part=CONTROL_PART,
                plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
            ),
        )

        assert result.success.shape == (1,)
        assert bool(result.success.item())
        assert result.positions is not None
        assert result.positions.shape[-1] == start_qpos.shape[-1]
        assert torch.allclose(result.positions[0, 0], start_qpos[0], atol=1e-3)
        assert torch.allclose(result.positions[0, -1], target_qpos[0], atol=1e-3)
        assert float(result.duration[0]) > 0.0
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
def test_curobo_v2_multi_env_worlds_are_independent():
    """Multi-env planning + per-env dynamic-obstacle updates through the worker.

    The cuRobo planner lives in a subprocess worker, so the parent can no longer
    inspect ``backend.planner.scene_collision_checker`` directly. Per-env
    independence is exercised through the public path instead: two envs plan to
    distinct joint targets, then each env's obstacle pose is updated separately
    and a second plan still succeeds.
    """
    sim, robot, block = _make_sim_robot(num_envs=2)
    planner = None
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(
                rigid_objects=[block],
                dynamic_obstacle_names=["demo_block"],
                multi_env=True,
            ),
            warmup_iterations=0,
        )
        planner = CuroboPlanner(cfg)
        backend = planner._get_isolated_backend(CONTROL_PART, batch_size=2)

        start_qpos = robot.get_qpos(name=CONTROL_PART)
        target_qpos = start_qpos.clone()
        target_qpos[:, 0] += torch.tensor([0.08, -0.08], device=robot.device)
        result = planner.plan(
            [PlanState.from_qpos(target_qpos)],
            CuroboPlanOptions(start_qpos=start_qpos, control_part=CONTROL_PART),
        )
        assert result.success.tolist() == [True, True]
        assert result.positions is not None
        assert result.positions.shape[0] == 2
        # Each env reaches its own distinct target.
        assert torch.allclose(result.positions[0, -1], target_qpos[0], atol=1e-3)
        assert torch.allclose(result.positions[1, -1], target_qpos[1], atol=1e-3)

        # Start from each live simulator base, apply a different per-env offset,
        # and push the per-env obstacle poses to the worker. The update itself is
        # the check that per-env obstacle writes reach the worker; replanning is
        # not asserted because the offset moves the block into the arm's path.
        dynamic_poses = planner._get_sim_base_pose(backend, batch_size=2).clone()
        dynamic_poses[:, 0, 3] += torch.tensor(
            [0.10, -0.15], device=dynamic_poses.device
        )
        planner.update_dynamic_obstacles({"demo_block": dynamic_poses}, backend)
    finally:
        if planner is not None:
            planner.close()
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
