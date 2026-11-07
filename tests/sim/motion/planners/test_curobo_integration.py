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
it builds a Panda profile with shared and per-environment collision worlds,
plans Cartesian and joint-space moves through the EmbodiChain motion APIs, and
verifies the resulting trajectories and dynamic obstacle updates.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

# Module-level guards: skip the whole file without cuRobo or CUDA. These must
# run before any cuRobo-only import.
pytest.importorskip("curobo")
if not torch.cuda.is_available():
    pytest.skip("cuRobo V2 requires CUDA", allow_module_level=True)

pytestmark = [pytest.mark.requires_sim, pytest.mark.gpu]

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg  # noqa: E402
from embodichain.lab.sim.objects import RigidObjectCfg  # noqa: E402
from embodichain_robots import FrankaPandaCfg  # noqa: E402
from embodichain.lab.sim.cfg import RigidBodyPhysicsCfg  # noqa: E402
from embodichain.lab.sim.shapes import CubeCfg  # noqa: E402
from embodichain.lab.sim.motion.motion_generator import (
    MotionGenCfg,
    MotionGenOptions,
    MotionGenerator,
)
from embodichain.lab.sim.motion.planners import PlanState
from embodichain.lab.sim.motion.planners.utils import MoveType
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (  # noqa: E402
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg,
    CuroboWorldCfg,
    _torch_cuda_graph_capture_mode,
)

ROBOT_UID = "curobo_franka"
CONTROL_PART = "arm"
DEMO_BLOCK_DIMS = [0.18, 0.40, 0.36]
DEMO_BLOCK_POS = [0.45, 0.0, 0.18]
# Small displacement from Panda's neutral ready pose. It is deliberately far
# from the joint limits so this smoke test exercises cuRobo planning rather
# than limit handling.
JOINT_1_TARGET_DELTA_RAD = 0.12


def _make_sim_robot(num_envs: int = 1, gravity: list[float] | None = None):
    sim_cfg = SimulationManagerCfg(headless=True, sim_device="cuda", num_envs=num_envs)
    if gravity is not None:
        sim_cfg.physics_cfg.gravity = gravity
    sim = SimulationManager(sim_cfg)
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": ROBOT_UID, "robot_type": "panda"})
    )
    # The block is both a DexSim obstacle and the source of the cuRobo collision
    # world (CuroboWorldCfg.rigid_objects), so sim and planner share geometry.
    block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="demo_block",
            shape=CubeCfg(size=DEMO_BLOCK_DIMS),
            attrs=RigidBodyPhysicsCfg(),
            body_type="static",
            init_pos=DEMO_BLOCK_POS,
            init_rot=[0.0, 0.0, 0.0],
        )
    )
    sim.prepare()
    return sim, robot, block


def test_curobo_v2_plans_around_a_static_cuboid_obstacle():
    sim, robot, block = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=[block]),
            # Skipping optional non-graph warmup keeps fresh CI runs practical.
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


def test_curobo_v2_plans_around_rigid_object_voxel_world():
    """Exercise convex-hull preprocessing and voxel ESDF planning end to end."""
    sim, robot, block = _make_sim_robot()
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=[block], representation="voxel"),
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


@pytest.mark.parametrize("gravity", [[0.0, 0.0, -9.81], [0.0, 0.0, 0.0]])
def test_curobo_v2_plans_a_joint_space_move(gravity):
    """Route a ``JOINT_MOVE`` through V2 ``plan_cspace`` on CUDA."""
    sim, robot, block = _make_sim_robot(gravity=gravity)
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
        backend = next(iter(mg.planner._backend_cache.values()))
        # Check the gravity actually consumed by native TimeOpt, not just the
        # adapter's request. Earth gravity must never leak into a zero-g solve.
        assert (
            backend.planner.trajopt_solver._timeopt.optimizer.config.gravity
            == pytest.approx(gravity)
        )
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
def test_curobo_v2_plans_batched_joint_space_moves_direct_first(monkeypatch):
    """Preserve direct-first planning and exercise the configured RRT fallback."""
    sim, robot, _ = _make_sim_robot(num_envs=8)
    try:
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            world=CuroboWorldCfg(rigid_objects=None, multi_env=False),
            max_attempts=5,
            use_cuda_graph=True,
            warmup_iterations=1,
        )
        mg = MotionGenerator(MotionGenCfg(planner_cfg=cfg))
        start_qpos = robot.get_qpos(name=CONTROL_PART)
        # Regression targets from atomic_franka_pgi_pose_batch seed 11.
        # All eight straight joint-space paths are comfortably inside the arm
        # limits and free of self-collision; the old graph-first batch call
        # nevertheless reported rows 2, 4, 5, and 7 as failed.
        offsets = torch.tensor(
            [
                [0.15976, -0.05098, 0.12690, 0.15966, -0.08830, 0.06692, -0.08060],
                [0.12048, -0.10187, 0.08644, 0.10465, -0.10477, 0.06865, -0.12116],
                [0.09504, -0.07512, 0.07663, 0.11244, -0.11973, 0.11481, -0.12281],
                [0.13500, -0.04382, 0.11807, 0.09559, -0.04747, 0.09605, -0.06925],
                [0.08764, -0.06679, 0.12766, 0.08694, -0.06389, 0.06160, -0.12890],
                [0.13917, -0.05947, 0.07191, 0.14824, -0.07624, 0.11442, -0.12609],
                [0.12398, -0.06021, 0.12909, 0.09491, -0.05128, 0.09065, -0.11569],
                [0.13496, -0.05047, 0.08403, 0.11617, -0.06094, 0.13774, -0.12086],
            ],
            dtype=start_qpos.dtype,
            device=start_qpos.device,
        )
        target_qpos = start_qpos + offsets

        result = mg.generate(
            [PlanState.from_qpos(target_qpos)],
            MotionGenOptions(
                sample_count=80,
                start_qpos=start_qpos,
                control_part=CONTROL_PART,
                plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
            ),
        )

        assert result.success.tolist() == [True] * 8
        assert result.positions is not None
        assert result.positions.shape == (8, 80, start_qpos.shape[-1])
        assert result.dt is not None
        assert result.dt.shape == (8, 80)
        assert torch.isfinite(result.positions).all()
        assert torch.allclose(result.positions[:, 0], start_qpos, atol=1e-3)
        assert torch.allclose(result.positions[:, -1], target_qpos, atol=1e-3)

        # Force the fallback once: easy direct-first queries otherwise need not
        # touch RRT's lane buffers or its separate collision rollout at all.
        planner = mg.planner
        backend = next(iter(planner._backend_cache.values()))

        def no_new_capture(*args, **kwargs):
            raise AssertionError("RRT fallback attempted capture after preparation")

        monkeypatch.setattr(torch.cuda, "graph", no_new_capture)
        with _torch_cuda_graph_capture_mode(cfg.cuda_graph_capture_error_mode):
            native = backend.planner.plan_cspace(
                planner._to_curobo_joint_goal(target_qpos, backend),
                planner._to_curobo_joint_state(start_qpos, backend),
                max_attempts=1,
                enable_graph_attempt=0,
            )
        assert native is not None
        assert native.success.reshape(-1).tolist() == [True] * 8
        _, positions, _, _, _ = planner._extract_segment(native, backend)
        assert torch.allclose(positions[:, -1], target_qpos, atol=1e-3)
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.slow
def test_curobo_v2_multi_env_worlds_are_independent():
    """Apply the first dynamic update to both in-process collision worlds."""
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

        start_qpos = robot.get_qpos(name=CONTROL_PART)
        target_qpos = start_qpos.clone()
        target_qpos[:, 0] += torch.tensor([0.08, -0.08], device=robot.device)
        dynamic_poses = block.get_local_pose(to_matrix=True).clone()
        result = planner.plan(
            [PlanState.from_qpos(target_qpos)],
            CuroboPlanOptions(
                start_qpos=start_qpos,
                control_part=CONTROL_PART,
                dynamic_obstacle_poses={"demo_block": dynamic_poses},
            ),
        )
        assert result.success.tolist() == [True, True]
        assert result.positions is not None
        assert result.positions.shape[0] == 2
        # Each env reaches its own distinct target.
        assert torch.allclose(result.positions[0, -1], target_qpos[0], atol=1e-3)
        assert torch.allclose(result.positions[1, -1], target_qpos[1], atol=1e-3)

        # Apply a different per-env offset. The update itself checks that both
        # collision worlds are independently addressable; replanning is
        # not asserted because the offset moves the block into the arm's path.
        dynamic_poses[:, 0, 3] += torch.tensor(
            [0.10, -0.15], device=dynamic_poses.device
        )
        backend = next(iter(planner._backend_cache.values()))
        planner.update_dynamic_obstacles({"demo_block": dynamic_poses}, backend)
    finally:
        if planner is not None:
            planner.close()
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


def test_observed_payload_buffers_and_query_lifetimes(
    record_property: Callable[[str, object], None],
) -> None:
    """Qualify singleton/multi-sphere attachment and GC at a host batch boundary."""
    import gc
    import weakref

    from embodichain.compute.geometry.mesh_bounds import aabb_sphere_cover
    from embodichain.lab.sim.motion.planners.curobo.payload import (
        CuroboPayload,
        CuroboQuerySnapshot,
        _attachment_models,
        curobo_motion_generator,
    )
    from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources

    sim, robot, block = _make_sim_robot()
    original_gc_enabled = gc.isenabled()
    try:
        full_qpos = robot.get_qpos().clone()
        start = robot.get_qpos(name=CONTROL_PART).clone()
        target = start.clone()
        target[:, 0] += JOINT_1_TARGET_DELTA_RAD
        world_to_base = torch.linalg.inv(
            robot.get_control_part_base_pose(CONTROL_PART, to_matrix=True)
        )
        # Nonzero sphere center and rotated/translated object pose detect a
        # workaround that drops the frame transform or attaches at the TCP.
        object_pose = robot.compute_fk(qpos=start, name=CONTROL_PART, to_matrix=True)
        offset = torch.eye(4, device=start.device).unsqueeze(0)
        offset[0, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            device=start.device,
        )
        offset[0, :3, 3] = torch.tensor([0.01, -0.02, 0.03], device=start.device)
        object_pose = object_pose @ offset
        world = snapshot_collision_sources(
            {"obstacle": block.get_collision_shapes()},
            {"obstacle": block.get_local_pose(to_matrix=True)},
            world_to_base,
        )
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            use_cuda_graph=False,
            preserve_plan_samples=True,
            warmup_iterations=0,
            world=CuroboWorldCfg(rigid_objects=world),
        )
        retained = []
        weak_buffers = []

        class CallerFailure(Exception):
            pass

        def run_query(sphere_count: int, fail_caller: bool = False) -> None:
            """Drop all native locals before testing the collection boundary."""
            spheres = torch.tensor([[0.002, 0.003, 0.004, 0.005]], device=start.device)
            if sphere_count == 27:
                axis = torch.tensor([-0.01, 0.01], device=start.device)
                spheres = aabb_sphere_cover(torch.cartesian_prod(axis, axis, axis))
            snapshot = CuroboQuerySnapshot(
                full_qpos=full_qpos,
                world_to_base=world_to_base,
                payload=CuroboPayload(
                    object_id="carried",
                    control_part=CONTROL_PART,
                    spheres=spheres,
                    object_pose=object_pose,
                ),
            )
            with curobo_motion_generator(cfg, query_snapshot=snapshot) as mg:
                mg.planner.prepare_backend(
                    control_part=CONTROL_PART,
                    batch_size=1,
                    move_type=MoveType.JOINT_MOVE,
                )
                backend = next(iter(mg.planner._backend_cache.values()))
                models = _attachment_models(backend.planner)
                for model in models:
                    params = model.config.kinematics_config
                    weak_buffers.append(weakref.ref(params.link_spheres))
                    indices = params.get_sphere_index_from_link_name("gen_sim_payload")
                    state = mg.planner._to_curobo_joint_state(start, backend)
                    tool = model.compute_kinematics(state).tool_poses.get_link_pose(
                        model.tool_frames[0]
                    )
                    transform = (
                        torch.linalg.inv(tool.get_matrix())
                        @ world_to_base
                        @ object_pose
                    )
                    expected = (
                        spheres[:, :3] @ transform[0, :3, :3].T + transform[0, :3, 3]
                    )
                    actual = params.link_spheres[0, indices]
                    torch.testing.assert_close(
                        actual[:, :3], expected, atol=1e-6, rtol=1e-5
                    )
                    torch.testing.assert_close(
                        actual[:, 3], spheres[:, 3], atol=0, rtol=0
                    )
                result = mg.generate(
                    [PlanState.from_qpos(target)],
                    MotionGenOptions(
                        start_qpos=start,
                        control_part=CONTROL_PART,
                        plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
                    ),
                )
                assert bool(result.success.all())
                assert torch.allclose(result.positions[:, -1], target, atol=0.02)
                retained.append(
                    (
                        result.positions,
                        result.positions.clone(),
                        result.dt,
                        result.dt.clone(),
                    )
                )
                if fail_caller:
                    raise CallerFailure
            assert not mg.planner._payload_managers
            assert not mg.planner._backend_cache
            mg.planner.close()  # Idempotent after detaching and native destruction.

        gc.collect()
        # Make delayed collection reproducible, independent of Python thresholds.
        gc.disable()
        settled_bytes = []
        for iteration in range(6):
            if iteration == 5:
                with pytest.raises(CallerFailure):
                    run_query(1, fail_caller=True)
            else:
                run_query(1 if iteration % 2 == 0 else 27)
            torch.cuda.synchronize()
            # The host owns this batch boundary; close does not promise to
            # collect native optimizer/graph-planner Python callback cycles.
            gc.collect()
            torch.cuda.synchronize()
            assert all(ref() is None for ref in weak_buffers)
            settled_bytes.append(torch.cuda.memory_allocated())
            assert all(
                torch.equal(a, b) and torch.equal(dt, copy_dt)
                for a, b, dt, copy_dt in retained
            )
            assert torch.equal(robot.get_qpos(), full_qpos)
        # Exclude first-use allocator/library initialization. Six tiny retained
        # trajectories are far below this allowance; one old native query was ~50 MiB.
        record_property("settled_allocated_bytes", settled_bytes)
        record_property("native_buffer_weakrefs", len(weak_buffers))
        assert max(settled_bytes[2:]) - min(settled_bytes[2:]) < 8 * 1024**2
    finally:
        if original_gc_enabled:
            gc.enable()
        gc.collect()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


@pytest.mark.parametrize("batch_size", [2, 4])
def test_observed_shared_world_native_batch(batch_size: int, monkeypatch) -> None:
    """Exercise native batch planning, explicit row frames and owned results."""
    import gc
    import weakref
    from embodichain.lab.sim.motion.planners.curobo.payload import (
        CuroboQuerySnapshot,
        curobo_motion_generator,
    )
    from embodichain.lab.sim.scene.collision_sources import snapshot_collision_sources

    sim, robot, block = _make_sim_robot(num_envs=batch_size)
    retained = []
    try:
        full = robot.get_qpos().clone()
        start = robot.get_qpos(name=CONTROL_PART).clone()
        target = start.clone()
        target[:, 0] += torch.linspace(0.04, 0.12, batch_size, device=start.device)
        real_frame = torch.linalg.inv(
            robot.get_control_part_base_pose(CONTROL_PART, to_matrix=True)
        )
        # Use distinct explicit observed frames to catch a row-zero broadcast.
        frames = torch.eye(4, device=start.device).repeat(batch_size, 1, 1)
        frames[:, 0, 3] = torch.arange(batch_size, device=start.device)
        frames[1, :3, :3] = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=start.device
        )
        base_goal = real_frame @ robot.compute_fk(
            qpos=target, name=CONTROL_PART, to_matrix=True
        )
        goals = torch.linalg.inv(frames) @ base_goal
        shared_pose = real_frame[:1] @ block.get_local_pose(to_matrix=True)[:1]
        world = snapshot_collision_sources(
            {"obstacle": block.get_collision_shapes()},
            {"obstacle": torch.linalg.inv(frames) @ shared_pose},
            frames,
        )
        cfg = CuroboPlannerCfg(
            robot_uid=ROBOT_UID,
            use_cuda_graph=False,
            preserve_plan_samples=True,
            warmup_iterations=0,
            max_attempts=2,
            world=CuroboWorldCfg(rigid_objects=world),
        )
        snapshot = CuroboQuerySnapshot(full_qpos=full, world_to_base=frames)

        def no_capture(*args, **kwargs):
            raise AssertionError("A non-graph backend attempted CUDA graph capture")

        monkeypatch.setattr(torch.cuda, "graph", no_capture)
        for kind in (MoveType.JOINT_MOVE, MoveType.EEF_MOVE):
            with curobo_motion_generator(cfg, query_snapshot=snapshot) as mg:
                meta = mg.planner.prepare_backend(
                    control_part=CONTROL_PART, batch_size=batch_size, move_type=kind
                )
                backend = next(iter(mg.planner._backend_cache.values()))
                assert isinstance(
                    backend.planner, mg.planner._bindings.BatchMotionPlanner
                )
                assert meta["batch_size"] == batch_size
                assert meta["use_cuda_graph"] is False
                assert meta["graph_planner_algorithm"] == "rrt"
                buffer_ref = weakref.ref(
                    backend.planner.kinematics.config.kinematics_config.link_spheres
                )
                result = mg.generate(
                    [
                        (
                            PlanState.from_qpos(target)
                            if kind == MoveType.JOINT_MOVE
                            else PlanState.from_xpos(goals)
                        )
                    ],
                    MotionGenOptions(
                        start_qpos=start,
                        control_part=CONTROL_PART,
                        plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
                    ),
                )
                assert result.success.tolist() == [True] * batch_size
                torch.testing.assert_close(
                    result.positions[:, 0], start, atol=1e-3, rtol=0
                )
                actual_goal = real_frame @ robot.compute_fk(
                    qpos=result.positions[:, -1], name=CONTROL_PART, to_matrix=True
                )
                torch.testing.assert_close(
                    actual_goal[:, :3, 3], base_goal[:, :3, 3], atol=0.02, rtol=0
                )
                if kind == MoveType.JOINT_MOVE:
                    torch.testing.assert_close(
                        result.positions[:, -1], target, atol=1e-3, rtol=0
                    )
                relative_rotation = (
                    actual_goal[:, :3, :3].transpose(-1, -2) @ base_goal[:, :3, :3]
                )
                angle = torch.acos(
                    (
                        (relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                    ).clamp(-1, 1)
                )
                assert (angle < 0.05).all()
                assert (
                    torch.isfinite(result.positions).all()
                    and torch.isfinite(result.dt).all()
                )
                assert (result.dt >= 0).all() and (result.duration > 0).all()
                assert mg.planner.validate_joint_trajectory(
                    result.positions, control_part=CONTROL_PART
                ).all()
                retained.append(
                    (
                        result.positions,
                        result.positions.clone(),
                        result.dt,
                        result.dt.clone(),
                    )
                )
                if batch_size == 2 and kind == MoveType.EEF_MOVE:
                    mixed_goals = goals.clone()
                    mixed_goals[1, :3, 3] = 50.0  # Beyond the arm workspace.
                    mixed = mg.generate(
                        [PlanState.from_xpos(mixed_goals)],
                        MotionGenOptions(
                            start_qpos=start,
                            control_part=CONTROL_PART,
                            plan_opts=CuroboPlanOptions(control_part=CONTROL_PART),
                        ),
                    )
                    assert mixed.success.tolist() == [True, False]
                    torch.testing.assert_close(
                        mixed.positions[1],
                        start[1].expand_as(mixed.positions[1]),
                        atol=0,
                        rtol=0,
                    )
                    assert (mixed.dt[1] == 0).all()
            del backend, mg
            gc.collect()
            torch.cuda.synchronize()
            assert buffer_ref() is None
            assert all(
                torch.equal(a, b) and torch.equal(c, d) for a, b, c, d in retained
            )
        torch.testing.assert_close(robot.get_qpos(), full, atol=0, rtol=0)
    finally:
        gc.collect()
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
