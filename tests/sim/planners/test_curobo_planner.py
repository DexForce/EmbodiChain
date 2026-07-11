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

"""Dependency-free unit tests for the optional cuRobo planner surface.

These tests never import the real ``curobo`` package. They cover config
validation, the public export behavior, the named-joint reorder helper, the
matrix -> position/quaternion conversion, the dynamic-obstacle validator, and
the actionable error raised when cuRobo is absent.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.planners import CuroboPlannerCfg, PlanState
from embodichain.lab.sim.planners.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg as CuroboPlannerCfgDirect,
    CuroboRobotProfileCfg,
    CuroboWorldCfg,
    _matrix_to_position_quaternion,
    _require_curobo,
    _reorder_by_names,
    _validate_dynamic_obstacles,
)


def _raise_module_not_found(*args, **kwargs):
    raise ModuleNotFoundError("curobo not installed")


def test_public_config_imports_without_curobo():
    """The planner package must export cuRobo configs without curobo installed."""
    assert CuroboPlannerCfg.__name__ == "CuroboPlannerCfg"
    assert CuroboPlannerCfgDirect is CuroboPlannerCfg
    assert CuroboPlannerCfg().planner_type == "curobo"


def test_reorder_by_names_preserves_batch_and_time_dimensions():
    values = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])  # (1, 2, 2)
    result = _reorder_by_names(values, ["joint_b", "joint_a"], ["joint_a", "joint_b"])
    assert torch.equal(result, torch.tensor([[[20.0, 10.0], [40.0, 30.0]]]))


def test_reorder_by_names_rejects_mismatched_name_sets():
    values = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="name"):
        _reorder_by_names(values, ["joint_a", "joint_b"], ["joint_a", "joint_c"])


def test_matrix_to_position_quaternion_uses_wxyz():
    matrix = torch.eye(4).unsqueeze(0)
    position, quaternion = _matrix_to_position_quaternion(matrix)
    assert torch.equal(position, torch.zeros(1, 3))
    assert torch.equal(quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert position.is_contiguous()
    assert quaternion.is_contiguous()


def test_matrix_to_position_quaternion_rejects_non_4x4_batch():
    with pytest.raises(ValueError, match="4, 4"):
        _matrix_to_position_quaternion(torch.zeros(3, 3))


def test_missing_curobo_is_actionable(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", _raise_module_not_found)
    with pytest.raises(ImportError, match=r"cu12.*cu13"):
        _require_curobo()


def test_unknown_dynamic_obstacle_is_rejected():
    with pytest.raises(ValueError, match="unknown obstacle"):
        _validate_dynamic_obstacles({"unknown": torch.eye(4)}, ["known"])


def test_dynamic_obstacle_shape_is_validated():
    # (4, 4) is not batched -> rejected; the API requires (B, 4, 4).
    with pytest.raises(ValueError, match="4, 4"):
        _validate_dynamic_obstacles({"known": torch.eye(4)}, ["known"])


def test_curobo_plan_options_carries_context_fields():
    opts = CuroboPlanOptions(
        start_qpos=torch.zeros(2, 7),
        control_part="arm",
        max_attempts=3,
    )
    assert opts.control_part == "arm"
    assert opts.max_attempts == 3
    assert opts.start_qpos.shape == (2, 7)


def test_curobo_planner_cfg_defaults():
    cfg = CuroboPlannerCfg(robot_uid="franka")
    assert cfg.planner_type == "curobo"
    assert cfg.warmup is True
    assert cfg.max_attempts == 5
    assert cfg.use_cuda_graph is False
    assert isinstance(cfg.world, CuroboWorldCfg)


def test_curobo_world_cfg_uses_v2_safe_default_collision_cache():
    cache = CuroboWorldCfg().collision_cache

    assert cache == {"cuboid": 8, "mesh": 2}


def test_curobo_robot_profile_cfg_requires_joint_map():
    cfg = CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names={"a": "b"},
    )
    assert cfg.robot_config_path == "franka.yml"
    assert cfg.sim_to_curobo_joint_names == {"a": "b"}


def test_curobo_planner_class_is_lazy_import_safe():
    """Referencing the class must not import curobo."""
    import sys

    sys.modules.pop("curobo", None)
    assert CuroboPlanner.__name__ == "CuroboPlanner"
    assert "curobo" not in sys.modules


# =============================================================================
# Fake cuRobo V2 bindings + backend planning tests
# =============================================================================

from embodichain.lab.sim.planners import curobo_planner as _curobo_mod
from embodichain.lab.sim.sim_manager import SimulationManager  # noqa: E402


class _FakeTrajectory:
    def __init__(self, position, joint_names, dt=None):
        self.position = position  # (B, 1, T, D)
        self.joint_names = list(joint_names)
        self.dt = dt


class _FakeV2Result:
    def __init__(self, success, trajectory, last_tstep, total_time=0.5):
        self.success = success  # (B, 1)
        self.interpolated_trajectory = trajectory
        self.interpolated_last_tstep = last_tstep  # (B, 1)
        self.total_time = total_time


class _FakeJointState:
    def __init__(self, position, joint_names):
        self.position = position
        self.joint_names = joint_names


class _FakePose:
    def __init__(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion


class _FakeGoalToolPose:
    def __init__(self, pose_dict, ordered_tool_frames):
        self.pose_dict = pose_dict
        self.ordered_tool_frames = ordered_tool_frames


class _FakeCollisionChecker:
    def __init__(self):
        self.updates = []

    def update_obstacle_pose(self, name, pose, env_idx=0):
        self.updates.append((name, pose, env_idx))


class _FakeKinematics:
    def __init__(self, joint_names, base_link="base"):
        self.joint_names = list(joint_names)
        self.base_link = base_link


class _FakeV2PlannerInstance:
    def __init__(self, bindings):
        self._bindings = bindings
        self.joint_names = list(bindings.full_joint_names)
        self.tool_frame = bindings.tool_frame
        self.tool_frames = list(bindings.tool_frames)
        self.scene_collision_checker = _FakeCollisionChecker()
        self.kinematics = _FakeKinematics(self.joint_names)
        self.plan_pose_calls = []
        self.plan_cspace_calls = []
        self.is_batch = False
        self.max_batch_size = None
        self.closed = False
        self.warmup_count = 0
        self.warmup_enable_graph = None

    def plan_pose(self, goal, current_state, max_attempts=5):
        self.plan_pose_calls.append((goal, current_state, max_attempts))
        return self._next_result()

    def plan_cspace(self, goal_state, current_state, max_attempts=5, **kwargs):
        self.plan_cspace_calls.append((goal_state, current_state, max_attempts))
        return self._next_result()

    def _next_result(self):
        if self._bindings.results:
            return self._bindings.results.pop(0)
        return self._bindings.next_result

    def warmup(self, enable_graph=True):
        self.warmup_count += 1
        self.warmup_enable_graph = enable_graph

    def close(self):
        self.closed = True


class _FakeMotionPlannerCfg:
    def __init__(self, bindings):
        self._bindings = bindings

    def create(self, **kwargs):
        self._bindings.create_kwargs = kwargs
        return ("fake_planner_cfg", kwargs)


class _FakeMotionPlanner:
    def __init__(self, bindings):
        self._bindings = bindings

    def __call__(self, cfg):
        inst = _FakeV2PlannerInstance(self._bindings)
        self._bindings.created_planners.append(inst)
        return inst


class _FakeBatchMotionPlanner:
    def __init__(self, bindings):
        self._bindings = bindings

    def __call__(self, cfg, max_batch_size=None):
        inst = _FakeV2PlannerInstance(self._bindings)
        inst.is_batch = True
        inst.max_batch_size = max_batch_size
        self._bindings.created_planners.append(inst)
        return inst


class _FakeJointStateFactory:
    def __init__(self, bindings):
        self._bindings = bindings

    def from_position(self, position, joint_names=None):
        if not isinstance(position, torch.Tensor):
            raise TypeError("JointState.from_position requires a tensor position")
        return _FakeJointState(position=position, joint_names=joint_names)


class _FakePoseFactory:
    def __init__(self, bindings):
        self._bindings = bindings

    def __call__(self, position, quaternion):
        return _FakePose(position=position, quaternion=quaternion)


class _FakeGoalToolPoseFactory:
    def __init__(self, bindings):
        self._bindings = bindings

    def from_poses(self, pose_dict, ordered_tool_frames=None, num_goalset=1):
        return _FakeGoalToolPose(
            pose_dict=pose_dict, ordered_tool_frames=ordered_tool_frames
        )


class _FakeDeviceCfgFactory:
    def __call__(self, device):
        return ("fake_device_cfg", device)


class _FakeCuroboBindings:
    """A minimal stand-in for the cuRobo V2 facade namespace."""

    def __init__(self, full_joint_names, tool_frame="tool"):
        self.full_joint_names = list(full_joint_names)
        self.tool_frame = tool_frame
        self.tool_frames = [tool_frame]
        self.warmup_count = 0
        self.create_kwargs = None
        self.created_planners: list = []
        self.results: list | None = None
        self.MotionPlannerCfg = _FakeMotionPlannerCfg(self)
        self.MotionPlanner = _FakeMotionPlanner(self)
        self.BatchMotionPlanner = _FakeBatchMotionPlanner(self)
        self.JointState = _FakeJointStateFactory(self)
        self.Pose = _FakePoseFactory(self)
        self.GoalToolPose = _FakeGoalToolPoseFactory(self)
        self.DeviceCfg = _FakeDeviceCfgFactory()
        self.next_result = self.make_result(
            position=torch.zeros(1, 1, 3, len(full_joint_names)),
            dt=torch.tensor([0.0, 0.025, 0.025]),
        )

    def make_result(
        self, position, success=None, last_tstep=None, total_time=0.5, dt=None
    ):
        B, _, T, D = position.shape
        if success is None:
            success = torch.ones(B, 1, dtype=torch.bool)
        if last_tstep is None:
            last_tstep = torch.full((B, 1), T - 1, dtype=torch.long)
        traj = _FakeTrajectory(
            position=position, joint_names=list(self.full_joint_names), dt=dt
        )
        return _FakeV2Result(
            success=success,
            trajectory=traj,
            last_tstep=last_tstep,
            total_time=total_time,
        )


class _FakeRobot:
    def __init__(self, device="cuda", num_instances=1, dof=2):
        self.uid = "fake_robot"
        self.device = torch.device(device)
        self.num_instances = num_instances
        self.dof = dof
        self.control_parts = {"arm": ["sim_a", "sim_b"]}
        self._solver = SimpleNamespace(root_link_name="sim_base")
        self.base_poses = torch.eye(4, device=self.device).repeat(num_instances, 1, 1)

    def get_qpos(self, name=None):
        return torch.zeros(self.num_instances, self.dof)

    def get_joint_ids(self, name=None):
        return list(range(self.dof))

    def get_solver(self, name=None):
        assert name == "arm"
        return self._solver

    def get_link_pose(self, link_name, env_ids=None, to_matrix=False):
        assert link_name == self._solver.root_link_name
        assert to_matrix is True
        if env_ids is None:
            return self.base_poses
        return self.base_poses[env_ids]


class _FakeSim:
    def __init__(self, robot):
        self.robot = robot

    def get_robot(self, uid):
        return self.robot


def _default_profile():
    return CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names={"sim_a": "cu_a", "sim_b": "cu_b"},
    )


def _make_planner(
    fake_curobo,
    fake_sim,
    *,
    profiles=None,
    world=None,
    **cfg_kw,
):
    """Construct a CuroboPlanner against fake bindings + fake sim."""
    if profiles is None:
        profiles = {"arm": _default_profile()}
    cfg = CuroboPlannerCfg(
        robot_uid="fake_robot",
        robot_profiles=profiles,
        world=world if world is not None else CuroboWorldCfg(),
        **cfg_kw,
    )
    return CuroboPlanner(cfg)


@pytest.fixture
def fake_sim(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("cuRobo backend requires a CUDA device")
    robot = _FakeRobot(device="cuda")
    sim = _FakeSim(robot)
    monkeypatch.setattr(SimulationManager, "get_instance", classmethod(lambda cls: sim))
    return sim


@pytest.fixture
def fake_curobo(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("cuRobo backend requires a CUDA device")
    bindings = _FakeCuroboBindings(full_joint_names=["cu_a", "cu_b"])
    monkeypatch.setattr(_curobo_mod, "_require_curobo", lambda: bindings)
    return bindings


def test_plan_pose_maps_curobo_full_output_to_control_part(fake_curobo, fake_sim):
    fake_curobo.next_result = fake_curobo.make_result(
        position=torch.tensor([[[[0.2, -0.1], [1.5, 0.5], [2.0, 1.0]]]]),
        dt=torch.tensor([0.0, 0.025, 0.025]),
    )
    planner = _make_planner(fake_curobo, fake_sim)
    result = planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.tensor([[0.2, -0.1]]), control_part="arm"),
    )
    assert result.success.tolist() == [True]
    assert result.positions.shape == (1, 3, 2)
    assert torch.equal(result.positions[0, -1].cpu(), torch.tensor([2.0, 1.0]))
    assert result.dt.shape == (1, 3)
    assert result.duration.shape == (1,)
    # warmup ran once and exactly one backend was built.
    assert fake_curobo.created_planners[0].warmup_count == 1


def test_failed_v2_result_holds_start_qpos(fake_curobo, fake_sim):
    fake_curobo.next_result.success = torch.tensor([[False]])
    planner = _make_planner(fake_curobo, fake_sim)
    start = torch.tensor([[0.3, -0.4]])
    result = planner.plan(
        [PlanState.from_qpos(start)],
        CuroboPlanOptions(start_qpos=start, control_part="arm"),
    )
    assert result.success.tolist() == [False]
    assert torch.equal(result.positions.cpu(), start.unsqueeze(1))


def test_two_waypoints_chain_segments_without_resample(fake_curobo, fake_sim):
    r1 = fake_curobo.make_result(
        position=torch.tensor([[[[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]]]),
        dt=torch.tensor([0.0, 0.025, 0.025]),
    )
    r2 = fake_curobo.make_result(
        position=torch.tensor([[[[1.0, 0.0], [2.0, 0.0]]]]),
        dt=torch.tensor([0.0, 0.025]),
    )
    fake_curobo.results = [r1, r2]
    planner = _make_planner(fake_curobo, fake_sim)
    start = torch.tensor([[0.0, 0.0]])
    result = planner.plan(
        [
            PlanState.from_xpos(torch.eye(4).unsqueeze(0)),
            PlanState.from_xpos(torch.eye(4).unsqueeze(0)),
        ],
        CuroboPlanOptions(start_qpos=start, control_part="arm"),
    )
    assert result.success.tolist() == [True]
    # seg1 (3) + seg2 without its duplicate junction (1) == 4 samples, unresampled.
    assert result.positions.shape == (1, 4, 2)
    assert torch.allclose(result.positions[0, 2].cpu(), torch.tensor([1.0, 0.0]))
    assert torch.allclose(result.positions[0, -1].cpu(), torch.tensor([2.0, 0.0]))
    # Segment 2 starts where segment 1 ended.
    planner_inst = fake_curobo.created_planners[0]
    assert len(planner_inst.plan_pose_calls) == 2
    second_current = planner_inst.plan_pose_calls[1][1].position
    assert torch.allclose(second_current[0].cpu(), torch.tensor([1.0, 0.0]))


def test_malformed_trajectory_joint_names_raise(fake_curobo, fake_sim):
    result = fake_curobo.make_result(position=torch.zeros(1, 1, 2, 2))
    result.interpolated_trajectory.joint_names = ["cu_a", "cu_x"]
    fake_curobo.next_result = result
    planner = _make_planner(fake_curobo, fake_sim)
    with pytest.raises(ValueError, match="missing active joint"):
        planner.plan(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )


def test_unknown_control_part_raises(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim)
    with pytest.raises(ValueError, match="No cuRobo profile"):
        planner.plan(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="leg"),
        )


def test_profile_base_link_must_match_curobo_model(fake_curobo, fake_sim):
    """A configured base frame must not silently disagree with the V2 model."""
    profile = _default_profile()
    profile.base_link_name = "unexpected_base"
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})

    with pytest.raises(ValueError, match="base_link_name"):
        planner.plan(
            [PlanState.from_qpos(torch.zeros(1, 2))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )


def test_active_joints_outside_control_part_are_rejected(fake_curobo, fake_sim):
    """Collision planning must not pin unexecuted joints only inside cuRobo."""
    fake_curobo.full_joint_names.append("cu_gripper")
    planner = _make_planner(fake_curobo, fake_sim)

    with pytest.raises(ValueError, match="outside the requested control part"):
        planner.plan(
            [PlanState.from_qpos(torch.zeros(1, 2))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )


def test_target_batch_must_match_planning_start_batch(fake_curobo, fake_sim):
    """Direct planner calls must reject a start/target batch mismatch early."""
    fake_sim.robot = _FakeRobot(num_instances=2)
    planner = _make_planner(fake_curobo, fake_sim)

    with pytest.raises(ValueError, match="target batch 2.*start batch 1"):
        planner.plan(
            [PlanState.from_qpos(torch.zeros(2, 2))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )


def test_pose_goal_is_expressed_in_curobo_base_frame(fake_curobo, fake_sim):
    """Simulator-world targets must be rebased before cuRobo sees them."""
    fake_sim.robot.base_poses[0, 0, 3] = 10.0
    profile = _default_profile()
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})
    backend = planner._get_backend(profile, "arm", batch_size=1)
    world_target = torch.eye(4).unsqueeze(0)
    world_target[0, 0, 3] = 10.5

    goal = planner._to_curobo_pose_goal(world_target, backend)

    assert torch.allclose(
        goal.pose_dict["tool"].position.cpu(), torch.tensor([[0.5, 0.0, 0.0]])
    )


def test_profile_fixed_base_transform_is_applied(fake_curobo, fake_sim):
    """A profile-specific simulator-base to cuRobo-base transform is composed."""
    fake_sim.robot.base_poses[0, 0, 3] = 10.0
    profile = _default_profile()
    profile.sim_base_to_curobo_base = [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})
    backend = planner._get_backend(profile, "arm", batch_size=1)
    world_target = torch.eye(4).unsqueeze(0)
    world_target[0, 0, 3] = 10.5

    goal = planner._to_curobo_pose_goal(world_target, backend)

    assert torch.allclose(
        goal.pose_dict["tool"].position.cpu(), torch.tensor([[1.5, 0.0, 0.0]])
    )


def test_profile_sim_base_link_works_without_local_solver(fake_curobo, fake_sim):
    """A profile can provide the simulator base link without an IK solver."""
    fake_sim.robot.get_solver = lambda name=None: None
    profile = _default_profile()
    profile.sim_base_link_name = "sim_base"
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})
    backend = planner._get_backend(profile, "arm", batch_size=1)

    goal = planner._to_curobo_pose_goal(torch.eye(4).unsqueeze(0), backend)

    assert torch.allclose(
        goal.pose_dict["tool"].position.cpu(), torch.tensor([[0.0, 0.0, 0.0]])
    )


def test_dynamic_obstacle_pose_is_expressed_in_curobo_base_frame(fake_curobo, fake_sim):
    """Dynamic simulator-world obstacle poses use the same base conversion."""
    fake_sim.robot.base_poses[0, 0, 3] = 10.0
    world = CuroboWorldCfg(dynamic_obstacle_names=["block"])
    planner = _make_planner(fake_curobo, fake_sim, world=world)
    profile = planner.cfg.robot_profiles["arm"]
    backend = planner._get_backend(profile, "arm", batch_size=1)
    world_pose = torch.eye(4).unsqueeze(0)
    world_pose[0, 0, 3] = 10.5

    planner.update_dynamic_obstacles({"block": world_pose}, backend)

    _, pose, _ = backend.planner.scene_collision_checker.updates[-1]
    assert torch.allclose(pose.position.cpu(), torch.tensor([0.5, 0.0, 0.0]))


def test_batched_pose_goals_rebase_each_simulator_arena(fake_curobo, fake_sim):
    """Parallel arenas retain one common local cuRobo target per environment."""
    fake_sim.robot = _FakeRobot(num_instances=2)
    fake_sim.robot.base_poses[1, 0, 3] = 10.0
    profile = _default_profile()
    planner = _make_planner(
        fake_curobo,
        fake_sim,
        profiles={"arm": profile},
        world=CuroboWorldCfg(multi_env=True),
    )
    backend = planner._get_backend(profile, "arm", batch_size=2)
    world_targets = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    world_targets[:, 0, 3] = torch.tensor([0.5, 10.5])

    goal = planner._to_curobo_pose_goal(world_targets, backend)

    expected = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert torch.allclose(goal.pose_dict["tool"].position.cpu(), expected)


def test_multi_env_materializes_one_scene_mapping_per_batch_row(
    fake_curobo, fake_sim, tmp_path
):
    """V2 needs a list of B scene mappings, not a singleton scene path."""
    scene_path = Path(tmp_path) / "world.yml"
    scene_path.write_text(
        "cuboid:\n"
        "  block:\n"
        "    dims: [0.1, 0.1, 0.1]\n"
        "    pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]\n",
        encoding="utf-8",
    )
    fake_sim.robot = _FakeRobot(num_instances=2)
    planner = _make_planner(
        fake_curobo,
        fake_sim,
        world=CuroboWorldCfg(world_config_path=str(scene_path), multi_env=True),
    )
    profile = planner.cfg.robot_profiles["arm"]

    planner._get_backend(profile, "arm", batch_size=2)

    scene_models = fake_curobo.create_kwargs["scene_model"]
    assert len(scene_models) == 2
    assert scene_models[0] == scene_models[1]
    assert scene_models[0] is not scene_models[1]
    assert scene_models[0]["cuboid"] is not scene_models[1]["cuboid"]


def test_multi_env_materializes_empty_scene_for_every_batch_row(fake_curobo, fake_sim):
    """An empty cached V2 collision world still needs B scene entries."""
    fake_sim.robot = _FakeRobot(num_instances=2)
    planner = _make_planner(
        fake_curobo,
        fake_sim,
        world=CuroboWorldCfg(multi_env=True),
    )
    profile = planner.cfg.robot_profiles["arm"]

    planner._get_backend(profile, "arm", batch_size=2)

    assert fake_curobo.create_kwargs["scene_model"] == [{}, {}]


def test_dynamic_update_requires_explicit_backend_for_mixed_batch_caches(
    fake_curobo, fake_sim
):
    """Avoid partially updating incompatible multi-env cuRobo cache entries."""
    world = CuroboWorldCfg(multi_env=True, dynamic_obstacle_names=["block"])
    planner = _make_planner(fake_curobo, fake_sim, world=world)
    profile = planner.cfg.robot_profiles["arm"]
    planner._get_backend(profile, "arm", batch_size=1)
    planner._get_backend(profile, "arm", batch_size=2)

    with pytest.raises(ValueError, match="different cached batch sizes"):
        planner.update_dynamic_obstacles({"block": torch.eye(4).unsqueeze(0)})

    assert all(
        not backend.planner.scene_collision_checker.updates
        for backend in planner._backend_cache.values()
    )


def test_non_cuda_device_is_rejected(monkeypatch):
    robot = _FakeRobot(device="cpu")
    sim = _FakeSim(robot)
    monkeypatch.setattr(SimulationManager, "get_instance", classmethod(lambda cls: sim))
    bindings = _FakeCuroboBindings(full_joint_names=["cu_a", "cu_b"])
    monkeypatch.setattr(_curobo_mod, "_require_curobo", lambda: bindings)
    with pytest.raises(RuntimeError, match="CUDA"):
        _make_planner(bindings, sim)


def test_total_time_over_budget_marks_unsuccessful(fake_curobo, fake_sim):
    fake_curobo.next_result = fake_curobo.make_result(
        position=torch.tensor([[[[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]]]]),
        total_time=0.5,
    )
    planner = _make_planner(fake_curobo, fake_sim, max_planning_time=0.1)
    start = torch.tensor([[0.3, -0.4]])
    result = planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=start, control_part="arm"),
    )
    assert result.success.tolist() == [False]
    assert torch.equal(result.positions.cpu(), start.unsqueeze(1))


def test_backend_is_cached_across_plans(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim)
    for _ in range(2):
        planner.plan(
            [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
            CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
        )
    assert len(fake_curobo.created_planners) == 1
    assert fake_curobo.created_planners[0].warmup_count == 1


def test_warmup_respects_use_cuda_graph_cfg(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim, use_cuda_graph=False)
    planner.plan(
        [PlanState.from_qpos(torch.zeros(1, 2))],
        CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )

    assert fake_curobo.created_planners[0].warmup_enable_graph is False


def test_update_dynamic_obstacles_reaches_backend(fake_curobo, fake_sim):
    world = CuroboWorldCfg(dynamic_obstacle_names=["block"])
    planner = _make_planner(fake_curobo, fake_sim, world=world)
    planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )
    backend = next(iter(planner._backend_cache.values()))
    planner.update_dynamic_obstacles(
        {"block": torch.eye(4).unsqueeze(0).repeat(1, 1, 1)}, backend
    )
    assert len(backend.planner.scene_collision_checker.updates) == 1
    name, _pose, env_idx = backend.planner.scene_collision_checker.updates[0]
    assert name == "block"
    assert env_idx == 0


def test_close_destroys_cached_planners(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim)
    planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )
    assert len(fake_curobo.created_planners) == 1
    planner.close()
    assert fake_curobo.created_planners[0].closed is True
    assert planner._backend_cache == {}


def test_joint_move_uses_plan_cspace(fake_curobo, fake_sim):
    fake_curobo.next_result = fake_curobo.make_result(
        position=torch.tensor([[[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]]),
        dt=torch.tensor([0.0, 0.025, 0.025]),
    )
    planner = _make_planner(fake_curobo, fake_sim)
    start = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 1.0]])
    result = planner.plan(
        [PlanState.from_qpos(target)],
        CuroboPlanOptions(start_qpos=start, control_part="arm"),
    )
    assert result.success.tolist() == [True]
    assert result.positions.shape == (1, 3, 2)
    assert torch.allclose(result.positions[0, -1].cpu(), target[0])
    planner_inst = fake_curobo.created_planners[0]
    assert len(planner_inst.plan_cspace_calls) == 1
    assert len(planner_inst.plan_pose_calls) == 0
    goal_state = planner_inst.plan_cspace_calls[0][0]
    assert torch.allclose(goal_state.position.cpu(), target)


def test_none_v2_result_holds_start_qpos(fake_curobo, fake_sim):
    fake_curobo.next_result = None
    planner = _make_planner(fake_curobo, fake_sim)
    start = torch.tensor([[0.3, -0.4]])

    result = planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=start, control_part="arm"),
    )

    assert result.success.tolist() == [False]
    assert torch.equal(result.positions.cpu(), start.unsqueeze(1))


def test_scalar_v2_dt_expands_over_trajectory_intervals(fake_curobo, fake_sim):
    fake_curobo.next_result = fake_curobo.make_result(
        position=torch.tensor([[[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]]]),
        dt=torch.tensor([[0.025]]),
    )
    planner = _make_planner(fake_curobo, fake_sim)

    result = planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )

    assert torch.allclose(result.dt.cpu(), torch.tensor([[0.0, 0.025, 0.025]]))
    assert torch.allclose(result.duration.cpu(), torch.tensor([0.05]))


def test_joint_mapping_uses_control_part_order_not_dict_insertion_order(
    fake_curobo, fake_sim
):
    profile = CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names={"sim_b": "cu_b", "sim_a": "cu_a"},
    )
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})
    backend = planner._get_backend(profile, "arm", batch_size=1)

    state = planner._to_curobo_joint_state(torch.tensor([[10.0, 20.0]]), backend)

    assert torch.allclose(state.position.cpu(), torch.tensor([[10.0, 20.0]]))


def test_missing_tool_frame_uses_single_backend_tool_frame(fake_curobo, fake_sim):
    profile = CuroboRobotProfileCfg(
        robot_config_path="franka.yml",
        sim_to_curobo_joint_names={"sim_a": "cu_a", "sim_b": "cu_b"},
        tool_frame_name=None,
    )
    planner = _make_planner(fake_curobo, fake_sim, profiles={"arm": profile})
    backend = planner._get_backend(profile, "arm", batch_size=1)

    goal = planner._to_curobo_pose_goal(torch.eye(4).unsqueeze(0), backend)

    assert goal.ordered_tool_frames == ["tool"]
    assert list(goal.pose_dict) == ["tool"]


def test_backend_normalizes_indexless_cuda_device_for_warp(fake_curobo, fake_sim):
    planner = _make_planner(fake_curobo, fake_sim)

    planner.plan(
        [PlanState.from_xpos(torch.eye(4).unsqueeze(0))],
        CuroboPlanOptions(start_qpos=torch.zeros(1, 2), control_part="arm"),
    )

    assert fake_curobo.create_kwargs["device_cfg"] == (
        "fake_device_cfg",
        torch.device("cuda:0"),
    )
