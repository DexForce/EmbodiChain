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

"""Unit and smoke tests for the optional cuRobo planner.

Most tests are dependency-free and cover planner configuration, conversion,
validation, and generated robot/world YAML. The two GPU-marked smoke tests
exercise cached in-process planning and CPU-physics interoperability. Full
collision-planning coverage remains in ``test_curobo_integration.py``.
"""

from __future__ import annotations

import importlib
import math

import pytest
import torch
import yaml

from embodichain.lab.sim.planners import CuroboPlannerCfg
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg as CuroboPlannerCfgDirect,
    CuroboWorldCfg,
    _matrix_to_position_quaternion,
    _require_curobo,
    _resolve_curobo_device,
    _torch_cuda_graph_capture_mode,
    _validate_dynamic_obstacles,
)
from embodichain.lab.sim.planners.curobo.curobo_yaml import (
    _mesh_to_obstacle_entry,
    _parse_mimic_joint_names,
    generate_curobo_world_yaml,
)

_SIM_ROBOT_UID = "curobo_franka_inprocess_test"
_SIM_CONTROL_PART = "arm"
_SIM_BLOCK_DIMS = [0.18, 0.40, 0.36]
_SIM_BLOCK_POS = (0.45, 0.0, 0.18)

# Minimal URDF mirroring the Franka Panda hand: finger joint 2 mimics joint 1.
_MIMIC_URDF = """\
<?xml version="1.0"?>
<robot name="panda_hand">
  <link name="base"/>
  <link name="fr3_hand"/>
  <link name="fr3_leftfinger"/>
  <link name="fr3_rightfinger"/>
  <joint name="fr3_hand_joint" type="fixed">
    <parent link="base"/>
    <child link="fr3_hand"/>
  </joint>
  <joint name="fr3_finger_joint1" type="prismatic">
    <parent link="fr3_hand"/>
    <child link="fr3_leftfinger"/>
    <axis xyz="0 1 0"/>
    <limit effort="100" lower="0.0" upper="0.04" velocity="0.2"/>
  </joint>
  <joint name="fr3_finger_joint2" type="prismatic">
    <parent link="fr3_hand"/>
    <child link="fr3_rightfinger"/>
    <axis xyz="0 1 0"/>
    <limit effort="100" lower="0.0" upper="0.04" velocity="0.2"/>
    <mimic joint="fr3_finger_joint1"/>
  </joint>
</robot>
"""

_NO_MIMIC_URDF = """\
<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-3.14" upper="3.14" velocity="2.0"/>
  </joint>
</robot>
"""


def _raise_module_not_found(*args, **kwargs):
    raise ModuleNotFoundError("curobo not installed")


def test_public_config_imports_without_curobo():
    """The planner package must export cuRobo configs without curobo installed."""
    assert CuroboPlannerCfg.__name__ == "CuroboPlannerCfg"
    assert CuroboPlannerCfgDirect is CuroboPlannerCfg
    assert CuroboPlannerCfg().planner_type == "curobo"


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
    assert cfg.warmup_iterations == 1
    assert cfg.max_attempts == 5
    assert cfg.cuda_device is None
    assert cfg.use_cuda_graph is True
    assert cfg.cuda_graph_fallback is True
    assert cfg.cuda_graph_capture_error_mode == "thread_local"
    assert cfg.capture_acquire_timeout == 2.0
    assert isinstance(cfg.world, CuroboWorldCfg)
    # No external-YAML / profile config; the base-frame override defaults to None.
    assert cfg.sim_base_to_curobo_base is None
    assert not hasattr(cfg, "robot_profiles")
    assert not hasattr(cfg.world, "world_config_path")


def test_curobo_world_cfg_uses_v2_safe_default_collision_cache():
    cfg = CuroboWorldCfg()

    assert cfg.collision_cache == {"cuboid": 8, "mesh": 2}
    assert cfg.obstacle_representation == "sphere"


def test_auto_gen_defaults_keep_sphere_count_low():
    """The voxel sphere estimate must be scaled down so planning stays fast."""
    auto = CuroboPlannerCfg(robot_uid="franka").auto_gen
    assert auto.fit_type == "voxel"
    assert auto.sphere_density == 0.1


def test_curobo_planner_class_is_lazy_import_safe():
    """Referencing the class must not import curobo."""
    import sys

    sys.modules.pop("curobo", None)
    assert CuroboPlanner.__name__ == "CuroboPlanner"
    assert "curobo" not in sys.modules


def test_cpu_sim_resolves_current_cuda_device(monkeypatch):
    """A CPU simulation defaults cuRobo to the current CUDA device."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert _resolve_curobo_device(None, torch.device("cpu")) == torch.device("cuda:2")


def test_planning_device_rejects_cpu_selection(monkeypatch):
    """The dedicated cuRobo device can never be configured as CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="must select a CUDA device"):
        _resolve_curobo_device("cpu", torch.device("cpu"))


def test_graph_capture_mode_is_forced_and_restored(monkeypatch):
    """The cuRobo adapter overrides capture mode only inside its context."""
    calls = []
    sentinel = object()

    def original_graph(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(torch.cuda, "graph", original_graph)

    with _torch_cuda_graph_capture_mode("thread_local"):
        result = torch.cuda.graph("graph", capture_error_mode="global")

    assert result is sentinel
    assert calls == [(("graph",), {"capture_error_mode": "thread_local"})]
    assert torch.cuda.graph is original_graph


def test_graph_capture_mode_rejects_unknown_value():
    """Invalid modes fail before the process-wide torch adapter is changed."""
    with pytest.raises(ValueError, match="capture_error_mode"):
        with _torch_cuda_graph_capture_mode("unsafe"):
            pass


# Robot YAML generation


def test_parse_mimic_joint_names_detects_mimic_joint(tmp_path):
    urdf = tmp_path / "panda_hand.urdf"
    urdf.write_text(_MIMIC_URDF, encoding="utf-8")

    assert _parse_mimic_joint_names(str(urdf)) == {"fr3_finger_joint2"}


def test_parse_mimic_joint_names_returns_empty_without_mimic(tmp_path):
    urdf = tmp_path / "arm.urdf"
    urdf.write_text(_NO_MIMIC_URDF, encoding="utf-8")

    assert _parse_mimic_joint_names(str(urdf)) == set()


def test_parse_mimic_joint_names_handles_missing_file(tmp_path):
    assert _parse_mimic_joint_names(str(tmp_path / "does_not_exist.urdf")) == set()


# World YAML generation


def _unit_cube_vertices() -> torch.Tensor:
    """Return eight vertices of a unit cube centered at the origin."""
    half_extent = 0.5
    return torch.tensor(
        [
            [-half_extent, -half_extent, -half_extent],
            [half_extent, -half_extent, -half_extent],
            [half_extent, half_extent, -half_extent],
            [-half_extent, half_extent, -half_extent],
            [-half_extent, -half_extent, half_extent],
            [half_extent, -half_extent, half_extent],
            [half_extent, half_extent, half_extent],
            [-half_extent, half_extent, half_extent],
        ],
        dtype=torch.float32,
    )


def _cube_faces() -> torch.Tensor:
    """Return twelve triangle indices for :func:`_unit_cube_vertices`."""
    return torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=torch.int32,
    )


def _identity_pose(
    translation: tuple[float, float, float] = (0.45, 0.0, 0.18),
) -> torch.Tensor:
    return torch.tensor(
        [*translation, 1.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )


class _FakeRigidObject:
    """Expose the mesh and pose API required by the world generator."""

    def __init__(
        self,
        uid: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        pose: torch.Tensor,
    ) -> None:
        self.uid = uid
        self._vertices = vertices
        self._faces = faces
        self._pose = pose

    def get_vertices(self, env_ids=None, scale=False):  # noqa: ARG002
        return self._vertices.unsqueeze(0)

    def get_triangles(self, env_ids=None):  # noqa: ARG002
        return self._faces.unsqueeze(0)

    def get_local_pose(self, to_matrix=False):  # noqa: ARG002
        return self._pose.unsqueeze(0)


def test_cuboid_entry_centered_mesh_matches_aabb_and_pose():
    entries = _mesh_to_obstacle_entry(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        representation="cuboid",
    )

    assert len(entries) == 1
    top_key, name, fields = entries[0]
    assert (top_key, name) == ("cuboid", "demo_block")
    assert fields["dims"] == pytest.approx([1.0, 1.0, 1.0])
    assert fields["pose"] == pytest.approx([0.45, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0])


def test_cuboid_entry_off_origin_mesh_offsets_center():
    vertices = _unit_cube_vertices() + 0.5
    _, _, fields = _mesh_to_obstacle_entry(
        "block",
        vertices,
        _cube_faces(),
        _identity_pose(),
        representation="cuboid",
    )[0]

    assert fields["dims"] == pytest.approx([1.0, 1.0, 1.0])
    assert fields["pose"][:3] == pytest.approx([0.95, 0.5, 0.68])


def test_cuboid_entry_rotated_pose_preserves_center():
    quaternion = torch.tensor(
        [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)],
        dtype=torch.float32,
    )
    pose = torch.cat([torch.tensor([0.45, 0.0, 0.18]), quaternion])
    _, _, fields = _mesh_to_obstacle_entry(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        pose,
        representation="cuboid",
    )[0]

    assert fields["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])
    assert fields["pose"][3:] == pytest.approx(quaternion.tolist())


def test_cuboid_entry_accepts_homogeneous_pose():
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([0.45, 0.0, 0.18])
    _, _, fields = _mesh_to_obstacle_entry(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        pose,
        representation="cuboid",
    )[0]

    assert fields["pose"] == pytest.approx([0.45, 0.0, 0.18, 1.0, 0.0, 0.0, 0.0])


def test_mesh_entry_serializes_flat_face_buffer():
    top_key, name, fields = _mesh_to_obstacle_entry(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        representation="mesh",
    )[0]

    assert (top_key, name) == ("mesh", "demo_block")
    assert len(fields["vertices"]) == 8
    assert len(fields["faces"]) == 36
    assert fields["pose"] == pytest.approx(_identity_pose().tolist())


def test_invalid_obstacle_representation_raises():
    with pytest.raises(ValueError, match="representation"):
        _mesh_to_obstacle_entry(
            "block",
            _unit_cube_vertices(),
            _cube_faces(),
            _identity_pose(),
            representation="banana",
        )


def test_empty_mesh_raises_for_cuboid():
    with pytest.raises(ValueError, match="no vertices"):
        _mesh_to_obstacle_entry(
            "block",
            torch.zeros((0, 3), dtype=torch.float32),
            torch.zeros((0, 3), dtype=torch.int32),
            _identity_pose(),
            representation="cuboid",
        )


def test_generate_cuboid_world_yaml_assembles_schema(tmp_path):
    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    output_path = tmp_path / "world.yml"

    result = generate_curobo_world_yaml(
        [rigid_object],
        str(output_path),
        representation="cuboid",
    )
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert result == str(output_path)
    assert list(data) == ["cuboid"]
    assert data["cuboid"]["demo_block"]["dims"] == pytest.approx([1.0, 1.0, 1.0])
    assert data["cuboid"]["demo_block"]["pose"][:3] == pytest.approx([0.45, 0.0, 0.18])


def test_generate_mesh_world_yaml_assembles_schema(tmp_path):
    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    output_path = tmp_path / "world_mesh.yml"

    generate_curobo_world_yaml(
        [rigid_object],
        str(output_path),
        representation="mesh",
    )
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert list(data) == ["mesh"]
    assert len(data["mesh"]["demo_block"]["vertices"]) == 8


def test_generate_world_yaml_supports_multiple_objects(tmp_path):
    rigid_objects = [
        _FakeRigidObject(
            "block_a",
            _unit_cube_vertices(),
            _cube_faces(),
            _identity_pose((0.45, 0.0, 0.18)),
        ),
        _FakeRigidObject(
            "block_b",
            _unit_cube_vertices(),
            _cube_faces(),
            _identity_pose((0.0, 0.3, 0.1)),
        ),
    ]
    output_path = tmp_path / "multi.yml"

    generate_curobo_world_yaml(
        rigid_objects,
        str(output_path),
        representation="cuboid",
    )
    data = yaml.safe_load(output_path.read_text(encoding="utf-8"))

    assert set(data["cuboid"]) == {"block_a", "block_b"}
    assert data["cuboid"]["block_b"]["pose"][:3] == pytest.approx([0.0, 0.3, 0.1])


def test_generate_world_yaml_rejects_empty_input(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        generate_curobo_world_yaml([], str(tmp_path / "world.yml"))


def test_generate_world_yaml_rejects_duplicate_names(tmp_path):
    pose = _identity_pose()
    first = _FakeRigidObject(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        pose,
    )
    second = _FakeRigidObject(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        pose,
    )

    with pytest.raises(ValueError, match="Duplicate"):
        generate_curobo_world_yaml(
            [first, second],
            str(tmp_path / "world.yml"),
        )


def test_generated_cuboid_yaml_loads_in_curobo_scene_cfg(tmp_path):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    output_path = tmp_path / "world.yml"
    generate_curobo_world_yaml(
        [rigid_object],
        str(output_path),
        representation="cuboid",
    )

    scene = SceneCfg.create(yaml.safe_load(output_path.read_text(encoding="utf-8")))

    assert len(scene.cuboid) == 1
    assert scene.cuboid[0].name == "demo_block"
    assert scene.cuboid[0].dims == pytest.approx([1.0, 1.0, 1.0])


def test_generated_mesh_yaml_loads_in_curobo_scene_cfg(tmp_path):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    output_path = tmp_path / "world_mesh.yml"
    generate_curobo_world_yaml(
        [rigid_object],
        str(output_path),
        representation="mesh",
    )

    scene = SceneCfg.create(yaml.safe_load(output_path.read_text(encoding="utf-8")))

    assert len(scene.mesh) == 1
    assert scene.mesh[0].name == "demo_block"
    assert len(scene.mesh[0].vertices) == 8


# Simulator smoke coverage


def _build_curobo_scene(sim_device: str = "cuda") -> tuple[object, object, object]:
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RigidBodyAttributesCfg
    from embodichain.lab.sim.objects import RigidObjectCfg
    from embodichain.lab.sim.robots import FrankaPandaCfg
    from embodichain.lab.sim.shapes import CubeCfg

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            sim_device=sim_device,
            num_envs=1,
            arena_space=2.0,
        )
    )
    robot = sim.add_robot(
        cfg=FrankaPandaCfg.from_dict({"uid": _SIM_ROBOT_UID, "robot_type": "panda"})
    )
    assert robot is not None
    block = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid="block",
            shape=CubeCfg(size=_SIM_BLOCK_DIMS),
            attrs=RigidBodyAttributesCfg(),
            body_type="kinematic",
            init_pos=_SIM_BLOCK_POS,
            init_rot=(0.0, 0.0, 0.0),
        )
    )
    return sim, robot, block


def _target_beyond_block(robot: object) -> torch.Tensor:
    qpos = robot.get_qpos(name=_SIM_CONTROL_PART)
    target = robot.compute_fk(
        qpos=qpos,
        name=_SIM_CONTROL_PART,
        to_matrix=True,
    )[0].clone()
    target[:3, 3] = torch.tensor(
        [0.55, 0.30, 0.45],
        device=robot.device,
    )
    return target


def _make_curobo_engine(
    block: object,
    *,
    use_cuda_graph: bool = False,
) -> object:
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        MoveEndEffector,
        MoveEndEffectorCfg,
    )
    from embodichain.lab.sim.planners import MotionGenCfg, MotionGenerator

    motion_generator = MotionGenerator(
        MotionGenCfg(
            planner_cfg=CuroboPlannerCfg(
                robot_uid=_SIM_ROBOT_UID,
                world=CuroboWorldCfg(rigid_objects=[block]),
                use_cuda_graph=use_cuda_graph,
            )
        )
    )
    engine = AtomicActionEngine(motion_generator)
    engine.register(
        MoveEndEffector(
            motion_generator,
            MoveEndEffectorCfg(
                motion_source="motion_gen",
                control_part=_SIM_CONTROL_PART,
                sample_interval=80,
            ),
        ),
        name="move_end_effector",
    )
    return engine


@pytest.mark.gpu
@pytest.mark.slow
def test_curobo_reuses_non_graph_backend():
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import EndEffectorPoseTarget

    pytest.importorskip("curobo", reason="cuRobo V2 not installed.")
    sim, robot, block = _build_curobo_scene()
    try:
        engine = _make_curobo_engine(block)
        target = _target_beyond_block(robot)

        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        assert bool(success.item()), "first plan failed"
        assert trajectory.shape[0] == 1

        planner = engine.motion_generator.planner
        assert planner.cfg.use_cuda_graph is False
        assert len(planner._backend_cache) == 1

        success, _, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )
        assert bool(success.item()), "second plan failed"
        assert len(planner._backend_cache) == 1
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.slow
def test_curobo_uses_accelerator_with_cpu_physics():
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import EndEffectorPoseTarget

    pytest.importorskip("curobo", reason="cuRobo V2 not installed.")
    sim, robot, block = _build_curobo_scene(sim_device="cpu")
    try:
        engine = _make_curobo_engine(block, use_cuda_graph=True)
        target = _target_beyond_block(robot)

        success, trajectory, _ = engine.run(
            [("move_end_effector", EndEffectorPoseTarget(xpos=target))]
        )

        planner = engine.motion_generator.planner
        backend = next(iter(planner._backend_cache.values()))
        assert robot.device.type == "cpu"
        assert planner._curobo_device.type == "cuda"
        assert backend.use_cuda_graph is True
        assert bool(success.item()), "CPU-physics cuRobo plan failed"
        assert trajectory.device.type == "cpu"
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
