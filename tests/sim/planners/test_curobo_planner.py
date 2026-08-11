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
validation, generated robot YAML, and mixed collision-world data. The GPU-marked smoke tests
exercise cached in-process planning and CPU-physics interoperability. Full
collision-planning coverage remains in ``test_curobo_integration.py``.
"""

from __future__ import annotations

import importlib
import logging
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from dexsim.types import RigidBodyShape

from embodichain.lab.sim.objects import CollisionShapeDesc
from embodichain.lab.sim.planners import CuroboPlannerCfg
from embodichain.lab.sim.planners.curobo.curobo_planner import (
    CuroboPlanOptions,
    CuroboPlanner,
    CuroboPlannerCfg as CuroboPlannerCfgDirect,
    CuroboWorldCfg,
    _CuroboProfile,
    _configure_curobo_logging,
    _matrix_to_position_quaternion,
    _require_curobo,
    _resolve_curobo_device,
    _torch_cuda_graph_capture_mode,
    _validate_dynamic_obstacles,
)
from embodichain.lab.sim.planners.curobo.curobo_yaml import (
    _convex_hulls_to_voxel_entry,
    _parse_mimic_joint_names,
    _world_collision_sphere_data,
    generate_curobo_robot_yaml,
    generate_curobo_world_scene,
    visualize_curobo_collision_models,
    visualize_curobo_robot_collision_model,
    visualize_curobo_world_collision_model,
)
from embodichain.lab.sim.planners.utils import MoveType

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


@pytest.fixture(scope="module", autouse=True)
def _restore_torch_precision_settings():
    """Keep cuRobo's process-wide TF32 changes local to this test module."""
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()

    yield

    torch.set_float32_matmul_precision(matmul_precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32


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
    assert cfg.log_level == "error"
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


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("debug", logging.DEBUG),
        ("INFO", logging.INFO),
        ("warn", logging.WARNING),
        ("warning", logging.WARNING),
        ("error", logging.ERROR),
    ],
)
def test_configure_curobo_logging_sets_package_logger(
    monkeypatch, configured, expected
):
    configured_levels = []
    curobo_logger = logging.getLogger("curobo")
    monkeypatch.setattr(curobo_logger, "setLevel", configured_levels.append)

    _configure_curobo_logging(configured)

    assert configured_levels == [expected]


def test_configure_curobo_logging_rejects_unknown_level():
    with pytest.raises(ValueError, match="CuroboPlannerCfg.log_level"):
        _configure_curobo_logging("silent")


def test_curobo_world_cfg_defaults_to_auto_collision_policy():
    cfg = CuroboWorldCfg()

    assert cfg.representation == "auto"
    assert cfg.overrides == {}
    assert cfg.voxel_size == pytest.approx(0.01)
    assert cfg.voxel_padding == pytest.approx(0.005)


def test_auto_gen_defaults_keep_sphere_count_low_and_fit_type_fixed():
    """MorphIt is fixed by the generator while density remains configurable."""
    auto = CuroboPlannerCfg(robot_uid="franka").auto_gen
    assert auto.sphere_density == 0.1
    assert not hasattr(auto, "fit_type")


def test_curobo_planner_class_is_lazy_import_safe():
    """Referencing the class must not import curobo."""
    import sys

    sys.modules.pop("curobo", None)
    assert CuroboPlanner.__name__ == "CuroboPlanner"
    assert "curobo" not in sys.modules


def test_backend_disables_curobo_self_collision(monkeypatch):
    create_kwargs = {}

    class FakeMotionPlannerCfg:
        @staticmethod
        def create(**kwargs):
            create_kwargs.update(kwargs)
            return SimpleNamespace(
                trajopt_solver_config=SimpleNamespace(interpolation_dt=None)
            )

    class FakeMotionPlanner:
        joint_names = ["joint"]

        def __init__(self, cfg):
            self.cfg = cfg

    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.cfg = SimpleNamespace(
        world=SimpleNamespace(multi_env=False),
        collision_activation_distance=0.01,
        interpolation_dt=0.025,
    )
    planner._curobo_device = torch.device("cuda:0")
    planner._bindings = SimpleNamespace(
        MotionPlannerCfg=FakeMotionPlannerCfg,
        DeviceCfg=lambda device: device,
        MotionPlanner=FakeMotionPlanner,
        BatchMotionPlanner=FakeMotionPlanner,
    )
    planner._validate_profile_joint_names = lambda *args: None
    planner._validate_base_link_name = lambda *args: None
    planner._resolve_tool_frame = lambda *args: "tool"
    planner._load_runtime_robot_config = lambda path: {
        "robot_cfg": {
            "kinematics": {
                "source": path,
                "self_collision_buffer": {},
                "self_collision_ignore": {},
            }
        }
    }
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    planner._build_backend(
        control_part="arm",
        batch_size=1,
        profile=_CuroboProfile(
            robot_config_path="robot.yml",
            sim_to_curobo_joint_names={"joint": "joint"},
        ),
        sim_joint_names=["joint"],
        scene_model=None,
        use_cuda_graph=False,
        planning_mode=MoveType.EEF_MOVE,
    )

    assert create_kwargs["self_collision_check"] is False
    assert create_kwargs["robot"]["robot_cfg"]["kinematics"] == {
        "source": "robot.yml",
        "self_collision_buffer": {},
        "self_collision_ignore": {},
    }


def test_runtime_robot_config_adds_only_curobo_compatibility_placeholders(tmp_path):
    config_path = tmp_path / "robot.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "base_link": "base",
                        "collision_spheres": {
                            "base": [{"center": [0.0, 0.0, 0.0], "radius": 0.1}]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    runtime_config = CuroboPlanner._load_runtime_robot_config(str(config_path))
    kinematics = runtime_config["robot_cfg"]["kinematics"]

    assert kinematics["self_collision_buffer"] == {}
    assert kinematics["self_collision_ignore"] == {}
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "self_collision_buffer" not in persisted["robot_cfg"]["kinematics"]
    assert "self_collision_ignore" not in persisted["robot_cfg"]["kinematics"]


def test_disable_self_collision_reaches_all_curobo_rollouts():
    class FakeCostCfg:
        def __init__(self):
            self.disable_calls = 0

        def disable_self_collision(self):
            self.disable_calls += 1

    class FakeRollout:
        def __init__(self):
            self.cost_cfg = FakeCostCfg()

        def get_cost_manager_configs(self):
            return [self.cost_cfg]

    ik_metrics = FakeRollout()
    ik_optimizer = FakeRollout()
    trajopt_metrics = FakeRollout()
    trajopt_optimizer = FakeRollout()
    graph_rollout = FakeRollout()
    planner_cfg = SimpleNamespace(
        ik_solver_config=SimpleNamespace(
            core_cfg=SimpleNamespace(
                metrics_rollout_config=ik_metrics,
                optimizer_rollout_configs=[ik_optimizer],
            )
        ),
        trajopt_solver_config=SimpleNamespace(
            core_cfg=SimpleNamespace(
                metrics_rollout_config=trajopt_metrics,
                optimizer_rollout_configs=[trajopt_optimizer],
            )
        ),
        graph_planner_config=SimpleNamespace(rollout_config=graph_rollout),
    )

    CuroboPlanner._disable_curobo_self_collision_rollouts(planner_cfg)

    assert all(
        rollout.cost_cfg.disable_calls == 1
        for rollout in (
            ik_metrics,
            ik_optimizer,
            trajopt_metrics,
            trajopt_optimizer,
            graph_rollout,
        )
    )


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


def test_robot_spheres_use_dexsim_morphit_with_two_hulls(tmp_path, monkeypatch):
    pytest.importorskip("curobo")
    import dexsim.kit.meshproc as meshproc

    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text(
        '<?xml version="1.0"?><robot name="test"><link name="base"/></robot>',
        encoding="utf-8",
    )

    class FakeRobot:
        cfg = type(
            "Cfg",
            (),
            {"fpath": str(urdf_path), "init_qpos": [], "base_link_name": "base"},
        )()
        joint_names = []
        control_parts = {"arm": []}

        def get_link_names(self):
            return ["base"]

        def get_link_vert_face(self, link_name):  # noqa: ARG002
            return _unit_cube_vertices(), _cube_faces()

        def get_control_part_link_names(self, control_part):  # noqa: ARG002
            return ["base"]

    calls = []

    def fake_sphere_fit(mesh, **kwargs):
        calls.append((mesh, kwargs))
        return (
            True,
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([0.25], dtype=torch.float32),
        )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(meshproc, "sphere_fit", fake_sphere_fit)
    output_path = tmp_path / "robot.yml"

    generate_curobo_robot_yaml(
        FakeRobot(),
        "arm",
        str(output_path),
        urdf_path=str(urdf_path),
        device="cuda:0",
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["fit_type"] is meshproc.SphereFitType.MORPHIT
    assert kwargs["max_convex_hull_num"] == 2
    kinematics = yaml.safe_load(output_path.read_text(encoding="utf-8"))["robot_cfg"][
        "kinematics"
    ]
    assert kinematics["collision_spheres"]["base"][0]["radius"] == pytest.approx(0.25)
    assert "self_collision_buffer" not in kinematics
    assert "self_collision_ignore" not in kinematics


def test_robot_collision_visualization_reads_cache_and_live_link_pose(tmp_path):
    robot_yaml_path = tmp_path / "robot_visual.yml"
    robot_yaml_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "collision_sphere_buffer": 0.0,
                        "collision_spheres": {
                            "base": [{"center": [0.0, 0.0, 0.0], "radius": 0.1}]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeRobot:
        def get_link_vert_face(self, link_name):  # noqa: ARG002
            return _unit_cube_vertices(), _cube_faces()

        def get_link_pose(
            self, link_name, env_ids=None, to_matrix=False  # noqa: ARG002
        ):
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
            return pose.unsqueeze(0)

    geometries = visualize_curobo_robot_collision_model(
        FakeRobot(), str(robot_yaml_path), draw=False
    )

    assert [geometry["name"] for geometry in geometries] == [
        "robot_mesh/base",
        "robot_spheres",
    ]
    sphere_bounds = geometries[-1]["geometry"].get_axis_aligned_bounding_box()
    assert sphere_bounds.get_center() == pytest.approx([1.0, 2.0, 3.0])


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
    """Expose the physical-shape and pose API required by the world generator."""

    def __init__(
        self,
        uid: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        pose: torch.Tensor,
        collision_shapes: list[CollisionShapeDesc] | None = None,
    ) -> None:
        self.uid = uid
        self._vertices = vertices
        self._faces = faces
        self._pose = pose
        self._collision_shapes = collision_shapes or [
            CollisionShapeDesc(
                name="shape_0",
                shape_type=RigidBodyShape.MESH,
                local_pose=torch.eye(4),
                vertices=vertices,
                triangles=faces,
            )
        ]

    def get_vertices(self, env_ids=None, scale=False):  # noqa: ARG002
        return self._vertices.unsqueeze(0)

    def get_triangles(self, env_ids=None):  # noqa: ARG002
        return self._faces.unsqueeze(0)

    def get_local_pose(self, to_matrix=False):
        if to_matrix:
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, 3] = self._pose[:3]
            return pose.unsqueeze(0)
        return self._pose.unsqueeze(0)

    def get_collision_shapes(self, env_id=0):  # noqa: ARG002
        return self._collision_shapes


def _mock_visacd_as_identity(monkeypatch, calls=None):
    import dexsim.kit.meshproc as meshproc

    def fake_visacd(mesh, **kwargs):
        if calls is not None:
            calls.append((mesh, kwargs))
        return True, (mesh,)

    monkeypatch.setattr(meshproc, "convex_decomposition_visacd", fake_visacd)


def test_voxel_entry_uses_visacd_with_sixteen_hulls(monkeypatch):
    calls = []
    _mock_visacd_as_identity(monkeypatch, calls)

    name, fields = _convex_hulls_to_voxel_entry(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        voxel_size=0.25,
        voxel_padding=0.25,
    )

    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["max_convex_hull_num"] == 16
    assert name == "block"
    assert fields["pose"] == pytest.approx(_identity_pose().tolist())
    assert fields["dims"] == pytest.approx([1.5, 1.5, 1.5])
    assert tuple(fields["feature_tensor"].shape) == (6, 6, 6)
    assert fields["feature_tensor"].amin() < 0.0
    assert fields["feature_tensor"].amax() > 0.0


def test_voxel_entry_preserves_homogeneous_object_pose(monkeypatch):
    _mock_visacd_as_identity(monkeypatch)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([0.45, 0.0, 0.18])

    _, fields = _convex_hulls_to_voxel_entry(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        pose,
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    assert fields["pose"] == pytest.approx(_identity_pose().tolist())


@pytest.mark.parametrize(
    ("voxel_size", "voxel_padding", "match"),
    [(0.0, 0.1, "voxel_size"), (0.1, -0.1, "voxel_padding")],
)
def test_voxel_entry_rejects_invalid_settings(voxel_size, voxel_padding, match):
    with pytest.raises(ValueError, match=match):
        _convex_hulls_to_voxel_entry(
            "block",
            _unit_cube_vertices(),
            _cube_faces(),
            _identity_pose(),
            voxel_size=voxel_size,
            voxel_padding=voxel_padding,
        )


class _FakeDexsimMaterial:
    def __init__(self, name, color):
        self.name = name
        self.color = color

    def set_base_color(self, color):
        self.color = color


class _FakeDexsimActor:
    def __init__(self, mesh):
        self.mesh = mesh
        self.material = None

    def set_material(self, material):
        self.material = material


class _FakeDexsimEnv:
    def __init__(self):
        self.materials = {}
        self.actors = []
        self.loaded_paths = []
        self.removed_actors = []

    def find_material(self, name):
        return self.materials.get(name)

    def create_color_material(self, color, name, has_alpha=False):  # noqa: ARG002
        material = _FakeDexsimMaterial(name, color)
        self.materials[name] = material
        return material

    def load_actor(self, mesh_path):
        import open3d as o3d

        self.loaded_paths.append(mesh_path)
        actor = _FakeDexsimActor(o3d.io.read_triangle_mesh(mesh_path))
        self.actors.append(actor)
        return actor

    def remove_actor(self, actor):
        self.removed_actors.append(actor)


def test_obstacle_collision_visualization_loads_one_combined_dexsim_actor():
    rigid_object = _FakeRigidObject(
        "block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    env = _FakeDexsimEnv()

    features = torch.ones((3, 3, 3), dtype=torch.float16)
    features[1, 1, 1] = 0.0
    world_scene = {
        "voxel": {
            "block": {
                "pose": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                "dims": [0.3, 0.3, 0.3],
                "voxel_size": 0.1,
                "feature_tensor": features,
            }
        }
    }
    actors = visualize_curobo_world_collision_model(
        [rigid_object], world_scene, env=env
    )

    assert actors == env.actors
    assert len(env.loaded_paths) == 1
    assert not Path(env.loaded_paths[0]).exists()
    bounds = actors[0].mesh.get_axis_aligned_bounding_box()
    assert bounds.get_center() == pytest.approx([1.0, 2.0, 3.0])
    assert bounds.get_extent() == pytest.approx([0.1, 0.1, 0.1])
    assert actors[0].material.name == "curobo_world_collision_material"
    assert actors[0].material.color == [1.0, 0.0, 0.0, 0.45]


def test_combined_collision_visualization_colors_and_cleans_two_actors(
    tmp_path, monkeypatch
):
    import dexsim

    robot_yaml_path = tmp_path / "robot_visual.yml"
    robot_yaml_path.write_text(
        yaml.safe_dump(
            {
                "robot_cfg": {
                    "kinematics": {
                        "collision_sphere_buffer": 0.01,
                        "collision_spheres": {
                            "hand": [{"center": [0.1, 0.0, 0.0], "radius": 0.1}]
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeRobot:
        def get_link_pose(
            self, link_name, env_ids=None, to_matrix=False  # noqa: ARG002
        ):
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
            return pose.unsqueeze(0)

    env = _FakeDexsimEnv()
    world = SimpleNamespace(get_env=lambda: env)
    prompts = []
    monkeypatch.setattr(dexsim, "default_world", lambda: world)
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or "")

    features = torch.ones((3, 3, 3), dtype=torch.float16)
    features[1, 1, 1] = 0.0
    world_scene = {
        "voxel": {
            "block": {
                "pose": [2.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                "voxel_size": 0.1,
                "feature_tensor": features,
            }
        }
    }
    rigid_object = _FakeRigidObject(
        "block", _unit_cube_vertices(), _cube_faces(), _identity_pose()
    )
    visualize_curobo_collision_models(
        FakeRobot(), str(robot_yaml_path), [rigid_object], world_scene
    )

    assert len(env.actors) == 2
    robot_actor, obstacle_actor = env.actors
    robot_bounds = robot_actor.mesh.get_axis_aligned_bounding_box()
    assert robot_bounds.get_center() == pytest.approx([1.1, 2.0, 3.0])
    assert robot_bounds.get_extent() == pytest.approx([0.22, 0.22, 0.22])
    assert robot_actor.material.name == "curobo_robot_collision_material"
    assert robot_actor.material.color == [0.0, 0.0, 1.0, 0.45]
    obstacle_bounds = obstacle_actor.mesh.get_axis_aligned_bounding_box()
    assert obstacle_bounds.get_center() == pytest.approx([2.0, 2.0, 3.0])
    assert obstacle_bounds.get_extent() == pytest.approx([0.1, 0.1, 0.1])
    assert obstacle_actor.material.name == "curobo_world_collision_material"
    assert obstacle_actor.material.color == [1.0, 0.0, 0.0, 0.45]
    assert env.removed_actors == list(reversed(env.actors))
    assert all(not Path(path).exists() for path in env.loaded_paths)
    assert "Showing 2 cuRobo collision spheres" in prompts[0]


def test_auto_world_scene_preserves_physical_box_as_cuboid():
    box = CollisionShapeDesc(
        name="physics_box",
        shape_type=RigidBodyShape.BOX,
        local_pose=torch.eye(4),
        half_extents=torch.tensor([0.1, 0.2, 0.3]),
    )
    rigid_object = _FakeRigidObject(
        "fixture",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose((1.0, 2.0, 3.0)),
        [box],
    )

    scene_data = generate_curobo_world_scene([rigid_object])

    assert list(scene_data) == ["cuboid"]
    assert scene_data["cuboid"]["fixture"]["dims"] == pytest.approx([0.2, 0.4, 0.6])
    assert scene_data["cuboid"]["fixture"]["pose"][:3] == pytest.approx([1.0, 2.0, 3.0])


def test_mixed_collision_visualization_supports_cuboid():
    centers, radii = _world_collision_sphere_data(
        {
            "cuboid": {
                "fixture": {
                    "pose": [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    "dims": [0.2, 0.4, 0.6],
                }
            }
        }
    )

    assert centers.shape == (8, 3)
    assert radii.shape == (8,)


def test_world_scene_object_override_can_force_voxel(monkeypatch):
    _mock_visacd_as_identity(monkeypatch)
    box = CollisionShapeDesc(
        name="physics_box",
        shape_type=RigidBodyShape.BOX,
        local_pose=torch.eye(4),
        half_extents=torch.tensor([0.5, 0.5, 0.5]),
    )
    rigid_object = _FakeRigidObject(
        "room_scan",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        [box],
    )

    scene_data = generate_curobo_world_scene(
        [rigid_object],
        overrides={"room_scan": "voxel"},
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    assert list(scene_data) == ["voxel"]
    assert set(scene_data["voxel"]) == {"room_scan"}


def test_auto_world_scene_preserves_compound_shape_names_and_local_poses():
    box_pose = torch.eye(4)
    box_pose[0, 3] = 0.25
    shapes = [
        CollisionShapeDesc(
            name="box",
            shape_type=RigidBodyShape.BOX,
            local_pose=box_pose,
            half_extents=torch.tensor([0.1, 0.1, 0.1]),
        ),
        CollisionShapeDesc(
            name="sphere",
            shape_type=RigidBodyShape.SPHERE,
            local_pose=torch.eye(4),
            radius=0.15,
        ),
    ]
    rigid_object = _FakeRigidObject(
        "compound",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose((1.0, 0.0, 0.0)),
        shapes,
    )

    scene_data = generate_curobo_world_scene([rigid_object])

    assert set(scene_data) == {"cuboid", "sphere"}
    assert set(scene_data["cuboid"]) == {"compound__shape_0"}
    assert set(scene_data["sphere"]) == {"compound__shape_1"}
    assert scene_data["cuboid"]["compound__shape_0"]["pose"][:3] == pytest.approx(
        [1.25, 0.0, 0.0]
    )


def test_dynamic_compound_object_fans_out_to_shape_local_poses():
    first_pose = torch.eye(4)
    first_pose[0, 3] = 0.25
    shapes = [
        CollisionShapeDesc(
            name="first",
            shape_type=RigidBodyShape.BOX,
            local_pose=first_pose,
            half_extents=torch.ones(3),
        ),
        CollisionShapeDesc(
            name="second",
            shape_type=RigidBodyShape.SPHERE,
            local_pose=torch.eye(4),
            radius=0.1,
        ),
    ]
    rigid_object = _FakeRigidObject(
        "compound",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        shapes,
    )
    planner = CuroboPlanner.__new__(CuroboPlanner)
    planner.cfg = SimpleNamespace(
        world=CuroboWorldCfg(
            rigid_objects=[rigid_object], dynamic_obstacle_names=["compound"]
        )
    )

    obstacle_shapes = planner._dynamic_obstacle_shapes("compound")

    assert [name for name, _ in obstacle_shapes] == [
        "compound__shape_0",
        "compound__shape_1",
    ]
    assert obstacle_shapes[0][1][:3, 3].tolist() == pytest.approx([0.25, 0.0, 0.0])


def test_generate_world_scene_supports_multiple_objects(monkeypatch):
    _mock_visacd_as_identity(monkeypatch)
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
    scene_data = generate_curobo_world_scene(
        rigid_objects,
        representation="voxel",
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    assert list(scene_data) == ["voxel"]
    assert set(scene_data["voxel"]) == {"block_a", "block_b"}
    assert scene_data["voxel"]["block_b"]["pose"][:3] == pytest.approx([0.0, 0.3, 0.1])


def test_generate_world_scene_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        generate_curobo_world_scene([])


def test_generate_world_scene_rejects_duplicate_names(monkeypatch):
    _mock_visacd_as_identity(monkeypatch)
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
        generate_curobo_world_scene([first, second], voxel_size=0.5)


def test_generated_voxel_data_loads_in_curobo_scene_cfg(monkeypatch):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    _mock_visacd_as_identity(monkeypatch)

    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    scene_data = generate_curobo_world_scene(
        [rigid_object],
        representation="voxel",
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    scene = SceneCfg.create(scene_data)

    assert len(scene.voxel) == 1
    assert scene.voxel[0].name == "demo_block"
    assert scene.voxel[0].voxel_size == pytest.approx(0.5)
    assert tuple(scene.voxel[0].feature_tensor.shape) == (2, 2, 2)


def test_generated_physical_mesh_loads_in_curobo_scene_cfg():
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    rigid_object = _FakeRigidObject(
        "collision_mesh",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )

    scene = SceneCfg.create(generate_curobo_world_scene([rigid_object]))

    assert len(scene.mesh) == 1
    assert scene.mesh[0].name == "collision_mesh"
    assert len(scene.mesh[0].vertices) == _unit_cube_vertices().shape[0]


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
