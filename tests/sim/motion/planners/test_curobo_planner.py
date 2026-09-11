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
import math
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from dexsim.types import RigidBodyShape

from embodichain.lab.sim.objects import CollisionShapeDesc
from embodichain.lab.sim.motion.planners import CuroboPlannerCfg, PlanState
from embodichain.lab.sim.motion.planners.curobo import curobo_yaml
from embodichain.lab.sim.motion.planners.curobo.curobo_planner import (
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
from embodichain.lab.sim.motion.planners.curobo.curobo_yaml import (
    _convex_hull_to_voxel_entry,
    _parse_mimic_joint_names,
    _world_collision_sphere_data,
    generate_curobo_robot_yaml,
    generate_curobo_world_scene,
    visualize_curobo_collision_models,
    visualize_curobo_robot_collision_model,
    visualize_curobo_world_collision_model,
)
from embodichain.lab.sim.motion.planners.utils import MoveType

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
    ("batch_size", "max_attempts"),
    [(1, None), (8, 3)],
)
def test_cspace_planning_uses_direct_seed_before_graph_seed(
    monkeypatch, batch_size, max_attempts
):
    """Use the same direct-first c-space policy for scalar and batched plans."""
    calls = []

    class _FakeV2Planner:
        def plan_cspace(self, goal, current, **kwargs):
            del goal, current
            calls.append(kwargs)
            return None

    planner = object.__new__(CuroboPlanner)
    planner.cfg = SimpleNamespace(
        max_attempts=5,
        max_planning_time=None,
        cuda_graph_capture_error_mode="thread_local",
    )
    planner.device = torch.device("cpu")
    planner._curobo_device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        planner,
        "_to_curobo_joint_state",
        lambda _qpos, _backend: object(),
    )
    monkeypatch.setattr(
        planner,
        "_to_curobo_joint_goal",
        lambda _qpos, _backend: object(),
    )
    backend = SimpleNamespace(
        planner=_FakeV2Planner(),
        use_cuda_graph=False,
    )
    start = torch.zeros(batch_size, 7)
    target = PlanState.from_qpos(
        torch.full((batch_size, 7), 0.1),
        move_type=MoveType.JOINT_MOVE,
    )

    result = planner._plan_segments(
        [target],
        start,
        {MoveType.JOINT_MOVE: backend},
        CuroboPlanOptions(max_attempts=max_attempts),
    )

    assert result.success.tolist() == [False] * batch_size
    assert calls == [
        {
            "max_attempts": 5 if max_attempts is None else max_attempts,
            "enable_graph_attempt": 1,
        }
    ]


@pytest.mark.parametrize(
    ("batch_size", "max_attempts"),
    [(1, None), (8, 3)],
)
def test_pose_planning_uses_direct_seed_before_graph_seed(
    monkeypatch, batch_size, max_attempts
):
    """Keep scalar and batched pose planning on the same seed schedule."""
    calls = []

    class _FakeV2Planner:
        def plan_pose(self, goal, current, **kwargs):
            del goal, current
            calls.append(kwargs)
            return None

    planner = object.__new__(CuroboPlanner)
    planner.cfg = SimpleNamespace(
        max_attempts=5,
        max_planning_time=None,
        cuda_graph_capture_error_mode="thread_local",
    )
    planner.device = torch.device("cpu")
    planner._curobo_device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(
        planner,
        "_to_curobo_joint_state",
        lambda _qpos, _backend: object(),
    )
    monkeypatch.setattr(
        planner,
        "_to_curobo_pose_goal",
        lambda _xpos, _backend, _base_inv: object(),
    )
    backend = SimpleNamespace(
        planner=_FakeV2Planner(),
        use_cuda_graph=False,
    )
    start = torch.zeros(batch_size, 7)
    target = PlanState.from_xpos(
        torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).clone(),
        move_type=MoveType.EEF_MOVE,
    )

    result = planner._plan_segments(
        [target],
        start,
        {MoveType.EEF_MOVE: backend},
        CuroboPlanOptions(max_attempts=max_attempts),
    )

    assert result.success.tolist() == [False] * batch_size
    assert calls == [
        {
            "max_attempts": 5 if max_attempts is None else max_attempts,
            "enable_graph_attempt": 1,
        }
    ]


def test_extract_segment_uses_curobo_exclusive_last_tstep(monkeypatch):
    """Do not admit cuRobo padding beyond each trajectory's slice end."""
    planner = object.__new__(CuroboPlanner)
    planner._curobo_device = torch.device("cpu")
    planner.cfg = SimpleNamespace(interpolation_dt=0.025)
    monkeypatch.setattr(
        planner,
        "_map_curobo_to_sim",
        lambda positions, _joint_names, _backend: positions,
    )

    positions = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [math.nan], [math.nan], [math.nan]],
            [[10.0], [11.0], [12.0], [13.0], [14.0], [math.nan]],
        ]
    )
    result = SimpleNamespace(
        success=torch.tensor([[True], [True]]),
        interpolated_last_tstep=torch.tensor([[3], [5]]),
        interpolated_trajectory=SimpleNamespace(
            position=positions,
            dt=torch.tensor([[0.1], [0.2]]),
            joint_names=["joint"],
        ),
    )

    success, extracted, dt = planner._extract_segment(result, SimpleNamespace())

    assert success.tolist() == [True, True]
    assert extracted.shape == (2, 5, 1)
    assert torch.isfinite(extracted).all()
    assert extracted[0, :, 0].tolist() == [0.0, 1.0, 2.0, 2.0, 2.0]
    assert extracted[1, :, 0].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    assert dt[0].tolist() == pytest.approx([0.0, 0.1, 0.1, 0.0, 0.0])
    assert dt[1].tolist() == pytest.approx([0.0, 0.2, 0.2, 0.2, 0.2])


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


def test_curobo_world_cfg_accepts_registered_dynamic_obstacle():
    obstacle = type("NamedObstacle", (), {"uid": "known"})()

    cfg = CuroboWorldCfg(
        rigid_objects=[obstacle],
        dynamic_obstacle_names=["known"],
    )

    assert cfg.dynamic_obstacle_names == ["known"]


def test_curobo_world_cfg_mapping_uses_registry_id_for_dynamic_obstacle():
    obstacle = type("NamedObstacle", (), {"uid": "legacy_uid"})()

    cfg = CuroboWorldCfg(
        rigid_objects={"registry_cube": obstacle},
        dynamic_obstacle_names=["registry_cube"],
    )

    assert cfg.dynamic_obstacle_names == ["registry_cube"]
    assert cfg.rigid_objects["registry_cube"] is obstacle


def test_curobo_world_cfg_mapping_does_not_accept_object_uid_as_alias():
    obstacle = type("NamedObstacle", (), {"uid": "legacy_uid"})()

    with pytest.raises(ValueError, match="not present in rigid_objects"):
        CuroboWorldCfg(
            rigid_objects={"registry_cube": obstacle},
            dynamic_obstacle_names=["legacy_uid"],
        )


def test_curobo_world_cfg_rejects_unregistered_dynamic_obstacle():
    obstacle = type("NamedObstacle", (), {"uid": "known"})()

    with pytest.raises(ValueError, match="not present in rigid_objects"):
        CuroboWorldCfg(
            rigid_objects=[obstacle],
            dynamic_obstacle_names=["unknown"],
        )


def test_curobo_world_cfg_rejects_duplicate_dynamic_obstacle_names():
    obstacle = type("NamedObstacle", (), {"uid": "known"})()

    with pytest.raises(ValueError, match="unique non-empty"):
        CuroboWorldCfg(
            rigid_objects=[obstacle],
            dynamic_obstacle_names=["known", "known"],
        )


def test_curobo_world_cfg_rejects_outer_whitespace_in_obstacle_ids():
    obstacle = type("NamedObstacle", (), {"uid": "known"})()

    with pytest.raises(ValueError, match="without outer whitespace"):
        CuroboWorldCfg(
            rigid_objects={"registry_cube": obstacle},
            dynamic_obstacle_names=[" registry_cube"],
        )
    with pytest.raises(ValueError, match="without outer whitespace"):
        CuroboWorldCfg(rigid_objects={" registry_cube": obstacle})


def test_curobo_world_cfg_rejects_string_dynamic_obstacle_collection():
    obstacle = type("NamedObstacle", (), {"uid": "known"})()

    with pytest.raises(TypeError, match="not a string"):
        CuroboWorldCfg(
            rigid_objects={"registry_cube": obstacle},
            dynamic_obstacle_names="registry_cube",  # type: ignore[arg-type]
        )


def test_curobo_world_cfg_rejects_duplicate_rigid_object_names():
    obstacle_type = type("NamedObstacle", (), {"uid": "duplicate"})

    with pytest.raises(ValueError, match="unique obstacle names"):
        CuroboWorldCfg(rigid_objects=[obstacle_type(), obstacle_type()])


@pytest.mark.parametrize(
    ("multi_env", "expected_mode"),
    [(False, "shared"), (True, "per_env")],
)
def test_curobo_planner_exposes_collision_world_contract(multi_env, expected_mode):
    planner = object.__new__(CuroboPlanner)
    planner.cfg = CuroboPlannerCfg(
        robot_uid="robot",
        world=CuroboWorldCfg(
            rigid_objects={"registry_cube": object()},
            dynamic_obstacle_names=["registry_cube"],
            multi_env=multi_env,
        ),
    )

    info = planner.collision_world_info

    assert info.dynamic_entity_ids == ("registry_cube",)
    assert info.entity_ids == ("registry_cube",)
    assert info.batch_mode == expected_mode
    assert info.supports_updates is True


def test_curobo_collision_world_binding_merges_owned_obstacle_poses():
    planner = object.__new__(CuroboPlanner)
    configured_pose = torch.eye(4).unsqueeze(0)
    observed_pose = torch.eye(4).unsqueeze(0)
    observed_pose[:, 0, 3] = 0.5
    options = CuroboPlanOptions(dynamic_obstacle_poses={"configured": configured_pose})

    bound = planner.with_collision_world(
        options,
        obstacle_poses={"observed": observed_pose},
    )

    assert bound is options
    assert set(bound.dynamic_obstacle_poses) == {"configured", "observed"}
    assert torch.equal(bound.dynamic_obstacle_poses["observed"], observed_pose)
    assert bound.dynamic_obstacle_poses["observed"] is not observed_pose


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


def test_multi_env_scene_model_clones_scene_dictionaries_independently():
    planner = object.__new__(CuroboPlanner)
    scene_model = {
        "voxel": {
            "block": {
                "feature_tensor": torch.ones((2, 2, 2), dtype=torch.float16),
            }
        }
    }

    copies = planner._materialize_multi_env_scene_model(scene_model, batch_size=2)

    assert all(isinstance(scene, dict) for scene in copies)
    assert copies[0] is not copies[1]
    copies[0]["voxel"]["block"]["feature_tensor"].zero_()
    assert torch.all(copies[1]["voxel"]["block"]["feature_tensor"] == 1.0)


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


def _track_convex_hull_preprocessing(monkeypatch, calls=None):
    original_compute_convex_hull = curobo_yaml._compute_convex_hull

    def tracked_compute_convex_hull(mesh):
        if calls is not None:
            calls.append(mesh)
        return original_compute_convex_hull(mesh)

    monkeypatch.setattr(
        curobo_yaml,
        "_compute_convex_hull",
        tracked_compute_convex_hull,
    )


def test_voxel_entry_computes_convex_hull_before_signed_distance(monkeypatch):
    calls = []
    _track_convex_hull_preprocessing(monkeypatch, calls)

    name, fields = _convex_hull_to_voxel_entry(
        "block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        voxel_size=0.25,
        voxel_padding=0.25,
    )

    assert len(calls) == 1
    assert name == "block"
    assert fields["pose"] == pytest.approx(_identity_pose().tolist())
    assert fields["dims"] == pytest.approx([1.5, 1.5, 1.5])
    assert tuple(fields["feature_tensor"].shape) == (6, 6, 6)
    assert fields["feature_tensor"].amin() < 0.0
    assert fields["feature_tensor"].amax() > 0.0


def test_voxel_entry_preserves_homogeneous_object_pose(monkeypatch):
    _track_convex_hull_preprocessing(monkeypatch)
    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([0.45, 0.0, 0.18])

    _, fields = _convex_hull_to_voxel_entry(
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
        _convex_hull_to_voxel_entry(
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
        self.pbr_type = None
        self.pbr_params = {}

    def set_base_color(self, color):
        self.color = color

    def get_inst(self):
        return self

    def update_pbr_material_type(self, material_type):
        self.pbr_type = material_type

    def set_pbr_param(self, name, *values):
        self.pbr_params[name] = values
        if name == "baseColor":
            self.color = list(values)


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

    def create_pbr_material(self, name):
        material = _FakeDexsimMaterial(name, None)
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
    assert actors[0].material.color == [1.0, 0.0, 0.0]


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
    assert robot_actor.material.color == [0.75, 0.75, 1.0]
    obstacle_bounds = obstacle_actor.mesh.get_axis_aligned_bounding_box()
    assert obstacle_bounds.get_center() == pytest.approx([2.0, 2.0, 3.0])
    assert obstacle_bounds.get_extent() == pytest.approx([0.1, 0.1, 0.1])
    assert obstacle_actor.material.name == "curobo_world_collision_material"
    assert obstacle_actor.material.color == [1.0, 0.75, 0.75]
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
    _track_convex_hull_preprocessing(monkeypatch)
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


def test_generate_world_scene_uses_mapping_key_instead_of_object_uid(monkeypatch):
    _track_convex_hull_preprocessing(monkeypatch)
    rigid_object = _FakeRigidObject(
        "legacy_uid",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )

    scene_data = generate_curobo_world_scene(
        {"registry_cube": rigid_object},
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    assert set(scene_data["voxel"]) == {"registry_cube"}


@pytest.mark.parametrize(
    "shape_type",
    [RigidBodyShape.MESH, RigidBodyShape.CONVEX, RigidBodyShape.SDF],
)
def test_auto_world_scene_voxelizes_every_mesh_backed_shape(monkeypatch, shape_type):
    _track_convex_hull_preprocessing(monkeypatch)
    mesh_shape = CollisionShapeDesc(
        name="mesh_backed_shape",
        shape_type=shape_type,
        local_pose=torch.eye(4),
        vertices=_unit_cube_vertices(),
        triangles=_cube_faces(),
    )
    rigid_object = _FakeRigidObject(
        "mesh_obstacle",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
        [mesh_shape],
    )

    scene_data = generate_curobo_world_scene(
        [rigid_object],
        voxel_size=0.5,
        voxel_padding=0.0,
    )

    assert list(scene_data) == ["voxel"]


def test_mesh_backed_world_scene_rejects_oversized_voxel_grid():
    rigid_object = _FakeRigidObject(
        "mesh_obstacle",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )

    with pytest.raises(ValueError, match="max_voxel_count"):
        generate_curobo_world_scene(
            [rigid_object],
            voxel_size=0.1,
            voxel_padding=0.0,
            max_voxel_count=8,
        )


def test_world_scene_cache_key_includes_registry_id():
    rigid_object = _FakeRigidObject(
        "legacy_uid",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    planner = object.__new__(CuroboPlanner)
    planner.cfg = CuroboPlannerCfg(
        robot_uid="robot",
        world=CuroboWorldCfg(rigid_objects={"registry_cube": rigid_object}),
    )
    registry_key = planner._world_scene_cache_key(planner.cfg.world)
    planner.cfg.world = CuroboWorldCfg(
        rigid_objects={"renamed_registry_cube": rigid_object}
    )

    renamed_key = planner._world_scene_cache_key(planner.cfg.world)

    assert registry_key != renamed_key


def test_dynamic_update_uses_registry_id_in_curobo_backend():
    rigid_object = _FakeRigidObject(
        "legacy_uid",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    planner = object.__new__(CuroboPlanner)
    planner.cfg = CuroboPlannerCfg(
        robot_uid="robot",
        world=CuroboWorldCfg(
            rigid_objects={"registry_cube": rigid_object},
            dynamic_obstacle_names=["registry_cube"],
        ),
    )
    planner._curobo_device = torch.device("cpu")
    planner._bindings = SimpleNamespace(Pose=lambda **kwargs: kwargs)
    updates = []
    collision_checker = SimpleNamespace(
        update_obstacle_pose=lambda name, pose, env_idx: updates.append(
            (name, pose, env_idx)
        )
    )
    backend = SimpleNamespace(
        batch_size=1,
        profile=SimpleNamespace(sim_base_to_curobo_base=None),
        sim_base_to_curobo_base_matrix=None,
        planner=SimpleNamespace(scene_collision_checker=collision_checker),
    )
    identity = torch.eye(4).unsqueeze(0)

    planner.update_dynamic_obstacles(
        {"registry_cube": identity},
        backend=backend,
        sim_base_pose_inv=identity,
    )

    assert [(name, env_idx) for name, _, env_idx in updates] == [("registry_cube", 0)]


def test_validate_joint_trajectory_checks_every_exact_sample_in_curobo_order():
    """The collision gate preserves samples and maps simulator joint order."""
    planner = object.__new__(CuroboPlanner)
    planner.cfg = CuroboPlannerCfg(
        robot_uid="robot",
        world=CuroboWorldCfg(multi_env=True),
    )
    planner._curobo_device = torch.device("cpu")
    joint_states = []
    collision_queries = []

    def from_position(position, *, joint_names):
        joint_states.append((position.clone(), tuple(joint_names)))
        return SimpleNamespace(position=position)

    def validate(sample, *, env_query_idx):
        collision_queries.append((sample.clone(), env_query_idx.clone()))
        validity = torch.ones(sample.shape[:2], dtype=torch.bool)
        if len(collision_queries) == 2:
            validity[1, 0] = False
        return validity

    planner._bindings = SimpleNamespace(
        JointState=SimpleNamespace(from_position=from_position),
    )
    backend = SimpleNamespace(
        sim_joint_names=["sim_left", "sim_right"],
        sim_to_curobo_col_idx=None,
        collision_checker=SimpleNamespace(validate=validate),
        profile=SimpleNamespace(
            sim_to_curobo_joint_names={
                "sim_left": "curobo_left",
                "sim_right": "curobo_right",
            },
        ),
        planner=SimpleNamespace(
            joint_names=["curobo_right", "curobo_left"],
        ),
    )
    planner._get_backend = lambda control_part, batch_size, move_type: backend
    trajectory = torch.tensor(
        (
            ((0.0, 1.0), (0.1, 1.1), (0.2, 1.2)),
            ((2.0, 3.0), (2.1, 3.1), (2.2, 3.2)),
        ),
        dtype=torch.float32,
    )

    validity = planner.validate_joint_trajectory(
        trajectory,
        control_part="dual_arm",
    )

    assert torch.equal(
        validity,
        torch.tensor(((True, True, True), (True, False, True))),
    )
    assert len(collision_queries) == trajectory.shape[1]
    for sample_index, (sample, env_query_idx) in enumerate(collision_queries):
        torch.testing.assert_close(
            sample[:, 0],
            trajectory[:, sample_index].flip(dims=(-1,)),
        )
        assert torch.equal(env_query_idx, torch.tensor((0, 1), dtype=torch.int32))
    torch.testing.assert_close(joint_states[0][0], trajectory[:, 0].flip(dims=(-1,)))
    assert joint_states[0][1] == ("curobo_right", "curobo_left")


def test_generate_world_scene_rejects_direct_mesh_representation():
    rigid_object = _FakeRigidObject(
        "demo_block",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )
    with pytest.raises(ValueError, match="representation policies"):
        generate_curobo_world_scene(
            [rigid_object],
            representation="mesh",
        )


def test_generate_world_scene_supports_multiple_objects(monkeypatch):
    _track_convex_hull_preprocessing(monkeypatch)
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


def test_registry_world_scene_rejects_missing_physical_shapes():
    rigid_object = _FakeRigidObject(
        "legacy_uid",
        torch.zeros((0, 3), dtype=torch.float32),
        torch.zeros((0, 3), dtype=torch.int64),
        _identity_pose(),
    )
    rigid_object._collision_shapes = []

    with pytest.raises(ValueError, match="Registry-backed obstacle.*no physical"):
        generate_curobo_world_scene(
            {"registry_cube": rigid_object},
        )


def test_generate_world_scene_rejects_duplicate_names(monkeypatch):
    _track_convex_hull_preprocessing(monkeypatch)
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


def test_generate_world_scene_rejects_outer_whitespace_in_mapping_id():
    rigid_object = _FakeRigidObject(
        "legacy_uid",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )

    with pytest.raises(ValueError, match="without outer whitespace"):
        generate_curobo_world_scene(
            {" registry_cube": rigid_object},
        )


def test_generated_voxel_data_loads_in_curobo_scene_cfg(monkeypatch):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    _track_convex_hull_preprocessing(monkeypatch)

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


def test_generated_physical_mesh_loads_as_voxel_in_curobo_scene_cfg(monkeypatch):
    pytest.importorskip("curobo")
    from curobo._src.geom.types import SceneCfg

    _track_convex_hull_preprocessing(monkeypatch)
    rigid_object = _FakeRigidObject(
        "collision_mesh",
        _unit_cube_vertices(),
        _cube_faces(),
        _identity_pose(),
    )

    scene = SceneCfg.create(
        generate_curobo_world_scene(
            [rigid_object],
            voxel_size=0.5,
            voxel_padding=0.0,
        )
    )

    assert not scene.mesh
    assert len(scene.voxel) == 1
    assert scene.voxel[0].name == "collision_mesh"


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
    )
    from embodichain.lab.sim.motion.motion_generator import (
        MotionGenCfg,
        MotionGenerator,
    )

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
    return engine


@pytest.mark.gpu
@pytest.mark.slow
def test_curobo_reuses_non_graph_backend():
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        EndEffectorPoseGoal,
        MotionPolicy,
    )

    pytest.importorskip("curobo", reason="cuRobo V2 not installed.")
    sim, robot, block = _build_curobo_scene()
    try:
        engine = _make_curobo_engine(block)
        target = _target_beyond_block(robot)
        binding = engine.bind_control_parts(
            "move_end_effector",
            {"primary": {"motion": _SIM_CONTROL_PART}},
        )

        result = engine.compile(
            (
                ActionInvocation(
                    "move_end_effector",
                    EndEffectorPoseGoal(xpos=target),
                    binding,
                    MotionPolicy(strategy="motion_gen", sample_count=80),
                ),
            )
        )
        success = result.plan_success
        trajectory = result.trajectory.positions
        assert bool(success.item()), "first plan failed"
        assert trajectory.shape[0] == 1

        planner = engine.motion_generator.planner
        assert planner.cfg.use_cuda_graph is False
        assert len(planner._backend_cache) == 1

        result = engine.compile(
            (
                ActionInvocation(
                    "move_end_effector",
                    EndEffectorPoseGoal(xpos=target),
                    binding,
                    MotionPolicy(strategy="motion_gen", sample_count=80),
                ),
            )
        )
        success = result.plan_success
        assert bool(success.item()), "second plan failed"
        assert len(planner._backend_cache) == 1
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.gpu
@pytest.mark.slow
def test_curobo_uses_accelerator_with_cpu_physics():
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.sim.atomic_actions import (
        ActionInvocation,
        EndEffectorPoseGoal,
        MotionPolicy,
    )

    pytest.importorskip("curobo", reason="cuRobo V2 not installed.")
    sim, robot, block = _build_curobo_scene(sim_device="cpu")
    try:
        engine = _make_curobo_engine(block, use_cuda_graph=True)
        target = _target_beyond_block(robot)
        binding = engine.bind_control_parts(
            "move_end_effector",
            {"primary": {"motion": _SIM_CONTROL_PART}},
        )

        result = engine.compile(
            (
                ActionInvocation(
                    "move_end_effector",
                    EndEffectorPoseGoal(xpos=target),
                    binding,
                    MotionPolicy(strategy="motion_gen", sample_count=80),
                ),
            )
        )
        success = result.plan_success
        trajectory = result.trajectory.positions

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
