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

from __future__ import annotations

from types import SimpleNamespace

import gymnasium
import pytest
import torch

from embodichain.gen_sim.action_agent_pipeline.contracts import (
    ACTION_AGENT_ENV_ID,
    DEFAULT_VIEWER_CAMERA_UID,
    LEFT_ARM_ACTION_KEY,
    LEFT_ARM_NAME,
    RIGHT_ARM_ACTION_KEY,
)
from embodichain.gen_sim.action_agent_pipeline.defaults import (
    DEFAULT_MAX_EPISODE_STEPS,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.agent_env import (
    AgenticGenSimEnv,
)
from embodichain.gen_sim.action_agent_pipeline.env_adapters.tableware.success import (
    _FALLBACKS,
    evaluate_configured_success,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_templates import (
    make_sensor_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.nominal_graph import (
    NominalGraphStep,
    build_nominal_task_graph,
)
from embodichain.gen_sim.action_agent_pipeline.runtime.graph_compiler import (
    compile_agent_graph_spec,
)


class _RigidObject:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self._pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
        self._pose[0, :3, 3] = torch.tensor(position)

    def get_local_pose(self, *, to_matrix: bool) -> torch.Tensor:
        assert to_matrix is True
        return self._pose


class _Simulation:
    def __init__(self, objects: dict[str, _RigidObject]) -> None:
        self._objects = objects

    def get_rigid_object(self, uid: str) -> _RigidObject | None:
        return self._objects.get(uid)


class _CompiledGraph:
    def __init__(self, start: str, goal: str, max_transitions: int) -> None:
        self.start = start
        self.goal = goal
        self.max_transitions = max_transitions
        self.nodes = []
        self.edges = []

    def add_node(self, node_id: str, semantic: str) -> None:
        self.nodes.append((node_id, semantic))

    def add_edge(self, edge_id: str, source: str, target: str, **actions) -> None:
        self.edges.append((edge_id, source, target, actions))


class _EventManager:
    def get_functor(self, name: str):
        assert name == "validation_cameras"
        return lambda env, env_ids: {
            "validation_rgb": torch.zeros((1, 2, 2, 3), dtype=torch.uint8)
        }


def test_registered_episode_limit_matches_packaged_default() -> None:
    assert DEFAULT_MAX_EPISODE_STEPS == 2000
    assert gymnasium.spec(ACTION_AGENT_ENV_ID).max_episode_steps == 2000


def test_default_sensor_uid_comes_from_runtime_contract() -> None:
    sensors = make_sensor_config()

    assert sensors[0]["uid"] == DEFAULT_VIEWER_CAMERA_UID


def test_agent_observation_honors_configured_viewer_camera_uid() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    env.viewer_camera_uid = "inspection_camera"
    env.get_obs = lambda: {
        "sensor": {
            "inspection_camera": {"color": torch.ones((1, 2, 2, 3), dtype=torch.uint8)}
        }
    }
    env.event_manager = _EventManager()

    observation = env.get_obs_for_agent()

    assert observation["rgb"].shape == (2, 2, 3)
    assert torch.all(observation["rgb"] == 1)


def test_agent_observation_reports_available_camera_uids() -> None:
    env = AgenticGenSimEnv.__new__(AgenticGenSimEnv)
    env.viewer_camera_uid = "missing_camera"
    env.get_obs = lambda: {"sensor": {"other_camera": {"color": torch.zeros(1)}}}

    with pytest.raises(KeyError, match="other_camera"):
        env.get_obs_for_agent()


def test_success_evaluator_fallback_values_preserve_legacy_behavior() -> None:
    assert _FALLBACKS == {
        "position_tolerance": 0.05,
        "xy_tolerance": 0.05,
        "container_xy_radius": 0.1,
        "container_min_z_offset": -0.03,
        "container_max_z_offset": 0.25,
        "support_xy_radius": 0.08,
        "support_min_z_offset": 0.02,
        "support_max_z_offset": 0.35,
        "max_tilt_degrees": 45.0,
        "axis_tolerance": 0.02,
        "collinearity_tolerance": 0.02,
        "ordering_tolerance": 0.02,
        "minimum_lift_height": 0.1,
        "single_gripper_max_distance": 0.12,
        "dual_gripper_max_distance": 0.10,
        "gripper_clear_min_distance": 0.05,
        "initial_qpos_tolerance": 0.05,
        "gripper_state_tolerance": 0.001,
    }


def test_success_value_precedence_is_predicate_then_environment_then_yaml() -> None:
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        sim=_Simulation(
            {
                "can": _RigidObject((0.11, 0.0, 0.0)),
                "basket": _RigidObject((0.0, 0.0, 0.0)),
            }
        ),
        agent_success_defaults={"xy_radius": 0.12},
    )
    base_spec = {
        "type": "object_in_container",
        "object": "can",
        "container": "basket",
    }

    assert evaluate_configured_success(env, base_spec).item() is True
    assert (
        evaluate_configured_success(env, {**base_spec, "radius": 0.105}).item() is False
    )


def test_nominal_graph_contract_round_trips_through_compiler() -> None:
    action = {
        "atomic_action_class": "MoveJoints",
        "robot_name": LEFT_ARM_NAME,
        "control": "hand",
        "target_qpos": {"source": "gripper_state", "state": "open"},
        "cfg": {},
    }
    graph_spec = build_nominal_task_graph(
        task_name="graph_contract",
        steps=[NominalGraphStep("Open the gripper", left_arm_action=action)],
    )

    edge = graph_spec["edges"][0]
    assert edge[LEFT_ARM_ACTION_KEY] == action
    assert edge[RIGHT_ARM_ACTION_KEY] is None

    graph = compile_agent_graph_spec(
        graph_spec,
        graph_cls=_CompiledGraph,
        action_module=SimpleNamespace(
            normalize_atomic_action_spec=lambda spec: dict(spec),
        ),
    )
    assert graph.edges[0][3]["left_arm_action"] == action
    assert graph.edges[0][3]["right_arm_action"] is None
