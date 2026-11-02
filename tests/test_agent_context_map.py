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

import importlib.util
from pathlib import Path
from types import ModuleType

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_HELPER_PATH = (
    _REPOSITORY_ROOT / ".agents/skills/project-dev-context/scripts/context.py"
)


def _load_helper() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_context_helper", _HELPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_context_map() -> dict:
    return _load_helper().load_map(_REPOSITORY_ROOT)


def _topics_by_id() -> dict[str, dict]:
    return {topic["id"]: topic for topic in _load_context_map()["topics"]}


def test_repository_context_map_is_valid() -> None:
    helper = _load_helper()
    data = helper.load_map(_REPOSITORY_ROOT)

    assert helper.validate_map(_REPOSITORY_ROOT, data) == []


def test_map_registers_the_supported_context_domains() -> None:
    topics = _topics_by_id()
    expected_topic_ids = {
        "simulation-system",
        "env-framework",
        "manager-functor",
        "ik-solvers",
        "robot-system",
        "sensor-system",
        "sim-visualization",
        "motion-planning",
        "rl-learning",
        "learning-layers",
        "supervised-learning",
        "neural-dynamics",
        "robot-world-model",
        "configclass-pattern",
        "randomization",
        "atomic-actions",
        "task-programs",
        "gen-sim",
        "data-assets",
        "asset-generalization",
        "data-pipeline",
        "robot-workspace",
        "scene-expansion",
        "scene-services",
        "differentiable-env",
    }

    assert set(topics) == expected_topic_ids


def test_new_topics_cover_their_owning_packages() -> None:
    topics = _topics_by_id()
    expected_source_prefixes = {
        "gen-sim": "embodichain/gen_sim/",
        "scene-services": "embodichain/lab/sim/scene/",
        "data-assets": "embodichain/data/",
        "data-pipeline": "embodichain/data_pipeline/",
        "robot-workspace": "embodichain/lab/sim/motion/workspace/",
        "differentiable-env": "embodichain/lab/gym/envs/differentiable_env.py",
    }

    for topic_id, prefix in expected_source_prefixes.items():
        assert any(
            path.startswith(prefix) for path in topics[topic_id]["source_of_truth"]
        )


def test_topic_paths_name_only_the_default_overview() -> None:
    topics = _topics_by_id()

    assert all(len(topic["paths"]) == 1 for topic in topics.values())


def test_representative_queries_route_against_the_repository_map() -> None:
    helper = _load_helper()
    data = helper.load_map(_REPOSITORY_ROOT)
    expected_routes = {
        "参考 env-framework 上下文，查 target_control_frequency 的配置优先级": [
            "env-framework"
        ],
        "SceneManifest 在哪里定义？": ["sim-visualization", "task-programs"],
        "visualization SceneManifest 在哪里定义？": ["sim-visualization"],
        "机器人工作空间缓存从哪里加载？": ["robot-workspace"],
        "OnlineDataEngine 采样失败后怎么处理？": ["data-pipeline"],
        "SimReady pipeline 的入口在哪里？": ["gen-sim"],
        "get_data_path 如何解析资产路径？": ["data-assets"],
        "Newton physics backend config 在哪里？": ["simulation-system"],
        "可微环境 APG 如何重置？": ["differentiable-env"],
        "可微 IK layer 的梯度": ["learning-layers"],
        "可微 FK layer 的梯度": ["learning-layers"],
        "DifferentiableFKLayer": ["learning-layers"],
        "DifferentiableIKLayer": ["learning-layers"],
        "sim.scene 的场景布局服务在哪里？": ["scene-services"],
        "程序化场景如何生成？": ["scene-services"],
        "SceneState 如何恢复？": ["scene-services"],
        "scene expansion 候选身份在哪里？": ["scene-expansion"],
    }

    assert {
        query: helper.route_topics(data, query) for query in expected_routes
    } == expected_routes


def test_project_context_adapters_reference_the_canonical_skill() -> None:
    canonical_path = ".agents/skills/project-dev-context/SKILL.md"
    adapter_paths = (
        _REPOSITORY_ROOT / ".claude/skills/project-dev-context/SKILL.md",
        _REPOSITORY_ROOT / ".github/copilot/project-dev-context.md",
    )
    missing_references = [
        str(path.relative_to(_REPOSITORY_ROOT))
        for path in adapter_paths
        if canonical_path not in path.read_text(encoding="utf-8")
    ]

    assert missing_references == []


def test_trajectory_test_changes_do_not_select_unrelated_ik_context() -> None:
    helper = _load_helper()
    data = helper.load_map(_REPOSITORY_ROOT)

    assert helper.affected_topics(
        data, ["tests/compute/test_trajectory_timing.py"]
    ) == ["motion-planning"]


def test_rl_only_config_changes_do_not_select_task_program_context() -> None:
    helper = _load_helper()
    data = helper.load_map(_REPOSITORY_ROOT)

    affected = helper.affected_topics(
        data,
        [
            "extensions/embodichain_tasks/configs/tasks/"
            "classic_control/cart_pole/agents/ppo.yaml"
        ],
    )

    assert "rl-learning" in affected
    assert "task-programs" not in affected


def test_task_program_deployments_and_shared_components_remain_watched() -> None:
    helper = _load_helper()
    data = helper.load_map(_REPOSITORY_ROOT)
    config_root = "extensions/embodichain_tasks/configs/"
    paths = [
        "tasks/manipulation/repeated_pick_place/task.ur5.yaml",
        "tasks/manipulation/repeated_pick_place/task_program/integration.yaml",
        "components/embodiments/franka_panda.yaml",
        "components/execution_policies/trajectory_open_loop.yaml",
        "components/expansion_policies/task_program_episode.yaml",
    ]

    for path in paths:
        assert "task-programs" in helper.affected_topics(data, [config_root + path])
