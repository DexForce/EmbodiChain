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

from embodichain.gen_sim.action_agent_pipeline.cli.run_agent import (
    _modify_gym_config_for_run_agent,
)
from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
    _make_stacking_dataset_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.config_blocks import (
    _make_arrangement_dataset_config,
    _make_dataset_config,
    _make_relative_dataset_config,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    resolve_robot_profile,
)


def test_all_task_routes_generate_manager_level_save_failed_episodes():
    profile = resolve_robot_profile("franka")
    placement = SimpleNamespace(
        active_side="left",
        moved_runtime_uid="interact_cup",
        relation="on",
        reference_runtime_uid="table",
    )
    configs = [
        _make_dataset_config(
            "gym_export",
            SimpleNamespace(
                container_runtime_uid="interact_basket",
                left_target_runtime_uid="interact_left_cube",
                right_target_runtime_uid="interact_right_cube",
                target_noun="cube",
                left_target_noun="cube",
                right_target_noun="cube",
                container_noun="basket",
            ),
            robot_profile=profile,
        ),
        _make_relative_dataset_config(
            "gym_export",
            SimpleNamespace(
                intent="place_relative",
                placements=(placement,),
                task_description="Place the cup on the table.",
            ),
            robot_profile=profile,
            relation_phrase=lambda relation: relation,
        ),
        _make_arrangement_dataset_config(
            "gym_export",
            SimpleNamespace(
                task_description="Arrange the cups in a line.",
                steps=(SimpleNamespace(runtime_uid="interact_cup"),),
            ),
            robot_profile=profile,
        ),
        _make_stacking_dataset_config(
            "gym_export",
            SimpleNamespace(
                anchor_runtime_uid=None,
                task_description="Stack the cups.",
                steps=(SimpleNamespace(runtime_uid="interact_cup"),),
            ),
            robot_profile=profile,
        ),
    ]

    for config in configs:
        recorder_config = config["lerobot"]
        assert recorder_config["save_failed_episodes"] is True
        assert "save_failed_episodes" not in recorder_config["params"]


def test_run_agent_normalizes_legacy_dataset_manager_parameter():
    gym_config = {
        "env": {
            "dataset": {
                "lerobot": {
                    "func": "LeRobotRecorder",
                    "mode": "save",
                    "params": {
                        "save_failed_episodes": True,
                        "use_videos": True,
                    },
                }
            }
        }
    }

    _modify_gym_config_for_run_agent(gym_config)

    recorder_config = gym_config["env"]["dataset"]["lerobot"]
    assert recorder_config["save_failed_episodes"] is True
    assert recorder_config["params"] == {"use_videos": True}


def test_run_agent_preserves_explicit_manager_level_value():
    gym_config = {
        "env": {
            "dataset": {
                "lerobot": {
                    "func": "LeRobotRecorder",
                    "mode": "save",
                    "save_failed_episodes": False,
                    "params": {"save_failed_episodes": True},
                }
            }
        }
    }

    _modify_gym_config_for_run_agent(gym_config)

    recorder_config = gym_config["env"]["dataset"]["lerobot"]
    assert recorder_config["save_failed_episodes"] is False
    assert "save_failed_episodes" not in recorder_config["params"]
