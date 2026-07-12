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

import pytest

from embodichain.gen_sim.action_agent_pipeline.defaults import (
    ACTION_AGENT_CONFIG_DEFAULTS,
    DEFAULT_MAX_EPISODES,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_SURFACE_RELEASE_CLEARANCE,
    DEFAULT_TARGET_BODY_SCALE,
    DEFAULT_TASK_NAME,
    generation_defaults_section,
)


def test_public_generation_defaults_are_loaded_from_yaml() -> None:
    task = ACTION_AGENT_CONFIG_DEFAULTS["task"]
    geometry = ACTION_AGENT_CONFIG_DEFAULTS["geometry"]

    assert DEFAULT_TASK_NAME == task["default_name"]
    assert DEFAULT_MAX_EPISODES == task["max_episodes"]
    assert DEFAULT_MAX_EPISODE_STEPS == task["max_episode_steps"]
    assert DEFAULT_TARGET_BODY_SCALE == geometry["target_body_scale"]
    assert DEFAULT_SURFACE_RELEASE_CLEARANCE == geometry["surface_release_clearance"]


def test_generation_defaults_expose_required_hyperparameter_sections() -> None:
    expected_sections = {
        "action",
        "arrangement",
        "geometry",
        "grasp",
        "physics",
        "relative_placement",
        "stacking",
        "task",
    }

    assert expected_sections <= ACTION_AGENT_CONFIG_DEFAULTS.keys()


def test_generation_defaults_section_rejects_unknown_section() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        generation_defaults_section("missing")
