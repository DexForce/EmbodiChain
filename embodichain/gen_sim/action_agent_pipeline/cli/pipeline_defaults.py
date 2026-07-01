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

import os
from pathlib import Path

__all__ = [
    "DEFAULT_ACTION_AGENT_WORKSPACE",
    "DEFAULT_CONFIG_OUTPUT_DIR",
    "DEFAULT_EXISTING_GYM_PROJECT",
    "DEFAULT_GYM_PROJECT_ROOT",
    "DEFAULT_IMAGE",
    "DEFAULT_IMAGE_DIR",
    "DEFAULT_IMAGE2SCENE_CONFIG",
    "DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR",
    "DEFAULT_IMAGE2SCENE_IMAGE",
    "DEFAULT_IMAGE2SCENE_OUTPUT_ROOT",
    "DEFAULT_IMAGE2SCENE_ROOT",
    "DEFAULT_JOB_TIMEOUT_S",
    "DEFAULT_PIPELINE_HISTORY",
    "DEFAULT_PROMPT2SCENE_LLM_CONFIG",
    "DEFAULT_PROMPT2SCENE_OUTPUT_ROOT",
    "DEFAULT_TASK_NAME",
    "DEFAULT_TASK_TEMPLATE_NAMES",
    "GYM_CONFIG_PREFERENCE",
    "IMAGE_SUFFIXES",
    "PIPELINE_HISTORY_SCHEMA_VERSION",
    "PIPELINE_MANIFEST_FILENAME",
    "REPO_ROOT",
]


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").is_file() and (parent / "embodichain").is_dir():
            return parent
    return Path.cwd().resolve()


REPO_ROOT = _repo_root()
DEFAULT_JOB_TIMEOUT_S = 1800.0
DEFAULT_GYM_PROJECT_ROOT = REPO_ROOT / "gym_project"
DEFAULT_ACTION_AGENT_WORKSPACE = DEFAULT_GYM_PROJECT_ROOT / "action_agent_pipeline"
DEFAULT_IMAGE = DEFAULT_ACTION_AGENT_WORKSPACE / "images/demo1.jpg"
DEFAULT_IMAGE_DIR = DEFAULT_IMAGE.parent
DEFAULT_EXISTING_GYM_PROJECT = DEFAULT_GYM_PROJECT_ROOT / "1780562837_gym_project"
DEFAULT_IMAGE2SCENE_ROOT = REPO_ROOT / "gym_project/environment/image2tabletop"
DEFAULT_IMAGE2SCENE_IMAGE = "scene_image/robotwin_example.png"
DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR = "./downloads"
DEFAULT_IMAGE2SCENE_OUTPUT_ROOT = "./generated"
DEFAULT_IMAGE2SCENE_CONFIG = "./gen_config.json"
DEFAULT_PROMPT2SCENE_OUTPUT_ROOT = DEFAULT_GYM_PROJECT_ROOT / "prompt2scene"
DEFAULT_PROMPT2SCENE_LLM_CONFIG = (
    REPO_ROOT / "embodichain/gen_sim/prompt2scene/configs/llm_config.json"
)
DEFAULT_CONFIG_OUTPUT_DIR = DEFAULT_ACTION_AGENT_WORKSPACE / "configs/demo3_text"
DEFAULT_PIPELINE_HISTORY = (
    DEFAULT_ACTION_AGENT_WORKSPACE / "configs/pipeline_history.json"
)
DEFAULT_TASK_NAME = os.getenv("ACTION_AGENT_DEFAULT_TASK_NAME", "ActionAgentTask")
DEFAULT_TASK_TEMPLATE_NAMES = frozenset(
    name.strip()
    for name in os.getenv("ACTION_AGENT_DEFAULT_TASK_TEMPLATE_NAMES", "").split(",")
    if name.strip()
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
GYM_CONFIG_PREFERENCE = ("gym_config_merged.json", "gym_config.json")
PIPELINE_HISTORY_SCHEMA_VERSION = 1
PIPELINE_MANIFEST_FILENAME = "pipeline_manifest.json"
