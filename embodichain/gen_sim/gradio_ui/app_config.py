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

"""Static settings for the engine-only Gradio application."""

from __future__ import annotations

from pathlib import Path

import app_env

APP_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = APP_ROOT / "assets"
DEXFORCE_LOGO = ASSETS_DIR / "dexforce.png"
GEN_SIM_ROOT = APP_ROOT / ".gen_sim"
GEN_SIM_ASSET_ROOT = GEN_SIM_ROOT / "assets"
GEN_SIM_SCENE_ROOT = GEN_SIM_ROOT / "scenes"

SCENE_ID = "current"
GYM_PROJECT_ROOT = app_env.EMBODICHAIN_ROOT / "gym_project"
ACTION_AGENT_ROOT = GYM_PROJECT_ROOT / "action_agent_pipeline"
CONFIG_DIR = ACTION_AGENT_ROOT / "configs" / SCENE_ID
FAST_GYM_CONFIG = CONFIG_DIR / "fast_gym_config.json"
AGENT_CONFIG = CONFIG_DIR / "agent_config.json"

OUTPUTS_DIR = app_env.EMBODICHAIN_ROOT / "outputs"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

PROCESS_STOP_TIMEOUT_S = 8.0
DEFAULT_CONCURRENCY_LIMIT = 1

DEBUG_ENGINE_ASSET = "asset_engine"
DEBUG_ENGINE_SCENE = "scene_engine"
DEBUG_ENGINE_ACTION = "action_engine"
DEBUG_ENGINES = (
    (DEBUG_ENGINE_ASSET, "Asset_engine"),
    (DEBUG_ENGINE_SCENE, "Scene_engine"),
    (DEBUG_ENGINE_ACTION, "Action_engine"),
)

SIMREADY_MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}

LANGUAGE_EN = "en"
UI_TEXT = {
    LANGUAGE_EN: {
        "robot": "Robot",
        "input_image": "Input image",
        "single_video_preview": "DexSim Video Preview",
        "current_task": "Current task",
        "progress": "Progress",
    }
}

ROBOT_PROFILE_FRANKA = "Franka"
ROBOT_PROFILE_UR5 = "UR5"
ROBOT_PROFILE_UR10 = "UR10"
ROBOT_PROFILES = [ROBOT_PROFILE_FRANKA, ROBOT_PROFILE_UR5, ROBOT_PROFILE_UR10]
DEFAULT_ROBOT_PROFILE = ROBOT_PROFILE_UR5

COMMANDS = {
    "agent": {
        "module": "embodichain.gen_sim.action_agent_pipeline.cli.run_agent",
        "help_args": ("--help",),
        "base_args": ("--regenerate", "--renderer", "fast-rt"),
        "single_num_envs": "1",
    },
    "scene_engine": {
        "module": "embodichain",
        "base_args": ("scene-engine",),
        "preview_script": "embodichain/gen_sim/scene_engine/cli/preview.py",
    },
}

PHASE_DEFINITIONS = {
    "idle": (0, "Idle"),
    "received": (5, "Input received"),
    "started": (10, "Scene generation started"),
    "scene_intake": (20, "Scene understanding"),
    "relations": (35, "Scene segmentation"),
    "asset_generation": (55, "Geometry generation"),
    "gym_export": (75, "Scene export"),
    "preview": (90, "Preview generation"),
    "complete": (100, "Complete"),
    "failed": (100, "Failed"),
}
