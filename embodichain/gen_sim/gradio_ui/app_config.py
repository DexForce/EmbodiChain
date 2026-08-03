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

"""Shared settings and helpers for the Gradio application.

Deployment-specific values are read from ``embodichain/gen_sim/.env``; this
module keeps UI constants, path derivation, and CLI command definitions close
to the application code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from embodichain.gen_sim.env import load_gen_sim_env

load_gen_sim_env()

PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "FTP_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "ftp_proxy",
)
DIRECT_NO_PROXY_VALUE = "*"


def _getenv(name: str, default: str) -> str:
    """Read a non-empty shared ``.env`` value, falling back to ``default``."""
    return os.environ.get(name) or default


# SimReady uses an OpenAI-compatible multimodal endpoint.  Configure these
# values here for a local deployment, or provide the matching SIMREADY_* env
# vars before launch.  Keep the API key out of commits; an empty value leaves
# any inherited OPENAI_* variables and SimReady's own JSON configuration intact.
SIMREADY_OPENAI_API_KEY = _getenv("SIMREADY_OPENAI_API_KEY", "")
SIMREADY_OPENAI_MODEL = _getenv("SIMREADY_OPENAI_MODEL", "")
SIMREADY_OPENAI_BASE_URL = _getenv("SIMREADY_OPENAI_BASE_URL", "")


def configure_direct_network_env(env: Any = None) -> None:
    """Disable proxy inheritance for local pipeline and Gradio processes."""
    if env is None:
        env = os.environ
    for key in PROXY_ENV_KEYS:
        env.pop(key, None)
    env["NO_PROXY"] = DIRECT_NO_PROXY_VALUE
    env["no_proxy"] = DIRECT_NO_PROXY_VALUE
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")


def configure_simready_llm_env(env: Any = None) -> None:
    """Map app-level SimReady settings to the upstream CLI's environment."""
    if env is None:
        env = os.environ
    configured_values = {
        "OPENAI_API_KEY": SIMREADY_OPENAI_API_KEY,
        "OPENAI_MODEL": SIMREADY_OPENAI_MODEL,
        "OPENAI_BASE_URL": SIMREADY_OPENAI_BASE_URL,
    }
    for key, value in configured_values.items():
        if value:
            env[key] = value


APP_ROOT = Path(__file__).resolve().parent
EMBODICHAIN_ROOT = Path(
    _getenv("EMBODICHAIN_ROOT", str(Path(__file__).resolve().parents[3]))
).expanduser()
ASSETS_DIR = APP_ROOT / "assets"
DEXFORCE_LOGO = ASSETS_DIR / "dexforce.png"
INTERACT_RANDOM_PREVIEW_DIR = APP_ROOT / ".gradio_previews"
DEBUG_ENGINE_ROOT = APP_ROOT / ".debug_engine"
DEBUG_ASSET_ENGINE_ROOT = DEBUG_ENGINE_ROOT / "assets"
ARTICRAFT_ROOT = Path(
    _getenv("ARTICRAFT_ROOT", str(APP_ROOT / ".articraft"))
).expanduser()
ARTICRAFT_REPOSITORY_URL = _getenv(
    "ARTICRAFT_REPOSITORY_URL", "https://github.com/mattzh72/articraft.git"
)
ARTICRAFT_CONDA_ENV = _getenv("ARTICRAFT_CONDA_ENV", "articraft")
# Keep every Articraft record, copied reference image, log, and downloadable
# result bundle under one app-owned directory rather than the source checkout.
ARTICRAFT_OUTPUT_ROOT = Path(
    _getenv("ARTICRAFT_OUTPUT_ROOT", str(DEBUG_ENGINE_ROOT / "articraft"))
).expanduser()
DEBUG_SCENE_ENGINE_ROOT = DEBUG_ENGINE_ROOT / "scenes"
SCENE_ENGINE_VISER_PORT = int(_getenv("SCENE_ENGINE_VISER_PORT", "8080"))
# Articulation previews run as a separate Viser process from scene previews,
# so they need their own externally configurable port.
ARTICRAFT_VISER_PORT = int(_getenv("ARTICRAFT_VISER_PORT", "8081"))
SCENE_ID = "current"

GYM_PROJECT_ROOT = EMBODICHAIN_ROOT / "gym_project"
ACTION_AGENT_ROOT = GYM_PROJECT_ROOT / "action_agent_pipeline"
IMAGE_DIR = ACTION_AGENT_ROOT / "images"
AUTO_LOG_DIR = ACTION_AGENT_ROOT / "auto_logs"
IMAGE_PATH = IMAGE_DIR / f"{SCENE_ID}.png"
PROMPT2SCENE_ROOT = GYM_PROJECT_ROOT / SCENE_ID
CONFIG_DIR = ACTION_AGENT_ROOT / "configs" / SCENE_ID
FAST_GYM_CONFIG = CONFIG_DIR / "fast_gym_config.json"
OUTPUTS_DIR = EMBODICHAIN_ROOT / "outputs"
CURRENT_GYM_EXPORT_DIR = PROMPT2SCENE_ROOT / "gym_export"
CURRENT_GYM_EXPORT_CONFIG = CURRENT_GYM_EXPORT_DIR / "gym_config.json"
GRADIO_SCENE_DIR = CONFIG_DIR / "gradio_scene"
GRADIO_SCENE_GLB = GRADIO_SCENE_DIR / "scene_current.glb"
GRADIO_INITIAL_SCENE_GLB = GRADIO_SCENE_DIR / "initial_scene.glb"
GRADIO_OBJECT_PREVIEW_GLB = GRADIO_SCENE_DIR / "object_preview.glb"
SCENE_MANIFEST = GRADIO_SCENE_DIR / "scene_manifest.json"
PENDING_PREFIX = "_gradio_pending_"
REPLACED_PREFIX = "_gradio_replaced_"
GRADIO_SCENE_TRANSFORM_POLICY = "dexsim_gltf_y_up_to_sim_z_up_v1"

PROCESS_STOP_TIMEOUT_S = 8.0
TEXT_REWRITE_SUFFIXES = {".json", ".jsonl", ".txt", ".yaml", ".yml", ".md", ".csv"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
LEROBOT_PREVIEW_DIR = OUTPUTS_DIR / "lerobot_previews"
COMBINED_PREVIEW_DIR = OUTPUTS_DIR / "combined_previews"
LEROBOT_PREVIEW_MAX_FRAMES = 360
COMBINED_VIDEO_FPS = 25

TOP_MODE_AUTO = "auto"
TOP_MODE_INTERACT = "interact"
TOP_MODE_PARALLEL_ENV = "parallel_env"
APP_MODE_DEMO = "demo"
APP_MODE_DEBUG = "debug"
DEBUG_ENGINE_ASSET = "asset_engine"
DEBUG_ENGINE_SCENE = "scene_engine"
DEBUG_ENGINE_ACTION = "action_engine"
DEBUG_ENGINES = (
    (DEBUG_ENGINE_ASSET, "Asset_engine"),
    (DEBUG_ENGINE_SCENE, "Scene_engine"),
    (DEBUG_ENGINE_ACTION, "Action_engine"),
)

# SimReady accepts one mesh plus optional material/texture sidecar files.  The
# File component deliberately permits the sidecars so OBJ/GLTF uploads retain
# their appearance during both preview and processing.
SIMREADY_MESH_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl"}

LANGUAGE_EN = "en"
LANGUAGE_ZH = "zh"
BUTTON_LABELS = {
    LANGUAGE_EN: {
        "auto": "Auto",
        "interact": "Interact",
        "parallel_env": "Parallel Simulation",
        "rerun_simulation": "Run Task",
        "generate": "Generate",
        "start": "Start",
        "random_input": "Random Task",
        "random_scene_input": "Random Scene",
        "reset": "Reset",
        "stop": "Stop",
        "language": "中文",
    },
    LANGUAGE_ZH: {
        "auto": "自动",
        "interact": "交互",
        "parallel_env": "并行仿真",
        "rerun_simulation": "运行任务",
        "generate": "生成",
        "start": "开始",
        "random_input": "随机任务",
        "random_scene_input": "随机场景",
        "reset": "重置",
        "stop": "停止",
        "language": "English",
    },
}
UI_TEXT = {
    LANGUAGE_EN: {
        "heading": "# Generative Simulation User Interface",
        "instruction": "Upload one image, enter one task, then EmbodiChain will generate simulation data what you want.",
        "robot": "Robot",
        "input_image": "Input image",
        "task_description": "Task description",
        "task_placeholder": "Put the middle bottle on the book",
        "scene_description": "Scene description",
        "scene_placeholder": "Optional: describe how to edit the current scene",
        "scene_mode": "Generation mode",
        "scene_mode_initial": "Initial generation",
        "scene_mode_edit": "Edit current scene",
        "scene_mode_task_only": "Change task only",
        "single_video_preview": "LeRobot Data Preview",
        "parallel_video_preview": "Parallel Env Data Preview",
        "current_task": "Current task",
        "progress": "Progress",
        "initial_preview": "Initial scene preview",
        "edited_preview": "Edited scene preview",
        "object_preview": "Generated object GLBs preview",
    },
    LANGUAGE_ZH: {
        "heading": "# 生成式仿真用户界面",
        "instruction": "上传一张图片，输入一个任务，EmbodiChain 将生成所需的仿真数据。",
        "robot": "机器人",
        "input_image": "输入图像",
        "task_description": "任务描述",
        "task_placeholder": "把中间的水瓶放到书上",
        "scene_description": "场景描述",
        "scene_placeholder": "可选：描述如何编辑当前场景",
        "scene_mode": "生成模式",
        "scene_mode_initial": "初始生成",
        "scene_mode_edit": "编辑当前场景",
        "scene_mode_task_only": "仅修改任务",
        "single_video_preview": "LeRobot 数据预览",
        "parallel_video_preview": "并行环境数据预览",
        "current_task": "当前任务",
        "progress": "进度",
        "initial_preview": "初始场景预览",
        "edited_preview": "编辑后场景预览",
        "object_preview": "生成对象 GLB 预览",
    },
}

PIPELINE_MODE_INITIAL = "initial"
PIPELINE_MODE_EDIT = "edit"
PIPELINE_MODE_TASK_ONLY = "task_only"
SCENE_MODE_INITIAL = "initial"
SCENE_MODE_EDIT = "edit"
SCENE_MODE_TASK_ONLY = "task_only"
ROBOT_PROFILE_FRANKA = "Franka"
ROBOT_PROFILE_UR5 = "UR5"
ROBOT_PROFILE_UR10 = "UR10"
ROBOT_PROFILES = [ROBOT_PROFILE_FRANKA, ROBOT_PROFILE_UR5, ROBOT_PROFILE_UR10]
DEFAULT_ROBOT_PROFILE = ROBOT_PROFILE_UR5
RUN_LOG_MODE_AUTO = "auto"
RUN_LOG_MODE_INTERACT = "interact"

# Command modules and immutable argument defaults. Dynamic values are added by
# command builders in app_commands.py.
COMMANDS = {
    "pipeline": {
        "module": "embodichain.gen_sim.action_agent_pipeline.cli.run_agent_pipeline",
        "base_args": (
            "--use-prompt2scene",
            "--overwrite-config",
            "--regenerate",
            "--skip-run-agent",
        ),
    },
    "config": {
        "module": "embodichain.gen_sim.action_agent_pipeline.cli.generate_action_agent_config",
        "base_args": ("--overwrite",),
    },
    "agent": {
        "module": "embodichain.gen_sim.action_agent_pipeline.cli.run_agent",
        "help_args": ("--help",),
        "base_args": ("--regenerate", "--renderer", "fast-rt"),
        "parallel_args": ("--arena_space", "2.2", "--filter_dataset_saving"),
        "parallel_num_envs": "9",
        "single_num_envs": "1",
    },
    # Scene Engine is dispatched by EmbodiChain's registered top-level CLI.
    "scene_engine": {
        "module": "embodichain",
        "base_args": ("scene-engine",),
        "preview_script": "embodichain/gen_sim/scene_engine/cli/preview.py",
    },
}

PHASE_DEFINITIONS = {
    "idle": (0, "Idle"),
    "received": (5, "Input received"),
    "started": (10, "Local pipeline started"),
    "scene_intake": (20, "Scene understanding"),
    "relations": (35, "Segmentation and spatial relations"),
    "asset_generation": (55, "3D asset generation"),
    "gym_export": (70, "Scene export"),
    "config": (82, "Action config generated"),
    "preview": (90, "3D preview loaded"),
    "complete": (100, "Complete"),
    "failed": (100, "Failed"),
}
TIMING_PHASE_LABELS = {
    "relations": "Segmentation / spatial relations",
    "asset_generation": "Object generation",
    "gym_export": "Scene generation / export",
    "action_graph_execution": "Action graph execution",
}
TIMING_PHASE_ORDER = (
    "relations",
    "asset_generation",
    "gym_export",
    "action_graph_execution",
)

SERVER_NAME = _getenv("GRADIO_SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(_getenv("GRADIO_SERVER_PORT", "7860"))
DEFAULT_CONCURRENCY_LIMIT = 1
