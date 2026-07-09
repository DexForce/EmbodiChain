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

import argparse

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    DEFAULT_CONFIG_OUTPUT_DIR,
    DEFAULT_EXISTING_GYM_PROJECT,
    DEFAULT_GYM_PROJECT_ROOT,
    DEFAULT_IMAGE,
    DEFAULT_IMAGE2SCENE_CONFIG,
    DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR,
    DEFAULT_IMAGE2SCENE_IMAGE,
    DEFAULT_IMAGE2SCENE_OUTPUT_ROOT,
    DEFAULT_IMAGE2SCENE_ROOT,
    DEFAULT_JOB_TIMEOUT_S,
    DEFAULT_PIPELINE_HISTORY,
    DEFAULT_PROMPT2SCENE_MESH_X_ROTATION_DEGREES,
    DEFAULT_PROMPT2SCENE_LLM_CONFIG,
    DEFAULT_PROMPT2SCENE_OUTPUT_ROOT,
    DEFAULT_PROMPT2SCENE_SCENE_Z_ROTATION_DEGREES,
    DEFAULT_TASK_NAME,
)
from embodichain.gen_sim.action_agent_pipeline.generation.robot_profiles import (
    DEFAULT_ROBOT_PROFILE_ID,
    available_robot_profile_choices,
)

__all__ = ["build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the one-shot action-agent pipeline argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a tabletop gym project from one image, generate action-agent "
            "configs from that project, then run the generated task."
        )
    )
    image_group = parser.add_mutually_exclusive_group()
    image_group.add_argument(
        "--image",
        default=None,
        help=(
            f"Input image path. If omitted, defaults to {DEFAULT_IMAGE.as_posix()} "
            f"or {DEFAULT_IMAGE2SCENE_IMAGE} with --use-image2scene."
        ),
    )
    image_group.add_argument(
        "--image-name",
        "--image_name",
        dest="image_name",
        default=None,
        help=(
            "Image file name under the default image directory. The suffix is "
            'optional, e.g. "demo6" resolves to demo6.jpg.'
        ),
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Image2Tabletop API server. Defaults to IMAGE2TABLETOP_SERVER.",
    )
    parser.add_argument(
        "--use-image2scene",
        action="store_true",
        default=False,
        help=(
            "Use gym_project/environment/image2tabletop/demo_api/client/"
            "image2scene_pipeline.py as the first stage and continue from its "
            "gym_config_merged.json output."
        ),
    )
    parser.add_argument(
        "--use-prompt2scene",
        action="store_true",
        default=False,
        help=(
            "Use embodichain.gen_sim.prompt2scene as the first stage and "
            "continue from its exported gym_config.json output."
        ),
    )
    parser.add_argument(
        "--background",
        default=None,
        help=(
            "Background description passed to image2scene_pipeline.py. Required "
            "with --use-image2scene."
        ),
    )
    parser.add_argument(
        "--image2scene-root",
        default=str(DEFAULT_IMAGE2SCENE_ROOT),
        help=(
            "Working directory for image2scene_pipeline.py. Defaults to "
            f"{DEFAULT_IMAGE2SCENE_ROOT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--image2scene-download-dir",
        default=DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR,
        help=(
            "Download directory passed to image2scene_pipeline.py. Relative "
            "paths are interpreted under --image2scene-root. Defaults to "
            f"{DEFAULT_IMAGE2SCENE_DOWNLOAD_DIR}."
        ),
    )
    parser.add_argument(
        "--image2scene-output-root",
        default=DEFAULT_IMAGE2SCENE_OUTPUT_ROOT,
        help=(
            "Generated EC project directory passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root. Defaults "
            f"to {DEFAULT_IMAGE2SCENE_OUTPUT_ROOT}."
        ),
    )
    parser.add_argument(
        "--image2scene-gen-config",
        default=DEFAULT_IMAGE2SCENE_CONFIG,
        help=(
            "Generation config passed to image2scene_pipeline.py. Relative "
            "paths are interpreted under --image2scene-root. Defaults to "
            f"{DEFAULT_IMAGE2SCENE_CONFIG}."
        ),
    )
    parser.add_argument(
        "--image2scene-client-url",
        default=None,
        help=(
            "MesaTask/TextToScene service URL passed to image2scene Stage B. "
            "If omitted, defaults to --server."
        ),
    )
    parser.add_argument(
        "--image2scene-llm-config",
        default=DEFAULT_IMAGE2SCENE_CONFIG,
        help=(
            "LLM config passed to image2scene_pipeline.py. Relative paths are "
            "interpreted under --image2scene-root. Defaults to "
            f"{DEFAULT_IMAGE2SCENE_CONFIG}."
        ),
    )
    parser.add_argument(
        "--image2scene-extract-dir",
        default=None,
        help=(
            "Optional extract directory passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root."
        ),
    )
    parser.add_argument(
        "--image2scene-merged-output",
        default=None,
        help=(
            "Optional merged output path passed to image2scene_pipeline.py. "
            "Relative paths are interpreted under --image2scene-root."
        ),
    )
    parser.add_argument(
        "--prompt2scene-output-root",
        "--prompt2scene_output_root",
        dest="prompt2scene_output_root",
        default=str(DEFAULT_PROMPT2SCENE_OUTPUT_ROOT),
        help=(
            "Output directory for the in-repo prompt2scene pipeline. Defaults "
            f"to {DEFAULT_PROMPT2SCENE_OUTPUT_ROOT.as_posix()}."
        ),
    )
    parser.add_argument(
        "--prompt2scene-llm-config",
        "--prompt2scene_llm_config",
        dest="prompt2scene_llm_config",
        default=str(DEFAULT_PROMPT2SCENE_LLM_CONFIG),
        help=(
            "LLM config JSON used by prompt2scene. Defaults to "
            f"{DEFAULT_PROMPT2SCENE_LLM_CONFIG.as_posix()}."
        ),
    )
    parser.add_argument(
        "--prompt2scene-text",
        "--prompt2scene_text",
        dest="prompt2scene_text",
        default=None,
        help=(
            "Deprecated. Text-only prompt2scene input was removed upstream and "
            "is rejected; use --prompt2scene-prompt for scene edit or "
            "randomization."
        ),
    )
    parser.add_argument(
        "--prompt2scene-prompt",
        "--prompt2scene_prompt",
        dest="prompt2scene_prompt",
        default=None,
        help=(
            "Prompt passed to prompt2scene for scene edit or scene "
            "randomization. This is independent from --task_description."
        ),
    )
    parser.add_argument(
        "--prompt2scene-gravity-settle-mode",
        "--prompt2scene_gravity_settle_mode",
        dest="prompt2scene_gravity_settle_mode",
        choices=("geometry", "physics"),
        default="geometry",
        help=(
            "Gravity settle mode passed to prompt2scene. 'geometry' translates "
            "each GLB by AABB; 'physics' runs simulation. Defaults to geometry."
        ),
    )
    parser.add_argument(
        "--prompt2scene-scene-z-rotation-degrees",
        "--prompt2scene_scene_z_rotation_degrees",
        dest="prompt2scene_scene_z_rotation_degrees",
        type=float,
        default=DEFAULT_PROMPT2SCENE_SCENE_Z_ROTATION_DEGREES,
        help=(
            "World-frame Z rotation applied when converting prompt2scene "
            "exports into action-agent configs. Defaults to -90."
        ),
    )
    parser.add_argument(
        "--prompt2scene-mesh-x-rotation-degrees",
        "--prompt2scene_mesh_x_rotation_degrees",
        dest="prompt2scene_mesh_x_rotation_degrees",
        type=float,
        default=DEFAULT_PROMPT2SCENE_MESH_X_ROTATION_DEGREES,
        help=(
            "Local X-axis rotation baked into prompt2scene GLB meshes during "
            "action-agent OBJ normalization. Defaults to 90."
        ),
    )
    parser.add_argument(
        "--gym-project-root",
        default=str(DEFAULT_GYM_PROJECT_ROOT),
        help=(
            "Directory where Image2Tabletop generated gym projects are written. "
            f"Defaults to {DEFAULT_GYM_PROJECT_ROOT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--use-existing-gym-project",
        action="store_true",
        default=False,
        help=(
            "Skip Image2Tabletop API and start from --gym-project. Defaults to "
            "false. For prompt2scene edits, use --use-prompt2scene with "
            "--prompt2scene-output-root instead."
        ),
    )
    parser.add_argument(
        "--base-task-name",
        "--base_task_name",
        dest="base_task_name",
        default=None,
        help=(
            "Start from the latest pipeline history entry with this task name. "
            "Use this to chain demos, e.g. demo2 based on Demo1_Text."
        ),
    )
    parser.add_argument(
        "--base-history-index",
        "--base_history_index",
        dest="base_history_index",
        type=int,
        default=None,
        help=(
            "Start from a specific pipeline history index. When used with "
            "--base-task-name, the history entry must match that task name."
        ),
    )
    parser.add_argument(
        "--gym-project",
        "--gym_project",
        dest="gym_project",
        default=str(DEFAULT_EXISTING_GYM_PROJECT),
        help=(
            "Existing gym project used with --use-existing-gym-project. "
            f"Defaults to {DEFAULT_EXISTING_GYM_PROJECT.as_posix()}"
        ),
    )
    parser.add_argument(
        "--config-output-dir",
        "--output_dir",
        dest="config_output_dir",
        default=str(DEFAULT_CONFIG_OUTPUT_DIR),
        help=(
            "Destination directory for generated config files. Defaults to "
            f"{DEFAULT_CONFIG_OUTPUT_DIR.as_posix()}"
        ),
    )
    parser.add_argument(
        "--pipeline-history-path",
        "--pipeline_history_path",
        dest="pipeline_history_path",
        default=str(DEFAULT_PIPELINE_HISTORY),
        help=(
            "Global pipeline history JSON path. Defaults to "
            f"{DEFAULT_PIPELINE_HISTORY.as_posix()}"
        ),
    )
    parser.add_argument(
        "--task_name",
        "--task-name",
        dest="task_name",
        default=DEFAULT_TASK_NAME,
        help=f"Task name passed to run_agent. Defaults to {DEFAULT_TASK_NAME}",
    )
    parser.add_argument(
        "--task_description",
        "--task-description",
        dest="task_description",
        default="",
        help=(
            'Task description passed to config generation. Defaults to "". '
            "Ignored for default-template tasks such as Demo1_Text."
        ),
    )
    parser.add_argument(
        "--robot-profile",
        "--robot_profile",
        dest="robot_profile",
        choices=available_robot_profile_choices(),
        default=DEFAULT_ROBOT_PROFILE_ID,
        help=(
            "Robot profile used by action-agent config generation. Defaults to "
            f"{DEFAULT_ROBOT_PROFILE_ID}."
        ),
    )
    parser.add_argument(
        "--target_body_scale",
        "--target-body-scale",
        dest="target_body_scale",
        type=float,
        default=None,
        help=(
            "Uniform body_scale for generated target objects. In prompt2scene "
            "mode, omit this option to preserve source body_scale by default. "
            "Other modes default to 0.8."
        ),
    )
    parser.add_argument(
        "--target_body_scale_mode",
        "--target-body-scale-mode",
        dest="target_body_scale_mode",
        choices=("preserve", "multiply", "absolute"),
        default=None,
        help=(
            "Prompt2scene body_scale policy for source-scene objects. "
            "preserve keeps source body_scale, multiply uses source_scale * "
            "--target_body_scale, and absolute sets objects directly to "
            "--target_body_scale. Defaults to preserve when --target_body_scale "
            "is omitted and multiply when it is provided."
        ),
    )
    parser.add_argument(
        "--inside-container-slot-distance-scale",
        "--inside_container_slot_distance_scale",
        dest="inside_container_slot_distance_scale",
        type=float,
        default=1.0,
        help=(
            "Scale factor for generated inside-container release slot offsets "
            "when multiple objects are placed into one container. Values below "
            "1.0 move release points closer to the container center. Defaults "
            "to 1.0."
        ),
    )
    parser.add_argument(
        "--target_replacement",
        "--target-replacement",
        dest="target_replacement",
        action="append",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=[],
        help=(
            "Generate one replacement foreground interactive object. Repeat for "
            "0-N replacements. Accepts either PROMPT for auto-selection from "
            "numbered rigid_object targets, or SOURCE_UID PROMPT for explicit "
            "selection."
        ),
    )
    parser.add_argument(
        "--target_replacement1",
        "--target-replacement1",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new1 from PROMPT. Accepts either "
            "PROMPT, which auto-selects the first duplicated foreground rigid "
            "object, or SOURCE_UID PROMPT for explicit selection."
        ),
    )
    parser.add_argument(
        "--target_replacement2",
        "--target-replacement2",
        nargs="+",
        metavar="SOURCE_OR_PROMPT",
        default=None,
        help=(
            "Generate <gym_project>/mesh_assets/new2 from PROMPT. Accepts either "
            "PROMPT, which auto-selects the second duplicated foreground rigid "
            "object, or SOURCE_UID PROMPT for explicit selection."
        ),
    )
    parser.add_argument(
        "--sync_replacement_names",
        "--sync-replacement-names",
        action="store_true",
        default=False,
        help=(
            "Also update replacement target runtime UIDs and generated prompts "
            "from the replacement prompts."
        ),
    )
    parser.add_argument(
        "--reuse-target-replacements",
        "--reuse_target_replacements",
        dest="reuse_target_replacements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse existing prompt-generated replacement GLBs when the prompt "
            "and expected output name match. Defaults to true."
        ),
    )
    parser.add_argument(
        "--prewarm-coacd-cache",
        "--prewarm_coacd_cache",
        dest="prewarm_coacd_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Precompute environment CoACD cache files during config generation. "
            "Defaults to true."
        ),
    )
    parser.add_argument(
        "--convex-decomposition-method",
        "--convex_decomposition_method",
        choices=("vhacd", "visacd", "coacd"),
        default="vhacd",
        help=(
            "Convex decomposition backend written to generated mesh objects. "
            "'visacd' is accepted as an alias for 'vhacd'. Defaults to vhacd."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Image2Tabletop job polling interval in seconds. Defaults to 10.0.",
    )
    parser.add_argument(
        "--job-timeout-s",
        "--job_timeout_s",
        dest="job_timeout_s",
        type=float,
        default=DEFAULT_JOB_TIMEOUT_S,
        help="Maximum seconds to wait for Image2Tabletop API jobs.",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        default=False,
        help="Skip GET /health before submitting the image.",
    )
    parser.add_argument(
        "--overwrite-gym-project",
        action="store_true",
        default=False,
        help="Replace an existing generated gym project with the same name.",
    )
    parser.add_argument(
        "--overwrite-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite generated config files. Defaults to true.",
    )
    parser.add_argument(
        "--regenerate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --regenerate to run_agent. Defaults to true.",
    )
    parser.add_argument(
        "--skip-run-agent",
        action="store_true",
        default=False,
        help="Stop after generating config files instead of launching run_agent.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Pass --headless to run_agent to avoid opening a simulator window.",
    )
    parser.add_argument(
        "--llm-usage-output",
        default=None,
        help=(
            "JSONL path for local LLM token usage records. Defaults to "
            "<config-output-dir>/llm_usage.jsonl."
        ),
    )
    parser.add_argument(
        "--llm-usage-summary-output",
        default=None,
        help=(
            "JSON path for the aggregated local LLM token usage summary. "
            "Defaults to <config-output-dir>/llm_usage_summary.json."
        ),
    )
    parser.add_argument(
        "--llm-usage-run-id",
        default=None,
        help="Optional run id written into local LLM token usage records.",
    )
    parser.add_argument(
        "--no-llm-usage",
        dest="llm_usage",
        action="store_false",
        default=True,
        help="Disable local LLM token usage recording for this pipeline run.",
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help=(
            "JSONL path for local stage timing records. Defaults to "
            "<config-output-dir>/timing.jsonl."
        ),
    )
    parser.add_argument(
        "--timing-summary-output",
        default=None,
        help=(
            "JSON path for the aggregated local timing summary. Defaults to "
            "<config-output-dir>/timing_summary.json."
        ),
    )
    parser.add_argument(
        "--no-timing",
        dest="timing",
        action="store_false",
        default=True,
        help="Disable local stage timing records for this pipeline run.",
    )
    return parser
