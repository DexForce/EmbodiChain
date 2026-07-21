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
import sys

from embodichain.gen_sim.action_agent_pipeline.cli.agent_run_stage import (
    run_agent_command,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_defaults import (
    PIPELINE_HISTORY_SCHEMA_VERSION,
    PIPELINE_MANIFEST_FILENAME,
    REPO_ROOT,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_records import (
    write_pipeline_manifests,
)
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_usage import (
    configure_llm_usage_tracking,
    write_llm_usage_summary,
)
from embodichain.gen_sim.action_agent_pipeline.cli.project_resolution import (
    resolve_gym_project,
    resolve_task_description_for_generation,
)
from embodichain.gen_sim.action_agent_pipeline.cli.target_replacements import (
    resolve_target_replacements,
)

__all__ = ["run_pipeline"]


def run_pipeline(args: argparse.Namespace) -> int:
    """Run image/project resolution, config generation, and optional task execution."""
    _ensure_repo_on_pythonpath()
    from embodichain.gen_sim.action_agent_pipeline.generation.action_agent_config import (
        TargetReplacementSpec,
        generate_action_agent_config_from_project,
    )

    resolution = resolve_gym_project(args)
    usage_paths = configure_llm_usage_tracking(args)
    target_replacements = resolve_target_replacements(
        args,
        TargetReplacementSpec,
        resolution.path,
    )
    task_description = resolve_task_description_for_generation(args)
    args.task_description = task_description or ""

    paths = generate_action_agent_config_from_project(
        gym_project=resolution.path,
        output_dir=args.config_output_dir,
        task_name=args.task_name,
        task_description=task_description,
        target_body_scale=args.target_body_scale,
        target_replacements=target_replacements,
        sync_replacement_names=args.sync_replacement_names,
        reuse_target_replacements=args.reuse_target_replacements,
        prewarm_coacd_cache=args.prewarm_coacd_cache,
        overwrite=args.overwrite_config,
    )
    write_pipeline_manifests(
        args=args,
        resolution=resolution,
        generated_paths=paths,
        target_replacements=target_replacements,
        repo_root=REPO_ROOT,
        schema_version=PIPELINE_HISTORY_SCHEMA_VERSION,
        manifest_filename=PIPELINE_MANIFEST_FILENAME,
    )

    print(f"Using gym project/config: {resolution.path}", flush=True)
    print(f"Generated gym config: {paths.gym_config}", flush=True)
    print(f"Generated agent config: {paths.agent_config}", flush=True)
    if args.skip_run_agent:
        write_llm_usage_summary(usage_paths)
        return 0

    return_code = run_agent_command(
        task_name=args.task_name,
        gym_config=paths.gym_config,
        agent_config=paths.agent_config,
        regenerate=args.regenerate,
    )
    write_llm_usage_summary(usage_paths)
    return return_code


def _ensure_repo_on_pythonpath() -> None:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
