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

"""CLI for preparing, scoring, or running mesh affordance inference."""

from __future__ import annotations

import argparse
import logging

from . import (
    MeshAffordanceCfg,
    analyze_mesh_affordance,
    prepare_mesh_affordance,
    score_mesh_affordance,
)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "prepare", "score"))
    parser.add_argument("--mesh", help="Local mesh path or get_data_path asset key")
    parser.add_argument("--object", dest="object_description")
    parser.add_argument("--task", dest="task_description")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--provider",
        choices=("openai", "deepseek"),
        default=None,
        help="Model provider; both use the Codex harness (default: openai)",
    )
    parser.add_argument(
        "--provider-config",
        default=None,
        help="Local DeepSeek JSON file containing base_url, api_key and optional model",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Default: gpt-6-astra for OpenAI; config model or deepseek-flash for DeepSeek",
    )
    parser.add_argument("--patch-count", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--blender-python", default=None)
    parser.add_argument(
        "--codex-executable", default=None, help="Codex CLI command or executable path"
    )
    parser.add_argument(
        "--target-part",
        default=None,
        help="Part to segment, e.g. 把手, independently of grasp scores",
    )
    parser.add_argument("--part-threshold", type=float, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.action == "score":
        import json
        from pathlib import Path

        cfg = MeshAffordanceCfg(
            **json.loads(
                (Path(args.output).expanduser() / "evidence.json").read_text()
            )["config"]
        )
    else:
        if not all((args.mesh, args.object_description, args.task_description)):
            parser.error("run/prepare require --mesh, --object and --task")
        cfg = MeshAffordanceCfg()
    # A provider switch should use its own default, not a saved model from the
    # previous provider. An explicit --model always takes precedence.
    if (
        args.provider is not None
        and args.provider != cfg.provider
        and args.model is None
    ):
        cfg.model = None
    for argument, field in (
        ("provider", "provider"),
        ("provider_config", "provider_config"),
        ("model", "model"),
        ("patch_count", "patch_count"),
        ("resolution", "render_resolution"),
        ("threshold", "threshold"),
        ("min_confidence", "min_confidence"),
        ("timeout", "timeout_seconds"),
        ("blender_python", "blender_python"),
        ("codex_executable", "codex_executable"),
        ("target_part", "target_part"),
        ("part_threshold", "part_threshold"),
    ):
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, value)
    if args.action == "score":
        result = score_mesh_affordance(args.output, cfg)
    else:
        function = (
            prepare_mesh_affordance
            if args.action == "prepare"
            else analyze_mesh_affordance
        )
        result = function(
            args.mesh, args.object_description, args.task_description, args.output, cfg
        )
    if args.action == "prepare":
        print(f"Evidence prepared: {result}")
    else:
        print(
            f"Selected {result.graspable_mask.sum()} / {len(result.scores)} vertices; output: {result.output_dir}"
        )
        if result.part_mask is not None:
            print(
                f"Target part: {result.part_mask.sum()} vertices; mesh: {result.output_dir / 'target_part.npz'}"
            )


if __name__ == "__main__":
    _main()
