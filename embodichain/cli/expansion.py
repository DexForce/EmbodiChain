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

"""CLI for preflight, collection, recovery and verification of expansion jobs."""

from __future__ import annotations
import argparse
import json
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run a configured expansion operation and print its machine-readable report.

    Args:
        argv: Optional command arguments without the executable name.
    Returns:
        Zero for successful preflight/inspection or a completed collection;
        two when a bounded collection stopped before its target.
    """
    parser = argparse.ArgumentParser(prog="embodichain expansion")
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--job", required=True)
    command = commands.add_parser("resume")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--job")
    source.add_argument("--output")
    command.add_argument("--stage", choices=("all", "export"), default="all")
    command = commands.add_parser("inspect")
    command.add_argument("--output", required=True)
    command = commands.add_parser(
        "preview", help="Replay confirmed results with native ghost rendering"
    )
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        "--inpu",
        dest="inputs",
        action="append",
        help="Existing job output or preview-plan directory; repeat for groups",
    )
    source.add_argument(
        "--job",
        help="Compute an endpoint/interior preview from a Task Program expansion job",
    )
    command.add_argument(
        "--output",
        help="New collection directory for --job; never an existing-result input",
    )
    command.add_argument(
        "--interior-samples",
        type=int,
        default=5,
        help="Stratified interior parameter points, in addition to center and both extrema",
    )
    command.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze and validate the preview plan without collection/rendering",
    )
    command.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser after --job collection",
    )
    command.add_argument(
        "--reference",
        help=(
            "Job directory containing one confirmed original trajectory for solid "
            "playback. Defaults to the first loaded input recording as a comparison."
        ),
    )
    command.add_argument(
        "--robot-asset", help="Matching URDF override if recorded asset paths moved"
    )
    command.add_argument(
        "--eef-link",
        help="URDF link used for measured FK paths; inferred from the solver when unique",
    )
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument(
        "--port",
        type=int,
        default=8769,
        help="Browser preview port; 0 selects an available port",
    )
    command.add_argument("--max-episodes", type=int, default=32)
    command.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Exit after these seconds; 0 waits for Ctrl+C",
    )
    command.add_argument(
        "--open",
        action="store_true",
        help="Open a browser after the first complete render",
    )
    args = parser.parse_args(argv)
    if args.operation == "preview":
        import math
        from pathlib import Path

        if (
            not 0 <= args.port <= 65535
            or args.gpu < 0
            or not math.isfinite(args.duration)
            or args.duration < 0
        ):
            parser.error("invalid preview port, GPU or duration")
        from embodichain.lab.expansion._preview_data import load_preview
        from embodichain.lab.expansion._preview_server import serve_preview

        try:
            if args.open and args.no_open:
                parser.error("--open and --no-open are mutually exclusive")
            coverage = None
            if args.job:
                if args.reference:
                    parser.error(
                        "--reference is only valid with existing --input results"
                    )
                from embodichain.lab.expansion._preview_job import (
                    plan_preview,
                    collect_preview,
                )

                coverage = plan_preview(
                    Path(args.job),
                    Path(args.output) if args.output else None,
                    interior=args.interior_samples,
                    max_episodes=args.max_episodes,
                )
                print(
                    json.dumps(
                        {
                            "preview_plan": str(
                                Path(coverage["output"]) / "preview-plan.json"
                            ),
                            "parameter_points": len(coverage["records"]),
                            "planned_attempts": coverage["planned_attempts"],
                        }
                    ),
                    flush=True,
                )
                if args.plan_only:
                    return 0
                import torch

                if args.gpu >= torch.cuda.device_count():
                    parser.error(
                        "preview GPU is outside CUDA_VISIBLE_DEVICES; use a visible device ordinal"
                    )
                coverage = collect_preview(coverage)
                print(
                    json.dumps(
                        {
                            "confirmed": coverage["confirmed"],
                            "planned_attempts": coverage["planned_attempts"],
                            "coverage_complete": coverage["coverage_complete"],
                            "output": coverage["output"],
                        }
                    ),
                    flush=True,
                )
                if not coverage["confirmed"]:
                    print(
                        "No qualified trajectories; inspect preview-plan.json and sample logs. No boundary was replaced.",
                        flush=True,
                    )
                    return 2
                inputs = [Path(coverage["output"])]
            else:
                if args.output or args.plan_only:
                    parser.error(
                        "--output and --plan-only require --job; use --input for existing results"
                    )
                inputs = [Path(p) for p in args.inputs]
            data = load_preview(
                inputs,
                Path(args.reference) if args.reference else None,
                robot_asset=Path(args.robot_asset) if args.robot_asset else None,
                eef_link=args.eef_link,
                max_episodes=args.max_episodes,
            )
            serve_preview(
                data,
                port=args.port,
                gpu=args.gpu,
                open_browser=(bool(args.job) or args.open) and not args.no_open,
                duration=args.duration,
            )
        except KeyboardInterrupt:
            return 0
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        return 2 if coverage and not coverage["coverage_complete"] else 0
    from embodichain.lab.expansion.configured import (
        load_expansion_job,
    )
    from embodichain.lab.expansion.workflow import (
        inspect_expansion_job,
    )

    if args.operation == "inspect":
        result = inspect_expansion_job(args.output)
    elif args.operation == "resume" and args.output is not None:
        if args.stage != "export":
            parser.error("--output recovery requires --stage export")
        from embodichain.lab.expansion.workflow import resume_expansion_export

        result = resume_expansion_export(args.output)
    else:
        if args.operation == "resume" and args.stage != "all":
            parser.error("use --output with --stage export")
        cfg = load_expansion_job(args.job)
        from embodichain.lab.expansion import ExpansionEngine

        engine = ExpansionEngine.from_job(cfg)
        result = getattr(engine, args.operation)()
    printable = {
        key: value for key, value in result.items() if key != "resolved_deployment"
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    if args.operation in {"run", "resume"} and not result.get(
        "stage_complete", result["complete"]
    ):
        raise SystemExit(2)
    return 0
