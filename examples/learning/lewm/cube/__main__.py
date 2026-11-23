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

"""Prepare, collect, train and evaluate the native OGBench Cube showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._resources import _CubeResources


def _write_video(evaluation: Path, query: int, output: Path, fps: int) -> None:
    import imageio.v2 as imageio
    import numpy as np

    with np.load(evaluation / f"preview-query-{query:02d}.npz") as recording:
        frames, goal = recording["frames"], recording["goal"]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(np.concatenate((frame, goal), axis=1))


def main() -> None:
    """Launch the pinned Cube example."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Fetch and verify the official inputs"
    )
    collect = commands.add_parser("collect", help="Collect native RGB for training")
    train = commands.add_parser(
        "train", help="Train and export a validation-selected LeWM model"
    )
    evaluate = commands.add_parser(
        "eval", help="Evaluate the official 50 queries with native RGB"
    )
    demo = commands.add_parser(
        "demo", help="Encode a recorded native inference rollout as MP4"
    )
    for command in (prepare, collect, train, evaluate):
        command.add_argument(
            "--resource-root", type=Path, default=Path("outputs/lewm-cube/resources")
        )
    prepare.add_argument("--local-files-only", action="store_true")
    prepare.add_argument(
        "--reference-python",
        type=Path,
        help="EmbodiChain Python environment used to compile fresh MuJoCo reference assets",
    )
    for command in (collect, train):
        command.add_argument(
            "--profile", choices=("smoke", "reproduce"), default="reproduce"
        )
        command.add_argument("--kind", choices=("paired", "expert"), default="paired")
        command.add_argument("--output", type=Path, required=True)
    collect.add_argument("--workers", type=int, default=8)
    collect.add_argument("--gpu-ids", type=int, nargs="+", default=[0])
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--cache-mib", type=int, default=0)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--gpu-id", type=int, default=0)
    evaluate.add_argument("--preview-query", type=int, action="append", default=[])
    demo.add_argument("--evaluation", type=Path, required=True)
    demo.add_argument("--query", type=int, default=12)
    demo.add_argument("--output", type=Path, required=True)
    demo.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if args.command == "demo":
        if not 0 <= args.query < 50 or args.fps < 1:
            parser.error("demo requires a query index in [0, 49] and positive fps")
        _write_video(args.evaluation, args.query, args.output, args.fps)
        return

    resources = _CubeResources(args.resource_root.resolve())
    if args.command == "prepare":
        from ._resources import prepare_cube

        report = prepare_cube(
            resources.root,
            local_files_only=args.local_files_only,
            reference_python=args.reference_python,
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        return

    resources.require_prepared()
    if args.command in ("collect", "train"):
        preset = Path(__file__).with_name("configs") / f"{args.profile}.json"
        profile = json.loads(preset.read_text())
    if args.command == "collect":
        if args.kind == "paired":
            from ._capture import collect_pairs as collect_data
        else:
            from ._expert import collect_expert as collect_data
        report = collect_data(
            resources,
            args.output.resolve(),
            train_episodes=profile[f"{args.kind}_train_episodes"],
            validation_episodes=profile[f"{args.kind}_validation_episodes"],
            gpu_ids=args.gpu_ids,
            workers=args.workers,
        )
    elif args.command == "train":
        from ._training import train_expert, train_visual

        if args.kind == "paired":
            report = train_visual(
                resources,
                args.data,
                args.output.resolve(),
                updates=profile["visual_updates"],
                device=args.device,
            )
        else:
            report = train_expert(
                resources,
                args.data,
                args.output.resolve(),
                updates=profile["expert_updates"],
                device=args.device,
                cache_mib=args.cache_mib,
            )
    else:
        from .._official_native import evaluate_dexuni_cube

        report = evaluate_dexuni_cube(
            reference=resources.reference,
            queries=resources.queries,
            checkpoint=(args.checkpoint or resources.weights).resolve(),
            swm_root=resources.swm,
            output=args.output.resolve(),
            device=args.device,
            gpu_id=args.gpu_id,
            preview_queries=tuple(args.preview_query),
        )
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (args.output / "report.json").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
