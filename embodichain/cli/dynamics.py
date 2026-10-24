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

"""Lazy commands for neural dynamics collection, learning and evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

__all__: list[str] = []


def _read_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Configuration file does not exist: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text())
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        value = yaml.safe_load(path.read_text())
    else:
        raise ValueError("Dynamics configuration must be .json, .yaml or .yml")
    if not isinstance(value, dict):
        raise ValueError("Dynamics configuration must contain a mapping")
    return value


def _path(value: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _resolve_paths(
    config: dict[str, Any], base: Path, operation: str
) -> dict[str, Any]:
    config = dict(config)
    inputs = {
        "dataset": "directory",
        "statistics": "file",
        "sample_order": "file",
        "resume": "file",
        "fine_tune": "directory",
    }
    if operation == "eval":
        inputs["model"] = "directory"
    if operation == "generate":
        inputs.pop("resume")  # generation resume is a Boolean, not a checkpoint.
    for name, kind in inputs.items():
        if name in config and config[name] is not None:
            path = _path(config[name], base)
            if not (path.is_file() if kind == "file" else path.is_dir()):
                raise ValueError(f"{name} {kind} does not exist: {path}")
            config[name] = str(path)
    for name in ("output_dir", "output"):
        if name in config:
            config[name] = str(_path(config[name], base))
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch one dynamics operation; offline paths never import simulation."""
    parser = argparse.ArgumentParser(prog="embodichain dynamics", description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name, help_text in (
        ("generate", "Generate committed teacher trajectory shards."),
        ("train", "Train, fine-tune or resume a NeRD model."),
        ("eval", "Evaluate an exported model on held-out trajectories."),
    ):
        child = subparsers.add_parser(name, help=help_text)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--device", help="Override the configured CPU/CUDA device.")
        if name in {"generate", "train"}:
            child.add_argument(
                "--seed", type=int, help="Override the collection/training seed."
            )
            child.add_argument(
                "--output-dir",
                type=Path,
                help="Override output directory (relative to current directory).",
            )
        if name == "generate":
            child.add_argument(
                "--resume",
                action="store_true",
                default=None,
                help="Continue from committed trajectory shards.",
            )
        elif name == "train":
            restart = child.add_mutually_exclusive_group()
            restart.add_argument(
                "--resume",
                type=Path,
                help="Restore an identical committed training run.",
            )
            restart.add_argument(
                "--fine-tune",
                type=Path,
                help="Initialize weights/statistics from a standard model export.",
            )
            child.add_argument(
                "--statistics",
                type=Path,
                help="Use a shared frozen normalization artifact.",
            )
            child.add_argument(
                "--sample-order",
                type=Path,
                help="Shared batch-index tensor (.pt) or memory-mapped integer matrix (.npy).",
            )
        else:
            child.add_argument(
                "--model", type=Path, help="Override the exported model directory."
            )
            child.add_argument(
                "--output",
                type=Path,
                help="Override JSON report path (relative to current directory).",
            )
    export = subparsers.add_parser(
        "export", help="Export portable model weights and metadata."
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    convert = subparsers.add_parser(
        "convert-checkpoint",
        help="Convert an explicitly trusted official Python-object checkpoint.",
    )
    convert.add_argument("--checkpoint", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON deployment manifest with codec and dt.",
    )
    convert.add_argument(
        "--trust-pickle",
        action="store_true",
        required=True,
        help="Explicitly trust the source pickle, which can execute Python code.",
    )
    args = parser.parse_args(argv)
    try:
        if args.operation in {"export", "convert-checkpoint"}:
            checkpoint = args.checkpoint.expanduser().resolve()
            if not checkpoint.is_file():
                raise ValueError(f"Checkpoint does not exist: {checkpoint}")
            output = args.output.expanduser().resolve()
            if args.operation == "export":
                from embodichain.learning.dynamics.workflow import run_export

                result = run_export(checkpoint, output)
            else:
                manifest = json.loads(args.manifest.expanduser().read_text())
                if (
                    not isinstance(manifest, dict)
                    or not isinstance(manifest.get("codec"), dict)
                    or "dt" not in manifest
                ):
                    raise ValueError(
                        "Conversion manifest requires codec mapping and dt"
                    )
                from embodichain.learning.models.nerd import convert_legacy_checkpoint

                convert_legacy_checkpoint(
                    checkpoint, output, manifest, trust_pickle=args.trust_pickle
                )
                result = {"output": str(output), "qualification": "unqualified"}
        else:
            path = args.config.expanduser().resolve()
            config = _read_config(path)
            if args.device is not None:
                if args.operation == "train":
                    config["train"] = {**config.get("train", {}), "device": args.device}
                else:
                    config["device"] = args.device
            if getattr(args, "seed", None) is not None:
                if args.operation == "train":
                    config["train"] = {**config.get("train", {}), "seed": args.seed}
                else:
                    config["seed"] = args.seed
            if getattr(args, "output_dir", None) is not None:
                config["output_dir"] = str(args.output_dir.expanduser().resolve())
            if getattr(args, "output", None) is not None:
                config["output"] = str(args.output.expanduser().resolve())
            for field in ("resume", "fine_tune", "statistics", "sample_order", "model"):
                value = getattr(args, field, None)
                if value is not None:
                    config[field] = (
                        str(value.expanduser().resolve())
                        if isinstance(value, Path)
                        else value
                    )
            config = _resolve_paths(config, path.parent, args.operation)
            if args.operation == "generate":
                from embodichain.data_pipeline.dynamics.generation import (
                    generate_dataset,
                )

                result = generate_dataset(config)
            elif args.operation == "train":
                from embodichain.learning.dynamics.workflow import run_train

                result = run_train(config)
            else:
                mode = config.pop("mode", "offline")
                if mode == "offline":
                    from embodichain.learning.dynamics.workflow import run_eval

                    result = run_eval(config)
                else:
                    raise ValueError("This workflow supports offline evaluation only")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
