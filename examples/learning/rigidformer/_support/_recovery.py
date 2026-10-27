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

"""Compare complete continuous and resumed native training checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from embodichain_models.dynamics.rigidformer import (
    build_rigidformer_trainer,
    export_rigidformer_checkpoint,
)
from ._evidence import evidence


def _exact(actual: Any, expected: Any) -> None:
    """Check every saved value, including optimizer, NumPy and device RNG state."""
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _exact(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _exact(a, b)
    else:
        assert actual == expected


def check_recovery(args: argparse.Namespace) -> None:
    """Run the public factory in three fresh destinations and compare full state."""
    if args.output.exists():
        raise FileExistsError(args.output)
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True)
    try:
        for name in ("full", "split", "resumed"):
            current = deepcopy(config)
            current.setdefault("trainer", {})["output_dir"] = str(args.output / name)
            trainer = build_rigidformer_trainer(current)
            if not 0 < args.stop_after < trainer.cfg.total_updates:
                raise ValueError("stop_after must be between zero and total_updates")
            if name == "resumed":
                trainer.resume(args.output / "split/last.pt")
            trainer.fit(stop_after=args.stop_after if name == "split" else None)
            del trainer
        # These are checkpoints produced by this process in fresh directories.
        full = torch.load(
            args.output / "full/last.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        resumed = torch.load(
            args.output / "resumed/last.pt",
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
        compared = ["learner", "training_states", "sampler", "rng"]
        for key in compared:
            _exact(full[key], resumed[key])
        tensors = {}
        for selection in ("raw", "ema"):
            path = export_rigidformer_checkpoint(
                args.output / "resumed/last.pt",
                args.output / f"{selection}.pt",
                weights=selection,
            )
            exported = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
            selected = (
                resumed["learner"]["model"]
                if selection == "raw"
                else resumed["training_states"]["ema"]["state"]["shadow"]
            )
            expected = {
                key.removeprefix("network."): value.to(
                    resumed["learner"]["model"][key].dtype
                )
                for key, value in selected.items()
            }
            _exact(exported["model_state_dict"], expected)
            tensors[selection] = len(expected)
        report = {
            **evidence(
                args.output / "resumed/last.pt",
                Path(config["data"]["root"]),
                config.get("model", {}).get("attention_backend", "sdpa"),
            ),
            "passed": True,
            "updates": resumed["learner"]["num_updates"],
            "stop_after": args.stop_after,
            "compared_exactly": compared,
            "exported_tensors": tensors,
        }
        (args.output / "summary.json").write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        )
        print(json.dumps(report, indent=2))
    except BaseException as exc:
        (args.output / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "message": str(exc)})
        )
        raise
