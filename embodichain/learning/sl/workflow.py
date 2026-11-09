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

"""Lazy recipe-based CLI for standalone supervised learning."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import copy
import importlib
import json
import os
from pathlib import Path
from typing import Any

import torch.distributed as dist

from .distributed import _DistributedContext, distributed_run
from .trainer import SupervisedTrainer

__all__ = ["train_from_config", "cli"]


def train_from_config(
    config: dict[str, Any], *, resume: str | None = None
) -> dict[str, Any]:
    """Build a recipe and run the common offline trainer.

    Args:
        config: JSON/YAML recipe with ``factory: module:function`` and model,
            data, objective, optimizer, trainer and evaluation recipe settings.
            The factory receives this mapping and returns a SupervisedTrainer.
        resume: Optional trusted training checkpoint.

    Returns:
        The trainer's committed update and artifact report.
    """
    factory = config.get("factory", "")
    module, separator, name = factory.partition(":")
    if not separator or not module or not name:
        raise ValueError("SL configuration requires factory: module:function")
    prepared = copy.deepcopy(config)
    device = prepared.get("trainer", {}).get("device", "cpu")
    trainer: SupervisedTrainer | None = None
    with distributed_run(device) as resolved:
        context = _DistributedContext(resolved)
        if context.enabled:
            prepared["trainer"] = {
                **prepared.get("trainer", {}),
                "device": str(resolved),
            }

        def build() -> SupervisedTrainer:
            nonlocal trainer
            candidate = getattr(importlib.import_module(module), name)(prepared)
            if not isinstance(candidate, SupervisedTrainer):
                raise TypeError("SL recipe factory must return a SupervisedTrainer")
            trainer = candidate
            return candidate

        try:
            trainer = context.run(build, "recipe factory")
            if resume is not None:
                trainer.resume(resume)
            return trainer.fit()
        finally:
            # Factories own acquisition, but this entry point owns the returned
            # source's consumption lifetime, including background prefetch work.
            if trainer is not None:
                close = getattr(trainer.source, "close", None)
                if close is not None:
                    close()


def cli(argv: Sequence[str] | None = None) -> None:
    """Train a supervised recipe from a YAML/JSON file and optional checkpoint.

    Args:
        argv: Explicit CLI arguments, or the current process arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("SL configuration must be a mapping")
    rank = (
        dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    )
    result = train_from_config(config, resume=args.resume)
    if rank == 0:
        print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    cli()
