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
import importlib
import json
from pathlib import Path
from typing import Any

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
    trainer = getattr(importlib.import_module(module), name)(config)
    if not isinstance(trainer, SupervisedTrainer):
        raise TypeError("SL recipe factory must return a SupervisedTrainer")
    if resume is not None:
        trainer.resume(resume)
    return trainer.fit()


def cli() -> None:
    """Train a supervised recipe from a YAML/JSON file and optional checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume")
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(args.config.read_text())
    if not isinstance(config, dict):
        raise ValueError("SL configuration must be a mapping")
    print(json.dumps(train_from_config(config, resume=args.resume), allow_nan=False))


if __name__ == "__main__":
    cli()
