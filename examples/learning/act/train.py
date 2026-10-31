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

"""Train visual ACT offline using EmbodiChain's supervised trainer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ._environment import task_identity


def _report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("mode") != "collect" or not report.get("dataset"):
        raise ValueError(f"{path} is not a completed collection report")
    if report.get("successes") != report.get("requested_episodes"):
        raise ValueError(f"{path} does not contain the requested demonstrations")
    if report.get("task_identity") != task_identity():
        raise ValueError(f"{path} was produced for a different task revision")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=19001)
    parser.add_argument("--updates", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train = _report(args.train_report)
    validation = _report(args.validation_report) if args.validation_report else None
    if not args.smoke and validation is None:
        parser.error("A separate --validation-report is required for full training")
    if validation is not None:
        if train["schema"] != validation["schema"]:
            raise ValueError("Training and validation schemas differ")
        if {r["seed"] for r in train["records"]} & {
            r["seed"] for r in validation["records"]
        }:
            raise ValueError("Training and validation scene seeds overlap")
    model = {} if not args.smoke else {"pretrained_backbone": False}
    config = {
        "factory": "embodichain_models.policy.act.sl_recipe:build_act_trainer",
        "task_identity": train["task_identity"],
        "scene_seeds": {
            "train": [r["seed"] for r in train["records"]],
            "validation": (
                [] if validation is None else [r["seed"] for r in validation["records"]]
            ),
        },
        "model": model,
        "data": {
            "train_path": train["dataset"],
            "validation_path": None if validation is None else validation["dataset"],
            "schema": train["schema"],
            "image_key": "observation.images.cam_high",
        },
        "objective": {"kl_weight": 10.0},
        "optimizer": {
            "learning_rate": 1e-4,
            "backbone_learning_rate": 1e-5,
            "weight_decay": 1e-4,
            "max_grad_norm": 1.0,
        },
        "trainer": {
            "total_updates": args.updates,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
            "output_dir": str(args.output.resolve()),
            "checkpoint_interval": args.eval_every,
            "validation_interval": args.eval_every if validation else 0,
        },
    }
    if args.resume is None:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "recipe.json").write_text(json.dumps(config, indent=2) + "\n")
    else:
        previous = json.loads((args.output / "recipe.json").read_text())
        if (
            previous.get("factory")
            == "embodichain_models.policies.act.sl_recipe:build_act_trainer"
        ):
            previous["factory"] = config["factory"]
        if previous != config:
            raise ValueError("Resume requires the original training recipe")
    from embodichain.learning.sl.workflow import train_from_config

    result = train_from_config(config, resume=str(args.resume) if args.resume else None)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
