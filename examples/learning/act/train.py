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

"""Train the official LeRobot ACT and retain EmbodiChain task/data provenance."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
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


def _policies(output: Path) -> list[Path]:
    return sorted(
        (
            p / "pretrained_model"
            for p in (output / "lerobot/checkpoints").glob("*")
            if p.name.isdigit() and (p / "pretrained_model/model.safetensors").is_file()
        ),
        key=lambda p: int(p.parent.name),
    )


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
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume", type=Path, help="Native checkpoint's pretrained_model directory"
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("This example is qualified on Python 3.12 with LeRobot 0.6.1")
    if min(args.updates, args.batch_size, args.eval_every) < 1:
        parser.error("updates, batch-size and eval-every must be positive")
    train = _report(args.train_report)
    validation = _report(args.validation_report) if args.validation_report else None
    if not args.smoke and (validation is None or train.get("smoke", False)):
        parser.error(
            "Full training requires qualified training and separate validation reports"
        )
    if validation is not None:
        if train["schema"] != validation["schema"]:
            raise ValueError("Training and validation schemas differ")
        if {r["seed"] for r in train["records"]} & {
            r["seed"] for r in validation["records"]
        }:
            raise ValueError("Training and validation scene seeds overlap")
        if not args.smoke and validation.get("smoke", False):
            raise ValueError("Full training requires a qualified validation corpus")
    from embodichain_models.policy.act import (
        build_act_train_config,
        dataset_fingerprint,
        load_act_policy,
        policy_fingerprint,
    )
    import torch

    # The wrapper performs CPU image conversion during post-training validation.
    # Bound its pool as well as the upstream subprocess's pool on large hosts.
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))

    output = args.output.resolve()
    cfg = build_act_train_config(
        train["dataset"],
        output / "lerobot",
        schema=train["schema"],
        device=args.device,
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        save_freq=args.eval_every,
        num_workers=(
            args.num_workers
            if args.num_workers is not None
            else (0 if args.smoke else 4)
        ),
        smoke=args.smoke,
    )
    native = cfg.to_dict()
    recipe = {
        "recipe": "lerobot-act-v1",
        "schema": train["schema"],
        "task_identity": train["task_identity"],
        "image_key": "observation.images.cam_high",
        "smoke": args.smoke,
        "backbone_initialization": cfg.policy.pretrained_backbone_weights,
        "scene_seeds": {
            "train": [r["seed"] for r in train["records"]],
            "validation": (
                [] if validation is None else [r["seed"] for r in validation["records"]]
            ),
        },
        "datasets": {
            "train": {
                "path": train["dataset"],
                "sha256": dataset_fingerprint(train["dataset"]),
            },
            "validation": (
                None
                if validation is None
                else {
                    "path": validation["dataset"],
                    "sha256": dataset_fingerprint(validation["dataset"]),
                }
            ),
        },
        "training_config": native,
        "versions": {
            p: version(p)
            for p in (
                "lerobot",
                "torch",
                "torchvision",
                "torchcodec",
                "dexsim-engine",
                "numpy",
            )
        },
    }
    if args.resume is None:
        output.mkdir(parents=True, exist_ok=False)
        resume_step = 0
    else:
        previous = json.loads((output / "recipe.json").read_text())
        original_steps = previous["training_config"].pop("steps")
        compared = json.loads(json.dumps(recipe))
        compared["training_config"].pop("steps")
        if previous != compared or args.updates < original_steps:
            raise ValueError(
                "Resume requires the same recipe, source contents and a nondecreasing update budget"
            )
        resume = args.resume.resolve()
        checkpoint_root = (output / "lerobot/checkpoints").resolve()
        if (
            not resume.is_relative_to(checkpoint_root)
            or resume.parent != (checkpoint_root / "last").resolve()
        ):
            raise ValueError("Resume must use this run's latest native checkpoint")
        if not (resume.parent / "training_state").is_dir():
            raise ValueError("Resume requires a native checkpoint with training_state")
        resume_step = json.loads(
            (resume.parent / "training_state/training_step.json").read_text()
        )["step"]
        if resume_step > args.updates:
            raise ValueError(
                "Native checkpoint already exceeds the requested update budget"
            )
        saved_config = json.loads((resume / "train_config.json").read_text())
        for key in ("dataset", "seed", "batch_size", "num_workers"):
            if saved_config[key] != native[key]:
                raise ValueError(f"Native resume configuration differs: {key}")
        for key, value in native["policy"].items():
            if key != "pretrained_path" and saved_config["policy"].get(key) != value:
                raise ValueError(f"Native resume policy configuration differs: {key}")
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    config_path = output / "lerobot_config.json"
    config_path.write_text(json.dumps(native, indent=2) + "\n")
    if args.resume is None:
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={config_path}",
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={resume / 'train_config.json'}",
            "--resume=true",
            f"--steps={args.updates}",
            f"--output_dir={output / 'lerobot'}",
        ]
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env["ACCELERATE_MIXED_PRECISION"] = "no"
    started = time.monotonic()
    if resume_step < args.updates:
        subprocess.run(command, check=True, env=env)
    for item in recipe["datasets"].values():
        if item is not None and dataset_fingerprint(item["path"]) != item["sha256"]:
            raise ValueError("Recorded source contents changed during training")
    policies = _policies(output)
    if not policies:
        raise RuntimeError("The official trainer did not publish a native checkpoint")
    for policy in policies:
        manifest = {
            **recipe,
            "update": int(policy.parent.name),
            "policy_sha256": policy_fingerprint(policy),
        }
        (policy / "embodichain_act.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    if validation is not None:
        diagnostics = []
        for policy in policies:
            loaded = load_act_policy(policy, args.device)
            diagnostics.append(
                {
                    "update": int(policy.parent.name),
                    **loaded.evaluate_dataset(validation["dataset"]),
                }
            )
            del loaded
        (output / "offline_validation.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
    report = {
        "trainer": "lerobot.scripts.lerobot_train",
        "wall_seconds": time.monotonic() - started,
        "resume_step": resume_step,
        "invocation_updates": args.updates - resume_step,
        "last_policy": str(policies[-1]),
        "updates": args.updates,
        "native_checkpoints": len(policies),
        "versions": recipe["versions"],
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
