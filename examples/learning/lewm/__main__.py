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

"""Prepare, train and evaluate the LeWM visual dynamics showcase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import torch

from embodichain_models.dynamics.lewm import (
    LeWM,
    LeWMCfg,
    LeWMCEMCfg,
    build_lewm_trainer,
    export_lewm,
)


def _write_report(report: dict, path: str | None) -> None:
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, flush=True)


def _train(args: argparse.Namespace) -> dict:
    if args.updates < 1 or args.batch_size < 2 or args.interval < 1:
        raise ValueError(
            "training requires positive updates/interval and batch size at least two"
        )
    with h5py.File(args.train_data, "r") as file:
        metadata = json.loads(file.attrs["metadata"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    interval = min(args.interval, args.updates)
    config = {
        "factory": "embodichain_models.dynamics.lewm.sl_recipe:build_lewm_trainer",
        "model": {"action_dim": metadata["action_dim"]},
        "data": {
            "train_path": str(Path(args.train_data).resolve()),
            "validation_path": str(Path(args.validation_data).resolve()),
        },
        "objective": {"sigreg_weight": 0.09, "knots": 17, "num_proj": 1024},
        "optimizer": {
            "learning_rate": 5e-5,
            "weight_decay": 1e-3,
            "max_grad_norm": 1.0,
            "amp_bfloat16": args.bf16,
        },
        "trainer": {
            "total_updates": args.updates,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
            "output_dir": str(output.resolve()),
            "checkpoint_interval": interval,
            "validation_interval": interval,
        },
    }
    trainer = build_lewm_trainer(config)
    if args.resume:
        trainer.resume(args.resume)
    recipe = output / "recipe.json"
    recipe.write_text(json.dumps(config, indent=2) + "\n")
    result = trainer.fit()
    candidates = []
    for line in (output / "metrics.jsonl").read_text().splitlines():
        record = json.loads(line)
        if "validation/prediction_loss" in record:
            checkpoint = output / f"step_{record['num_updates']}.pt"
            if checkpoint.is_file():
                candidates.append(
                    (
                        record["validation/prediction_loss"],
                        record["num_updates"],
                        checkpoint,
                    )
                )
    if not candidates:
        raise RuntimeError("No validation-aligned checkpoint was produced")
    loss, step, checkpoint = min(candidates)
    # Training checkpoints contain optimizer/RNG Python objects; only explicitly
    # selected local training checkpoints are deserialized here.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = LeWM(LeWMCfg(**state["signature"]["model"]))
    model.load_state_dict(state["learner"]["model"], strict=True)
    export_path = output / "export"
    export_lewm(
        export_path,
        model,
        {
            "data_contract": state["signature"]["data_contract"],
            "training_source": state["source_id"],
            "validation_source": state["signature"]["validation_source"],
            "train_seeds": state["signature"]["train_seeds"],
            "validation_seeds": state["signature"]["validation_seeds"],
            "selected_step": step,
            "validation_prediction_loss": loss,
        },
    )
    return {
        "training": result,
        "selected_step": step,
        "validation_prediction_loss": loss,
        "checkpoint": str(checkpoint),
        "export": str(export_path),
        "recipe": str(recipe),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Download pinned upstream reference resources"
    )
    prepare.add_argument("--resource-root", required=True)
    prepare.add_argument("--local-files-only", action="store_true")
    reference = commands.add_parser(
        "reference", help="Compare official pretrained weights"
    )
    reference.add_argument("--resource-root", required=True)
    reference.add_argument("--output")
    collect = commands.add_parser(
        "collect", help="Collect a simulator image/action split"
    )
    collect.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    collect.add_argument("--episodes", type=int)
    collect.add_argument("--output", required=True)
    collect.add_argument("--device", default="cuda:0")
    train = commands.add_parser(
        "train", help="Train and export the best validation checkpoint"
    )
    train.add_argument("--train-data", required=True)
    train.add_argument("--validation-data", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--updates", type=int, default=10_000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--interval", type=int, default=500)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--resume")
    offline = commands.add_parser(
        "eval", help="Evaluate held-out multi-step latent predictions"
    )
    offline.add_argument("--model", required=True)
    offline.add_argument("--data", required=True)
    offline.add_argument("--split", choices=("validation", "test"), default="test")
    offline.add_argument("--device", default="cuda:0")
    offline.add_argument("--batch-size", type=int, default=16)
    offline.add_argument("--output")
    sim = commands.add_parser("sim-eval", help="Compare visual MPC with random control")
    sim.add_argument("--model", required=True)
    sim.add_argument("--device", default="cuda:0")
    sim.add_argument("--episodes", type=int, default=50)
    sim.add_argument("--start-index", type=int, default=0)
    sim.add_argument("--split", choices=("validation", "test"), default="test")
    sim.add_argument("--samples", type=int, default=300)
    sim.add_argument("--iterations", type=int, default=30)
    sim.add_argument("--topk", type=int, default=30)
    sim.add_argument("--candidate-batch-size", type=int, default=300)
    sim.add_argument("--output")
    merge = commands.add_parser(
        "merge-sim", help="Combine disjoint paired simulation shards"
    )
    merge.add_argument("--reports", nargs="+", required=True)
    merge.add_argument("--episodes", type=int, default=50)
    merge.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command in {"prepare", "reference"}:
        from ._reference import compare, prepare

        report = (
            prepare(args.resource_root, args.local_files_only)
            if args.command == "prepare"
            else compare(args.resource_root)
        )
        _write_report(report, getattr(args, "output", None))
    elif args.command == "collect":
        from ._sim import collect

        count = (
            args.episodes
            if args.episodes is not None
            else {"train": 512, "validation": 64, "test": 50}[args.split]
        )
        _write_report(collect(args.output, args.split, count, args.device), None)
    elif args.command == "train":
        _write_report(_train(args), str(Path(args.output) / "summary.json"))
    else:
        from ._evaluation import evaluate_offline, evaluate_sim, merge_sim_reports

        if args.command == "eval":
            report = evaluate_offline(
                args.model, args.data, args.split, args.device, args.batch_size
            )
        elif args.command == "sim-eval":
            cfg = LeWMCEMCfg(
                num_samples=args.samples,
                iterations=args.iterations,
                topk=args.topk,
                candidate_batch_size=args.candidate_batch_size,
            )
            report = evaluate_sim(
                args.model,
                args.device,
                args.episodes,
                cfg,
                args.split,
                args.start_index,
            )
        else:
            report = merge_sim_reports(args.reports, args.episodes)
        _write_report(report, args.output)


if __name__ == "__main__":
    main()
