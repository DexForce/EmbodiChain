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
    if (
        args.updates < 1
        or args.batch_size < 2
        or args.interval < 1
        or args.action_block < 1
        or args.cache_mib < 0
    ):
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
        "model": {
            "action_dim": metadata["action_dim"] * args.action_block,
            "action_block": args.action_block,
        },
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
    if args.cache_mib:
        config["data"].update(
            cache_device=args.device, cache_max_bytes=args.cache_mib * 1024**2
        )
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
    reference.add_argument(
        "--cube-weights", help="Verify the pinned official Cube checkpoint"
    )
    reference.add_argument(
        "--cube-swm-root", help="Use the original Cube-compatible upstream checkout"
    )
    reference.add_argument("--output")
    official = commands.add_parser(
        "official-reference", help="Audit the pinned MuJoCo Cube task"
    )
    official.add_argument("--swm-root", type=Path, required=True)
    official.add_argument("--ogbench-root", type=Path, required=True)
    official.add_argument("--output", type=Path, required=True)
    official.add_argument("--steps", type=int, default=100)
    official.add_argument("--compare-cem", action="store_true")
    native = commands.add_parser(
        "official-kinematics", help="Check the Cube robot in DexSim"
    )
    native.add_argument("--reference", type=Path, required=True)
    native.add_argument("--device", default="cuda:0")
    official_eval = commands.add_parser(
        "official-eval-reference", help="Run the original 50-query MuJoCo evaluator"
    )
    for name in ("swm-root", "ogbench-root", "data", "queries", "weights", "output"):
        official_eval.add_argument(f"--{name}", type=Path, required=True)
    official_eval.add_argument("--device", default="cuda:0")
    native_eval = commands.add_parser(
        "official-eval-dexsim",
        help="Evaluate the Cube prototype on the official queries",
    )
    for name in ("reference", "queries", "weights", "output"):
        native_eval.add_argument(f"--{name}", type=Path, required=True)
    native_eval.add_argument("--device", default="cuda:0")
    native_eval.add_argument("--worker-devices", nargs=2)
    native_eval.add_argument("--reference-planner-root", type=Path)
    native_eval.add_argument(
        "--replay", type=Path, help="Replay a reference trace as a physics diagnostic"
    )
    collect = commands.add_parser(
        "collect", help="Collect a simulator image/action split"
    )
    collect.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    collect.add_argument("--episodes", type=int)
    collect.add_argument("--start-index", type=int, default=0)
    collect.add_argument(
        "--render-mode", choices=("default", "legacy", "stable"), default="default"
    )
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
    train.add_argument("--action-block", type=int, default=1)
    train.add_argument("--cache-mib", type=int, default=0)
    train.add_argument("--resume")
    offline = commands.add_parser(
        "eval", help="Evaluate held-out multi-step latent predictions"
    )
    offline.add_argument("--model", required=True)
    offline.add_argument("--data", required=True)
    offline.add_argument("--split", choices=("validation", "test"), default="test")
    offline.add_argument("--device", default="cuda:0")
    offline.add_argument("--batch-size", type=int, default=16)
    offline.add_argument("--cache-mib", type=int, default=0)
    offline.add_argument("--allow-render-mismatch", action="store_true")
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
    sim.add_argument("--horizon", type=int, default=5)
    sim.add_argument(
        "--sampling-distribution", choices=("bounds", "training"), default="bounds"
    )
    sim.add_argument("--action-selection", choices=("best", "mean"), default="best")
    sim.add_argument("--goal-render-frames", type=int, default=32)
    sim.add_argument(
        "--render-mode", choices=("default", "legacy", "stable"), default="default"
    )
    sim.add_argument("--allow-render-mismatch", action="store_true")
    sim.add_argument("--trace", help="Write per-step observations, actions and states")
    sim.add_argument("--output")
    merge = commands.add_parser(
        "merge-sim", help="Combine disjoint paired simulation shards"
    )
    merge.add_argument("--reports", nargs="+", required=True)
    merge.add_argument("--episodes", type=int, default=50)
    merge.add_argument("--start-index", type=int, default=0)
    merge.add_argument("--output", required=True)
    replay = commands.add_parser("check-trace", help="Recompute recorded MPC decisions")
    replay.add_argument("--model", required=True)
    replay.add_argument("--trace", required=True)
    replay.add_argument("--device", default="cuda:0")
    replay.add_argument("--max-plans", type=int, default=10)
    replay.add_argument("--output")
    diagnose = commands.add_parser(
        "diagnose-candidates",
        help="Check visual action rankings against physical outcomes",
    )
    diagnose.add_argument("--model", required=True)
    diagnose.add_argument("--data", required=True)
    diagnose.add_argument("--seeds", nargs="+", type=int, required=True)
    diagnose.add_argument("--device", default="cuda:0")
    diagnose.add_argument("--candidates", type=int, default=32)
    diagnose.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 20])
    diagnose.add_argument("--goal-render-frames", type=int, default=32)
    diagnose.add_argument("--trace", required=True)
    diagnose.add_argument("--output", required=True)
    rescore = commands.add_parser(
        "rescore-candidates",
        help="Re-score identical physical rollouts after predictor fine-tuning",
    )
    rescore.add_argument("--model", required=True)
    rescore.add_argument("--reference-model", required=True)
    rescore.add_argument("--trace", required=True)
    rescore.add_argument("--device", default="cuda:0")
    rescore.add_argument("--output", required=True)
    finetune = commands.add_parser(
        "finetune-predictor",
        help="Fit multi-step dynamics with a frozen visual encoder",
    )
    finetune.add_argument("--model", required=True)
    finetune.add_argument("--train-data", required=True)
    finetune.add_argument("--validation-data", required=True)
    finetune.add_argument("--output", required=True)
    finetune.add_argument("--device", default="cuda:0")
    finetune.add_argument("--updates", type=int, default=2000)
    finetune.add_argument("--horizon", type=int, default=5)
    finetune.add_argument("--batch-size", type=int, default=128)
    finetune.add_argument("--interval", type=int, default=200)
    finetune.add_argument("--seed", type=int, default=0)
    finetune.add_argument("--successful-tail-fraction", type=float, default=0.0)
    goals = commands.add_parser(
        "prepare-goals", help="Collect nontrivial measured future-image goals"
    )
    goals.add_argument("--output", required=True)
    goals.add_argument("--device", default="cuda:0")
    goals.add_argument("--episodes", type=int, default=20)
    goals.add_argument("--start-index", type=int, default=100)
    goals.add_argument("--max-proposals", type=int, default=200)
    goals.add_argument("--goal-offset", type=int, default=25)
    goals.add_argument("--history-steps", type=int, default=10)
    goals.add_argument("--split", choices=("validation", "test"), default="validation")
    reachable = commands.add_parser(
        "reachable-eval", help="Compare model timing on measured reachable future goals"
    )
    reachable.add_argument(
        "--models", nargs="+", required=True, help="name=export pairs"
    )
    reachable.add_argument("--pairs", required=True)
    reachable.add_argument("--output", required=True)
    reachable.add_argument("--device", default="cuda:0")
    reachable.add_argument("--start-index", type=int, default=0)
    reachable.add_argument("--episodes", type=int, default=5)
    reachable.add_argument("--horizon", type=int, default=5)
    reachable.add_argument("--receding-horizon", type=int, default=1)
    reachable.add_argument("--max-steps", type=int, default=50)
    reachable.add_argument(
        "--sampling-distribution", choices=("bounds", "training"), default="bounds"
    )
    reachable.add_argument(
        "--action-selection", choices=("best", "mean"), default="best"
    )
    args = parser.parse_args()
    if args.command == "reachable-eval":
        from ._reachable import evaluate_reachable

        exports = dict(value.split("=", 1) for value in args.models)
        _write_report(
            evaluate_reachable(
                exports,
                args.pairs,
                args.output,
                args.device,
                start_index=args.start_index,
                episodes=args.episodes,
                horizon=args.horizon,
                receding_horizon=args.receding_horizon,
                max_steps=args.max_steps,
                sampling_distribution=args.sampling_distribution,
                action_selection=args.action_selection,
            ),
            str(Path(args.output) / "summary.json"),
        )
    elif args.command == "prepare-goals":
        from ._reachable import prepare_goals

        _write_report(
            prepare_goals(
                args.output,
                args.device,
                episodes=args.episodes,
                start_index=args.start_index,
                max_proposals=args.max_proposals,
                goal_offset=args.goal_offset,
                history_steps=args.history_steps,
                split=args.split,
            ),
            args.output + ".json",
        )
    elif args.command == "official-reference":
        from ._official_cube import _compare_cem, audit_reference

        report = audit_reference(
            args.output, args.swm_root, args.ogbench_root, steps=args.steps
        )
        if args.compare_cem:
            report["cem"] = _compare_cem(args.output)
        _write_report(report, str(args.output / "reference-report.json"))
    elif args.command == "official-eval-dexsim":
        from ._official_eval import evaluate_dexsim

        _write_report(
            evaluate_dexsim(
                reference=args.reference,
                queries=args.queries,
                weights=args.weights,
                output=args.output,
                device=args.device,
                replay=args.replay,
                worker_devices=args.worker_devices,
                reference_planner_root=args.reference_planner_root,
            ),
            None,
        )
    elif args.command == "official-eval-reference":
        from ._official_eval import evaluate_reference

        _write_report(
            evaluate_reference(
                swm_root=args.swm_root,
                ogbench_root=args.ogbench_root,
                data=args.data,
                queries=args.queries,
                weights=args.weights,
                output=args.output,
                device=args.device,
            ),
            None,
        )
    elif args.command == "official-kinematics":
        from ._official_dexsim import audit_kinematics

        _write_report(audit_kinematics(args.reference, device=args.device), None)
    elif args.command in {"prepare", "reference"}:
        from ._reference import compare, prepare

        report = (
            prepare(args.resource_root, args.local_files_only)
            if args.command == "prepare"
            else compare(
                args.resource_root,
                cube_weights=args.cube_weights,
                cube_swm_root=args.cube_swm_root,
            )
        )
        _write_report(report, getattr(args, "output", None))
    elif args.command == "collect":
        from ._sim import collect

        count = (
            args.episodes
            if args.episodes is not None
            else {"train": 512, "validation": 64, "test": 50}[args.split]
        )
        _write_report(
            collect(
                args.output,
                args.split,
                count,
                args.device,
                args.render_mode,
                args.start_index,
            ),
            None,
        )
    elif args.command == "train":
        _write_report(_train(args), str(Path(args.output) / "summary.json"))
    elif args.command == "check-trace":
        from ._trace import check_trace

        _write_report(
            check_trace(args.model, args.trace, args.device, args.max_plans),
            args.output,
        )
    elif args.command == "diagnose-candidates":
        from ._diagnostics import diagnose_candidates

        _write_report(
            diagnose_candidates(
                args.model,
                args.data,
                args.device,
                args.seeds,
                args.trace,
                candidates=args.candidates,
                horizons=tuple(args.horizons),
                goal_render_frames=args.goal_render_frames,
            ),
            args.output,
        )
    elif args.command == "finetune-predictor":
        from ._finetune import finetune_predictor

        _write_report(
            finetune_predictor(
                args.model,
                args.train_data,
                args.validation_data,
                args.output,
                args.device,
                updates=args.updates,
                horizon=args.horizon,
                batch_size=args.batch_size,
                interval=args.interval,
                seed=args.seed,
                successful_tail_fraction=args.successful_tail_fraction,
            ),
            str(Path(args.output) / "summary.json"),
        )
    elif args.command == "rescore-candidates":
        from ._diagnostics import rescore_candidates

        _write_report(
            rescore_candidates(
                args.model, args.reference_model, args.trace, args.device
            ),
            args.output,
        )
    else:
        from ._evaluation import evaluate_offline, evaluate_sim, merge_sim_reports

        if args.command == "eval":
            report = evaluate_offline(
                args.model,
                args.data,
                args.split,
                args.device,
                args.batch_size,
                allow_render_mismatch=args.allow_render_mismatch,
                cache_max_bytes=args.cache_mib * 1024**2,
            )
        elif args.command == "sim-eval":
            cfg = LeWMCEMCfg(
                horizon=args.horizon,
                num_samples=args.samples,
                iterations=args.iterations,
                topk=args.topk,
                candidate_batch_size=args.candidate_batch_size,
                sampling_distribution=args.sampling_distribution,
                action_selection=args.action_selection,
            )
            report = evaluate_sim(
                args.model,
                args.device,
                args.episodes,
                cfg,
                args.split,
                args.start_index,
                render_mode=args.render_mode,
                allow_render_mismatch=args.allow_render_mismatch,
                trace_path=args.trace,
                goal_render_frames=args.goal_render_frames,
            )
        else:
            report = merge_sim_reports(args.reports, args.episodes, args.start_index)
        _write_report(report, args.output)


if __name__ == "__main__":
    main()
