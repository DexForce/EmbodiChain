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

"""Compare independent NeRD trainers on immutable published data.

Run:
    python scripts/experiments/supervised/nerd_acceptance.py \
        --reference-source /path/to/pr855 --datasets /path/to/nerd-datasets \
        --models /path/to/nerd-models --device cuda:0 --output /path/to/results

Dataset and model roots contain ``cartpole/v1`` and ``ant/v1``. No data is
generated, no ODS engine is started, and no physics rollout result is inferred
from these offline prediction and paired-training checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identity(source: Path) -> dict[str, Any]:
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(source), *args])

    modules = (
        "embodichain/learning/dynamics/trainer.py",
        "embodichain/learning/dynamics/supervised.py",
        "embodichain/learning/sl/learner.py",
        "embodichain/learning/sl/types.py",
    )
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD")).hexdigest(),
        "modules": {
            name: _sha(source / name) for name in modules if (source / name).is_file()
        },
    }


def _worker(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    if "runner_sha256" in request and _sha(Path(__file__)) != request["runner_sha256"]:
        raise RuntimeError("Acceptance runner changed after protocol freeze")
    if (
        "source_identity" in request
        and _source_identity(Path(request["source"])) != request["source_identity"]
    ):
        raise RuntimeError("Trainer source changed after protocol freeze")
    sys.path.insert(0, request["source"])
    import torch

    from embodichain.compute.dynamics import NeRDStateCodec
    from embodichain.data_pipeline.dynamics.shards import (
        DynamicsWindowDataset,
        dataset_fingerprint,
        load_manifest,
    )
    from embodichain.learning.dynamics.trainer import (
        DynamicsTrainCfg,
        DynamicsTrainer,
        evaluate_model,
    )
    from embodichain.learning.models.nerd import NeRDModel, load_model

    torch.set_num_threads(request["threads"])
    started = time.perf_counter()
    device = request["device"]
    cuda = torch.device(device).type == "cuda"
    if cuda:
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    pretrained, _ = load_model(request["model"])
    manifest = load_manifest(request["dataset"])
    codec = NeRDStateCodec(**manifest["metadata"]["codec"])
    result: dict[str, Any] = {"mode": request["mode"]}
    train = validation = test = None
    try:
        if request["mode"] == "offline":
            test = DynamicsWindowDataset(
                request["dataset"], "test", pretrained.cfg.history_length
            )
            pretrained.to(device)
            result["metrics"] = evaluate_model(
                pretrained, codec, test, request["batch_size"], device
            )
        else:
            seed = request["seed"]
            torch.manual_seed(seed)
            model = NeRDModel(pretrained.cfg)
            statistics = {
                "schema_version": 1,
                "weighting": "shared_published_training_statistics",
                "dataset_fingerprint": dataset_fingerprint(manifest),
                "inputs": {
                    name: {"mean": rms.mean.clone(), "var": rms.var.clone()}
                    for name, rms in pretrained.input_rms.items()
                },
                "output": {
                    "mean": pretrained.output_rms.mean.clone(),
                    "var": pretrained.output_rms.var.clone(),
                },
            }
            train = DynamicsWindowDataset(
                request["dataset"], "train", model.cfg.history_length
            )
            validation = DynamicsWindowDataset(
                request["dataset"], "validation", model.cfg.history_length
            )
            cfg = DynamicsTrainCfg(
                total_steps=4_995_000,
                batch_size=request["batch_size"],
                learning_rate=1e-3,
                final_learning_rate=1e-4,
                max_grad_norm=1.0,
                seed=seed,
                device=device,
                validation_interval=5000,
                validation_batches=100,
                checkpoint_interval=5000,
                lr_update_interval=5000,
                lr_offset=1,
                log_interval=100,
                resident_cache_bytes=request["resident_cache_bytes"],
                numerical_profile="reference",
            )
            trainer = DynamicsTrainer(
                model,
                codec,
                train,
                validation,
                cfg,
                output,
                manifest=manifest["metadata"],
                statistics=statistics,
                initialize_normalization=not request.get("resume"),
            )
            if request.get("resume"):
                trainer.resume(output / "last.pt")
            last_update: dict[str, Any] = {}
            if trainer.global_step == 0:
                captured: dict[str, Any] = {"raw_gradients": {}}
                handles = [
                    parameter.register_hook(
                        lambda gradient, name=name: captured["raw_gradients"].update(
                            {name: gradient.detach().cpu().clone()}
                        )
                    )
                    for name, parameter in model.named_parameters()
                ]
                handles.append(
                    model.register_forward_hook(
                        lambda module, inputs, prediction: captured.update(
                            prediction=prediction.detach().cpu().clone()
                        )
                    )
                )
                first = trainer.step()
                last_update = first
                for handle in handles:
                    handle.remove()
                captured.update(
                    loss=torch.tensor(first["loss"]),
                    grad_norm=torch.tensor(first["grad_norm"]),
                    clipped_gradients={
                        name: parameter.grad.detach().cpu().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    },
                    updated_parameters={
                        name: parameter.detach().cpu().clone()
                        for name, parameter in model.named_parameters()
                    },
                )
                torch.save(captured, output / "first_update.pt")
            result["training"] = trainer.train(stop_after=request["stop_after"])
            last_update = result["training"]["last"] or last_update
            # Diagnostic only: read already clipped gradients after the final
            # update, without another forward, backward, hook or optimizer call.
            torch.save(
                {
                    "num_updates": trainer.global_step,
                    "pre_clip_grad_norm": last_update.get("grad_norm"),
                    "clipped_gradients": {
                        name: parameter.grad.detach().cpu().clone()
                        for name, parameter in model.named_parameters()
                        if parameter.grad is not None
                    },
                },
                output / "final_gradients.pt",
            )
            result["metrics"] = evaluate_model(
                model, codec, validation, request["batch_size"], device
            )
        if cuda:
            torch.cuda.synchronize(device)
        result.update(
            elapsed_seconds=time.perf_counter() - started,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            torch_peak_gpu_mib=(
                torch.cuda.max_memory_allocated(device) / 1024**2 if cuda else None
            ),
            source_identity=_source_identity(Path(request["source"])),
        )
        _write(output / "result.json", result)
    finally:
        for dataset in (train, validation, test):
            if dataset is not None:
                dataset.close()


def _run(request: dict[str, Any]) -> dict[str, Any]:
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"{request.get('stop_after', 'offline')}-{bool(request.get('resume'))}"
    request_path = output / f"request-{suffix}.json"
    _write(request_path, request)
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=request["source"],
        OMP_NUM_THREADS=str(request["threads"]),
        MKL_NUM_THREADS=str(request["threads"]),
        OPENBLAS_NUM_THREADS=str(request["threads"]),
    )
    with (output / f"worker-{suffix}.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(request_path),
            ],
            cwd=request["source"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((output / "result.json").read_text())


def _state_equal(first: Any, second: Any) -> bool:
    import torch

    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _state_equal(value, second[key]) for key, value in first.items()
        )
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _state_equal(a, b) for a, b in zip(first, second)
        )
    return first == second


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return seeds


def main() -> None:
    """Run the full offline and five-seed matched-update acceptance matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument(
        "--candidate-source", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seeds", type=_parse_seeds, default="0,1,2,3,4")
    parser.add_argument("--robots", nargs="+", default=["cartpole", "ant"])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resident-cache-bytes", type=int, default=4 * 1024**3)
    args = parser.parse_args()
    if args.worker:
        _worker(args.worker)
        return
    if any(
        value is None
        for value in (args.reference_source, args.datasets, args.models, args.output)
    ):
        parser.error("reference-source, datasets, models and output are required")
    if args.steps < 2 or args.batch_size < 1:
        parser.error("steps must be >=2 and batch-size positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "output must be a fresh directory; completed results are immutable"
        )
    import torch

    sys.path.insert(0, str(args.candidate_source.resolve()))
    from embodichain.data_pipeline.dynamics.shards import load_manifest
    from embodichain.learning.dynamics.qualification import (
        compare_training_steps,
        paired_noninferiority,
    )

    seeds = args.seeds
    sources = {
        "reference": str(args.reference_source.resolve()),
        "candidate": str(args.candidate_source.resolve()),
    }
    source_identities = {
        name: _source_identity(Path(source)) for name, source in sources.items()
    }
    rows, checks = [], {}
    for robot in args.robots:
        dataset, model = args.datasets / robot / "v1", args.models / robot / "v1"
        load_manifest(dataset, verify=True)
        base = {
            "dataset": str(dataset.resolve()),
            "model": str(model.resolve()),
            "device": args.device,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "resident_cache_bytes": args.resident_cache_bytes,
            "runner_sha256": _sha(Path(__file__)),
        }
        _write(
            args.output / robot / "artifacts.json",
            {
                "dataset_manifest_sha256": _sha(dataset / "manifest.json"),
                "model_manifest_sha256": _sha(model / "manifest.json"),
                "model_weights_sha256": _sha(model / "weights.pt"),
            },
        )
        offline = {}
        for variant, source in sources.items():
            request = dict(
                base,
                source=source,
                source_identity=source_identities[variant],
                mode="offline",
                output=str((args.output / robot / variant / "offline").resolve()),
            )
            offline[variant] = _run(request)
            rows.append(
                dict(robot=robot, variant=variant, seed="published", **offline[variant])
            )
        metric_names = [
            key
            for key, value in offline["reference"]["metrics"].items()
            if isinstance(value, (int, float))
        ]
        checks[f"{robot}/published_offline"] = all(
            abs(
                offline["reference"]["metrics"][key]
                - offline["candidate"]["metrics"][key]
            )
            <= 1e-5 + 1e-4 * abs(offline["reference"]["metrics"][key])
            for key in metric_names
        )
        errors = {variant: [] for variant in sources}
        for seed in seeds:
            paths = {}
            for variant, source in sources.items():
                output = (args.output / robot / variant / f"seed-{seed}").resolve()
                paths[variant] = output
                report = _run(
                    dict(
                        base,
                        source=source,
                        source_identity=source_identities[variant],
                        mode="train",
                        seed=seed,
                        stop_after=args.steps,
                        output=str(output),
                    )
                )
                rows.append(dict(robot=robot, variant=variant, seed=seed, **report))
                errors[variant].append(report["metrics"]["normalized_mse"])
            captures = [
                torch.load(paths[variant] / "first_update.pt", weights_only=True)
                for variant in sources
            ]
            parity = compare_training_steps(*captures)
            _write(args.output / robot / f"seed-{seed}-first-update.json", parity)
            checks[f"{robot}/seed-{seed}/first_update"] = parity["passed"]
            states = [
                torch.load(
                    paths[variant] / "last.pt", weights_only=True, map_location="cpu"
                )
                for variant in sources
            ]
            checks[f"{robot}/seed-{seed}/final_parameters_exact"] = _state_equal(
                states[0]["model"], states[1]["model"]
            )
            if seed == seeds[0]:
                resumed = (
                    args.output / robot / "candidate" / f"seed-{seed}-resumed"
                ).resolve()
                request = dict(
                    base,
                    source=sources["candidate"],
                    source_identity=source_identities["candidate"],
                    mode="train",
                    seed=seed,
                    stop_after=args.steps // 2,
                    output=str(resumed),
                )
                _run(request)
                _run(dict(request, stop_after=args.steps, resume=True))
                state = torch.load(
                    resumed / "last.pt", weights_only=True, map_location="cpu"
                )
                checks[f"{robot}/resume_exact"] = all(
                    _state_equal(state[key], states[1][key])
                    for key in (
                        "model",
                        "optimizer",
                        "global_step",
                        "sampler_state",
                        "rng",
                    )
                )
        gate = paired_noninferiority(errors["reference"], errors["candidate"])
        _write(args.output / robot / "paired-validation.json", gate)
        checks[f"{robot}/paired_noninferiority"] = gate["status"] == "passed"
    passed = all(checks.values())
    _write(
        args.output / "summary.json", {"passed": passed, "checks": checks, "rows": rows}
    )
    lines = [
        "# NeRD SL acceptance",
        "",
        f"Passed: {passed}",
        "",
        "## Time & Memory",
        "",
        "| Robot | Trainer | Seed | Seconds | Process peak RSS MiB | Torch peak GPU MiB |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {row['robot']} | {row['variant']} | {row['seed']} | {row['elapsed_seconds']:.3f} | {row['process_peak_rss_mib']:.1f} | {row['torch_peak_gpu_mib']} |"
        for row in rows
    ]
    lines += [
        "",
        "## Success & Other Metrics",
        "",
        "| Check | Passed |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {value} |" for name, value in checks.items()]
    lines += [
        "",
        "## Leaderboard",
        "",
        "| Robot | Trainer | Mean validation normalized MSE |",
        "| --- | --- | --- |",
    ]
    for robot in args.robots:
        ranked = []
        for variant in sources:
            values = [
                row["metrics"]["normalized_mse"]
                for row in rows
                if row["robot"] == robot
                and row["variant"] == variant
                and row["seed"] != "published"
            ]
            ranked.append((sum(values) / len(values), variant))
        lines += [
            f"| {robot} | {variant} | {value:.9g} |"
            for value, variant in sorted(ranked)
        ]
    lines += [
        "",
        "Peak RSS is the worker-process high water mark; GPU memory covers Torch only. Setup, training and validation are included in elapsed time. Published normalization is shared; these fits do not claim full-budget convergence or physics rollout reproduction.",
        "",
    ]
    (args.output / "report.md").write_text("\n".join(lines))
    print(args.output / "summary.json", flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
