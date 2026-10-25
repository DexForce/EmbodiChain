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

"""Reproducible, immutable two-teacher/two-trainer NeRD experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ._reference import OFFICIAL_COMMIT, _hash, reference_identity

__all__ = ["prepare_comparison", "run_comparison", "run_candidate"]
_TEACHERS = ("official_warp", "current_newton")
_TRAINERS = ("official", "embodichain")


def _json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    text = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
    else:
        temporary = path.with_suffix(path.suffix + ".pending")
        temporary.write_text(text)
        os.replace(temporary, path)


def prepare_comparison(config: dict[str, Any]) -> Path:
    """Freeze common initialization, statistics, batch orders and run manifests.

    ``datasets`` maps both teacher names to canonical dataset paths or ``None``.
    Missing inputs create explicitly pending cells. ``scope=smoke`` runs cannot
    satisfy full-budget qualification, even if every training process succeeds.
    No official source is executed during preparation.
    """
    import numpy as np
    import torch
    from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
    from embodichain_models.dynamics.nerd.data import DynamicsWindowDataset
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg, export_model
    from ._trainer import (
        DynamicsTrainCfg,
        compute_statistics,
        save_statistics,
        load_statistics,
    )
    from ._sample_order import _write_checkpointed_order

    allowed = {
        "output_dir",
        "datasets",
        "source",
        "reference_dependencies",
        "python",
        "seeds",
        "model",
        "train",
        "statistics",
        "scope",
        "cpu_threads",
        "dataset_origins",
    }
    if config.keys() - allowed:
        raise ValueError(f"Unknown comparison options: {config.keys()-allowed}")
    if set(config["datasets"]) != set(_TEACHERS):
        raise ValueError(
            "datasets must name official_warp and current_newton, using null for unavailable data"
        )
    scope = config.get("scope", "smoke")
    if scope not in {"smoke", "full"}:
        raise ValueError("scope must be smoke or full")
    seeds = config.get("seeds", [0, 1, 2, 3, 4])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    root = Path(config["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Comparison output must be a new empty directory")
    source = reference_identity(config["source"])
    # Resolving the executable symlink discards pyvenv.cfg and silently selects
    # the base interpreter's packages instead of the requested virtualenv.
    python = str(Path(config.get("python", sys.executable)).expanduser().absolute())
    dependencies = (
        str(Path(config["reference_dependencies"]).resolve())
        if config.get("reference_dependencies")
        else None
    )
    architecture = NeRDModelCfg(**config["model"])
    cfg = DynamicsTrainCfg(**config["train"])
    architecture.validate()
    cfg.validate()
    if cfg.total_steps < 1 or cfg.batch_size < 1:
        raise ValueError("Training sizes must be positive")
    cells = []
    datasets = {}
    artifacts = {}
    for teacher in _TEACHERS:
        path = config["datasets"][teacher]
        if path is None:
            datasets[teacher] = {"status": "pending", "reason": "dataset unavailable"}
            for seed in seeds:
                for trainer in _TRAINERS:
                    cells.append(
                        {
                            "id": f"{teacher}-{trainer}-{seed}",
                            "teacher": teacher,
                            "trainer": trainer,
                            "seed": seed,
                            "status": "pending",
                            "reason": "dataset unavailable",
                        }
                    )
            continue
        dataset = DynamicsWindowDataset(
            path, "train", architecture.history_length, verify=True
        )
        if not len(dataset):
            raise ValueError(f"No full training windows: {teacher}")
        codec = NeRDStateCodec(**dataset.manifest["metadata"]["codec"])
        teacher_dir = root / teacher
        teacher_dir.mkdir()
        stats_source = config.get("statistics", {}).get(teacher)
        stats = (
            load_statistics(stats_source)
            if stats_source
            else compute_statistics(dataset, codec, cfg.batch_size)
        )
        stats_path = teacher_dir / "statistics.pt"
        save_statistics(stats_path, stats)
        artifacts[str(stats_path.relative_to(root))] = _hash(stats_path)
        datasets[teacher] = {
            "status": "available",
            "path": str(Path(path).resolve()),
            "fingerprint": dataset.fingerprint,
            "origin": config.get("dataset_origins", {}).get(teacher),
            "normalizer": (
                "shared_artifact"
                if stats_source
                else "shared_window_population_moments"
            ),
        }
        for seed in seeds:
            seed_dir = teacher_dir / f"seed-{seed}"
            seed_dir.mkdir()
            order_path, rng_path = _write_checkpointed_order(
                seed_dir / "order.json",
                dataset_size=len(dataset),
                steps=cfg.total_steps,
                batch_size=cfg.batch_size,
                seed=seed,
            )
            artifacts[str(rng_path.relative_to(root))] = _hash(rng_path)
            order_hash = _hash(order_path)
            artifacts[str(order_path.relative_to(root))] = order_hash
            torch.manual_seed(seed)
            model = NeRDModel(architecture)
            model.set_normalization(
                {
                    key: (item["mean"], item["var"])
                    for key, item in stats["inputs"].items()
                },
                stats["output"]["mean"],
                stats["output"]["var"],
            )
            initial = seed_dir / "initial-model"
            export_model(initial, model, dataset.manifest["metadata"])
            for filename in ("manifest.json", "weights.pt"):
                artifacts[str((initial / filename).relative_to(root))] = _hash(
                    initial / filename
                )
            for trainer in _TRAINERS:
                run_id = f"{teacher}-{trainer}-{seed}"
                directory = seed_dir / trainer
                spec = {
                    "source": source["root"],
                    "dataset": str(Path(path).resolve()),
                    "dataset_fingerprint": dataset.fingerprint,
                    "initial_model": str(initial),
                    "initial_model_sha256": _hash(initial / "weights.pt"),
                    "sample_order": str(order_path),
                    "sample_order_sha256": order_hash,
                    "train": {**cfg.to_dict(), "seed": seed},
                    "output_dir": str(directory),
                    "cpu_threads": config.get("cpu_threads", 1),
                }
                spec_path = seed_dir / f"{trainer}.json"
                _json(spec_path, spec, exclusive=True)
                artifacts[str(spec_path.relative_to(root))] = _hash(spec_path)
                worker = "_reference" if trainer == "official" else "_comparison"
                command = [
                    python,
                    str(Path(__file__).with_name("_worker.py")),
                    worker,
                    "--spec",
                    str(spec_path),
                ]
                if trainer == "official":
                    command.append("--trust-external-code")
                else:
                    command.append("--candidate")
                cells.append(
                    {
                        "id": run_id,
                        "teacher": teacher,
                        "trainer": trainer,
                        "seed": seed,
                        "status": "prepared",
                        "spec": str(spec_path),
                        "output_dir": str(directory),
                        "command": command,
                    }
                )
        dataset.close()
    plan = {
        "format": "embodichain.nerd.comparison.v1",
        "scope": scope,
        "source": source,
        "reference_dependencies": dependencies,
        "python": python,
        "seeds": seeds,
        "model": architecture.to_dict(),
        "feature_order": list(architecture.input_dims),
        "train": cfg.to_dict(),
        "comparison_protocol": {
            "kind": "controlled_matched_data_noninferiority",
            "initialization": "shared exported weights and normalizers",
            "training_order": "shared frozen torch.randint-with-replacement indices",
            "statistics": "shared provided artifact or train-window population moments",
            "checkpoint_selection": "one validation split; ordered first configured batches; partial final batch included; no epoch-zero best selection",
            "preprocessing_execution": "shared current frame; scripted coordinate math; three real-layout CUDA preprocessing warmups without optimizer/RNG advancement",
            "paper_reproduction": "not established by this controlled protocol",
        },
        "datasets": datasets,
        "cells": cells,
        "artifacts": artifacts,
        "adapter_source_sha256": _hash(Path(__file__)),
        "worker_bootstrap_sha256": _hash(Path(__file__).with_name("_worker.py")),
        "acceptance": {
            "relative_margin": 0.05,
            "absolute_tolerance": 1e-5,
            "resamples": 10000,
            "bootstrap_seed": 0,
            "required_seeds": 5,
            "official_updates": 4995000,
            "primary_metrics": ["q_error_norm", "qd_error_norm"],
            "rollout_primary_metrics": ["q_mae", "qd_mae"],
            "policy_return_relative_margin": 0.05,
            "failure_rate_increase": 0.0,
            "nonfinite_rate_increase": 0.0,
        },
    }
    _json(root / "experiment.json", plan, exclusive=True)
    _json(
        root / "experiment.identity.json",
        {"sha256": _hash(root / "experiment.json")},
        exclusive=True,
    )
    return root / "experiment.json"


def _read_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent
    identity = json.loads((root / "experiment.identity.json").read_text())
    if _hash(path) != identity["sha256"]:
        raise ValueError("Experiment manifest changed after preparation")
    plan = json.loads(path.read_text())
    if plan.get("format") != "embodichain.nerd.comparison.v1":
        raise ValueError("Unsupported comparison manifest")
    for name, digest in plan["artifacts"].items():
        file = (root / name).resolve()
        if not file.is_relative_to(root) or not file.is_file() or _hash(file) != digest:
            raise ValueError(f"Immutable comparison artifact changed: {name}")
    return root, plan


def run_candidate(
    spec: dict[str, Any],
    *,
    resume: str | Path | None = None,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Execute the migrated trainer on the same frozen run specification."""
    import torch
    from embodichain_models.dynamics.nerd.data import (
        load_manifest,
        dataset_fingerprint,
    )
    from ._workflow import run_train, run_export, run_eval

    torch.set_num_threads(int(spec.get("cpu_threads", 1)))
    torch.set_float32_matmul_precision("highest")
    if (
        dataset_fingerprint(load_manifest(spec["dataset"], verify=True))
        != spec["dataset_fingerprint"]
    ):
        raise ValueError("Candidate dataset changed")
    if _hash(Path(spec["sample_order"])) != spec["sample_order_sha256"]:
        raise ValueError("Candidate sample order changed")
    if (
        _hash(Path(spec["initial_model"]) / "weights.pt")
        != spec["initial_model_sha256"]
    ):
        raise ValueError("Common initialization changed")
    directory = Path(spec["output_dir"])
    initial_manifest = json.loads(
        (Path(spec["initial_model"]) / "manifest.json").read_text()
    )
    execution = (
        {"resume": str(resume)}
        if resume is not None
        else {"fine_tune": spec["initial_model"]}
    )
    result = run_train(
        {
            "dataset": spec["dataset"],
            "output_dir": str(directory),
            "model": initial_manifest["model_config"],
            "sample_order": spec["sample_order"],
            "train": spec["train"],
            "stop_after": stop_after,
            **execution,
        }
    )
    if result["status"] != "completed":
        result["completed_updates"] = result["global_step"]
        _json(directory / "result.json", result)
        return result
    for name in ("best", "last"):
        run_export(directory / f"{name}.pt", directory / f"{name}-export")
    result.update(
        completed_updates=result["global_step"],
        test=run_eval(
            {
                "dataset": spec["dataset"],
                "model": str(directory / "best-export"),
                "device": spec["train"]["device"],
                "batch_size": spec["train"]["batch_size"],
            }
        ),
        dataset_fingerprint=spec["dataset_fingerprint"],
        sample_order_sha256=spec["sample_order_sha256"],
        seed=spec["train"]["seed"],
        qualification="matched_training_run_only",
        best_model_sha256=_hash(directory / "best-export" / "weights.pt"),
        last_model_sha256=_hash(directory / "last-export" / "weights.pt"),
    )
    _json(directory / "result.json", result)
    return result


def run_comparison(
    path: str | Path, *, trust_external_code: bool = False, max_runs: int | None = None
) -> dict[str, Any]:
    """Execute prepared cells sequentially, retaining real command/stdout evidence.

    Completed verified outputs are reused. Incomplete directories are not silently
    restarted; create a new experiment or recover its run explicitly. This function
    does not claim rollout or paper qualification from offline training results.
    """
    root, plan = _read_plan(path)
    if not trust_external_code:
        raise ValueError("Comparison requires explicit trust_external_code=True")
    if max_runs is not None and max_runs < 1:
        raise ValueError("max_runs must be positive")
    statuses = []
    executed = 0
    for cell in plan["cells"]:
        if cell["status"] != "prepared":
            statuses.append(
                {"id": cell["id"], "status": "pending", "reason": cell["reason"]}
            )
            continue
        output = Path(cell["output_dir"])
        receipt = root / (cell["id"] + ".receipt.json")
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if (
                previous["status"] == "completed"
                and _hash(output / "result.json") == previous["result_sha256"]
            ):
                statuses.append(previous)
                continue
            statuses.append(
                {
                    "id": cell["id"],
                    "status": "pending",
                    "reason": "previous run needs explicit recovery",
                }
            )
            continue
        if max_runs is not None and executed >= max_runs:
            statuses.append(
                {"id": cell["id"], "status": "pending", "reason": "execution limit"}
            )
            continue
        if output.exists() and any(output.iterdir()):
            statuses.append(
                {
                    "id": cell["id"],
                    "status": "pending",
                    "reason": "unreceipted output needs explicit recovery",
                }
            )
            continue
        output.mkdir(parents=True, exist_ok=True)
        spec = json.loads(Path(cell["spec"]).read_text())
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": str(spec["cpu_threads"]),
            "MKL_NUM_THREADS": str(spec["cpu_threads"]),
        }
        pythonpaths = []
        if cell["trainer"] == "official":
            if plan["reference_dependencies"]:
                pythonpaths.append(plan["reference_dependencies"])
            pythonpaths.append(plan["source"]["root"])
        root = Path(__file__).resolve().parents[4]
        if (root / "pyproject.toml").is_file():
            pythonpaths.extend((str(root / "extensions/embodichain_models"), str(root)))
        if environment.get("PYTHONPATH"):
            pythonpaths.append(environment["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(pythonpaths)
        if spec["train"]["device"] == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        _json(
            output / "command.json",
            {
                "argv": cell["command"],
                "PYTHONPATH": environment["PYTHONPATH"],
                "source_commit": OFFICIAL_COMMIT,
            },
            exclusive=True,
        )
        with (
            (output / "stdout.log").open("w") as stdout,
            (output / "stderr.log").open("w") as stderr,
        ):
            process = subprocess.run(
                cell["command"],
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        executed += 1
        result_file = output / "result.json"
        status = "failed"
        if process.returncode == 0 and result_file.is_file():
            result = json.loads(result_file.read_text())
            if (
                result.get("status") == "completed"
                and result.get("completed_updates") == spec["train"]["total_steps"]
            ):
                status = "completed"
        receipt_value = {
            "id": cell["id"],
            "status": status,
            "returncode": process.returncode,
            "command": cell["command"],
            "result_sha256": _hash(result_file) if result_file.exists() else None,
            "stdout_sha256": _hash(output / "stdout.log"),
            "stderr_sha256": _hash(output / "stderr.log"),
        }
        _json(receipt, receipt_value, exclusive=True)
        statuses.append(receipt_value)
    _json(root / "execution.json", {"runs": statuses})
    return {"runs": statuses, "executed": executed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--prepare", type=Path, help="JSON comparison configuration")
    parser.add_argument("--run", type=Path, help="Prepared experiment.json")
    parser.add_argument("--trust-external-code", action="store_true")
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    if args.candidate and args.spec:
        value = run_candidate(
            json.loads(args.spec.read_text()),
            resume=args.resume,
            stop_after=args.stop_after,
        )
    elif args.prepare:
        value = {
            "manifest": str(prepare_comparison(json.loads(args.prepare.read_text())))
        }
    elif args.run:
        value = run_comparison(
            args.run,
            trust_external_code=args.trust_external_code,
            max_runs=args.max_runs,
        )
    else:
        parser.error("Choose --prepare, --run or --candidate --spec")
    print(json.dumps(value, indent=2, allow_nan=False))
    if args.run and any(run["status"] == "failed" for run in value["runs"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
