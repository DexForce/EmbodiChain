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

"""Observe the original NeRD training entry in its supported Python environment.

The source constructor, statistics, samplers, losses, validation, simulation
evaluation and checkpoint selectors execute unchanged. Observers capture actual
consumed indices, initial normalizers/weights and minibatch metrics. A budget
guard stops before the second trained epoch, after the first epoch's original
validation/evaluation/checkpoint selection has finished. This file supports
Python 3.8 and never imports EmbodiChain or the current SDK.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
from typing import Any

__all__: list[str] = []


class _BudgetReached(Exception):
    pass


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_first_batch(output: Path, data: dict[str, Any]) -> None:
    """Capture physical features before the original model normalizes its input."""
    import torch

    path = output / "first-preprocessed-batch.pt"
    torch.save(
        {
            key: value.detach().cpu().clone()
            for key, value in data.items()
            if torch.is_tensor(value)
        },
        path,
    )
    _json(
        output / "first-preprocessed-batch.json",
        {
            "format_version": 1,
            "feature_stage": "physical_before_model_forward",
            "tensor_sha256": _hash(path),
        },
    )


def _observed_loss(
    trainer: Any,
    data: dict[str, Any],
    train: bool,
    original_loss: Any,
    state: dict[str, Any],
    output: Path,
) -> Any:
    """Observe the first training batch before an in-place model forward."""
    if train and state["steps"] == 0:
        _save_first_batch(output, data)
    result = original_loss(trainer, data, train)
    if train:
        state["last_loss"] = float(result[0].detach())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trust-external-code", action="store_true", required=True)
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    source, output = Path(plan["source"]), Path(plan["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import warp as wp

    expected_warp = "1.5.1" if plan["robot"] == "ant" else "1.8.0"
    if (
        platform.python_version() != "3.8.20"
        or torch.__version__.split("+")[0] != "2.2.2"
        or np.__version__ != "1.24.4"
        or wp.__version__ != expected_warp
    ):
        raise RuntimeError(
            "Official baseline requires the declared supported environment"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != "370e89d2b8ce68aa3578aa7812859c681b7502bc":
        raise ValueError("Unexpected official source commit")
    if subprocess.check_output(["git", "-C", str(source), "diff", "--name-only"]):
        raise ValueError("Official tracked source must be unmodified")
    torch.set_num_threads(plan.get("cpu_threads", 8))
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    # Upstream simulation unconditionally asks for a CUDA stream even on CPU.
    # CPU work is synchronous; the existing torch.cuda.stream(None) is a no-op.
    # Keep its numerical integrator and evaluation/selection code unchanged.
    original_stream_to_torch = wp.stream_to_torch

    def stream_to_torch(device: Any) -> Any:
        if wp.get_device(device).is_cpu:
            return None
        return original_stream_to_torch(device)

    wp.stream_to_torch = stream_to_torch
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "train"))
    from torch.utils.data import BatchSampler, DataLoader
    from algorithms.vanilla_trainer import VanillaTrainer

    state: dict[str, Any] = {
        "steps": 0,
        "stage": "initialization",
        "trainer": None,
        "last_loss": None,
        "validation": [],
        "saves": [],
        "started": time.time(),
    }
    metrics = (output / "metrics.jsonl").open("w", buffering=1)
    original_batch_iter = BatchSampler.__iter__
    original_loader_iter = DataLoader.__iter__

    def batch_iter(sampler: Any) -> Any:
        for indices in original_batch_iter(sampler):
            record = getattr(sampler, "_nerd_observation", None)
            if record is not None:
                record.append(indices)
            yield indices

    def loader_iter(loader: Any) -> Any:
        # The original iterator still owns workers, RNG, shuffling and prefetch.
        # Pair observations at consumption, rather than counting prefetched rows.
        recorded: list[Any] = []
        loader.batch_sampler._nerd_observation = recorded
        iterator = original_loader_iter(loader)

        def observe() -> Any:
            consumed = 0
            while True:
                try:
                    data = next(iterator)
                except StopIteration:
                    return
                indices = recorded[consumed]
                consumed += 1
                if state["stage"] != "initialization":
                    state["indices"].append(np.asarray(indices, dtype=np.int64))
                yield data

        # Create the original iterator eagerly: worker prefetch can consume the
        # global sampling RNG before the validation iterators are constructed.
        return observe()

    BatchSampler.__iter__ = batch_iter
    DataLoader.__iter__ = loader_iter

    original_init = VanillaTrainer.__init__
    original_epoch = VanillaTrainer.one_epoch
    original_loss = VanillaTrainer.compute_loss
    original_save = VanillaTrainer.save_model

    def init(trainer: Any, *positional: Any, **keywords: Any) -> None:
        original_init(trainer, *positional, **keywords)
        state["trainer"] = trainer
        torch.save(
            [trainer.neural_model, trainer.neural_env.robot_name], output / "initial.pt"
        )
        torch.save(
            {
                key: {"mean": rms.mean, "var": rms.var, "count": float(rms.count)}
                for key, rms in trainer.dataset_rms.items()
            },
            output / "original-statistics.pt",
        )
        original_step = trainer.optimizer.step

        def step(*pos: Any, **kw: Any) -> Any:
            result = original_step(*pos, **kw)
            state["steps"] += 1
            record = {
                "step": state["steps"],
                "loss": state["last_loss"],
                "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - state["started"],
            }
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            if state["steps"] in (1, 10, 100, 250, 500, 750, plan["steps"]):
                torch.save(
                    [trainer.neural_model, trainer.neural_env.robot_name],
                    output / ("step-%04d.pt" % state["steps"]),
                )
                print("OBSERVED_UPDATE", json.dumps(record), flush=True)
            return result

        trainer.optimizer.step = step
        _json(
            output / "initial-ready.json",
            {"initial_sha256": _hash(output / "initial.pt")},
        )

    def loss(trainer: Any, data: Any, train: bool) -> Any:
        return _observed_loss(trainer, data, train, original_loss, state, output)

    def epoch(trainer: Any, *positional: Any, **keywords: Any) -> Any:
        train = keywords.get("train", positional[0] if positional else None)
        if train and state["steps"] >= plan["steps"]:
            raise _BudgetReached()
        state["stage"] = "train" if train else "validation"
        state["indices"] = []
        result = original_epoch(trainer, *positional, **keywords)
        label = "train" if train else "validation-%04d" % state["steps"]
        np.save(str(output / (label + "-indices.npy")), np.stack(state["indices"]))
        if not train:
            state["validation"].append(
                {"step": state["steps"], "loss": result[0], "itemized": result[1]}
            )
            _json(output / "validation.json", state["validation"])
        state["stage"] = "initialization"
        return result

    def save(trainer: Any, filename: Any = None) -> None:
        original_save(trainer, filename)
        state["saves"].append({"step": state["steps"], "filename": filename})
        _json(output / "checkpoint-selection.json", state["saves"])

    VanillaTrainer.__init__ = init
    VanillaTrainer.compute_loss = loss
    VanillaTrainer.one_epoch = epoch
    VanillaTrainer.save_model = save
    entry = source / "train" / "train.py"
    original_files = {
        str(path.relative_to(source)): _hash(path)
        for path in (
            entry,
            source / "algorithms" / "vanilla_trainer.py",
            source / "algorithms" / "sequence_model_trainer.py",
            source / "utils" / "datasets.py",
            source / "utils" / "running_mean_std.py",
            source / "models" / "models.py",
            source / "models" / "model_transformer.py",
        )
    }
    _json(
        output / "provenance.json",
        {
            "plan": plan,
            "source_commit": commit,
            "original_files": original_files,
            "observer_sha256": _hash(Path(__file__)),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "warp": wp.__version__,
            "device": "cpu",
            "entry": str(entry),
            "cpu_compatibility": "wp.stream_to_torch(cpu) returns None; upstream torch.cuda.stream(None) remains a no-op; no numerical or selector changes",
            "budget_adaptation": "1000 original updates in first trained epoch; preserve 1000-epoch LR denominator and epoch-zero selection; stop before next trained epoch",
        },
    )
    sys.argv = [
        str(entry),
        "--cfg",
        plan["training_config"],
        "--logdir",
        str(output / "original-log"),
        "--no-time-stamp",
        "--device",
        "cpu",
        "--seed",
        str(plan["seed"]),
        "--save-interval",
        "1",
    ]
    previous_cwd = Path.cwd()
    os.chdir(source / "train")
    try:
        runpy.run_path(str(entry), run_name="__main__")
    except _BudgetReached:
        trainer = state["trainer"]
        trainer.save_model("final_model")
        torch.save(trainer.optimizer.state_dict(), output / "final-optimizer.pt")
        trainer.logger.finish()
    finally:
        metrics.close()
        os.chdir(previous_cwd)
    if state["steps"] != plan["steps"]:
        raise RuntimeError(
            "Original entry did not reach the requested optimizer budget"
        )
    _json(
        output / "completed.json",
        {
            "steps": state["steps"],
            "elapsed_seconds": time.time() - state["started"],
            "validation": state["validation"],
            "checkpoint_selection": state["saves"],
            "source_files_unchanged": all(
                _hash(source / p) == h for p, h in original_files.items()
            ),
        },
    )


if __name__ == "__main__":
    main()
