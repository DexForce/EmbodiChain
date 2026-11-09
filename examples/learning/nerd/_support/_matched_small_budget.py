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

"""Run the migrated trainer against a captured, literal original-entry protocol.

Unlike the general controlled comparison, the original trainer supplies model
initialization, FP32 RunningMeanStd statistics and actual consumed DataLoader
indices. Epoch-zero and final validation use its actual emitted validation order.
The original run retains its independent best-evaluation checkpoint as well.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import torch

from embodichain_models.dynamics.nerd.codec import NeRDStateCodec
from embodichain_models.dynamics.nerd.data import (
    DynamicsWindowDataset,
    dataset_fingerprint,
    load_manifest,
)
from examples.learning.nerd._support._trainer import (
    DynamicsTrainCfg,
    DynamicsTrainer,
    encode_batch,
    evaluate_model,
    normalized_mse,
)
from examples.learning.nerd._support._workflow import run_export
from embodichain_models.dynamics.nerd import convert_legacy_checkpoint, load_model

__all__: list[str] = []


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


def _first_batch_differences(
    model: Any,
    inputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    official: Path,
) -> tuple[dict[str, float], str]:
    """Compare features with an explicit pre/post-normalization observation stage."""
    path = official / "first-preprocessed-batch.pt"
    sidecar = official / "first-preprocessed-batch.json"
    if sidecar.exists():
        observation = json.loads(sidecar.read_text())
        stage = observation.get("feature_stage")
        if (
            observation.get("format_version") != 1
            or stage != "physical_before_model_forward"
            or observation.get("tensor_sha256") != _hash(path)
        ):
            raise ValueError("Invalid original first-batch observation metadata")
    else:
        # Historical observer snapshots were taken after ModelMixedInput.forward
        # normalized its input dictionary in place. Preserve those frozen files.
        stage = "legacy_unlabeled_normalized_after_model_forward"
    original = torch.load(path, map_location="cpu", weights_only=True)
    differences = {}
    for key, value in inputs.items():
        original_key = "joint_acts" if key == "joint_f" else key
        rms = model.input_rms[key]
        mean, denominator = rms.mean.cpu(), torch.sqrt(rms.var.cpu() + 1e-5)
        normalized = (value.detach().cpu() - mean) / denominator
        expected = original[original_key]
        if stage == "physical_before_model_forward":
            expected = (expected - mean) / denominator
        differences[key] = float((normalized - expected).abs().max())
    differences["target"] = float((targets.cpu() - original["target"]).abs().max())
    return differences, stage


@torch.no_grad()
def _validation(trainer: DynamicsTrainer, order: np.ndarray) -> float:
    trainer.model.eval()
    accumulated = torch.zeros((), device=trainer.cfg.device)
    for indices in order:
        inputs, targets, _ = encode_batch(
            trainer.validation_dataset.get_batch(indices),
            trainer.codec,
            trainer.cfg.device,
        )
        accumulated += normalized_mse(
            trainer.model(inputs), targets, trainer.model.output_rms.var
        )
    return float(accumulated) / len(order)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    official, output = args.official_dir, args.output_dir
    while not (official / "completed.json").exists():
        if not args.wait:
            raise ValueError("The original entry must finish its 1000-update run first")
        time.sleep(2)
    completed = json.loads((official / "completed.json").read_text())
    if completed["steps"] != 1000 or not completed["source_files_unchanged"]:
        raise ValueError("Invalid original-entry completion receipt")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1 if args.device.startswith("cuda") else 8)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    metadata = load_manifest(args.dataset, verify=True)["metadata"]
    convert_legacy_checkpoint(
        official / "initial.pt",
        output / "initial-export",
        metadata,
        trust_pickle=True,
    )
    model, manifest = load_model(output / "initial-export", args.device)
    train = DynamicsWindowDataset(args.dataset, "train", 10, verify=False)
    validation = DynamicsWindowDataset(args.dataset, "validation", 10, verify=False)
    order = np.load(official / "train-indices.npy", allow_pickle=False)
    if order.shape != (1000, 512) or len(np.unique(order)) != order.size:
        raise ValueError(
            "Expected the first 1000 full, nonreplacement official batches"
        )
    # This is the exact rate of the first trained official epoch. Keeping both
    # endpoints equal preserves it across this truncated 1000-update experiment.
    cfg = DynamicsTrainCfg(
        total_steps=1000,
        batch_size=512,
        learning_rate=0.001 * (1 - 1 / 1000) + 0.0001 / 1000,
        final_learning_rate=0.001 * (1 - 1 / 1000) + 0.0001 / 1000,
        device=args.device,
        seed=0,
        validation_interval=1000,
        checkpoint_interval=250,
        max_grad_norm=1.0,
        resident_cache_bytes=4 * 1024**3,
    )
    trainer = DynamicsTrainer(
        model,
        NeRDStateCodec(**metadata["codec"]),
        train,
        validation,
        cfg,
        output,
        manifest=manifest,
        initialize_normalization=False,
        sample_order=torch.from_numpy(order.copy()),
    )
    # Verify initialization and statistics rather than relying on a shared seed.
    fresh, _ = load_model(output / "initial-export", "cpu")
    if any(
        not torch.equal(v.detach().cpu(), fresh.state_dict()[k])
        for k, v in trainer.model.state_dict().items()
    ):
        raise RuntimeError("Model initialization or normalization changed")
    first_inputs, first_targets, _ = encode_batch(
        train.get_batch(order[0]), trainer.codec, args.device
    )
    differences, observation_stage = _first_batch_differences(
        model, first_inputs, first_targets, official
    )
    _json(
        output / "provenance.json",
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": args.device,
            "dataset_fingerprint": dataset_fingerprint(load_manifest(args.dataset)),
            "original_completion_sha256": _hash(official / "completed.json"),
            "original_initial_sha256": _hash(official / "initial.pt"),
            "sample_order_sha256": _hash(official / "train-indices.npy"),
            "validation_orders": {
                p.name: _hash(p)
                for p in sorted(official.glob("validation-*-indices.npy"))
            },
            "cfg": cfg.to_dict(),
            "first_batch_feature_target_max_abs_difference": differences,
            "feature_comparison": "normalized input features; unnormalized physical target",
            "original_first_batch_feature_stage": observation_stage,
            "known_differences": [
                "official Python3.8/Torch2.2 CPU versus current Python3.11/Torch2.7 and selected candidate device",
                "native COM/spatial velocity transport can introduce FP32 roundoff in Ant",
                "the same normalized-MSE validation rule selects epoch-zero or update1000; official also retains its original independent rollout-best checkpoint",
            ],
        },
    )
    selected = []
    best = float("inf")
    for step in (0, 1000):
        if step:
            with (output / "metrics.jsonl").open("w", buffering=1) as metrics:
                while trainer.global_step < 1000:
                    record = trainer.step()
                    metrics.write(json.dumps(record, allow_nan=False) + "\n")
                    if trainer.global_step in (1, 10, 100, 250, 500, 750, 1000):
                        trainer.save(output / ("step-%04d.pt" % trainer.global_step))
                        print("MIGRATED_UPDATE", json.dumps(record), flush=True)
        loss = _validation(
            trainer,
            np.load(
                official / ("validation-%04d-indices.npy" % step), allow_pickle=False
            ),
        )
        selected.append({"step": step, "loss": loss})
        if loss < best:
            best = loss
            trainer.best_validation = best
            trainer.save(output / "best.pt")
    trainer.save(output / "last.pt")
    for name in ("best", "last"):
        run_export(output / (name + ".pt"), output / (name + "-export"))
    best_model, _ = load_model(output / "best-export", args.device)
    test = DynamicsWindowDataset(args.dataset, "test", 10, verify=False)
    result = {
        "steps": trainer.global_step,
        "validation": selected,
        "selected_step": min(selected, key=lambda x: x["loss"])["step"],
        "test": evaluate_model(best_model, trainer.codec, test, 512, args.device),
        "best_weights_sha256": _hash(output / "best-export/weights.pt"),
        "last_weights_sha256": _hash(output / "last-export/weights.pt"),
    }
    _json(output / "completed.json", result)
    train.close()
    validation.close()
    test.close()


if __name__ == "__main__":
    main()
