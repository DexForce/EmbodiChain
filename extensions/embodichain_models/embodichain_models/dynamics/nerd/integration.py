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

"""NeRD encoding, target losses, frozen statistics and offline evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .data import DynamicsWindowDataset

__all__ = [
    "encode_batch",
    "normalized_mse",
    "compute_statistics",
    "save_statistics",
    "load_statistics",
    "evaluate_model",
]

_PREPROCESSING_EXECUTION = "nerd-shared-frame-scripted-cuda-warmup-v2"


def encode_batch(
    batch: dict[str, Any], codec: Any, device: str | torch.device
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Encode a collated raw trajectory window through the shared codec."""
    states = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in batch["states"].items()
    }
    transitions = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in batch["transitions"].items()
    }
    effort = transitions.get("joint_f", transitions.get("effort"))
    if effort is None:
        raise ValueError("Dataset requires transitions/joint_f")
    contacts = {
        key: value for key, value in transitions.items() if key.startswith("contact_")
    }
    q, qd = states["q"], states["qd"]
    legacy_velocity = states.get("legacy_velocity")
    current_velocity = None if legacy_velocity is None else legacy_velocity[:, :-1]
    inputs = codec.encode(
        q[:, :-1],
        qd[:, :-1],
        effort,
        contacts or None,
        transitions.get("gravity_dir"),
        legacy_velocity=current_velocity,
    )
    target = codec.targets(
        q[:, :-1],
        qd[:, :-1],
        q[:, 1:],
        qd[:, 1:],
        current_embedding=inputs["states_embedding"],
        legacy_velocity=current_velocity,
        next_legacy_velocity=(
            None if legacy_velocity is None else legacy_velocity[:, 1:]
        ),
    )
    return inputs, target, batch["valid_mask"].to(device)


def normalized_mse(
    prediction: Tensor,
    target: Tensor,
    variance: Tensor | None,
    mask: Tensor | None = None,
) -> Tensor:
    """Official MSE over normalized physical targets, excluding padded steps."""
    dtype = torch.promote_types(prediction.dtype, target.dtype)
    prediction = prediction.to(dtype=dtype)
    target = target.to(dtype=dtype)
    if variance is None:
        scaled_prediction, scaled_target = prediction, target
    else:
        weights = 1.0 / torch.sqrt(variance + 1e-5)
        # Preserve official arithmetic ordering for numerical parity.
        scaled_prediction, scaled_target = prediction * weights, target * weights
    if mask is None:
        return torch.nn.functional.mse_loss(scaled_prediction, scaled_target)
    squared = torch.nn.functional.mse_loss(
        scaled_prediction, scaled_target, reduction="none"
    )
    valid = mask.to(squared.dtype).unsqueeze(-1)
    return (squared * valid).sum() / (valid.sum() * squared.shape[-1]).clamp_min(1)


class _Moments:
    def __init__(self) -> None:
        self.count = 0
        self.mean: Tensor | None = None
        self.m2: Tensor | None = None

    def add(self, values: Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        count = values.shape[0]
        if not count:
            return
        mean = values.mean(0)
        m2 = (values - mean).square().sum(0)
        if self.mean is None:
            self.count, self.mean, self.m2 = count, mean, m2
        else:
            delta = mean - self.mean
            total = self.count + count
            self.m2 += m2 + delta.square() * self.count * count / total
            self.mean += delta * count / total
            self.count = total

    def result(self) -> tuple[Tensor, Tensor]:
        if self.mean is None:
            raise ValueError("Cannot normalize an empty dataset")
        return self.mean.float(), (self.m2 / self.count).float()


def compute_statistics(
    dataset: DynamicsWindowDataset, codec: Any, batch_size: int = 64
) -> dict[str, Any]:
    """Compute bounded training-only, overlapping-window-weighted moments.

    Statistical weighting is explicit. Official comparison runs can instead load
    the exact shared reference artifact; no claim of matching an upstream running
    estimator's epsilon initialization is implied by these population moments.
    """
    if dataset.split != "train":
        raise ValueError("Normalization statistics may only use the training split")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    moments: dict[str, _Moments] = {}
    targets = _Moments()
    for start in range(0, len(dataset), batch_size):
        batch = dataset.get_batch(range(start, min(len(dataset), start + batch_size)))
        inputs, target, mask = encode_batch(batch, codec, "cpu")
        for key, value in inputs.items():
            moments.setdefault(key, _Moments()).add(value[mask])
        targets.add(target[mask])
    mean, variance = targets.result()
    return {
        "schema_version": 1,
        "weighting": "valid_window_occurrences",
        "dataset_fingerprint": dataset.fingerprint,
        "inputs": {
            key: {"mean": item.result()[0], "var": item.result()[1]}
            for key, item in moments.items()
        },
        "output": {"mean": mean, "var": variance},
    }


def save_statistics(path: str | Path, statistics: dict[str, Any]) -> None:
    """Write a tensor-only statistics artifact usable by both trainers."""
    _atomic_save(Path(path), statistics)


def load_statistics(path: str | Path) -> dict[str, Any]:
    """Load a tensor-only shared reference statistics artifact."""
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported statistics schema")
    for item in [*value["inputs"].values(), value["output"]]:
        if (
            not torch.isfinite(item["mean"]).all()
            or not torch.isfinite(item["var"]).all()
            or (item["var"] < 0).any()
        ):
            raise ValueError("Invalid normalizer moments")
    return value


def _atomic_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _model_variance(model: torch.nn.Module) -> Tensor | None:
    return model.output_rms.var if model.cfg.normalize_output else None


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    codec: Any,
    dataset: DynamicsWindowDataset,
    batch_size: int = 64,
    device: str = "cpu",
    max_batches: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate all valid held-out positions without model selection on test."""
    if not len(dataset):
        raise ValueError(f"No eligible windows in {dataset.split} split")
    if batch_size < 1 or (max_batches is not None and max_batches < 1):
        raise ValueError("Evaluation batch limits must be positive")
    window_count = (
        len(dataset)
        if max_batches is None
        else min(len(dataset), max_batches * batch_size)
    )
    was_training = model.training
    model.eval()
    accumulated: dict[str, float] = {
        "normalized_mse": 0.0,
        "target_mse": 0.0,
        "q_error_norm": 0.0,
        "qd_error_norm": 0.0,
    }
    count = 0
    try:
        for start in range(0, window_count, batch_size):
            batch = dataset.get_batch(
                range(start, min(window_count, start + batch_size))
            )
            inputs, target, mask = encode_batch(batch, codec, device)
            prediction = model(inputs)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite evaluation prediction")
            n = int(mask.sum())
            accumulated["normalized_mse"] += (
                float(
                    normalized_mse(
                        prediction,
                        target,
                        _model_variance(model),
                        mask if dataset.include_prefix else None,
                    )
                )
                * n
            )
            accumulated["target_mse"] += (
                float((prediction - target).square()[mask].mean()) * n
            )
            q = batch["states"]["q"].to(device=device, dtype=prediction.dtype)
            qd = batch["states"]["qd"].to(device=device, dtype=prediction.dtype)
            next_q, next_qd = codec.decode(q[:, :-1], qd[:, :-1], prediction)
            q_error = next_q - q[:, 1:]
            periodic = codec.continuous_q_indices
            if periodic:
                q_error[..., list(periodic)] = torch.atan2(
                    torch.sin(q_error[..., list(periodic)]),
                    torch.cos(q_error[..., list(periodic)]),
                )
            if codec.robot == "ant":
                position = q_error[..., :3].norm(dim=-1)[mask].sum()
                dot = (
                    (
                        torch.nn.functional.normalize(next_q[..., 3:7], dim=-1)
                        * torch.nn.functional.normalize(q[:, 1:, 3:7], dim=-1)
                    )
                    .sum(-1)
                    .abs()
                    .clamp(max=1)
                )
                angle = (2 * torch.acos(dot))[mask].sum()
                accumulated["base_position_error"] = accumulated.get(
                    "base_position_error", 0.0
                ) + float(position)
                accumulated["base_orientation_error_rad"] = accumulated.get(
                    "base_orientation_error_rad", 0.0
                ) + float(angle)
                q_error = q_error[..., 7:]
            accumulated["q_error_norm"] += float(q_error.norm(dim=-1)[mask].sum())
            accumulated["qd_error_norm"] += float(
                (next_qd - qd[:, 1:]).norm(dim=-1)[mask].sum()
            )
            count += n
    finally:
        model.train(was_training)
    return {
        **{key: value / count for key, value in accumulated.items()},
        "valid_positions": count,
        "windows": window_count,
        "split": dataset.split,
    }
