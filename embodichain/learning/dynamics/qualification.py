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

"""Explicit numerical parity and seed-paired accuracy acceptance gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from .trainer import normalized_mse

__all__ = ["capture_training_step", "compare_training_steps", "paired_noninferiority"]


def capture_training_step(
    model: torch.nn.Module,
    inputs: dict[str, Tensor],
    targets: Tensor,
    *,
    optimizer_state: dict[str, Any] | None = None,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 1.0,
    mask: Tensor | None = None,
) -> dict[str, Any]:
    """Capture one real Adam update; caller owns model/RNG initialization.

    This mutates the supplied model. Run the official implementation separately
    and serialize the same named tensors; comparison never substitutes a second
    invocation of this function for independent official evidence.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = normalized_mse(
        prediction,
        targets,
        model.output_rms.var if model.cfg.normalize_output else None,
        mask,
    )
    loss.backward()
    raw = {
        key: parameter.grad.detach().cpu().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_grad_norm, error_if_nonfinite=True
    )
    clipped = {
        key: parameter.grad.detach().cpu().clone()
        for key, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    return {
        "prediction": prediction.detach().cpu(),
        "loss": loss.detach().cpu(),
        "grad_norm": norm.detach().cpu(),
        "raw_gradients": raw,
        "clipped_gradients": clipped,
        "updated_parameters": {
            key: parameter.detach().cpu().clone()
            for key, parameter in model.named_parameters()
        },
        "optimizer": optimizer.state_dict(),
    }


def compare_training_steps(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> dict[str, Any]:
    """Compare independently captured forward, gradient and Adam-update tensors."""
    results: dict[str, Any] = {}

    def compare(name: str, expected: Tensor, actual: Tensor) -> None:
        same_shape = expected.shape == actual.shape
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        passed = (
            same_shape
            and finite
            and torch.allclose(expected, actual, atol=atol, rtol=rtol)
        )
        results[name] = {
            "passed": bool(passed),
            "max_abs": (
                float((expected - actual).abs().max())
                if same_shape and finite and expected.numel()
                else None
            ),
        }

    for name in ("prediction", "loss", "grad_norm"):
        compare(name, reference[name], candidate[name])
    for group in ("raw_gradients", "clipped_gradients", "updated_parameters"):
        if reference[group].keys() != candidate[group].keys():
            results[group] = {"passed": False, "reason": "parameter_names_differ"}
        for name in reference[group].keys() & candidate[group].keys():
            compare(f"{group}/{name}", reference[group][name], candidate[group][name])
    return {
        "passed": all(value["passed"] for value in results.values()),
        "atol": atol,
        "rtol": rtol,
        "tensors": results,
    }


def paired_noninferiority(
    reference: list[float],
    candidate: list[float],
    *,
    margin: float = 0.05,
    absolute_tolerance: float = 1e-5,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """One-sided 95% seed-paired bootstrap gate for lower-is-better metrics.

    Fewer than five seeds is pending qualification. Near-zero references use
    absolute error, and non-finite measurements always fail qualification.
    """
    baseline, measured = np.asarray(reference, dtype=float), np.asarray(
        candidate, dtype=float
    )
    if baseline.shape != measured.shape or baseline.ndim != 1 or baseline.size == 0:
        raise ValueError("Expected equally sized nonempty paired seed vectors")
    if margin < 0 or absolute_tolerance <= 0 or resamples < 1:
        raise ValueError("Invalid acceptance thresholds")
    if not np.isfinite(baseline).all() or not np.isfinite(measured).all():
        return {"status": "failed", "reason": "nonfinite_metrics"}
    if (baseline < 0).any() or (measured < 0).any():
        raise ValueError("Error metrics must be nonnegative")
    absolute = bool((baseline <= absolute_tolerance).any())
    differences = measured - baseline if absolute else (measured - baseline) / baseline
    rng = np.random.default_rng(seed)
    # Bounded chunks also support larger external qualification runs.
    estimates = np.empty(resamples)
    for start in range(0, resamples, 1024):
        count = min(1024, resamples - start)
        indices = rng.integers(0, baseline.size, size=(count, baseline.size))
        estimates[start : start + count] = differences[indices].mean(axis=1)
    upper = float(np.quantile(estimates, 0.95))
    threshold = absolute_tolerance if absolute else margin
    status = (
        "pending" if baseline.size < 5 else "passed" if upper <= threshold else "failed"
    )
    return {
        "status": status,
        "seeds": int(baseline.size),
        "mean_difference": float(differences.mean()),
        "upper_95": upper,
        "threshold": threshold,
        "difference_mode": "absolute" if absolute else "relative",
        "resamples": resamples,
        "seed": seed,
    }
