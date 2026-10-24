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

"""Immutable ChunkBC checkpoints and optimizer/random-state recovery."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from examples.ods.simple_bc_demo.models import ChunkBC


def save_checkpoint(
    path: Path,
    model: ChunkBC,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    run_id: int,
    *,
    sampling_rng_state: torch.Tensor | None = None,
) -> None:
    """Save immutable model, optimizer, normalization and random-state data."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {path}")
    state = {
        "model": model.state_dict(),
        "policy_config": {
            "type": "chunk_pose",
            "observation_mode": model.observation_mode,
            "arm_command_mode": model.arm_command_mode,
            "schema": getattr(model, "schema", None),
            "prediction_horizon": model.prediction_horizon,
            "action_horizon": model.action_horizon,
            "observation_noise": model.observation_noise,
        },
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": vars(args),
        "run_id": run_id,
        "torch_rng": torch.get_rng_state(),
        "sampling_rng": sampling_rng_state,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": (
            {"train": torch.cuda.get_rng_state(args.train_device)}
            if torch.cuda.is_initialized()
            and str(getattr(args, "train_device", "cpu")).startswith("cuda:")
            else None
        ),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_policy(path: str, device: str) -> tuple[ChunkBC, dict[str, Any]]:
    """Load a supported chunk-policy checkpoint; reject legacy model schemas."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["model"]
    policy_config = state.get("policy_config", {})
    if policy_config.get("type") != "chunk_pose":
        raise ValueError("Only ChunkBC pose checkpoints are supported")
    if (
        policy_config.get("observation_mode") != "geometry"
        or policy_config.get("arm_command_mode") != "pose"
    ):
        raise ValueError("Chunk BC requires geometry observations and pose requests")
    model = ChunkBC(
        weights["mean"],
        weights["std"],
        weights["target_mean"],
        weights["target_std"],
        policy_config["prediction_horizon"],
        policy_config["action_horizon"],
        policy_config["observation_noise"],
    ).to(device)
    model.load_state_dict(weights)
    model.schema = policy_config.get("schema")
    model.eval()
    return model, state
