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
"""Checkpoint-compatible stateless particle transitions in world coordinates."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import torch
from torch import Tensor
from ._runtime import load_assets, load_network

__all__ = ["WorldParticleModel", "load_model"]


class WorldParticleModel:
    """Own frozen weights, authored features/topology and model-frame conversion.

    Inputs and outputs use world-space particle order and SI units. Acceleration
    includes gravity; model-specific BF16 arithmetic matches official inference.
    Callers own execution streams and must create/use this object on that stream.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        inputs: tuple[Tensor, ...],
        rotation: Any,
        cfg: Mapping[str, Any],
    ) -> None:
        self.network = network
        self.inputs = inputs
        self.rotation = torch.as_tensor(rotation, device=inputs[0].device)
        self.dt = float(cfg["timestep"])
        self.k_partite = int(cfg.get("k_partite", -1))

    def __call__(
        self, position: Tensor, velocity: Tensor, acceleration: Tensor, dt: float
    ) -> tuple[Tensor, Tensor]:
        import math

        if not math.isclose(dt, self.dt, rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("WorldParticle step dt must match the checkpoint timestep")
        feature, box, box_feats = self.inputs[:3]
        arguments = dict(
            k_partite=self.k_partite, external_force=acceleration @ self.rotation
        )
        if len(self.inputs) == 5:
            arguments.update(
                topology_neighbors_index=self.inputs[3],
                topology_neighbors_row_splits=self.inputs[4],
            )
        model_position = position @ self.rotation
        model_velocity = velocity @ self.rotation
        with torch.autocast("cuda", dtype=torch.bfloat16):
            q, qd = self.network(
                (model_position, model_velocity, feature, box, box_feats), **arguments
            )
        return q.float() @ self.rotation.T, qd.float() @ self.rotation.T


def load_model(
    options: Mapping[str, Any], device: str = "cuda:0"
) -> WorldParticleModel:
    """Load validated static inputs and strict official checkpoint tensors."""
    cfg, arrays, rotation = load_assets(options)
    inputs = tuple(torch.as_tensor(value, device=device) for value in arrays)
    return WorldParticleModel(
        load_network(cfg, options["checkpoint"], device), inputs, rotation, cfg
    )
