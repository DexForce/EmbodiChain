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

"""Joint latent prediction and single-device SIGReg objective."""

from __future__ import annotations

import torch
from torch import nn

from embodichain.learning.sl import LossResult, SupervisedBatch

from ._architecture import SIGReg
from .model import LeWM, preprocess_rgb

__all__ = ["LeWMObjective"]


class LeWMObjective:
    """Reference next-embedding objective with a Gaussian regularizer.

    Args:
        sigreg_weight: Weight of the SIGReg term.
        knots: Number of numerical integration knots.
        num_proj: Number of random projection directions.
    """

    def __init__(
        self, sigreg_weight: float = 0.09, knots: int = 17, num_proj: int = 1024
    ) -> None:
        if not 0 <= sigreg_weight < float("inf") or knots < 2 or num_proj < 1:
            raise ValueError("invalid SIGReg configuration")
        self.weight = float(sigreg_weight)
        self.sigreg = SIGReg(knots=knots, num_proj=num_proj)

    def compute_loss(self, model: nn.Module, batch: SupervisedBatch) -> LossResult:
        """Predict all next frames in a complete aligned context window.

        Args:
            model: LeWM model.
            batch: uint8 HWC images and outgoing raw actions.

        Returns:
            Joint differentiable loss and its separately reported terms.
        """
        if not isinstance(model, LeWM):
            raise TypeError("LeWMObjective requires LeWM")
        frames, actions = batch.inputs["pixels"], batch.inputs["actions"]
        h = model.cfg.history_size
        if frames.shape[1] != h + 1 or actions.shape[1] != h:
            raise ValueError("training windows must contain H+1 frames and H actions")
        self.sigreg.to(frames.device)
        embeddings = model.encode(preprocess_rgb(frames, model.cfg.image_size))
        predicted = model.predict(embeddings[:, :-1], actions)
        prediction_loss = (predicted - embeddings[:, 1:]).square().mean()
        with torch.autocast(device_type=frames.device.type, enabled=False):
            regularization = self.sigreg(embeddings.float().transpose(0, 1))
        loss = prediction_loss + self.weight * regularization
        return LossResult(
            loss,
            {
                "prediction_loss": prediction_loss.detach(),
                "sigreg_loss": regularization.detach(),
            },
            float(frames.shape[0]),
        )
