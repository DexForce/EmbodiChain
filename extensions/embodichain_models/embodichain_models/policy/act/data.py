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

"""Causal visual/action sequence adaptation for ACT."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from tensordict import TensorDict

from embodichain.data_pipeline.datasets import (
    LeRobotDataSource,
    LeRobotImageReader,
    SequenceSpecCfg,
)
from embodichain.learning.sl import SequenceBatchSource, SupervisedBatch

__all__ = ["ACTSequenceSource"]


class ACTSequenceSource:
    """Join current RGB frames to immutable numeric action windows.

    Both readers must refer to the same unmodified dataset, using SDK
    episode/frame identities. Only the anchor image is decoded per window.

    Args:
        numeric: Resident numeric LeRobot source.
        images: Bounded image reader from the same dataset.
        horizon: Current and future actions per training sample.
    """

    def __init__(
        self, numeric: LeRobotDataSource, images: LeRobotImageReader, horizon: int
    ) -> None:
        root = numeric.metadata.get("storage", {}).get("root")
        if not root or Path(root).resolve() != images.root:
            raise ValueError("ACT numeric and image readers must share a dataset root")
        images.validate_frame_order(numeric.data["frame_ids"][numeric.data["valid"]])
        self.images = images

        def encode(context: TensorDict) -> SupervisedBatch:
            return SupervisedBatch(
                inputs={
                    "rgb": images.read(context["frame_ids"][:, 0]),
                    "state": context["obs"][:, 0],
                },
                targets=context["actions"],
                masks={"valid": context["loss_mask"]},
            )

        self.bridge = SequenceBatchSource(
            numeric,
            SequenceSpecCfg(history_length=1, prediction_horizon=horizon),
            encoder=encode,
            encoder_version="act-rgb-current-action-chunk-v1",
            immutable=True,
        )
        self.source_id = (
            "act:"
            + hashlib.sha256(
                f"{self.bridge.source_id}:{images.content_id}".encode()
            ).hexdigest()
        )

    def sample(self, batch_size: int, *, generator: torch.Generator) -> SupervisedBatch:
        """Sample owned aligned RGB, state and action windows.

        Args:
            batch_size: Number of windows.
            generator: Caller-owned sampling RNG.

        Returns:
            A supervised batch with one RGB observation per action chunk.
        """
        return self.bridge.sample(batch_size, generator=generator)

    def validate_resume(self) -> None:
        """Reject recovery if numeric or image data have changed."""
        self.bridge.validate_resume()
        if self.images.fingerprint() != self.images.content_id:
            raise ValueError("ACT image dataset content changed")
