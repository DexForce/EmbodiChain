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

"""Native author-based RigidFormer and explicit geometry/time contracts."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from importlib.resources import files
from typing import Any
import torch
from torch import nn
from embodichain.utils import configclass

__all__ = ["RIGIDFORMER_REVISION", "RigidFormerCfg", "RigidFormerModel"]
RIGIDFORMER_REVISION = "160fe0852c51cffae7b7440452b58d0b49db6df6"


@configclass
class RigidFormerCfg:
    """Configure a native model using an author architecture.

    Args:
        preset: Author MOVi architecture/configuration to load.
        architecture: Optional full author configuration. Checkpoint imports
            supply their saved configuration rather than guessing a preset.
        attention_backend: Explicit sdpa or optional flash backend.
        max_batch_bytes: Bound on geometry and context tensors before forwarding.

    Physical intervals are step_size / frame_rate seconds. The author model
    integrates in its learned unit system with dt=1; these are distinct units.
    """

    preset: str = "movi_b"
    architecture: dict[str, Any] | None = None
    attention_backend: str = "sdpa"
    max_batch_bytes: int = 512 * 1024 * 1024

    def reference_config(self) -> dict[str, Any]:
        """Return an owned author configuration without opening data assets."""
        self.validate()
        if self.preset not in ("movi_b", "movi_a", "movi_sphere"):
            raise ValueError("preset must be movi_b, movi_a, or movi_sphere")
        if self.attention_backend not in ("sdpa", "flash"):
            raise ValueError("attention_backend must be sdpa or flash")
        if type(self.max_batch_bytes) is not int or self.max_batch_bytes < 1:
            raise ValueError("max_batch_bytes must be a positive integer")
        if self.architecture is not None:
            result = deepcopy(self.architecture)
        else:
            import yaml

            result = yaml.safe_load(
                files(__package__)
                .joinpath("configs", self.preset + ".yaml")
                .read_text()
            )
        if not result.get("model", {}).get("no_global_state", False):
            raise ValueError(
                "The supported author MOVi family requires no_global_state=true"
            )
        if result["model"].get("kabsch_reference") != "fixed":
            raise ValueError(
                "The supported author MOVi family requires fixed Kabsch reference geometry"
            )
        return result


class RigidFormerModel(nn.Module):
    """Predict rigid geometry with the author architecture and weight layout.

    Args:
        cfg: Native architecture, optional attention backend and geometry budget.

    Construction loads optional numerical dependencies, but no data, weights,
    examples or simulation. Inputs follow the author's batch layout. Training
    uses shifted conditioning frames; free rollout takes only warmup frames.
    """

    action_conditioned: bool = False
    qualified_state_gradients: bool = False

    def __init__(self, cfg: RigidFormerCfg | None = None) -> None:
        super().__init__()
        self.cfg = cfg or RigidFormerCfg()
        self.reference_config = self.cfg.reference_config()
        try:
            import roma
            from ._native.network import CUT3RDynamicsModel
        except ImportError as exc:
            raise ImportError(
                "Native RigidFormer requires embodichain[rigidformer] (einops and roma); the external rigidformer package is not used."
            ) from exc
        self.network = CUT3RDynamicsModel(deepcopy(self.reference_config))
        for module in self.network.modules():
            module.attention_backend = self.cfg.attention_backend

    def _check(self, batch: Mapping[str, Any], *, min_frames: int) -> None:
        positions = batch.get("obj_positions")
        if (
            not isinstance(positions, torch.Tensor)
            or positions.ndim != 5
            or positions.shape[-1] != 3
            or min(positions.shape[:4]) < 1
        ):
            raise ValueError("obj_positions must be floating geometry [B,T,O,P,3]")
        if not positions.is_floating_point() or positions.shape[1] < min_frames:
            raise ValueError(f"Geometry requires at least {min_frames} frames")
        b, t, o, p, _ = positions.shape
        for key, shape in (
            ("valid_obj_mask", (b, o)),
            ("valid_vtx_mask", (b, t, o, p)),
        ):
            value = batch.get(key)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != torch.bool
                or value.device != positions.device
            ):
                raise ValueError(
                    f"{key} must be a bool tensor {shape} on the geometry device"
                )
        valid = batch["valid_obj_mask"]
        if not (valid.sum(dim=-1) >= 2).all():
            raise ValueError(
                "The author nearest-body geometry requires at least two valid objects per scene"
            )
        anchor = batch.get("anchor_idx")
        count = self.reference_config["data"]["anchor_k"]
        if (
            not isinstance(anchor, torch.Tensor)
            or anchor.shape != (b, o, count)
            or anchor.dtype != torch.long
            or anchor.device != positions.device
        ):
            raise ValueError(
                "anchor_idx must be int64 [B,O,anchor_k] on the geometry device"
            )
        if (anchor < 0).any() or (anchor >= p).any():
            raise ValueError("Anchors must index stored points")
        anchor_mask = torch.gather(
            batch["valid_vtx_mask"], 3, anchor[:, None].expand(b, t, o, count)
        )
        if not (anchor_mask | ~valid[:, None, :, None]).all():
            raise ValueError(
                "Every valid object anchor must index valid points at every frame"
            )
        if not torch.isfinite(positions).all():
            raise ValueError("Geometry, including padding, must be finite")
        ref = batch.get("ref_obj_positions")
        if (
            not isinstance(ref, torch.Tensor)
            or ref.shape != (b, o, p, 3)
            or ref.device != positions.device
            or not torch.isfinite(ref).all()
        ):
            raise ValueError(
                "ref_obj_positions must be finite aligned geometry [B,O,P,3]"
            )
        props = batch.get("phys_params")
        if (
            not isinstance(props, torch.Tensor)
            or props.shape != (b, o, 3)
            or props.device != positions.device
            or not torch.isfinite(props).all()
        ):
            raise ValueError(
                "phys_params must be finite [B,O,3] mass/friction/restitution"
            )
        step = batch.get("skip_step_size")
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != (b,)
            or step.device != positions.device
            or not torch.isfinite(step).all()
            or (step <= 0).any()
        ):
            raise ValueError("skip_step_size must be positive finite [B]")
        required = sum(
            v.numel() * v.element_size()
            for v in batch.values()
            if isinstance(v, torch.Tensor)
        )
        if required > self.cfg.max_batch_bytes:
            raise ValueError("Input geometry exceeds max_batch_bytes")
        if positions.device.type != "cuda":
            raise ValueError(
                "The author geometry/KNN path requires CUDA; CPU supports import and checkpoint inspection only"
            )

    def forward(
        self, batch: Mapping[str, Any], *, teacher_forcing_ratio: float = 1.0
    ) -> dict[str, torch.Tensor]:
        """Predict shifted training frames using explicit causal conditioning.

        Args:
            batch: Author-layout geometry, masks, anchors and physical context.
            teacher_forcing_ratio: Ratio in [0,1]; one matches released recipes.

        Returns:
            Author prediction fields with frames aligned to positions[:,2:].
        """
        self._check(batch, min_frames=3)
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0,1]")
        return self.network(dict(batch), teacher_forcing_ratio=teacher_forcing_ratio)

    def rollout(
        self, batch: Mapping[str, Any], *, num_frames: int, warmup_frames: int = 4
    ) -> dict[str, torch.Tensor]:
        """Predict a free trajectory from warmup geometry.

        Args:
            batch: Geometry containing exactly the declared warmup frames.
            num_frames: Total returned frames including warmup.
            warmup_frames: Initial ground-truth frames; four matches evaluation.

        Returns:
            Author rollout fields. This inference method does not retain autograd.
        """
        self._check(batch, min_frames=warmup_frames)
        if batch["obj_positions"].shape[1] != warmup_frames:
            raise ValueError("Rollout inputs must contain exactly warmup_frames")
        if (
            type(num_frames) is not int
            or num_frames <= warmup_frames
            or warmup_frames < 2
        ):
            raise ValueError("num_frames must exceed at least two warmup frames")
        modes = [(m, m.training) for m in self.network.modules()]
        try:
            return self.network.rollout(
                dict(batch),
                num_steps=num_frames,
                warmup_frames=warmup_frames,
                return_extras=False,
            )
        finally:
            for module, mode in modes:
                module.training = mode
