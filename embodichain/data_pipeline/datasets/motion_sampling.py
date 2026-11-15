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

"""Adaptive reference-start sampling from per-bin tracking returns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import torch
from embodichain.utils import configclass
from .motion import MotionLibrary

__all__ = ["MotionStartSampler", "MotionStartSamplerCfg"]


@configclass
class MotionStartSamplerCfg:
    """Mixture of uniform, current low-return and past low-return probabilities."""

    bin_seconds: float = 1.0
    uniform_weight: float = 0.5
    current_weight: float = 0.4
    past_weight: float = 0.1
    temperature: float = 25.0
    probability_cap: float = 50.0
    return_ema: float = 0.1
    past_decay: float = 0.999


class MotionStartSampler:
    """Sample clip-local time bins and learn their normalized return statistics.

    The cap is a multiple of one bin's uniform share, applied to the current
    softmax component before mixing. Scores from simultaneous episodes in the
    same bin are averaged before their exponential moving average is updated.
    """

    def __init__(
        self, motion: MotionLibrary, cfg: MotionStartSamplerCfg | None = None
    ) -> None:
        self.motion, self.cfg = motion, cfg or MotionStartSamplerCfg()
        c = self.cfg
        if (
            c.bin_seconds <= 0
            or c.temperature < 0
            or c.probability_cap < 1
            or not 0 < c.return_ema <= 1
            or not 0 < c.past_decay <= 1
            or min(c.uniform_weight, c.current_weight, c.past_weight) < 0
            or abs(c.uniform_weight + c.current_weight + c.past_weight - 1) > 1e-8
        ):
            raise ValueError("Invalid reference-start sampling configuration.")
        self.width = max(1, round(c.bin_seconds * motion.fps))
        lengths = motion.lengths.cpu().tolist()
        starts = motion.starts.cpu().tolist()
        lo = []
        hi = []
        clip = []
        base = []
        for i, (start, length) in enumerate(zip(starts, lengths)):
            base.append(len(lo))
            for offset in range(0, length, self.width):
                lo.append(start + offset)
                hi.append(min(start + offset + self.width, start + length))
                clip.append(i)
        self.lower = torch.tensor(lo, device=motion.device)
        self.upper = torch.tensor(hi, device=motion.device)
        self.clips = torch.tensor(clip, device=motion.device)
        self.base = torch.tensor(base, device=motion.device)
        self.scores = torch.zeros(len(lo), device=motion.device)
        self.visits = torch.zeros(len(lo), device=motion.device)
        self.past = torch.zeros(len(lo), device=motion.device)

    def bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Map reference frames to their clip-local time bins."""
        clips = self.motion.clip_ids(frames)
        return self.base[clips] + (frames - self.motion.starts[clips]) // self.width

    @torch.no_grad()
    def probabilities(self) -> torch.Tensor:
        """Advance the past-softmax envelope and return the sampling mixture."""
        c = self.cfg
        n = self.scores.numel()
        soft = (-c.temperature * self.scores).softmax(0)
        cap = min(1.0, c.probability_cap / n)
        for _ in range(8):
            excess = (soft - cap).clamp_min(0).sum()
            free = torch.where(soft > cap, 0.0, soft)
            soft = soft.clamp_max(cap) + excess * free / free.sum().clamp_min(1e-30)
        self.past.copy_(torch.maximum(self.past * c.past_decay, soft))
        return (
            c.uniform_weight / n
            + c.current_weight * soft
            + c.past_weight * self.past / self.past.sum()
        )

    @torch.no_grad()
    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        """Draw frames leaving at least one transition before the clip ends."""
        bins = torch.multinomial(
            self.probabilities(), count, replacement=True, generator=generator
        )
        last = self.motion.ends[self.clips[bins]] - 1
        lower = self.lower[bins]
        upper = torch.minimum(self.upper[bins], last).clamp_min(lower + 1)
        frames = (
            lower
            + (
                torch.rand(count, device=self.motion.device, generator=generator)
                * (upper - lower)
            ).long()
        )
        return torch.minimum(frames, last - 1).clamp_min(
            self.motion.starts[self.clips[bins]]
        )

    @torch.no_grad()
    def update(self, frames: torch.Tensor, normalized_returns: torch.Tensor) -> None:
        """Merge completed episodes by their start frames, including repeated bins."""
        bins = self.bins(frames)
        count = torch.zeros_like(self.scores).index_add_(
            0, bins, torch.ones_like(normalized_returns)
        )
        sums = torch.zeros_like(self.scores).index_add_(0, bins, normalized_returns)
        mean = sums / count.clamp_min(1)
        merged = torch.where(
            self.visits > 0,
            (1 - self.cfg.return_ema) * self.scores + self.cfg.return_ema * mean,
            mean,
        )
        self.scores.copy_(torch.where(count > 0, merged.clamp(0, 1), self.scores))
        self.visits.add_(count)

    def state_dict(self) -> dict[str, Any]:
        """Save distribution statistics with the motion-layout identity."""
        return {
            "config": self.cfg.to_dict(),
            "names": self.motion.names,
            "lengths": self.motion.lengths.cpu().tolist(),
            "content_digest": self.motion.content_digest,
            "scores": self.scores.clone(),
            "visits": self.visits.clone(),
            "past": self.past.clone(),
        }

    def validate_state_dict(self, state: Mapping[str, Any]) -> None:
        """Validate saved bins and statistics before mutating the sampler."""
        if (
            state["config"] != self.cfg.to_dict()
            or tuple(state["names"]) != self.motion.names
            or state["lengths"] != self.motion.lengths.cpu().tolist()
            or state.get("content_digest") != self.motion.content_digest
        ):
            raise ValueError("Motion sampler checkpoint layout or content differs.")
        for key in ("scores", "visits", "past"):
            value = state[key]
            if (
                value.shape != self.scores.shape
                or not bool(torch.isfinite(value).all())
                or bool((value < 0).any())
            ):
                raise ValueError("Invalid motion sampler statistics.")

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore the current and historical curriculum distribution."""
        self.validate_state_dict(state)
        for key in ("scores", "visits", "past"):
            getattr(self, key).copy_(state[key])
