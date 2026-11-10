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

"""Storage-independent contracts for causal training sequences."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from tensordict import TensorDict

from embodichain.utils import configclass

__all__ = [
    "SequenceSpecCfg",
    "BatchSource",
    "DataBudgetExhaustedError",
    "DataUnavailableError",
]


class DataBudgetExhaustedError(RuntimeError):
    """A complete requested batch exceeds eligible source delivery allowances."""


class DataUnavailableError(RuntimeError):
    """No eligible unlocked window exists under the current source/filter state.

    This is distinct from quota depletion and a terminal producer/read failure.
    The caller decides whether new publication can help and whether to wait
    with a deadline; a fixed offline source may never satisfy the request.
    Legacy consumers catching RuntimeError remain compatible.
    """


@configclass
class SequenceSpecCfg:
    """Specify causal history and current/future supervision.

    Args:
        history_length: Number of observation frames, including the current one.
        prediction_horizon: Number of target frames, including the current one.
        sampling_mode: Allow segment crossings within an episode, or restrict
            context to one accepted segment. Continuity changes always pad.
        generations: Optional source-generation filter. Missing generation
            metadata raises rather than silently ignoring this filter.

    The current index is ``history_length - 1`` in a sequence of length
    ``history_length + prediction_horizon - 1``. Numeric padding is zero,
    source/language padding is -1, and only real current/future frames enter
    ``loss_mask``. Boundary-crossing chunk sampling remains an engine chunk API.
    """

    history_length: int = 2
    prediction_horizon: int = 64
    sampling_mode: Literal["episode", "segment"] = "episode"
    generations: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self._check()

    def _check(self) -> None:
        if any(
            type(v) is not int or v < 1
            for v in (self.history_length, self.prediction_horizon)
        ):
            raise ValueError("Context lengths must be positive integers")
        if self.sampling_mode not in {"episode", "segment"}:
            raise ValueError("Context sampling_mode must be episode or segment")
        if self.generations is not None and (
            not isinstance(self.generations, tuple)
            or not self.generations
            or any(type(v) is not int or v < 1 for v in self.generations)
        ):
            raise ValueError(
                "generations must be a nonempty tuple of positive integers"
            )


@runtime_checkable
class BatchSource(Protocol):
    """A source of owned training batches with explicit data semantics.

    ``schema`` declares ordered observations/actions, their timing and units.
    ``source_id`` identifies the source dataset/run independently of cache slots.
    Production/refill and loading lifecycles belong to concrete sources and the
    caller, not to this sampling protocol. Sampling never performs hidden waits.
    """

    schema: Mapping[str, Any]
    source_id: str

    def sample_context(
        self,
        batch_size: int,
        spec: SequenceSpecCfg,
        *,
        generator: torch.Generator | None = None,
    ) -> TensorDict:
        """Return copied ``obs/actions/valid/loss_mask`` and source metadata.

        Args:
            batch_size: Number of episode-uniform contexts, with replacement.
            spec: Window, boundary and optional generation requirements.
            generator: Sampling RNG; ``None`` uses the current Torch RNG.

        Returns:
            TensorDict with batch axes ``[B, H + F - 1]``. Returned storage
            remains valid after subsequent sampling or source replacement.

        Raises:
            DataBudgetExhaustedError: Eligible delivery credits cannot provide
                the entire batch. No partial debit is performed.
            DataUnavailableError: No eligible anchor exists in the current data.
            RuntimeError: A producer or source read failed.
            ValueError: Schema or requested capabilities are incompatible.
        """
        ...
