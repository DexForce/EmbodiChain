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

"""Host policy and row provenance for offline Affordance collection.

This module does not schedule trajectory candidates or change physical slots.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Literal, TYPE_CHECKING

from embodichain.utils import configclass

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv

__all__ = ["AffordanceAugmentationCfg"]


@configclass
class AffordanceAugmentationCfg:
    """Opt-in policy for synchronous, measured offline Affordance episodes.

    Args:
        branches: Sampling branches including nominal; one disables variation.
        max_batches: Bound on all attempted vector rollouts, including failures.
        required_assurance: Required physical acceptance, currently measured only.
    """

    branches: int = 1
    max_batches: int = 200
    required_assurance: Literal["measured"] = "measured"

    def __post_init__(self) -> None:
        for name in ("branches", "max_batches"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.required_assurance != "measured":
            raise ValueError("Affordance collection requires measured acceptance.")

    def validate_host(self, *, num_envs: int, seed: int | None) -> None:
        """Validate the resolved physical batch and explicit sampling seed.

        Args:
            num_envs: Number of physical simulation rows.
            seed: Explicit, non-negative environment seed.
        """
        self.__post_init__()
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer.")
        if self.branches != 1 and self.branches != num_envs:
            raise ValueError(
                "Affordance branches must equal num_envs (or be nominal 1)."
            )
        if type(seed) is not int or seed < 0:
            raise ValueError(
                "Affordance collection requires an explicit non-negative seed."
            )


@contextmanager
def _sampling_attempt(
    env: EmbodiedEnv,
    cfg: AffordanceAugmentationCfg,
    *,
    run_id: str,
    batch_id: int,
    attempt_id: int,
) -> Iterator[None]:
    """Install caller-owned identity only for one offline collection attempt."""
    from embodichain.lab.sim.atomic_actions.affordance_sampling import (
        AffordanceSamplingContext,
    )

    if getattr(env, "_affordance_collection_metadata", None) is not None:
        raise RuntimeError("An Affordance collection attempt is already active.")
    cfg.validate_host(num_envs=env.num_envs, seed=env.cfg.seed)
    sampling = AffordanceSamplingContext(
        count=cfg.branches,
        seed=env.cfg.seed,
        episode_id=batch_id,
        attempt_id=attempt_id,
    )
    env._affordance_sampling_context = sampling if sampling.enabled else None
    env._affordance_collection_metadata = {
        "augmentation_schema_version": 1,
        "run_id": run_id,
        "collection_batch_id": batch_id,
        "sampling_attempt_id": attempt_id,
        "sampling": sampling.metadata(),
        "required_assurance": cfg.required_assurance,
    }
    try:
        yield
    finally:
        env._affordance_sampling_context = None
        env._affordance_collection_metadata = None


def _select_accepted_rows(
    *,
    completed: tuple[bool, ...],
    success: tuple[bool, ...],
    measured: tuple[bool, ...],
    lengths: tuple[int, ...],
    remaining: int,
) -> tuple[int, ...]:
    """Select successful, nonempty measured episodes within the remaining quota."""
    if type(remaining) is not int or remaining < 0:
        raise ValueError("remaining must be a non-negative integer.")
    return _eligible_accepted_rows(
        completed=completed,
        success=success,
        measured=measured,
        lengths=lengths,
    )[:remaining]


def _eligible_accepted_rows(
    *,
    completed: tuple[bool, ...],
    success: tuple[bool, ...],
    measured: tuple[bool, ...],
    lengths: tuple[int, ...],
) -> tuple[int, ...]:
    """Return all rows eligible before applying the collection quota.

    Keeping eligibility separate from quota selection lets the collector report
    measured acceptance and quota discards without changing the legacy selector
    contract.
    """
    if len({len(completed), len(success), len(measured), len(lengths)}) != 1:
        raise ValueError("All acceptance masks must have one value per row.")
    return tuple(
        row
        for row in range(len(success))
        if completed[row] and success[row] and measured[row] and lengths[row] > 0
    )


_SAMPLING_ROW_FIELDS = frozenset(
    {
        "candidate_ids",
        "valid_candidate_counts",
        "unique_candidate_counts",
        "reused",
        "roll",
        "selected_poses",
        "reference_poses",
        "success",
        "env_ids",
        "control_parts",
    }
)


def _project_affordance_metadata(value: Any, *, env_id: int) -> Any:
    """Copy metadata, projecting only explicit Affordance sampling row fields."""

    def project_samples(samples: Any) -> Any:
        if isinstance(samples, Mapping):
            if "key" in samples and "sampling" in samples:
                identities = samples.get("env_ids")
                if isinstance(identities, (list, tuple)):
                    if env_id not in identities:
                        return {
                            "key": samples["key"],
                            "sampling": deepcopy(samples["sampling"]),
                            "participating": False,
                        }
                    row = identities.index(env_id)
                else:
                    row = env_id
                selected = {
                    key: deepcopy(
                        item[row]
                        if key in _SAMPLING_ROW_FIELDS
                        and isinstance(item, (list, tuple))
                        else item
                    )
                    for key, item in samples.items()
                }
                if "candidate_ids" in selected:
                    selected["pool_candidate_index"] = selected["candidate_ids"]
                return selected
            return {key: project_samples(item) for key, item in samples.items()}
        return deepcopy(samples)

    if isinstance(value, Mapping):
        return {
            key: (
                project_samples(item)
                if key == "affordance_sample"
                else _project_affordance_metadata(item, env_id=env_id)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_project_affordance_metadata(item, env_id=env_id) for item in value]
    return deepcopy(value)
