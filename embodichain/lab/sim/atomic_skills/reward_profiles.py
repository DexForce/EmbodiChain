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

"""Execution-independent reward templates owned by stable Atomic Skill identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

__all__ = [
    "SkillRewardProfile",
    "default_skill_reward_profile",
    "register_skill_reward_profile",
    "skill_reward_profiles",
]


@dataclass(frozen=True, slots=True)
class SkillRewardProfile:
    """Immutable measured potential template, independent of action producers.

    Args:
        skill_id: Canonical Atomic Skill identifier.
        revision: Positive revision persisted with effective reward configs.
        roles: Explicit scalar measured scores, each in [0, 1].
        reduction: Bounded combination. None selects the built-in skill formula.
            A trusted subclass can override ``potential`` and declare only the
            roles consumed by its custom formula. Inherited built-in formulas
            require their complete role set.
    """

    skill_id: str
    revision: int
    roles: tuple[str, ...]
    reduction: Literal["mean", "product"] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.skill_id, str)
            or not self.skill_id
            or self.skill_id != self.skill_id.strip()
        ):
            raise ValueError("Reward profile requires an exact skill identifier")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Reward profile revision must be a positive integer")
        roles = tuple(self.roles)
        if (
            not roles
            or len(set(roles)) != len(roles)
            or any(not isinstance(r, str) or not r or r != r.strip() for r in roles)
        ):
            raise ValueError("Reward profile requires unique nonempty measured roles")
        if self.reduction not in (None, "mean", "product"):
            raise ValueError("Unknown potential reduction")
        required = _DEFAULT_ROLES.get(self.skill_id)
        if (
            self.reduction is None
            and type(self).potential is SkillRewardProfile.potential
            and required is not None
            and set(roles) != set(required)
        ):
            raise ValueError(
                f"Built-in {self.skill_id} formula requires roles {required}"
            )
        object.__setattr__(self, "roles", roles)

    def potential(
        self, scores: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Combine measured scores without acquiring observations or changing state.

        Args:
            scores: Exact role-to-score mapping, with aligned floating batch vectors.

        Returns:
            Bounded potential and per-row validity. Nonfinite/out-of-range scores
            produce unavailable evidence; they are never converted into success.
        """
        if set(scores) != set(self.roles):
            raise ValueError(f"{self.skill_id} requires roles {self.roles}")
        values = [scores[role] for role in self.roles]
        first = values[0]
        if first.ndim != 1 or not first.is_floating_point():
            raise ValueError("Skill scores must be floating point batch vectors")
        if any(
            v.shape != first.shape or v.device != first.device or v.dtype != first.dtype
            for v in values
        ):
            raise ValueError("Skill scores must share shape, dtype and device")
        stacked = torch.stack(values)
        valid = (torch.isfinite(stacked) & (stacked >= 0) & (stacked <= 1)).all(0)
        if self.reduction == "product":
            result = stacked.prod(0)
        elif self.reduction == "mean":
            result = stacked.mean(0)
        elif self.skill_id == "pick_up":
            reach, held, lifted = (scores[role] for role in ("reach", "held", "lifted"))
            result = (reach + held + held * lifted) / 3
        elif self.skill_id == "place":
            pose, released, stable = (
                scores[role] for role in ("pose", "released", "stable")
            )
            result = (pose + pose * released + pose * released * stable) / 3
        elif self.skill_id == "move_held_object":
            pose, held = scores["pose"], scores["held"]
            result = pose * held
        else:
            result = stacked.mean(0)
        return torch.where(valid, result, 0.0), valid


_DEFAULT_ROLES = {
    "pick_up": ("reach", "held", "lifted"),
    "place": ("pose", "released", "stable"),
    "move_held_object": ("pose", "held"),
    "move_end_effector": ("position", "orientation"),
    "move_joints": ("joints",),
}
_PROFILES: dict[str, dict[str, SkillRewardProfile]] = {
    skill: {"default": SkillRewardProfile(skill, 1, roles)}
    for skill, roles in _DEFAULT_ROLES.items()
}


def register_skill_reward_profile(name: str, profile: SkillRewardProfile) -> None:
    """Register a trusted named reward template before configuration assembly.

    Args:
        name: Unique profile name within this skill; default is reserved.
        profile: Immutable template, optionally a trusted subclass.

    Raises:
        ValueError: A name would overwrite an existing template.
    """
    if not isinstance(profile, SkillRewardProfile):
        raise TypeError("Reward profiles must implement SkillRewardProfile")
    if (
        not isinstance(name, str)
        or not name
        or name != name.strip()
        or name == "measured_potential"
    ):
        raise ValueError("Profile name must be an exact nonempty identifier")
    profiles = _PROFILES.setdefault(profile.skill_id, {})
    if name in profiles:
        raise ValueError(f"Reward profile {profile.skill_id}/{name} already exists")
    profiles[name] = profile


def skill_reward_profiles(skill_id: str) -> dict[str, SkillRewardProfile]:
    """Return installed named templates without constructing a planner.

    Args:
        skill_id: Exact skill identifier from the existing Atomic Skill catalog.

    Returns:
        Owned mapping of immutable templates. Empty means no declared template.
    """
    return dict(_PROFILES.get(skill_id, {}))


def default_skill_reward_profile(skill_id: str) -> SkillRewardProfile:
    """Resolve a skill default; missing templates reject explicit reward requests.

    Args:
        skill_id: Exact Atomic Skill identifier.

    Returns:
        Shared immutable default template.
    """
    try:
        return _PROFILES[skill_id]["default"]
    except KeyError as error:
        raise ValueError(f"No default reward profile for {skill_id!r}") from error
