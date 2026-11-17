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

"""Expand measured skill recipes into ordinary RewardManager terms."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any, TYPE_CHECKING
from collections.abc import Sequence

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from .observation_types import ObservationFrame
import torch

from embodichain.compute.rewards import potential_shaping
from embodichain.lab.sim.atomic_skills.reward_profiles import (
    default_skill_reward_profile,
    skill_reward_profiles,
)
from .cfg import RewardCfg
from .manager_base import Functor

__all__ = ["ComposedPotentialReward", "compose_reward_terms"]


def _measure(frame, declaration):
    field = declaration["field"]
    sample = frame.fields.get(field)
    if sample is None:
        return None, None
    value = sample.values
    if "index" in declaration:
        value = value[..., declaration["index"]]
    valid = sample.valid.clone()
    if "target" in declaration:
        target = value.new_tensor(declaration["target"])
        if target.shape != value.shape[1:]:
            raise ValueError("Reward target must match the described observation shape")
        error = value - target
        value = (
            error.abs()
            if error.ndim == 1
            else torch.linalg.vector_norm(error.flatten(1), dim=-1)
        )
    if value.ndim != 1:
        raise ValueError(
            "A reward score must be scalar per environment; supply an explicit target for vector errors"
        )
    if "scale" in declaration:
        value = 1 - torch.tanh(value / declaration["scale"])
    return value.float(), valid


class ComposedPotentialReward(Functor):
    """One potential state owner for an entire skill or Task Program objective.

    Instances are constructed by RewardManager from generated RewardCfg values.
    All measurements are read from the host's canonical frame. Success remains
    owned by TerminationManager, independently of this shaping term.
    """

    def __init__(self, cfg: RewardCfg, env: EmbodiedEnv) -> None:
        super().__init__(cfg, env)
        self.discount = cfg.params["discount"]
        self.skills = deepcopy(cfg.params["skills"])
        self.program = cfg.params.get("program", False)
        self.profiles = [
            (
                None
                if s.get("profile") == "measured_potential"
                else skill_reward_profiles(s["skill_id"])[s.get("profile", "default")]
            )
            for s in self.skills
        ]
        self._prepared = None
        self._prepared_identity = None
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        self.valid = torch.ones_like(self.initialized)

    def _potential(self, env, frame):
        if self.program:
            progress = env.task_program_progress
            current, count = progress.ordered_progress()
            potential = current.float() / count
        else:
            count = 1
            current = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
            potential = torch.zeros_like(self.previous)
        valid = torch.ones_like(self.initialized)
        for declaration, profile in zip(self.skills, self.profiles, strict=True):
            scores, measured = {}, torch.ones_like(valid)
            for role, binding in declaration["metrics"].items():
                scores[role], available = _measure(frame, binding)
                if available is None:
                    scores[role] = torch.zeros_like(self.previous)
                    available = torch.zeros_like(measured)
                measured &= available
            if profile is None:
                score = scores["potential"]
                available = torch.isfinite(score) & (score >= 0) & (score <= 1)
            else:
                score, available = profile.potential(scores)
            if self.program:
                indices = progress.indices(declaration["stage"])
                active = torch.stack([current == index for index in indices]).any(0)
            else:
                active = torch.ones_like(valid)
            available &= torch.isfinite(score) & (score >= 0) & (score <= 1)
            score = torch.where(available & measured, score, 0.0)
            potential += torch.where(active, score / count, 0)
            valid &= ~active | (measured & available)
        return potential, valid

    def initialize_observation(
        self,
        env: EmbodiedEnv,
        frame: ObservationFrame,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Seed selected baselines from reset observations without paying rewards.

        Args:
            env: Environment owning the canonical measurement frame.
            frame: Final acquisition after reset and scene initialization.
            env_ids: Episode rows whose potential baselines must be replaced.
        """
        potential, valid = self._potential(env, frame)
        self.previous[env_ids] = potential[env_ids]
        self.initialized[env_ids] = valid[env_ids]
        self.valid[env_ids] = valid[env_ids]

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear only the selected reward histories.

        Args:
            env_ids: Rows starting new episodes; None selects the whole batch.
        """
        ids = slice(None) if env_ids is None else env_ids
        self.previous[ids] = 0
        self.initialized[ids] = False
        self.valid[ids] = True
        self._prepared = None
        self._prepared_identity = None

    def prepare_observation(
        self, env: EmbodiedEnv, frame: ObservationFrame
    ) -> torch.Tensor:
        """Prepare measured potentials before the host finalizes termination.

        Args:
            env: Owning environment and its measured program progress.
            frame: Canonical post-step acquisition, before any automatic reset.

        Returns:
            Per-row evidence validity, including the initialized reset baseline.
        """
        self._prepared = self._potential(env, frame)
        self._prepared_identity = (frame.revision, frame.episode_ids)
        return self._prepared[1] & self.initialized

    def __call__(
        self,
        env: EmbodiedEnv,
        obs: Any,
        action: Any,
        info: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        context = env.reward_manager.transition_context
        if context is None:
            raise ValueError("Composed rewards require a host RewardTransitionContext")
        if self._prepared is not None and self._prepared_identity == context.identity:
            current, valid = self._prepared
        else:
            current, valid = self._potential(env, context.frame)
        self._prepared = None
        self._prepared_identity = None
        self.valid.copy_(valid & self.initialized)
        value = potential_shaping(
            self.previous, current, context.terminated, discount=self.discount
        )
        self.previous.copy_(current)
        self.initialized.copy_(valid)
        # Invalid evidence is reported explicitly; it cannot pay shaping.
        info.setdefault("reward_validity", {})[self.name] = self.valid.clone()
        return torch.where(self.valid | context.terminated, value, 0)


def compose_reward_terms(
    declaration: Mapping[str, Any], *, program: bool
) -> dict[str, RewardCfg]:
    """Compile one explicit recipe selection into standard manager terms.

    Args:
        declaration: Discount, weight and skill/metric bindings. Metrics select
            exact raw observation fields, optional vector targets and scales.
        program: Whether stage selectors bind TaskProgramProgress occurrences.

    Returns:
        Named RewardCfg values for the existing RewardManager.
    """
    if not isinstance(declaration, Mapping) or set(declaration) - {
        "discount",
        "weight",
        "skills",
        "success",
        "failure",
        "milestone_budget",
        "time_cost",
        "profiles",
    }:
        raise ValueError("reward_composition accepts discount, weight and skills")
    discount = declaration.get("discount")
    weight = declaration.get("weight", 1.0)
    if (
        isinstance(discount, bool)
        or not isinstance(discount, (float, int))
        or not math.isfinite(discount)
        or not 0 <= discount <= 1
    ):
        raise ValueError("reward_composition requires an explicit discount in [0, 1]")
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (float, int))
        or not math.isfinite(weight)
    ):
        raise ValueError("Reward weight must be finite")
    skills = deepcopy(declaration.get("skills", []))
    if not isinstance(skills, list) or (not program and len(skills) != 1):
        raise ValueError("Standalone composition requires exactly one skill")
    stages = set()
    for skill in skills:
        if (
            not isinstance(skill, dict)
            or set(skill)
            - (
                {"skill_id", "metrics", "stage", "profile", "source_profile"}
                if program
                else {"skill_id", "metrics", "profile", "source_profile"}
            )
            or not {"skill_id", "metrics"} <= set(skill)
            or (program and "stage" not in skill)
        ):
            raise ValueError(
                "Each skill requires skill_id, metrics, and a stage for programs"
            )
        profiles = skill_reward_profiles(skill["skill_id"])
        selected = skill.get("profile", "default")
        if selected == "measured_potential":
            profile = default_skill_reward_profile(skill["skill_id"])
        elif selected not in profiles:
            raise ValueError(f"Unknown reward profile {skill['skill_id']}/{selected}")
        else:
            profile = profiles[selected]
        if "source_profile" in skill and (
            selected != "measured_potential" or skill["source_profile"] not in profiles
        ):
            raise ValueError(
                "Measured source_profile must name an installed skill template"
            )
        if set(skill["metrics"]) != (
            {"potential"}
            if skill.get("profile") == "measured_potential"
            else set(profile.roles)
        ):
            raise ValueError(
                f"{profile.skill_id} requires measured roles {profile.roles}"
            )
        if program:
            if skill["stage"] in stages:
                raise ValueError("Duplicate skill shaping for one stage")
            stages.add(skill["stage"])
        for metric in skill["metrics"].values():
            if (
                not isinstance(metric, dict)
                or "field" not in metric
                or set(metric) - {"field", "target", "scale", "index"}
            ):
                raise ValueError(
                    "Metric binding requires field, optional target and scale"
                )
            if not isinstance(metric["field"], str) or not metric["field"]:
                raise ValueError("Metric field must be a nonempty raw channel name")
            if "index" in metric and (
                type(metric["index"]) is not int or metric["index"] < 0
            ):
                raise ValueError("Metric index must be a nonnegative integer")
            if "scale" in metric:
                scale = metric["scale"]
                if (
                    isinstance(scale, bool)
                    or not isinstance(scale, (int, float))
                    or not math.isfinite(scale)
                    or scale <= 0
                ):
                    raise ValueError("Metric scale must be finite and positive")
    for name in ("success", "failure", "milestone_budget", "time_cost"):
        value = declaration.get(name, 0.0)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Reward {name} must be finite")
    if (
        declaration.get("milestone_budget", 1.0) < 0
        or declaration.get("time_cost", 0.0) < 0
    ):
        raise ValueError("Milestone budget and time cost must be nonnegative")
    revisions = []
    for skill in skills:
        selected = skill.get("profile", "default")
        if selected == "measured_potential":
            source = skill.get("source_profile", "default")
            source_profile = skill_reward_profiles(skill["skill_id"])[source]
            label = selected if source == "default" else f"measured:{source}"
            revisions.append((skill["skill_id"], label, source_profile.revision))
        else:
            revisions.append(
                (
                    skill["skill_id"],
                    selected,
                    skill_reward_profiles(skill["skill_id"])[selected].revision,
                )
            )
    expected = declaration.get("profiles")
    if expected is not None and list(map(tuple, expected)) != revisions:
        raise ValueError(
            "Installed skill reward profiles differ from the saved reward contract"
        )
    return {
        "composed_potential": RewardCfg(
            func=ComposedPotentialReward,
            weight=weight,
            params={"discount": discount, "skills": skills, "program": program},
            extra={"profile_revisions": revisions},
        )
    }


def _bind_skill_objectives(cfg):
    from .skill_objective import _bind_objective

    composition = cfg.rl_task.reward_composition
    if cfg.rl_task.skill is not None:
        if composition is None:
            raise ValueError(
                "Standalone skill requires reward_composition with explicit discount"
            )
        generated = _bind_objective(cfg, "standalone", cfg.rl_task.skill)
        if "skills" in composition and composition["skills"] != [generated]:
            raise ValueError("Standalone skill owns its generated reward bindings")
        composition["skills"] = [generated]
    elif composition is not None:
        generated = []
        for index, item in enumerate(composition.get("skills", [])):
            if "objective" in item:
                if (
                    set(item)
                    not in ({"stage", "objective"}, {"call_index", "objective"})
                    or cfg.task_program is None
                ):
                    raise ValueError(
                        "Program objectives require stage or call_index, and objective"
                    )
                if "call_index" in item and (
                    type(item["call_index"]) is not int or item["call_index"] < 0
                ):
                    raise ValueError(
                        "call_index must be a nonnegative compiled call occurrence"
                    )
                selector = item.get("stage", f"call:{item.get('call_index')}")
                generated.append(
                    _bind_objective(cfg, str(index), item["objective"], stage=selector)
                )
            else:
                generated.append(item)
        composition["skills"] = generated


class _OutcomeReward(Functor):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._paid = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Clear the selected episodes' once-only outcome payment flags."""
        self._paid[slice(None) if env_ids is None else env_ids] = False

    def __call__(self, env, obs, action, info, **params):
        outcome = info["task_outcome"]
        success = outcome["accepted"]
        failure = outcome["failure"]
        terminal = success | failure
        reward = torch.where(
            success, params["success"], torch.where(failure, params["failure"], 0.0)
        )
        reward = torch.where(self._paid, 0.0, reward)
        self._paid |= terminal
        return reward - params["time_cost"] * env.step_dt


def _add_common_rewards(cfg, generated, stages):
    from .rewards import success_reward, program_stage_reward

    declaration = cfg.rl_task.reward_composition
    explicit = [
        term for term in vars(cfg.rewards).values() if isinstance(term, RewardCfg)
    ]
    success = declaration.get(
        "success", 0.0 if any(t.func is success_reward for t in explicit) else 10.0
    )
    generated["composed_outcome"] = RewardCfg(
        func=_OutcomeReward,
        params={
            "success": success,
            "failure": declaration.get("failure", -10.0),
            "time_cost": declaration.get("time_cost", 0.0),
        },
    )
    if stages:
        budget = declaration.get("milestone_budget", 1.0)
        for index, stage in enumerate(stages):
            if any(
                t.func is program_stage_reward
                and t.params.get("mode") == "first_achievement"
                and t.params.get("stage") in (stage.name, stage.segment_id)
                for t in explicit
            ):
                continue
            generated[f"composed_stage_{index}"] = RewardCfg(
                func=program_stage_reward,
                weight=budget / len(stages),
                params={
                    "stage": stage.segment_id,
                    "mode": "first_achievement",
                    "scope": "occurrence",
                },
            )


def _snapshot_reward_composition(cfg, declaration):
    """Bind the effective template revisions while retaining authored objective bindings."""
    result = deepcopy(getattr(cfg.task_program, "rewards", None) or {})
    result.update(deepcopy(declaration or {}))
    effective = cfg.rl_task.reward_composition
    for name in (
        "discount",
        "weight",
        "success",
        "failure",
        "milestone_budget",
        "time_cost",
    ):
        if name in effective:
            result[name] = deepcopy(effective[name])
    generated = compose_reward_terms(effective, program=cfg.task_program is not None)
    if "skills" in result:
        result["skills"] = [
            (
                {
                    **item,
                    "objective": deepcopy(
                        getattr(cfg.observations, f"skill_objective_{index}").params[
                            "objective"
                        ]
                    ),
                }
                if "objective" in item
                else item
            )
            for index, item in enumerate(result["skills"])
        ]
    result["profiles"] = [
        list(item)
        for item in generated["composed_potential"].extra["profile_revisions"]
    ]
    return result
