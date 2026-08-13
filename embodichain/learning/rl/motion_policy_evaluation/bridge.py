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

"""Run an EmbodiChain Motion Profile through DexSim Motion Policy Kit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from dexsim.kit.motion_policy import (
    MotionPolicyEvaluator,
    PolicyAdapter,
    PolicySpec,
    ResolvedPolicy,
    ResourceResolver,
    RunOptions,
    create_motion_policy_evaluator,
    load_scene_config,
    parse_policy_spec,
    policy_spec_to_dict,
    resolve_policy_spec,
    run_motion_policy,
    scene_config_to_dict,
)
from dexsim.kit.motion_policy.environment import PolicyEnvironment
from dexsim.kit.motion_policy.evaluator import InputProvider

from .profile import MotionProfile

__all__ = [
    "MotionEvaluationResult",
    "create_motion_profile_evaluator",
    "evaluate_motion_profile",
]


@dataclass(frozen=True)
class MotionEvaluationResult:
    """Normalized inputs and per-episode motion evaluation results."""

    profile: MotionProfile
    policy_spec: Mapping[str, Any]
    scene_config: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]
    viewer: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "policy_spec", MappingProxyType(dict(self.policy_spec))
        )
        object.__setattr__(
            self,
            "scene_config",
            MappingProxyType(dict(self.scene_config)),
        )
        object.__setattr__(
            self,
            "episodes",
            tuple(MappingProxyType(dict(value)) for value in self.episodes),
        )
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


def evaluate_motion_profile(
    profile: MotionProfile,
    *,
    episodes: int = 1,
    viewer: bool = False,
    control_steps: int | None = None,
    duration: float | None = None,
    command: tuple[float, ...] | None = None,
    scene_config: str | Path = "standard",
    physics_backend: str | None = None,
    simulation_device: str = "cpu",
    renderer: str = "hybrid",
    gpu_id: int = 0,
    termination_behavior: str | None = None,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    input_provider: InputProvider | None = None,
    environment: PolicyEnvironment | None = None,
) -> MotionEvaluationResult:
    """Resolve one Motion Profile and run its visual evaluation.

    Args:
        profile: Provider-built profile containing the DexSim Policy Spec.
        episodes: Number of independent runs.
        viewer: Open the DexSim Viewer.
        control_steps: Exact number of applied policy commands per run.
        duration: Convenience duration converted by DexSim to policy steps.
        command: Optional task command override.
        scene_config: Built-in scene style or custom YAML path.
        physics_backend: Optional DexSim physics backend override.
        simulation_device: ``cpu`` or ``gpu``.
        renderer: DexSim renderer.
        gpu_id: Selected GPU index.
        termination_behavior: Policy termination handling override.
        cache_dir: Motion Policy Kit resource cache.
        offline: Use resources already available in the cache.
        input_provider: Build per-frame task inputs for the Adapter.
        environment: Prebuilt task environment owned by the Evaluator.

    Returns:
        Normalized inputs, episode results, and aggregate metrics.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if viewer and episodes != 1:
        raise ValueError("Viewer evaluation supports one episode")
    if environment is not None and episodes != 1:
        raise ValueError("A prebuilt environment supports one episode")
    parsed, resolved = _resolve_profile(profile, cache_dir, offline)
    resolved_scene = load_scene_config(scene_config)
    options = RunOptions(
        physics_backend=physics_backend,
        simulation_device=simulation_device,
        renderer=renderer,
        gpu_id=gpu_id,
        headless=not viewer,
        control_steps=control_steps,
        duration=duration,
        command=command,
        termination_behavior=termination_behavior,
        scene_config=resolved_scene,
    )
    results = tuple(
        _episode(
            index,
            run_motion_policy(
                resolved,
                options,
                environment=environment,
                input_provider=input_provider,
            ),
        )
        for index in range(episodes)
    )
    return MotionEvaluationResult(
        profile=profile,
        policy_spec=policy_spec_to_dict(parsed),
        scene_config=scene_config_to_dict(resolved_scene),
        episodes=results,
        summary=_summary(results),
        viewer=viewer,
    )


def create_motion_profile_evaluator(
    profile: MotionProfile,
    options: RunOptions | None = None,
    *,
    cache_dir: str | Path | None = None,
    offline: bool = False,
    adapter: PolicyAdapter | None = None,
    environment: PolicyEnvironment | None = None,
) -> MotionPolicyEvaluator:
    """Create a DexSim Evaluator for an EmbodiChain Motion Profile.

    Args:
        profile: Provider-built profile containing the DexSim Policy Spec.
        options: DexSim evaluation options.
        cache_dir: Motion Policy Kit resource cache.
        offline: Use resources already available in the cache.
        adapter: Prebuilt policy adapter owned by the returned Evaluator.
        environment: Prebuilt task environment owned by the returned Evaluator.

    Returns:
        Configured evaluator ready for ``reset()``, ``step()``, or ``run()``.
    """
    _parsed, resolved = _resolve_profile(profile, cache_dir, offline)
    return create_motion_policy_evaluator(
        resolved,
        options,
        adapter=adapter,
        environment=environment,
    )


def _resolve_profile(
    profile: MotionProfile,
    cache_dir: str | Path | None,
    offline: bool,
) -> tuple[PolicySpec, ResolvedPolicy]:
    parsed = parse_policy_spec(profile.policy_spec)
    resolved = resolve_policy_spec(
        parsed,
        ResourceResolver(
            None if cache_dir is None else Path(cache_dir),
            offline=offline,
        ),
    )
    return parsed, resolved


def _episode(index: int, result: Any) -> dict[str, Any]:
    return {
        "index": index,
        "reason": str(result.reason),
        "simulation_time": float(result.simulation_time),
        "simulation_steps": int(result.simulation_steps),
        "control_steps": int(result.control_steps),
        "physics_backend": str(result.physics_backend),
        "requested_duration": (
            None
            if result.requested_duration is None
            else float(result.requested_duration)
        ),
        "effective_duration": float(result.effective_duration),
        "metrics": {name: float(value) for name, value in result.metrics.items()},
    }


def _summary(episodes: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    count = len(episodes)
    metric_names = set.intersection(*(set(episode["metrics"]) for episode in episodes))
    metrics = {
        name: sum(episode["metrics"][name] for episode in episodes) / count
        for name in sorted(metric_names)
    }
    result: dict[str, Any] = {
        "episodes": count,
        "avg_simulation_time": sum(episode["simulation_time"] for episode in episodes)
        / count,
        "avg_control_steps": sum(episode["control_steps"] for episode in episodes)
        / count,
        "avg_effective_duration": sum(
            episode["effective_duration"] for episode in episodes
        )
        / count,
    }
    if metrics:
        result["metrics"] = metrics
    return result
