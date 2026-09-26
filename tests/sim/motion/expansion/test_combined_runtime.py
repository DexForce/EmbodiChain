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

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateIdentity,
    CombinedEpisodeCoordinator,
    CombinedGenerationProfile,
    CubeInitialPoseProvider,
    ExpertEpisode,
    MeasuredValidator,
    PhysicalSlotPool,
    ValidationCheck,
    ValidationResult,
    VisualProfileRegistry,
    enumerate_candidate_recipes,
    round_robin_recipes,
)

_ROOT = Path(__file__).parents[4]
_TASK_ROOT = _ROOT / "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place"


def _profile() -> CombinedGenerationProfile:
    from embodichain.lab.sim.motion.expansion.profile import load_generation_profile

    profile = load_generation_profile(_TASK_ROOT / "generation.combined.yaml")
    assert isinstance(profile, CombinedGenerationProfile)
    return profile


def test_visual_registry_resolves_seeded_per_environment_profile() -> None:
    profile = _profile()
    registry = VisualProfileRegistry.from_yaml(
        _TASK_ROOT / "generation_profiles/rgb_visual.yaml"
    )
    assert registry.profile_ids == profile.visual.profiles
    resolved = registry.resolve("rgb_camera_01", seed=123)
    assert resolved.seed == 125
    assert all(
        operation["scope"] == "per_environment" for operation in resolved.operations
    )


def test_round_robin_coordinator_releases_slots_and_counts_terminal_states() -> None:
    profile = _profile()
    recipes = enumerate_candidate_recipes(
        profile,
        families=CubeInitialPoseProvider().enumerate(4),
    )
    ordered = round_robin_recipes(recipes, family_count=4)
    assert ordered[0].reference_family_id != ordered[1].reference_family_id

    coordinator = CombinedEpisodeCoordinator(
        recipes,
        slot_pool=PhysicalSlotPool(1),
        family_count=4,
        compatibility_key="strict:combined",
    )
    assignment = coordinator.acquire_next()
    assert assignment is not None
    assert coordinator.acquire_next() is None
    coordinator.finish(assignment, status="accepted")
    assert coordinator.snapshot()["accepted"] == 1
    assert coordinator.snapshot()["available_slots"] == 1


def test_measured_validator_requires_three_cycle_target_evidence() -> None:
    identity = CandidateIdentity(
        "case",
        "initial",
        "candidate",
        "geometry",
        "source",
        "revision",
        "template",
    )
    episode = ExpertEpisode(
        identity=identity,
        observations={"rgb": torch.zeros(3, 2)},
        actions=torch.zeros(2, 1),
        timestamps=torch.tensor([0.0, 0.1, 0.2]),
        action_representation="qpos",
        validation=ValidationResult((ValidationCheck("configured", "passed"),)),
        episode_id="episode",
        commit_id="commit",
        metadata={
            "cycle_poses": [[[0.0]], [[0.0]], [[0.0]]],
            "target_matches": [True, True, True],
        },
    )
    result = MeasuredValidator().validate(episode)
    assert result.accepted

    missing = ExpertEpisode(
        identity=identity,
        observations=episode.observations,
        actions=episode.actions,
        timestamps=episode.timestamps,
        action_representation=episode.action_representation,
        validation=episode.validation,
        episode_id="episode-missing",
        commit_id="commit-missing",
    )
    assert not MeasuredValidator().validate(missing).accepted
