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


def test_visual_registry_resolves_seeded_visual_profiles() -> None:
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

    light = registry.resolve("rgb_light_01", seed=123)
    assert light.operations == (
        {
            "kind": "global_sun",
            "scope": "global",
            "light_uid": "main_light",
            "color": [1.0, 0.82, 0.62],
            "intensity": 5.75,
        },
    )


def test_visual_registry_rejects_global_sun_intensity_above_ten() -> None:
    with pytest.raises(ValueError, match=r"intensity.*\[0, 10\]"):
        VisualProfileRegistry(
            "rgb_visual",
            "rgb_visual:v1",
            {
                "invalid": {
                    "operations": [
                        {
                            "kind": "global_sun",
                            "scope": "global",
                            "light_uid": "main_light",
                            "color": [1.0, 1.0, 1.0],
                            "intensity": 10.01,
                        }
                    ]
                }
            },
        )


def test_visual_registry_applies_global_sun_once_for_all_rows() -> None:
    class Sun:
        is_global = True

        def __init__(self) -> None:
            self.reset_count = 0
            self.color = None
            self.intensity = None

        def reset(self) -> None:
            self.reset_count += 1

        def set_color(self, color) -> None:
            self.color = color

        def set_intensity(self, intensity) -> None:
            self.intensity = intensity

    class Sim:
        def __init__(self, sun) -> None:
            self.sun = sun

        def get_light(self, uid):
            assert uid == "main_light"
            return self.sun

        def get_rigid_object(self, uid):
            assert uid == "cube"
            return None

    class Env:
        def __init__(self, sim) -> None:
            self.sim = sim

    sun = Sun()
    registry = VisualProfileRegistry.from_yaml(
        _TASK_ROOT / "generation_profiles/rgb_visual.yaml"
    )
    applications = registry.apply_to_environment(
        Env(Sim(sun)),
        {0: "rgb_light_01", 1: "rgb_canonical"},
        seed=7,
    )

    assert tuple(item.profile_id for item in applications) == (
        "rgb_light_01",
        "rgb_canonical",
    )
    assert sun.reset_count == 1
    assert torch.equal(sun.color, torch.tensor([1.0, 0.82, 0.62]))
    assert torch.equal(sun.intensity, torch.tensor(5.75))


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
