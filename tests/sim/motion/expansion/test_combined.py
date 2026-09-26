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
import yaml

from embodichain.lab.sim.motion.expansion import (
    CombinedGenerationProfile,
    CubeInitialPoseProvider,
    PhysicalSlotPool,
    enumerate_candidate_recipes,
    schedule_digest,
)
from embodichain.lab.sim.motion.expansion.combined import load_visual_profile_registry
from embodichain.lab.sim.motion.expansion.profile import load_generation_profile

_PROFILE = Path(__file__).parents[4] / (
    "embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/"
    "generation.combined.yaml"
)


def test_combined_profile_decodes_canonical_shape() -> None:
    profile = load_generation_profile(_PROFILE)

    assert isinstance(profile, CombinedGenerationProfile)
    assert profile.scene_randomization.reference_family_count == 4
    assert profile.affordance.branches_per_family == 4
    assert profile.trajectory.variants_per_family == 4
    assert profile.execution.max_inflight == 16
    profile.validate_for_num_envs(16)
    visual_ids = load_visual_profile_registry(
        _PROFILE.parent / "generation_profiles/rgb_visual.yaml",
        requested_profile_ids=profile.visual.profiles,
    )
    profile.validate_registries(
        visual_profile_ids=visual_ids,
        observation_profile_ids=profile.observation.profiles,
    )


def test_combined_profile_rejects_capacity_and_environment_mismatch() -> None:
    payload = _PROFILE.read_text(encoding="utf-8")
    payload = payload.replace("candidate_budget: 64", "candidate_budget: 63")
    with pytest.raises(ValueError, match="m.*a.*t"):
        CombinedGenerationProfile.from_mapping(yaml.safe_load(payload))

    profile = load_generation_profile(_PROFILE)
    assert isinstance(profile, CombinedGenerationProfile)
    with pytest.raises(ValueError, match="num_envs"):
        profile.validate_for_num_envs(8)


def test_recipe_schedule_is_complete_nominal_and_stable() -> None:
    profile = load_generation_profile(_PROFILE)
    assert isinstance(profile, CombinedGenerationProfile)
    families = CubeInitialPoseProvider().enumerate(4)
    recipes = enumerate_candidate_recipes(profile, families=families)

    assert len(recipes) == 64
    assert recipes[0].recipe_index == 0
    assert recipes[0].reference_family_id == "cube_pose_00"
    assert recipes[0].visual_profile_id == "rgb_canonical"
    assert all(
        (cycle.affordance_requested, cycle.trajectory_requested) == (0, 0)
        for cycle in recipes[0].cycle_schedule
    )
    assert len({item.candidate_id for item in recipes}) == 64
    assert schedule_digest(recipes) == schedule_digest(
        enumerate_candidate_recipes(profile, families=families)
    )
    assert {item.reference_family_id for item in recipes} == {
        "cube_pose_00",
        "cube_pose_01",
        "cube_pose_02",
        "cube_pose_03",
    }
    assert {item.visual_profile_id for item in recipes} == set(profile.visual.profiles)


def test_physical_slot_pool_releases_only_exact_reservations() -> None:
    pool = PhysicalSlotPool(2, compatibility_keys={0: "strict", 1: "strict"})
    first = pool.reserve("candidate-a", "strict")
    second = pool.reserve("candidate-b", "strict")
    assert {first.slot_id, second.slot_id} == {0, 1}
    with pytest.raises(BufferError):
        pool.reserve("candidate-c", "strict")
    with pytest.raises(ValueError, match="stale"):
        pool.release(
            first.__class__(
                first.slot_id, "candidate-c", first.compatibility_key, first.token
            )
        )
    pool.release(first)
    assert pool.available_count == 1
    replacement = pool.reserve("candidate-c", "strict")
    assert replacement.slot_id == first.slot_id
