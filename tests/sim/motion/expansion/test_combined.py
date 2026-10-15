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
    CombinedExpansionProfile,
    CubeInitialPoseProvider,
    PhysicalSlotPool,
    enumerate_candidate_recipes,
    schedule_digest,
)
from embodichain.lab.sim.motion.expansion.combined import load_visual_profile_registry
from embodichain.lab.sim.motion.expansion.profile import load_expansion_profile
from embodichain.utils.config_paths import resolve_config_path


def test_combined_profile_decodes_canonical_shape() -> None:
    # Exercise the combined schema independently of packaged task policies.
    profile = CombinedExpansionProfile.from_mapping({})

    assert isinstance(profile, CombinedExpansionProfile)
    assert profile.scene_randomization.reference_family_count == 4
    assert profile.affordance.branches_per_family == 4
    assert profile.trajectory.variants_per_family == 4
    assert profile.execution.max_inflight == 16
    profile.validate_for_num_envs(16)
    visual_ids = load_visual_profile_registry(
        resolve_config_path(profile.visual.profile_file),
        requested_profile_ids=profile.visual.profiles,
    )
    profile.validate_registries(
        visual_profile_ids=visual_ids,
        observation_profile_ids=profile.observation.profiles,
    )


def test_combined_profile_round_trips_through_yaml(tmp_path: Path) -> None:
    profile = CombinedExpansionProfile.from_mapping({})
    profile_path = tmp_path / "combined.yaml"
    profile_path.write_text(yaml.safe_dump(profile.to_dict()), encoding="utf-8")

    decoded = load_expansion_profile(profile_path)

    assert isinstance(decoded, CombinedExpansionProfile)
    assert decoded.to_dict() == profile.to_dict()


def test_combined_profile_rejects_capacity_and_environment_mismatch() -> None:
    profile = CombinedExpansionProfile.from_mapping({})
    payload = profile.to_dict()
    payload["scheduling"]["candidate_budget"] = 63
    with pytest.raises(ValueError, match="m.*a.*t"):
        CombinedExpansionProfile.from_mapping(payload)

    profile = CombinedExpansionProfile.from_mapping({})
    assert isinstance(profile, CombinedExpansionProfile)
    with pytest.raises(ValueError, match="num_envs"):
        profile.validate_for_num_envs(8)


def test_spatial_dedup_tolerance_travels_with_the_offset_scale() -> None:
    profile = CombinedExpansionProfile.from_mapping(
        {
            "trajectory": {
                "spatial": {"joint_offset_scale": 0.005, "dedup_tolerance": 0.002}
            }
        }
    )
    assert isinstance(profile, CombinedExpansionProfile)
    spatial = profile.trajectory.spatial
    # A conservative amplitude needs an explicitly smaller deduplication tolerance.
    assert spatial.joint_offset_scale == 0.005
    assert spatial.dedup_tolerance == 0.002

    payload = profile.to_dict()
    payload["trajectory"]["spatial"]["dedup_tolerance"] = 0.005
    with pytest.raises(ValueError, match="must stay below"):
        CombinedExpansionProfile.from_mapping(payload)


def test_spatial_dedup_tolerance_defaults_to_the_contract_tolerance() -> None:
    profile = CombinedExpansionProfile.from_mapping(
        {"trajectory": {"spatial": {"enabled": True, "joint_offset_scale": 0.05}}}
    )
    assert profile.trajectory.spatial.dedup_tolerance is None


def test_recipe_schedule_is_complete_nominal_and_stable() -> None:
    profile = CombinedExpansionProfile.from_mapping({})
    assert isinstance(profile, CombinedExpansionProfile)
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


def test_authored_pose_profile_controls_family_geometry_and_identity() -> None:
    profile = CombinedExpansionProfile.from_mapping({})
    assert isinstance(profile, CombinedExpansionProfile)
    provider = CubeInitialPoseProvider.from_yaml(
        resolve_config_path(profile.scene_randomization.profile_file)
    )
    families = provider.enumerate(4)
    recipes = enumerate_candidate_recipes(profile, families=families)
    assert recipes[0].reference_family_id == "cube_pose_00"
    altered = list(families)
    altered[0] = altered[0].__class__(
        reference_family_id=altered[0].reference_family_id,
        cube_position=(1.0, 2.0, 3.0),
        cube_quaternion_xyzw=altered[0].cube_quaternion_xyzw,
        drop_targets=altered[0].drop_targets,
        deterministic_seed=altered[0].deterministic_seed,
    )
    altered_recipes = enumerate_candidate_recipes(profile, families=altered)
    assert recipes[0].candidate_id != altered_recipes[0].candidate_id
    assert schedule_digest(recipes) != schedule_digest(altered_recipes)


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
