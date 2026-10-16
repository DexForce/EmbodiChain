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

"""Configuration tests for batched random-pose Atomic Task benchmarks."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from scripts.benchmark.motion_generation.config import (
    TrackCfg,
    load_suite,
    resolve_atomic_batch_sizes,
    resolve_atomic_pose_randomization,
)
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    _apply_target_rotation_randomization,
    _randomization_parameters,
    _randomized_vector_batch,
    _seeded_jitter,
    _workspace_shell_profile,
    atomic_skill_provider_names,
    create_atomic_skill_provider,
)
from scripts.benchmark.motion_generation.scenarios.atomic_dual_arm_providers import (
    _support_randomization_config,
)


def test_pose_batch_suite_declares_translation_only_batched_trials() -> None:
    """The shipped pose-batch suite maps one random pose to one env row."""
    suite = load_suite("atomic_franka_pgi_curobo_pose_batch")
    track = suite.enabled_tracks()[0]

    settings = resolve_atomic_pose_randomization(track)
    assert suite.suite_version == "atomic_franka_pgi_curobo_pose_batch_v3"
    assert settings.enabled is True
    assert settings.mode == "translation"
    assert settings.pose_batch_size == 8
    assert settings.object_translation_jitter_m == (0.03, 0.03, 0.0)
    assert settings.target_translation_jitter_m == (0.03, 0.03, 0.02)
    assert resolve_atomic_batch_sizes(track) == [8]

    skills = {entry["id"]: entry for entry in track.config["skills"]}
    assert set(skills) == {
        "move_end_effector",
        "move_joints",
        "pick_up",
        "move_held_object",
        "place",
        "press",
        "slide",
        "twist",
    }
    assert skills["pick_up"]["grasp_source"] == "fixed"
    assert skills["press"]["press_distance_m"] == pytest.approx(0.008)
    assert {
        skill_id: (
            skills[skill_id]["sample_count"],
            skills[skill_id]["hand_interp_steps"],
        )
        for skill_id in ("press", "twist")
    } == {"press": (256, 12), "twist": (256, 20)}
    assert skills["pick_up"]["object_position_jitter_m"] == [0.03, 0.03, 0.0]
    assert skills["move_end_effector"]["target_offset_jitter_m"] == [
        0.03,
        0.03,
        0.02,
    ]
    assert {
        entry["id"]: entry["asset_path"] for entry in track.config["articulations"]
    } == {
        "microwave": "Microwave/microwave.urdf",
        "drawer": "Drawer/model_split_links_with_inertials.urdf",
    }
    assert {
        skill_id: skills[skill_id]["articulation_position_jitter_m"]
        for skill_id in ("press", "slide", "twist")
    } == {
        "press": [0.03, 0.03, 0.02],
        "slide": [0.03, 0.03, 0.02],
        "twist": [0.03, 0.03, 0.02],
    }
    assert {
        skill_id: (
            skills[skill_id]["articulation"],
            skills[skill_id]["target_link"],
            skills[skill_id]["target_joint"],
        )
        for skill_id in ("press", "slide", "twist")
    } == {
        "press": ("microwave", "button_link", "button_joint"),
        "slide": ("drawer", "large_handle_bar", "cabinet_to_drawer"),
        "twist": ("microwave", "knob_link", "knob_joint"),
    }


def test_spatial_generalization_suite_covers_task_level_single_arm_skills() -> None:
    """Every Franka task skill gets its meaningful spatial or asset domains."""
    suite = load_suite("atomic_franka_pgi_spatial_generalization")
    asset_domain_ids = {
        "nominal",
        "cube_uniform_40mm",
        "cube_uniform_60mm",
        "cube_aspect_x",
        "cube_aspect_y",
        "cube_aspect_z",
    }
    expected = {
        "spatial-rigid-object": (
            {"pick_up", "pour", "axis_align"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "task_frame_planar",
                "object_planar_pose",
                "object_workspace_shell",
            },
        ),
        "spatial-rigid-object-held-pose-target": (
            {"move_held_object"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "target_yaw",
                "target_z",
                "target_roll_pitch",
                "task_frame_planar",
                "object_target_spatial_pose",
                "target_relative_shell",
            },
        ),
        "spatial-rigid-object-place-target": (
            {"place"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "target_yaw",
                "task_frame_planar",
                "object_target_planar_pose",
                "target_relative_shell",
            },
        ),
        "spatial-rigid-object-position-target": (
            {"push_object"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "task_frame_planar",
                "object_target_position",
                "target_relative_shell",
            },
        ),
        "asset-cube-rigid-object": (
            {"pick_up", "pour", "axis_align"},
            asset_domain_ids,
        ),
        "asset-cube-held-pose-target": (
            {"move_held_object"},
            asset_domain_ids,
        ),
        "asset-cube-place-target": ({"place"}, asset_domain_ids),
        "asset-cube-position-target": ({"push_object"}, asset_domain_ids),
        "spatial-microwave-controls": (
            {"press", "twist"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "object_z",
                "object_roll_pitch",
                "task_frame_planar",
                "articulation_spatial_pose",
                "articulation_workspace_shell",
            },
        ),
        "spatial-drawer": (
            {"slide"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "object_z",
                "object_roll_pitch",
                "task_frame_planar",
                "articulation_spatial_pose",
                "articulation_workspace_shell",
            },
        ),
        "spatial-door": (
            {"open_door"},
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "object_z",
                "object_roll_pitch",
                "task_frame_planar",
                "articulation_spatial_pose",
                "articulation_workspace_shell",
            },
        ),
    }

    assert suite.suite_version == "atomic_franka_pgi_spatial_generalization_v8"
    assert len(suite.evaluation_tracks()) == 79
    assert set(expected) == {track.id for track in suite.tracks}
    for track in suite.tracks:
        skills, domain_ids = expected[track.id]
        assert {entry["id"] for entry in track.config["skills"]} == skills
        assert {domain.id for domain in track.domains} == domain_ids
        assert all(set(domain.applicability) == skills for domain in track.domains)


def test_cube_asset_domains_override_only_geometry_for_every_cube_skill() -> None:
    """Every cube skill receives one nominal and five isolated asset variants."""
    suite = load_suite("atomic_franka_pgi_spatial_generalization")
    expected_sizes = {
        "nominal": [0.05, 0.05, 0.05],
        "cube_uniform_40mm": [0.04, 0.04, 0.04],
        "cube_uniform_60mm": [0.06, 0.06, 0.06],
        "cube_aspect_x": [0.07, 0.042258, 0.042258],
        "cube_aspect_y": [0.042258, 0.07, 0.042258],
        "cube_aspect_z": [0.042258, 0.042258, 0.07],
    }
    cube_groups = {
        "asset-cube-rigid-object",
        "asset-cube-held-pose-target",
        "asset-cube-place-target",
        "asset-cube-position-target",
    }
    cube_skills = {
        "pick_up",
        "move_held_object",
        "place",
        "pour",
        "axis_align",
        "push_object",
    }
    nominal_object = suite.tracks[0].config["objects"][0]
    asset_tracks = [
        track
        for track in suite.evaluation_tracks()
        if track.domain.metadata.get("generalization_axis") == "asset_geometry"
    ]

    assert len(asset_tracks) == len(cube_groups) * len(expected_sizes)
    assert {track.group_id for track in asset_tracks} == cube_groups
    assert {
        skill["id"] for track in asset_tracks for skill in track.config["skills"]
    } == cube_skills
    for group_id in cube_groups:
        assert {
            track.domain.id for track in asset_tracks if track.group_id == group_id
        } == set(expected_sizes)

    for track in asset_tracks:
        object_config = track.config["objects"]
        assert len(object_config) == 1
        object_config = object_config[0]
        expected_size = expected_sizes[track.domain.id]
        assert object_config["id"] == "cube_50mm"
        assert object_config["kind"] == "cube"
        assert object_config["size"] == expected_size
        assert track.domain.objects == ["cube_50mm"]
        assert track.domain.perturbations == {"asset_size_m": expected_size}
        assert track.domain.metadata["asset_family"] == "cube"
        assert track.domain.metadata["nominal_size_m"] == [0.05, 0.05, 0.05]
        assert track.domain.case_generator == "configured_asset_geometry"
        assert resolve_atomic_batch_sizes(track) == [1]
        assert resolve_atomic_pose_randomization(track).enabled is False
        for field in (
            "position",
            "mass",
            "dynamic_friction",
            "static_friction",
            "contact_offset",
            "rest_offset",
            "enable_ccd",
        ):
            assert object_config[field] == nominal_object[field]
        assert {
            case["object"]
            for skill in track.config["skills"]
            for case in skill["cases"]
        } == {"cube_50mm"}


@pytest.mark.parametrize(
    ("suite_name", "skill_id", "domain_ids"),
    [
        (
            "atomic_dual_ur5_pgi_hand_over_spatial_generalization",
            "hand_over",
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "task_frame_planar",
                "object_target_position",
                "target_relative_shell",
                "bilateral_role_mirror",
            },
        ),
        (
            "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization",
            "coordinated_pickment",
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "target_yaw",
                "target_z",
                "target_roll_pitch",
                "task_frame_planar",
                "object_target_spatial_pose",
                "target_relative_shell",
                "bilateral_role_mirror",
            },
        ),
        (
            "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization",
            "coordinated_placement",
            {
                "nominal",
                "object_xy",
                "object_yaw",
                "target_relative_pose",
                "target_yaw",
                "target_z",
                "support_object_xy",
                "support_object_yaw",
                "task_frame_xy",
                "task_frame_yaw",
                "task_frame_planar",
                "object_support_target_task_frame_pose",
                "support_relative_shell",
                "bilateral_role_mirror",
            },
        ),
    ],
)
def test_dual_arm_spatial_suites_cover_meaningful_domains(
    suite_name: str, skill_id: str, domain_ids: set[str]
) -> None:
    """Each dual-arm task receives its applicable isolated spatial shifts."""
    suite = load_suite(suite_name)
    tracks = suite.evaluation_tracks()

    assert {track.domain.id for track in tracks} == domain_ids
    assert all(track.domain.applicability == [skill_id] for track in tracks)
    assert all(
        [entry["id"] for entry in track.config["skills"]] == [skill_id]
        for track in tracks
    )


def test_spatial_suites_cover_every_non_motion_atomic_skill() -> None:
    """Only pure joint and end-effector moves are outside spatial evaluation."""
    suite_names = (
        "atomic_franka_pgi_spatial_generalization",
        "atomic_dual_ur5_pgi_hand_over_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization",
    )
    configured = {
        skill["id"]
        for suite_name in suite_names
        for track in load_suite(suite_name).tracks
        for skill in track.config["skills"]
    }

    expected = set(atomic_skill_provider_names()) - {
        "move_end_effector",
        "move_joints",
    }
    assert configured == expected


def test_every_spatial_domain_matches_its_runtime_randomization() -> None:
    """Declared domain provenance and executable overrides cannot drift."""
    suite_names = (
        "atomic_franka_pgi_spatial_generalization",
        "atomic_dual_ur5_pgi_hand_over_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization",
    )
    perturbation_fields = {
        "object_translation_jitter_m": "object_translation_jitter_m",
        "object_yaw_jitter_rad": "object_yaw_jitter_rad",
        "object_roll_pitch_jitter_rad": "object_roll_pitch_jitter_rad",
        "target_translation_jitter_m": "target_translation_jitter_m",
        "target_yaw_jitter_rad": "target_yaw_jitter_rad",
        "target_roll_pitch_jitter_rad": "target_roll_pitch_jitter_rad",
        "support_object_translation_jitter_m": ("support_object_translation_jitter_m"),
        "support_object_yaw_jitter_rad": "support_object_yaw_jitter_rad",
        "support_object_roll_pitch_jitter_rad": (
            "support_object_roll_pitch_jitter_rad"
        ),
        "task_frame_translation_jitter_m": "task_frame_translation_jitter_m",
        "task_frame_yaw_jitter_rad": "task_frame_yaw_jitter_rad",
        "task_frame_roll_pitch_jitter_rad": "task_frame_roll_pitch_jitter_rad",
    }
    for suite_name in suite_names:
        for track in load_suite(suite_name).evaluation_tracks():
            settings = resolve_atomic_pose_randomization(track)
            perturbations = track.domain.perturbations
            assert track.config["seeds"] == [11, 23, 47]
            if track.domain.kind == "nominal":
                assert settings.enabled is False
                assert resolve_atomic_batch_sizes(track) == [1]
                continue
            if track.domain.case_generator == "mirrored_control_assignment":
                assert settings.enabled is False
                assert resolve_atomic_batch_sizes(track) == [1]
                assert "control_parts" in track.config
                continue
            if track.domain.case_generator == "configured_asset_geometry":
                assert settings.enabled is False
                assert resolve_atomic_batch_sizes(track) == [1]
                assert track.domain.metadata["generalization_axis"] == "asset_geometry"
                assert track.config["objects"][0]["size"] == (
                    track.domain.perturbations["asset_size_m"]
                )
                continue
            assert settings.enabled is True
            assert resolve_atomic_batch_sizes(track) == [8]
            for field, attribute in perturbation_fields.items():
                if field not in perturbations:
                    continue
                expected = perturbations[field]
                actual = getattr(settings, attribute)
                if isinstance(expected, list):
                    assert actual == tuple(expected)
                else:
                    assert actual == pytest.approx(expected)


def test_composed_spatial_domains_use_stratified_boundary_sampling() -> None:
    """Every new multi-factor domain exercises all configured pose bounds."""
    suite_names = (
        "atomic_franka_pgi_spatial_generalization",
        "atomic_dual_ur5_pgi_hand_over_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization",
    )
    composed_domain_ids = {
        "task_frame_planar",
        "object_planar_pose",
        "object_target_position",
        "object_target_planar_pose",
        "object_target_spatial_pose",
        "articulation_spatial_pose",
        "object_support_target_task_frame_pose",
    }
    seen: set[str] = set()
    for suite_name in suite_names:
        for track in load_suite(suite_name).evaluation_tracks():
            if track.domain.id not in composed_domain_ids:
                continue
            seen.add(track.domain.id)
            settings = resolve_atomic_pose_randomization(track)
            pose_fields = [
                value
                for key, value in track.domain.perturbations.items()
                if key.endswith(("_jitter_m", "_jitter_rad"))
                and (any(value) if isinstance(value, list) else value > 0.0)
            ]
            assert len(pose_fields) >= 2
            assert track.domain.case_generator == "deterministic_stratified_pose_batch"
            assert track.domain.perturbations["distribution"] == "stratified"
            assert settings.sampling_pattern == "stratified"
    assert seen == composed_domain_ids


def test_workspace_shell_domains_declare_radial_buckets() -> None:
    """Every shell domain uses the executable radial sampler and three buckets."""
    suite_names = (
        "atomic_franka_pgi_spatial_generalization",
        "atomic_dual_ur5_pgi_hand_over_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization",
        "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization",
    )
    shell_ids = {
        "object_workspace_shell",
        "target_relative_shell",
        "articulation_workspace_shell",
        "support_relative_shell",
    }
    seen: set[str] = set()
    for suite_name in suite_names:
        for track in load_suite(suite_name).evaluation_tracks():
            if track.domain.id not in shell_ids:
                continue
            seen.add(track.domain.id)
            settings = resolve_atomic_pose_randomization(track)
            assert track.domain.case_generator == "deterministic_workspace_shell_batch"
            assert track.domain.perturbations["distribution"] == "workspace_shell"
            assert track.domain.perturbations["workspace_buckets"] == [
                "near",
                "mid",
                "far",
            ]
            assert settings.sampling_pattern == "workspace_shell"
            assert resolve_atomic_batch_sizes(track) == [8]
    assert seen == shell_ids


def test_dual_arm_mirror_domains_swap_spatial_roles() -> None:
    """Mirrored domains route each semantic slot to the opposite physical arm."""
    expected = {
        "atomic_dual_ur5_pgi_hand_over_spatial_generalization": {
            "source": {"motion": "right_arm", "grasp": "right_eef"},
            "destination": {"motion": "left_arm", "grasp": "left_eef"},
        },
        "atomic_dual_ur5_pgi_coordinated_pickment_spatial_generalization": {
            "left": {"motion": "right_arm", "grasp": "right_eef"},
            "right": {"motion": "left_arm", "grasp": "left_eef"},
        },
        "atomic_dual_ur5_pgi_coordinated_placement_spatial_generalization": {
            "placing": {"motion": "right_arm", "grasp": "right_eef"},
            "support": {"motion": "left_arm", "grasp": "left_eef"},
        },
    }
    for suite_name, control_parts in expected.items():
        mirror = next(
            track
            for track in load_suite(suite_name).evaluation_tracks()
            if track.domain.id == "bilateral_role_mirror"
        )
        prepared = AtomicTaskScenario._prepare_pose_batch_config(
            {}, mirror, batch_size=1
        )
        assert prepared["control_parts"] == control_parts
        assert mirror.domain.perturbations["mirror_axis"] == "y"


def test_pose_batch_aliases_are_normalized() -> None:
    """Readable aliases work without changing the canonical manifest."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "random_pose_count": 4,
            "randomization": {
                "random_pose_count": 4,
                "object_position_jitter_m": [0.01, 0.02, 0.0],
                "target_offset_jitter_m": [0.03, 0.0, 0.01],
            },
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    assert settings.enabled is True
    assert settings.pose_batch_size == 4
    assert settings.object_translation_jitter_m == (0.01, 0.02, 0.0)
    assert settings.target_translation_jitter_m == (0.03, 0.0, 0.01)
    assert resolve_atomic_batch_sizes(track) == [4]


def test_planar_pose_mode_injects_all_strict_spatial_controls() -> None:
    """The planar-pose mode injects all normalized planar dimensions."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "mode": "planar_pose",
                "pose_batch_size": 4,
                "object_yaw_jitter_rad": 0.5,
                "target_yaw_jitter_rad": 0.4,
                "support_object_translation_jitter_m": [0.03, 0.02, 0.0],
                "support_object_yaw_jitter_rad": 0.3,
                "task_frame_translation_jitter_m": [0.01, 0.02, 0.0],
                "task_frame_yaw_jitter_rad": 0.2,
            }
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    prepared = AtomicTaskScenario._prepare_pose_batch_config({}, track, batch_size=4)

    assert settings.enabled is True
    assert settings.object_yaw_jitter_rad == pytest.approx(0.5)
    assert settings.target_yaw_jitter_rad == pytest.approx(0.4)
    assert settings.support_object_translation_jitter_m == (0.03, 0.02, 0.0)
    assert settings.support_object_yaw_jitter_rad == pytest.approx(0.3)
    assert settings.task_frame_translation_jitter_m == (0.01, 0.02, 0.0)
    assert settings.task_frame_yaw_jitter_rad == pytest.approx(0.2)
    assert prepared["_pose_batch_object_yaw_jitter_rad"] == pytest.approx(0.5)
    assert prepared["_pose_batch_target_yaw_jitter_rad"] == pytest.approx(0.4)
    assert prepared["_pose_batch_support_object_translation_jitter_m"] == [
        0.03,
        0.02,
        0.0,
    ]
    assert prepared["_pose_batch_support_object_yaw_jitter_rad"] == pytest.approx(0.3)
    assert prepared["_pose_batch_task_frame_translation_jitter_m"] == [
        0.01,
        0.02,
        0.0,
    ]
    assert prepared["_pose_batch_task_frame_yaw_jitter_rad"] == pytest.approx(0.2)
    assert prepared["_pose_batch_object_translation_jitter_m"] == [0.0, 0.0, 0.0]
    assert prepared["_pose_batch_translation_jitter_m"] == [0.0, 0.0, 0.0]


def test_spatial_pose_mode_injects_all_roll_pitch_controls() -> None:
    """The spatial-pose mode exposes 3D orientation for every pose role."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "mode": "spatial_pose",
                "pose_batch_size": 4,
                "object_roll_pitch_jitter_rad": [0.1, 0.2],
                "target_roll_pitch_jitter_rad": [0.2, 0.3],
                "support_object_roll_pitch_jitter_rad": [0.3, 0.4],
                "task_frame_roll_pitch_jitter_rad": [0.4, 0.5],
            }
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    prepared = AtomicTaskScenario._prepare_pose_batch_config({}, track, batch_size=4)

    assert settings.object_roll_pitch_jitter_rad == (0.1, 0.2)
    assert settings.target_roll_pitch_jitter_rad == (0.2, 0.3)
    assert settings.support_object_roll_pitch_jitter_rad == (0.3, 0.4)
    assert settings.task_frame_roll_pitch_jitter_rad == (0.4, 0.5)
    assert prepared["_pose_batch_object_roll_pitch_jitter_rad"] == [0.1, 0.2]
    assert prepared["_pose_batch_target_roll_pitch_jitter_rad"] == [0.2, 0.3]
    assert prepared["_pose_batch_support_object_roll_pitch_jitter_rad"] == [
        0.3,
        0.4,
    ]
    assert prepared["_pose_batch_task_frame_roll_pitch_jitter_rad"] == [0.4, 0.5]


def test_stratified_pose_sampling_is_deterministic_and_covers_bounds() -> None:
    """Each dimension independently visits both configured interval limits."""
    kwargs = {
        "amplitude": (0.03, 0.02, 0.5),
        "seed": 23,
        "stream": 101,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "batch_size": 8,
        "seed_stride": 1,
        "sampling_pattern": "stratified",
    }
    first = _seeded_jitter(**kwargs)
    second = _seeded_jitter(**kwargs)

    torch.testing.assert_close(first, second)
    for dimension, amplitude in enumerate(kwargs["amplitude"]):
        expected = torch.linspace(-amplitude, amplitude, 8)
        torch.testing.assert_close(torch.sort(first[:, dimension]).values, expected)
    assert not torch.allclose(
        first[:, 0] / kwargs["amplitude"][0],
        first[:, 1] / kwargs["amplitude"][1],
    )


def test_workspace_shell_sampling_covers_radius_buckets_and_height() -> None:
    """Workspace shells cover near/mid/far XY radii and the configured Z range."""
    amplitude = (0.04, 0.02, 0.03)
    kwargs = {
        "amplitude": amplitude,
        "seed": 47,
        "stream": 203,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "batch_size": 8,
        "seed_stride": 1,
        "sampling_pattern": "workspace_shell",
    }
    first = _seeded_jitter(**kwargs)
    second = _seeded_jitter(**kwargs)
    radii, labels = _workspace_shell_profile(8)
    realized_radii = torch.sqrt(
        (first[:, 0] / amplitude[0]).square() + (first[:, 1] / amplitude[1]).square()
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(realized_radii, radii)
    torch.testing.assert_close(
        torch.sort(first[:, 2]).values,
        torch.linspace(-amplitude[2], amplitude[2], 8),
    )
    assert set(labels) == {"near", "mid", "far"}


def test_pose_sampling_pattern_is_validated_and_injected() -> None:
    """The YAML sampling selector reaches runtime pose generation unchanged."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "enabled": True,
                "mode": "translation",
                "sampling_pattern": "stratified",
                "pose_batch_size": 4,
                "object_translation_jitter_m": [0.02, 0.02, 0.0],
            }
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    prepared = AtomicTaskScenario._prepare_pose_batch_config({}, track, batch_size=4)
    assert settings.sampling_pattern == "stratified"
    assert prepared["_pose_batch_sampling_pattern"] == "stratified"

    track.config["randomization"]["sampling_pattern"] = "grid"
    with pytest.raises(ValueError, match="sampling_pattern"):
        resolve_atomic_pose_randomization(track)


@pytest.mark.parametrize(
    "field",
    [
        "object_yaw_jitter_rad",
        "target_yaw_jitter_rad",
        "support_object_yaw_jitter_rad",
        "task_frame_yaw_jitter_rad",
    ],
)
def test_translation_mode_rejects_yaw_fields(field: str) -> None:
    """A translation-only domain rejects every orientation control."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "enabled": True,
                "mode": "translation",
                field: 0.5,
            }
        },
    )

    with pytest.raises(ValueError, match="mode='translation'"):
        resolve_atomic_pose_randomization(track)


@pytest.mark.parametrize(
    "field",
    [
        "object_roll_pitch_jitter_rad",
        "target_roll_pitch_jitter_rad",
        "support_object_roll_pitch_jitter_rad",
        "task_frame_roll_pitch_jitter_rad",
    ],
)
@pytest.mark.parametrize("mode", ["translation", "planar_pose"])
def test_non_spatial_modes_reject_roll_pitch_fields(field: str, mode: str) -> None:
    """Roll/pitch controls require the explicit spatial-pose mode."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "enabled": True,
                "mode": mode,
                field: [0.1, 0.2],
            }
        },
    )

    with pytest.raises(ValueError, match=f"mode='{mode}'"):
        resolve_atomic_pose_randomization(track)


def test_atomic_pose_config_rejects_null_track_mapping() -> None:
    """Report malformed YAML track entries with a useful type error."""
    track = TrackCfg(id="atomic-task", scenario="atomic_task", config=None)
    with pytest.raises(TypeError, match="track config must be a mapping"):
        resolve_atomic_pose_randomization(track)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pose_batch_size", 0, "pose_batch_size"),
        ("pose_batch_size", -2, "pose_batch_size"),
        ("object_translation_jitter_m", [0.01, -0.01, 0.0], "object_translation"),
        ("object_yaw_jitter_rad", -0.1, "object_yaw"),
        ("object_roll_pitch_jitter_rad", [0.1, -0.1], "object_roll_pitch"),
        ("target_translation_jitter_m", [0.01, 0.0], "target_translation"),
        ("target_yaw_jitter_rad", -0.1, "target_yaw"),
        ("target_roll_pitch_jitter_rad", [0.1], "target_roll_pitch"),
        (
            "support_object_translation_jitter_m",
            [0.01, -0.01, 0.0],
            "support_object_translation",
        ),
        ("support_object_yaw_jitter_rad", -0.1, "support_object_yaw"),
        (
            "support_object_roll_pitch_jitter_rad",
            [0.1, float("inf")],
            "support_object_roll_pitch",
        ),
        (
            "task_frame_translation_jitter_m",
            [0.01, 0.0],
            "task_frame_translation",
        ),
        ("task_frame_yaw_jitter_rad", -0.1, "task_frame_yaw"),
        (
            "task_frame_roll_pitch_jitter_rad",
            "invalid",
            "task_frame_roll_pitch",
        ),
    ],
)
def test_invalid_pose_randomization_is_rejected(
    field: str, value: object, message: str
) -> None:
    """Reject malformed batch and translation perturbation controls early."""
    randomization = {field: value}
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={"randomization": randomization},
    )

    with pytest.raises((TypeError, ValueError), match=message):
        resolve_atomic_pose_randomization(track)


def test_explicit_batch_sizes_must_match_random_pose_batch() -> None:
    """Avoid silently running a different batch size than the pose matrix."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "batch_sizes": [1],
            "randomization": {"enabled": True, "pose_batch_size": 8},
        },
    )

    with pytest.raises(ValueError, match="exactly"):
        resolve_atomic_batch_sizes(track)


def test_none_mode_can_batch_identical_poses_without_jitter() -> None:
    """``none`` disables perturbations but may still exercise a batched planner."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "mode": "none",
                "pose_batch_size": 3,
            }
        },
    )

    settings = resolve_atomic_pose_randomization(track)
    assert settings.enabled is True
    assert settings.mode == "none"
    assert settings.pose_batch_size == 3
    assert resolve_atomic_batch_sizes(track) == [3]


def test_none_mode_rejects_nonzero_jitter_when_enabled() -> None:
    """A disabled perturbation mode must not silently ignore configured jitter."""
    track = TrackCfg(
        id="atomic-task",
        scenario="atomic_task",
        config={
            "randomization": {
                "enabled": True,
                "mode": "none",
                "pose_batch_size": 3,
                "target_translation_jitter_m": [0.01, 0.0, 0.0],
            }
        },
    )

    with pytest.raises(ValueError, match="mode='none'"):
        resolve_atomic_pose_randomization(track)


def test_b1_offset_manifest_keeps_legacy_waypoint_shape() -> None:
    """B=1 metadata remains ``[waypoint][xyz]`` after batched generation."""
    robot = Mock(device=torch.device("cpu"))
    robot.get_joint_ids.return_value = list(range(7))
    robot.get_qpos.return_value = torch.zeros(1, 7)
    robot.compute_fk.return_value = torch.eye(4).unsqueeze(0)
    scenario = Mock(robot=robot, control_part="arm")
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], 7
    )

    case = create_atomic_skill_provider("move_end_effector").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        {
            "name": "legacy_shape",
            "target_offsets_m": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        },
        seed=11,
        batch_size=1,
    )

    offsets = case.case_parameters["target_offsets_m"]
    assert len(offsets) == 2
    assert len(offsets[0]) == 3
    assert not isinstance(offsets[0][0], list)


def test_pose_batch_moveeef_shares_translation_across_waypoints() -> None:
    """One pose sample translates a whole MoveEEF waypoint path together."""
    batch_size = 4
    robot = Mock(device=torch.device("cpu"))
    robot.get_joint_ids.return_value = list(range(7))
    robot.get_qpos.side_effect = lambda name=None: torch.zeros(
        batch_size, 7 if name == "arm" else 9
    )
    robot.compute_fk.return_value = torch.eye(4).repeat(batch_size, 1, 1)
    scenario = Mock(robot=robot, control_part="arm")
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], start.shape[-1]
    )
    config = {
        "name": "shared_target",
        "target_offsets_m": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]],
        "target_offset_jitter_m": [0.02, 0.02, 0.02],
        "_pose_batch_enabled": True,
        "_pose_batch_seed_stride": 1,
    }

    case = create_atomic_skill_provider("move_end_effector").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        config,
        seed=11,
        batch_size=batch_size,
    )

    base = torch.tensor(config["target_offsets_m"], dtype=torch.float32)
    actual = case.target_waypoints[:, :, :3, 3]
    jitter = actual - base.unsqueeze(0)
    torch.testing.assert_close(jitter[:, 0], jitter[:, 1])
    assert not torch.allclose(jitter[0], jitter[1])
    assert len(case.case_parameters["target_offsets_m"]) == batch_size


def test_randomize_object_pose_returns_live_pose_after_optional_settle() -> None:
    """PickUp can freeze the post-gravity pose used by its grasp target."""
    pose = torch.eye(4).unsqueeze(0)
    settled = pose.clone()
    settled[:, 2, 3] = 0.025
    entity = Mock()
    entity.get_local_pose.return_value = settled
    handle = Mock(initial_pose=pose, entity=entity)
    simulation = Mock()
    scenario = AtomicTaskScenario()
    scenario.simulation = simulation

    result = scenario.randomize_object_pose(
        handle,
        {},
        seed=11,
        stream=201,
        settle_steps=50,
    )

    simulation.update.assert_called_once_with(step=50)
    entity.get_local_pose.assert_called_once_with(to_matrix=True)
    torch.testing.assert_close(result, settled)
    torch.testing.assert_close(entity.set_local_pose.call_args.args[0], pose)


def test_track_level_object_yaw_is_batched_and_deterministic() -> None:
    """Each environment gets a reproducible yaw while translation stays fixed."""
    batch_size = 4
    initial_pose = torch.eye(4).repeat(batch_size, 1, 1)
    initial_pose[:, :3, 3] = torch.tensor([-0.42, -0.08, 0.05])
    handle = Mock(initial_pose=initial_pose, entity=Mock())
    scenario = AtomicTaskScenario()
    config = {
        "_pose_batch_object_yaw_jitter_rad": 0.5,
        "_pose_batch_seed_stride": 1,
    }

    first = scenario.randomize_object_pose(handle, config, seed=11, stream=201)
    second = scenario.randomize_object_pose(handle, config, seed=11, stream=201)

    yaw = torch.atan2(first[:, 1, 0], first[:, 0, 0])
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :3, 3], initial_pose[:, :3, 3])
    assert torch.all(yaw.abs() <= 0.5)
    assert torch.unique(yaw).numel() == batch_size


def test_target_spatial_rotation_is_batched_without_translating_the_goal() -> None:
    """Target 3D orientation rows vary reproducibly around a fixed position."""
    batch_size = 4
    target = torch.eye(4).repeat(batch_size, 1, 1)
    target[:, :3, 3] = torch.tensor([0.4, -0.2, 0.7])
    config = {
        "_pose_batch_target_yaw_jitter_rad": 0.5,
        "_pose_batch_target_roll_pitch_jitter_rad": [0.2, 0.3],
        "_pose_batch_seed_stride": 1,
    }

    first = target.clone()
    second = target.clone()
    _apply_target_rotation_randomization(first, config, seed=11, stream=312)
    _apply_target_rotation_randomization(second, config, seed=11, stream=312)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, :3, 3], target[:, :3, 3])
    assert torch.unique(first[:, :3, :3], dim=0).shape[0] == batch_size
    identity = torch.eye(3).expand(batch_size, -1, -1)
    assert not torch.allclose(first[:, :3, :3], identity)


def test_task_frame_transform_is_shared_by_scene_objects() -> None:
    """A task-frame sample preserves relative distance between scene objects."""
    batch_size = 4
    first_pose = torch.eye(4).repeat(batch_size, 1, 1)
    second_pose = torch.eye(4).repeat(batch_size, 1, 1)
    first_pose[:, :3, 3] = torch.tensor([0.4, -0.2, 0.1])
    second_pose[:, :3, 3] = torch.tensor([-0.1, 0.3, 0.1])
    first_handle = Mock(initial_pose=first_pose, entity=Mock())
    second_handle = Mock(initial_pose=second_pose, entity=Mock())
    scenario = AtomicTaskScenario()
    config = {
        "_pose_batch_task_frame_translation_jitter_m": [0.03, 0.03, 0.0],
        "_pose_batch_task_frame_yaw_jitter_rad": 0.2,
        "_pose_batch_task_frame_roll_pitch_jitter_rad": [0.1, 0.1],
        "_pose_batch_seed_stride": 1,
    }

    first = scenario.randomize_object_pose(first_handle, config, seed=11, stream=901)
    second = scenario.randomize_object_pose(second_handle, config, seed=11, stream=903)

    initial_distance = torch.linalg.vector_norm(
        first_pose[:, :3, 3] - second_pose[:, :3, 3], dim=-1
    )
    transformed_distance = torch.linalg.vector_norm(
        first[:, :3, 3] - second[:, :3, 3], dim=-1
    )
    torch.testing.assert_close(transformed_distance, initial_distance)
    torch.testing.assert_close(first[:, :3, :3], second[:, :3, :3])


def test_support_object_ranges_do_not_inherit_primary_object_ranges() -> None:
    """Primary, support, and shared-frame perturbations remain independent."""
    config = {
        "_pose_batch_object_translation_jitter_m": [0.03, 0.03, 0.0],
        "_pose_batch_object_yaw_jitter_rad": 0.5,
        "_pose_batch_support_object_translation_jitter_m": [0.02, 0.01, 0.0],
        "_pose_batch_support_object_yaw_jitter_rad": 0.3,
        "_pose_batch_support_object_roll_pitch_jitter_rad": [0.2, 0.15],
        "_pose_batch_task_frame_translation_jitter_m": [0.01, 0.01, 0.0],
        "_pose_batch_task_frame_yaw_jitter_rad": 0.1,
    }

    support = _support_randomization_config(config)

    assert support["object_position_jitter_m"] == [0.02, 0.01, 0.0]
    assert support["object_yaw_jitter_rad"] == pytest.approx(0.3)
    assert support["object_roll_pitch_jitter_rad"] == [0.2, 0.15]
    assert "_pose_batch_object_translation_jitter_m" not in support
    assert "_pose_batch_object_yaw_jitter_rad" not in support
    assert "_pose_batch_object_roll_pitch_jitter_rad" not in support
    assert support["_pose_batch_task_frame_translation_jitter_m"] == [
        0.01,
        0.01,
        0.0,
    ]
    assert support["_pose_batch_task_frame_yaw_jitter_rad"] == pytest.approx(0.1)


def test_randomize_articulation_root_pose_is_deterministic() -> None:
    """Articulation roots vary reproducibly over enabled 3D pose axes."""
    batch_size = 4
    initial_pose = torch.eye(4).repeat(batch_size, 1, 1)
    initial_pose[:, :3, 3] = torch.tensor([-1.0, -0.3, 0.4])
    initial_qpos = torch.zeros(batch_size, 4)
    state = {"pose": initial_pose.clone()}
    entity = Mock()
    entity.get_local_pose.side_effect = lambda **_: state["pose"].clone()
    handle = Mock(
        initial_pose=initial_pose,
        initial_qpos=initial_qpos,
        entity=entity,
        config={"reset_settle_steps": 1},
    )

    def reset(*, pose: torch.Tensor, qpos: torch.Tensor) -> None:
        state["pose"] = pose.clone()
        torch.testing.assert_close(qpos, initial_qpos)

    handle.reset.side_effect = reset
    scenario = AtomicTaskScenario()
    scenario.simulation = Mock()
    config = {
        "articulation_position_jitter_m": [0.03, 0.02, 0.01],
        "_pose_batch_object_yaw_jitter_rad": 0.4,
        "_pose_batch_object_roll_pitch_jitter_rad": [0.1, 0.2],
        "_pose_batch_seed_stride": 1,
    }

    first = scenario.randomize_articulation_pose(
        handle,
        config,
        seed=11,
        stream=32,
    )
    second = scenario.randomize_articulation_pose(
        handle,
        config,
        seed=11,
        stream=32,
    )

    offsets = first[:, :3, 3] - initial_pose[:, :3, 3]
    torch.testing.assert_close(first, second)
    assert torch.all(offsets[:, 0].abs() <= 0.03)
    assert torch.all(offsets[:, 1].abs() <= 0.02)
    assert torch.all(offsets[:, 2].abs() <= 0.01)
    assert torch.unique(offsets, dim=0).shape[0] == batch_size
    assert torch.unique(first[:, :3, :3], dim=0).shape[0] == batch_size
    assert scenario.simulation.update.call_count == 2
    scenario.simulation.update.assert_called_with(step=1)


def test_resolve_articulation_grasp_broadcasts_one_local_candidate_over_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Slide samples PGI once and screens the translated batch together."""
    from scripts.tutorials.atomic_action import tutorial_utils

    batch_size = 4
    target_pose = torch.eye(4).repeat(batch_size, 1, 1)
    target_pose[:, :3, 3] = torch.tensor(
        [
            [-0.82, -0.02, 0.46],
            [-0.80, 0.01, 0.46],
            [-0.84, 0.02, 0.46],
            [-0.81, -0.01, 0.46],
        ]
    )
    candidate = target_pose[0].clone()
    candidate[:3, 3] += torch.tensor([0.01, 0.0, 0.0])
    generator = Mock()
    generator.get_valid_grasp_poses.return_value = [
        (candidate.unsqueeze(0), torch.zeros(1))
    ]
    monkeypatch.setattr(
        tutorial_utils,
        "create_parallel_jaw_grasp_pose_generator",
        lambda **_: generator,
    )
    robot = Mock(device=torch.device("cpu"))
    robot.compute_ik.side_effect = lambda pose, joint_seed, name: (
        torch.ones(batch_size, dtype=torch.bool),
        joint_seed.clone(),
    )
    scenario = AtomicTaskScenario()
    scenario.robot = robot
    scenario.control_part = "arm"

    grasp = scenario.resolve_articulation_grasp(
        mesh_vertices=torch.zeros(3, 3),
        mesh_triangles=torch.tensor([[0, 1, 2]]),
        target_pose=target_pose,
        approach_direction=torch.tensor([0.0, 0.0, 1.0]).repeat(batch_size, 1),
        start_qpos=torch.zeros(batch_size, 7),
        direction="pull",
        approach_distance=0.10,
        translation_distance=0.18,
        seed=11,
        n_sample=32,
        max_candidates=8,
        alignment_max_angle_deg=10.0,
    )

    expected = target_pose.clone()
    expected[:, 0, 3] += 0.01
    torch.testing.assert_close(grasp, expected)
    sampled = generator.get_valid_grasp_poses.call_args.kwargs
    assert sampled["obj_poses"].shape == (1, 4, 4)
    assert sampled["approach_direction"].shape == (1, 3)
    assert robot.compute_ik.call_count == 3
    assert all(
        call.kwargs["pose"].shape == (batch_size, 4, 4)
        for call in robot.compute_ik.call_args_list
    )


def test_pickup_case_pins_settle_steps_in_pose_manifest() -> None:
    """PickUp forwards its settle window so reset/replay can reproduce it."""
    pose = torch.eye(4).unsqueeze(0)
    pose[:, 2, 3] = 0.025
    handle = Mock(
        object_id="cube",
        entity=Mock(),
        config={"settle_steps": 10},
    )
    robot = Mock(device=torch.device("cpu"))
    robot.get_joint_ids.return_value = list(range(7))
    robot.get_qpos.side_effect = lambda name=None: (
        torch.zeros(1, 7) if name == "arm" else torch.zeros(1, 9)
    )
    robot.compute_fk.return_value = pose
    robot.compute_ik.side_effect = lambda pose, joint_seed, name: (
        torch.ones(1, dtype=torch.bool),
        joint_seed,
    )
    scenario = Mock(robot=robot, control_part="arm")
    scenario.activate_object.return_value = handle
    scenario.randomize_object_pose.return_value = pose.clone()
    scenario.solve_reference_qpos.side_effect = lambda start, targets: torch.zeros(
        targets.shape[0], targets.shape[1], 7
    )

    case = create_atomic_skill_provider("pick_up").generate_case(
        scenario,
        Mock(suite_version="test_v1", robot=Mock(id="franka_pgi")),
        Mock(id="atomic-task"),
        {
            "name": "settled_pick",
            "object": "cube",
            "grasp_source": "fixed",
            "pre_action_settle_steps": 50,
        },
        seed=11,
        batch_size=1,
    )

    scenario.randomize_object_pose.assert_called_once_with(
        handle,
        {
            "name": "settled_pick",
            "object": "cube",
            "grasp_source": "fixed",
            "pre_action_settle_steps": 50,
        },
        seed=11,
        stream=201,
        settle_steps=50,
    )
    assert case.case_parameters["pre_action_settle_steps"] == 50
    torch.testing.assert_close(
        torch.tensor(case.case_parameters["object_initial_pose"]), pose
    )


def test_reset_case_reapplies_frozen_object_pose_after_settle() -> None:
    """Repeated planner attempts start from the manifest pose, not drifted state."""
    pose = torch.eye(4).unsqueeze(0)
    entity = Mock()
    handle = Mock(object_id="cube", entity=entity)
    simulation = Mock()
    robot = Mock(device=torch.device("cpu"))
    scenario = AtomicTaskScenario()
    scenario._objects = {"cube": handle}

    case = Mock(
        full_start_qpos=torch.zeros(1, 7),
        object_id="cube",
        case_parameters={
            "object_initial_pose": pose.tolist(),
            "pre_action_settle_steps": 50,
        },
    )

    scenario.reset_case(simulation, robot, case, "arm")

    simulation.update.assert_called_once_with(step=50)
    assert entity.set_local_pose.call_count == 2
    torch.testing.assert_close(entity.set_local_pose.call_args.args[0], pose)
    assert entity.clear_dynamics.call_count == 2


def test_reset_case_restores_batched_articulation_root_and_joint_state() -> None:
    """Each translated articulation row is restored from the frozen manifest."""
    batch_size = 4
    pose = torch.eye(4).repeat(batch_size, 1, 1)
    pose[:, 0, 3] = torch.tensor([-1.02, -0.98, -1.01, -0.99])
    qpos = torch.zeros(batch_size, 4)
    qpos[:, 0] = torch.tensor([0.0, 0.001, 0.002, 0.003])
    handle = Mock(
        object_id="microwave",
        config={"reset_settle_steps": 10},
    )
    simulation = Mock()
    robot = Mock(device=torch.device("cpu"))
    scenario = AtomicTaskScenario()
    scenario._articulations = {"microwave": handle}
    case = Mock(
        full_start_qpos=torch.zeros(batch_size, 8),
        object_id="microwave",
        case_parameters={
            "articulation_initial_pose": pose.tolist(),
            "articulation_initial_qpos": qpos.tolist(),
        },
    )

    scenario.reset_case(simulation, robot, case, "arm")

    assert handle.reset.call_count == 2
    restored = handle.reset.call_args.kwargs
    torch.testing.assert_close(restored["pose"], pose)
    torch.testing.assert_close(restored["qpos"], qpos)
    simulation.update.assert_called_once_with(step=10)


def test_pose_fallback_does_not_leak_into_local_grasp_offsets() -> None:
    """Track pose ranges apply only to target/object pose jitter keys."""
    config = {
        "_pose_batch_translation_jitter_m": [0.25, 0.25, 0.25],
        "_pose_batch_object_translation_jitter_m": [0.25, 0.25, 0.25],
        "_pose_batch_seed_stride": 1,
    }
    target = _randomized_vector_batch(
        [0.0, 0.0, 0.0],
        config,
        jitter_name="target_offset_jitter_m",
        seed=11,
        stream=101,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    held = _randomized_vector_batch(
        [0.0, 0.0, 0.0],
        config,
        jitter_name="held_object_offset_jitter_m",
        seed=11,
        stream=101,
        batch_size=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.allclose(held, torch.zeros_like(held))
    assert not torch.allclose(target, torch.zeros_like(target))


def test_pose_batch_internal_ranges_are_not_exposed_as_legacy_jitter() -> None:
    """Manifest ranges retain user keys while dedicated pose metadata is explicit."""
    parameters = _randomization_parameters(
        {
            "_pose_batch_enabled": True,
            "_pose_batch_seed_stride": 1,
            "_pose_batch_sampling_pattern": "stratified",
            "_pose_batch_translation_jitter_m": [0.1, 0.1, 0.0],
            "_pose_batch_object_translation_jitter_m": [0.2, 0.2, 0.0],
            "_pose_batch_object_yaw_jitter_rad": 0.5,
            "_pose_batch_object_roll_pitch_jitter_rad": [0.1, 0.2],
            "_pose_batch_target_yaw_jitter_rad": 0.4,
            "_pose_batch_target_roll_pitch_jitter_rad": [0.2, 0.3],
            "_pose_batch_support_object_translation_jitter_m": [0.03, 0.03, 0.0],
            "_pose_batch_support_object_yaw_jitter_rad": 0.3,
            "_pose_batch_support_object_roll_pitch_jitter_rad": [0.3, 0.4],
            "_pose_batch_task_frame_translation_jitter_m": [0.01, 0.01, 0.0],
            "_pose_batch_task_frame_yaw_jitter_rad": 0.2,
            "_pose_batch_task_frame_roll_pitch_jitter_rad": [0.4, 0.5],
            "target_offset_jitter_m": [0.1, 0.1, 0.0],
        },
        seed=11,
    )
    assert all(not key.startswith("_pose_batch_") for key in parameters["ranges"])
    assert parameters["pose_batch"] is True
    assert parameters["distribution"] == "deterministic_stratified"
    assert parameters["pose_batch_sampling_pattern"] == "stratified"
    assert parameters["pose_batch_object_yaw_jitter_rad"] == pytest.approx(0.5)
    assert parameters["pose_batch_object_roll_pitch_jitter_rad"] == [0.1, 0.2]
    assert parameters["pose_batch_target_yaw_jitter_rad"] == pytest.approx(0.4)
    assert parameters["pose_batch_target_roll_pitch_jitter_rad"] == [0.2, 0.3]
    assert parameters["pose_batch_support_object_translation_jitter_m"] == [
        0.03,
        0.03,
        0.0,
    ]
    assert parameters["pose_batch_support_object_yaw_jitter_rad"] == pytest.approx(0.3)
    assert parameters["pose_batch_support_object_roll_pitch_jitter_rad"] == [
        0.3,
        0.4,
    ]
    assert parameters["pose_batch_task_frame_translation_jitter_m"] == [
        0.01,
        0.01,
        0.0,
    ]
    assert parameters["pose_batch_task_frame_yaw_jitter_rad"] == pytest.approx(0.2)
    assert parameters["pose_batch_task_frame_roll_pitch_jitter_rad"] == [0.4, 0.5]


def test_workspace_shell_manifest_records_row_buckets() -> None:
    """Case metadata maps every shell row to its normalized radial bucket."""
    parameters = _randomization_parameters(
        {
            "_pose_batch_enabled": True,
            "_pose_batch_size": 8,
            "_pose_batch_seed_stride": 1,
            "_pose_batch_sampling_pattern": "workspace_shell",
        },
        seed=23,
    )

    assert parameters["distribution"] == "deterministic_workspace_shell"
    assert parameters["spatial_bucket_axis"] == "normalized_xy_offset_radius"
    assert parameters["spatial_buckets"] == [
        "near",
        "near",
        "mid",
        "mid",
        "mid",
        "far",
        "far",
        "far",
    ]
    assert parameters["spatial_bucket_radii"] == pytest.approx(
        [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )
