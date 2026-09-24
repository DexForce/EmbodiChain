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

import pytest

from embodichain.lab.sim.motion.expansion import (
    TrajectoryAugmentationCfg,
    TrajectoryGenerationJobCfg,
)


def _capabilities() -> dict[str, set[str]]:
    return {
        "source_ids": {"handwritten_qpos"},
        "validator_ids": {"task_success"},
        "profile_ids": {
            "fixed_scene_initial_state",
            "fixed_scene_tolerances",
            "verified_motion",
            "robot_execution_limits",
        },
        "sink_ids": {"lerobot"},
        "operators": {"joint_residual", "retime"},
    }


def test_job_decodes_nested_schema_and_round_trips_without_imports() -> None:
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "seed": 17,
                "factors": {
                    "timing": {"enabled": True, "duration_scales": [0.8, 1.0, 1.2]}
                },
            },
            "collection": {"target_committed_episodes": 4},
        }
    )
    cfg.validate_capabilities(**_capabilities())
    assert cfg.planning.batch_mode == "env_rows"
    assert cfg.execution.pool_mode == "per_env_case"
    assert cfg.persistence.sink == "lerobot"
    assert cfg.augmentation.factors.timing.duration_scales == (0.8, 1.0, 1.2)
    assert (
        TrajectoryGenerationJobCfg.from_mapping(cfg.to_dict()).to_dict()
        == cfg.to_dict()
    )


def test_generation_profile_is_source_neutral_and_uses_one_family_budget():
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "source": {
                "kind": "motion_generator",
            },
            "affordance": {
                "enabled": True,
                "branches_per_family": 3,
            },
            "scheduling": {
                "policy": "fifo",
                "reference_family_budget": 4,
                "exploration_fraction": 0.2,
            },
        }
    )
    assert cfg.source.kind == "motion_generator"
    assert cfg.affordance.branches_per_family == 3
    assert cfg.scheduling.policy == "fifo"
    assert cfg.scheduling.reference_family_budget == 4


def test_generation_profile_rejects_unimplemented_scheduler() -> None:
    with pytest.raises(ValueError, match="coverage_per_cost.*not implemented"):
        TrajectoryGenerationJobCfg.from_mapping(
            {"scheduling": {"policy": "coverage_per_cost"}}
        )


@pytest.mark.parametrize(
    "kind",
    ["handwritten", "motion_generator", "atomic_action", "task_program"],
)
def test_generation_profile_accepts_all_source_adapters(kind):
    cfg = TrajectoryGenerationJobCfg.from_mapping({"source": {"kind": kind}})
    assert cfg.source.kind == kind


@pytest.mark.parametrize(
    "payload",
    [
        {"seed": 4},
        {"source": {"callable": "os:system"}},
        {"execution": {"unknown": 1}},
        {"augmentation": {"factors": {"timing": {"unknown": 1}}}},
        {"collection": {"max_proposals": True}},
        {"persistence": {"async_write": "false"}},
        {"augmentation": {"seed": -1}},
        {"augmentation": {"seed": 2**63}},
        {
            "augmentation": {
                "factors": {"timing": {"enabled": True, "duration_scales": [0.0, 1.0]}}
            }
        },
        {
            "augmentation": {
                "factors": {
                    "timing": {"enabled": True, "duration_scales": [float("nan")]}
                }
            }
        },
        {"augmentation": {"factors": {"timing": {"duration_scales": [2.0]}}}},
        {"augmentation": {"factors": {"spatial": {"joint_offset_scale": 2.0}}}},
        {"execution": {"ready_low_watermark": 16, "ready_high_watermark": 16}},
        {"execution": {"ready_max_bytes": 0}},
        {"collection": {"max_wall_time_s": float("inf")}},
        {"collection": {"target_committed_episodes": 101}},
        {"persistence": {"pending_max_bytes": 0}},
        {"augmentation": {"factors": {"manipulability": {"band_edges": []}}}},
        {"augmentation": {"factors": {"manipulability": {"band_edges": [0.9, 0.5]}}}},
        {"augmentation": {"factors": {"manipulability": {"band_edges": [0.5, 0.5]}}}},
        {"augmentation": {"factors": {"manipulability": {"band_edges": [0.0]}}}},
        {"augmentation": {"factors": {"manipulability": {"jacobian_rows": "linear"}}}},
        {"augmentation": {"factors": {"manipulability": {"target_per_band": 0}}}},
        {"augmentation": {"factors": {"manipulability": {"guided_proposals": 0}}}},
    ],
)
def test_decoder_rejects_unknown_fields_types_and_invalid_budgets(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TrajectoryGenerationJobCfg.from_mapping(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"planning": {"batch_mode": "candidate_buckets"}},
        {"execution": {"pool_mode": "env_rows"}},
        {"collection": {"max_attempts_per_candidate": 2}},
        {"execution": {"pool_mode": "grouped_replicas"}},
        {"execution": {"scheduler": "slot_refill"}},
        {"execution": {"overlap_planning_and_physics": True}},
        {"reset": {"outer_mode": "host_reset"}},
        {"reset": {"on_initial_state_mismatch": "ignore"}},
        {"persistence": {"async_write": True}},
        {"validation": {"require_path_collision": False}},
        {"validation": {"require_task_success": False}},
        {"augmentation": {"factors": {"contact": {"enabled": True}}}},
        {"augmentation": {"factors": {"contact_timing": {"enabled": True}}}},
        {"augmentation": {"factors": {"recovery": {"enabled": True}}}},
    ],
)
def test_unsupported_modes_cannot_be_silently_enabled(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        TrajectoryGenerationJobCfg.from_mapping(payload)


@pytest.mark.parametrize(
    "registry", ["source_ids", "validator_ids", "profile_ids", "sink_ids"]
)
def test_preflight_requires_explicit_registered_implementations(registry: str) -> None:
    capabilities = _capabilities()
    capabilities[registry] = set()
    with pytest.raises(ValueError, match="unregistered"):
        TrajectoryGenerationJobCfg().validate_capabilities(**capabilities)


def test_spatial_and_timing_factors_require_available_operators() -> None:
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {"spatial": {"enabled": True, "method": "via_points"}}
            }
        }
    )
    with pytest.raises(ValueError, match="via_points"):
        cfg.validate_capabilities(**_capabilities())
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {"timing": {"enabled": True, "duration_scales": [1.0, 2.0]}}
            }
        }
    )
    capabilities = _capabilities()
    capabilities["operators"] = set()
    with pytest.raises(ValueError, match="retime"):
        cfg.validate_capabilities(**capabilities)


def test_manipulability_guidance_requires_its_operator_capability() -> None:
    payload = {
        "augmentation": {
            "factors": {
                "manipulability": {
                    "enabled": True,
                    "band_edges": [0.5, 0.9],
                    "guided_proposals": 4,
                }
            }
        }
    }
    cfg = TrajectoryGenerationJobCfg.from_mapping(payload)
    with pytest.raises(ValueError, match="manipulability guided residual"):
        cfg.validate_capabilities(**_capabilities())
    capabilities = _capabilities()
    capabilities["operators"] |= {"manipulability_guided_residual"}
    cfg.validate_capabilities(**capabilities)
    assert cfg.augmentation.factors.manipulability.band_edges == (0.5, 0.9)
    # Banded coverage alone selects among unguided proposals and needs no operator.
    payload["augmentation"]["factors"]["manipulability"]["guided_proposals"] = 1
    TrajectoryGenerationJobCfg.from_mapping(payload).validate_capabilities(
        **_capabilities()
    )


def test_manipulability_is_disabled_by_default() -> None:
    factors = TrajectoryGenerationJobCfg().augmentation.factors
    assert not factors.manipulability.enabled
    assert factors.manipulability.guided_proposals == 1


def test_configuration_instances_do_not_share_nested_values() -> None:
    first = TrajectoryGenerationJobCfg()
    second = TrajectoryGenerationJobCfg()
    first.augmentation.coverage.target_per_cell = 2
    assert second.augmentation.coverage.target_per_cell == 1


def test_semantic_revalidation_rejects_mutated_configuration() -> None:
    cfg = TrajectoryGenerationJobCfg()
    cfg.persistence.async_write = True
    with pytest.raises(ValueError, match="not implemented"):
        cfg.validate_semantics()
    with pytest.raises(ValueError, match="typed configuration"):
        TrajectoryAugmentationCfg(factors={})


def test_variant_factors_decode_and_declare_their_operator_capabilities() -> None:
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": "via_points",
                        "via_count": 3,
                    },
                    "ik": {"enabled": True, "task_rows": [0, 1, 2, 3, 4]},
                    "approach": {
                        "enabled": True,
                        "cone_half_angle_rad": 0.3,
                        "directions": 4,
                    },
                    "timing": {
                        "enabled": True,
                        "duration_scales": [0.8, 1.0],
                        "profiles": ["uniform", "ease_in"],
                    },
                }
            }
        }
    )
    factors = cfg.augmentation.factors
    assert factors.spatial.method == ("via_points",)
    assert factors.spatial.via_count == 3
    assert factors.ik.task_rows == (0, 1, 2, 3, 4)
    assert factors.approach.directions == 4
    assert factors.timing.profiles == ("uniform", "ease_in")

    registries = {
        "source_ids": ("handwritten_qpos",),
        "validator_ids": ("task_success",),
        "profile_ids": (
            "fixed_scene_initial_state",
            "fixed_scene_tolerances",
            "verified_motion",
            "robot_execution_limits",
        ),
        "sink_ids": ("lerobot",),
    }
    cfg.validate_capabilities(
        operators=(
            "via_points",
            "nullspace_residual",
            "perturb_approach_direction",
            "retime",
        ),
        **registries,
    )
    for missing in ("nullspace_residual", "perturb_approach_direction"):
        available = tuple(
            name
            for name in (
                "via_points",
                "nullspace_residual",
                "perturb_approach_direction",
                "retime",
            )
            if name != missing
        )
        with pytest.raises(ValueError, match="capability unavailable"):
            cfg.validate_capabilities(operators=available, **registries)


@pytest.mark.parametrize(
    "payload",
    [
        {"augmentation": {"factors": {"spatial": {"via_count": 0}}}},
        {"augmentation": {"factors": {"spatial": {"method": []}}}},
        {"augmentation": {"factors": {"spatial": {"method": ["teleport"]}}}},
        {
            "augmentation": {
                "factors": {"spatial": {"method": ["via_points", "via_points"]}}
            }
        },
        {"augmentation": {"factors": {"spatial": {"method": [1]}}}},
        {"augmentation": {"factors": {"spatial": {"via_count": 9}}}},
        {"augmentation": {"factors": {"ik": {"method": "damped_least_squares"}}}},
        {"augmentation": {"factors": {"ik": {"task_rows": []}}}},
        {"augmentation": {"factors": {"ik": {"task_rows": [0, 6]}}}},
        {"augmentation": {"factors": {"ik": {"normalized_scale": 1.5}}}},
        {"augmentation": {"factors": {"approach": {"enabled": True}}}},
        {"augmentation": {"factors": {"approach": {"cone_half_angle_rad": 2.0}}}},
        {"augmentation": {"factors": {"approach": {"directions": 0}}}},
        {
            "augmentation": {
                "factors": {"timing": {"enabled": True, "profiles": ["bounce"]}}
            }
        },
        {
            "augmentation": {
                "factors": {
                    "timing": {"enabled": True, "profiles": ["uniform", "uniform"]}
                }
            }
        },
        {"augmentation": {"factors": {"timing": {"profiles": ["ease_in"]}}}},
    ],
)
def test_variant_factor_settings_are_validated(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TrajectoryGenerationJobCfg.from_mapping(payload)


def test_spatial_method_accepts_one_name_or_several() -> None:
    def decode(value: object) -> tuple[str, ...]:
        cfg = TrajectoryGenerationJobCfg.from_mapping(
            {"augmentation": {"factors": {"spatial": {"method": value}}}}
        )
        return cfg.augmentation.factors.spatial.method

    # A single name keeps its existing spelling and normalizes to one entry.
    assert decode("via_points") == ("via_points",)
    assert decode(["joint_residual"]) == ("joint_residual",)
    assert decode(["joint_residual", "via_points"]) == ("joint_residual", "via_points")
    assert TrajectoryGenerationJobCfg().augmentation.factors.spatial.method == (
        "joint_residual",
    )


def test_every_requested_spatial_method_needs_its_operator() -> None:
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual", "via_points"],
                    }
                }
            }
        }
    )
    registries = _capabilities()
    registries["operators"] = ("joint_residual", "retime")
    with pytest.raises(ValueError, match="via_points"):
        cfg.validate_capabilities(**registries)
    registries["operators"] = ("joint_residual", "via_points", "retime")
    cfg.validate_capabilities(**registries)


@pytest.mark.parametrize(
    "payload",
    [
        {"augmentation": {"factors": {"timing": {"profiles": "uniform"}}}},
        {"augmentation": {"factors": {"ik": {"task_rows": "0"}}}},
    ],
)
def test_only_spatial_method_accepts_a_bare_name(payload: dict[str, object]) -> None:
    # The single-name spelling is a compatibility shim for spatial.method. Any
    # other sequence field must reach its own validator as a sequence, or an
    # unsupported name would be split into characters or silently wrapped.
    with pytest.raises(ValueError, match="must be a sequence"):
        TrajectoryGenerationJobCfg.from_mapping(payload)
