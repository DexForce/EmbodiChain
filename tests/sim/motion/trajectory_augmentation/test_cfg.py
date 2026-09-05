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

from embodichain.lab.sim.motion.trajectory_augmentation import (
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
        {"augmentation": {"factors": {"ik": {"enabled": True}}}},
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
