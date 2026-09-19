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

"""Contracts for opt-in Affordance offline collection."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.envs.augmentation import (
    AffordanceAugmentationCfg,
    _sampling_attempt,
    _select_accepted_rows,
    _project_affordance_metadata,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
from embodichain.lab.gym.envs.expert_trajectory import ExpertTrajectoryCfg


def test_expert_config_decodes_optional_augmentation():
    assert ExpertTrajectoryCfg().affordance_augmentation is None
    cfg = ExpertTrajectoryCfg(affordance_augmentation={"branches": 4, "max_batches": 9})
    assert isinstance(cfg.affordance_augmentation, AffordanceAugmentationCfg)
    assert cfg.affordance_augmentation.branches == 4
    assert cfg.affordance_augmentation.max_batches == 9
    with pytest.raises(TypeError):
        ExpertTrajectoryCfg(affordance_augmentation={"branhes": 4})


@pytest.mark.parametrize(
    "values",
    [
        {"branches": 0},
        {"branches": True},
        {"max_batches": -1},
        {"required_assurance": "projected"},
    ],
)
def test_augmentation_rejects_invalid_config(values):
    with pytest.raises((TypeError, ValueError)):
        AffordanceAugmentationCfg(**values)


def test_augmentation_requires_explicit_seed_and_matching_parallelism():
    cfg = AffordanceAugmentationCfg(branches=4)
    cfg.validate_host(num_envs=4, seed=7)
    with pytest.raises(ValueError, match="branches"):
        cfg.validate_host(num_envs=3, seed=7)
    with pytest.raises(ValueError, match="seed"):
        cfg.validate_host(num_envs=4, seed=None)
    AffordanceAugmentationCfg(branches=1).validate_host(num_envs=4, seed=7)


def test_sampling_attempt_is_stable_and_clears_even_after_exception():
    env = SimpleNamespace(cfg=SimpleNamespace(seed=7), num_envs=4)
    env.get_affordance_sampling_context = (
        lambda: EmbodiedEnv.get_affordance_sampling_context(env)
    )
    cfg = AffordanceAugmentationCfg(branches=4)
    with pytest.raises(RuntimeError, match="planner"):
        with _sampling_attempt(env, cfg, run_id="run", batch_id=2, attempt_id=1):
            context = env.get_affordance_sampling_context()
            assert env.get_affordance_sampling_context() is context
            assert (
                context.count,
                context.seed,
                context.episode_id,
                context.attempt_id,
            ) == (4, 7, 2, 1)
            assert env._affordance_collection_metadata["run_id"] == "run"
            raise RuntimeError("planner")
    assert env.get_affordance_sampling_context() is None
    assert env._affordance_collection_metadata is None


def test_select_accepted_rows_requires_all_row_evidence_and_exact_quota():
    assert _select_accepted_rows(
        completed=(True, True, True, False),
        success=(False, True, True, True),
        measured=(True, True, True, True),
        lengths=(5, 7, 9, 3),
        remaining=1,
    ) == (1,)
    assert (
        _select_accepted_rows(
            completed=(True,),
            success=(True,),
            measured=(False,),
            lengths=(3,),
            remaining=2,
        )
        == ()
    )
    with pytest.raises(ValueError):
        _select_accepted_rows(
            completed=(True,), success=(), measured=(True,), lengths=(3,), remaining=1
        )


def test_provenance_projects_only_declared_sampling_rows_and_owns_copy():
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = 0.25
    record = {
        "runtime": {
            "affordance_sample": {
                "grasp": {
                    "key": "pick/grasp",
                    "sampling": {"count": 2},
                    "candidate_ids": [2, 5],
                    "success": [True, True],
                    "selected_poses": poses.tolist(),
                    "reference_poses": None,
                    "pose_frame": "world",
                    "env_ids": [0, 1],
                }
            }
        },
        "unrelated": ["keep", "both"],
    }
    selected = _project_affordance_metadata(record, env_id=1)
    grasp = selected["runtime"]["affordance_sample"]["grasp"]
    assert grasp["candidate_ids"] == 5
    assert grasp["selected_poses"][0][3] == 0.25
    assert selected["unrelated"] == ["keep", "both"]
    grasp["sampling"]["count"] = 9
    assert record["runtime"]["affordance_sample"]["grasp"]["sampling"]["count"] == 2


def test_augmented_episode_metadata_has_distinct_row_identity():
    env = SimpleNamespace(num_envs=2, cfg=SimpleNamespace(seed=7))
    with _sampling_attempt(
        env,
        AffordanceAugmentationCfg(branches=2),
        run_id="run",
        batch_id=4,
        attempt_id=2,
    ):
        first = EmbodiedEnv._new_demo_episode_metadata(env, 0)["augmentation"]
        second = EmbodiedEnv._new_demo_episode_metadata(env, 1)["augmentation"]
        assert first["episode_id"] != second["episode_id"]
        assert first["commit_id"] != second["commit_id"]
        assert second["physical_env_id"] == 1
        assert second["branch"] == 1
        second["sampling"]["count"] = 999
        assert env._affordance_collection_metadata["sampling"]["count"] == 2


def test_augmentation_row_selection_records_measured_acceptance():
    from embodichain.lab.gym.envs.demo import DemoEpisodeResult

    metadata = [{"augmentation": {}} for _ in range(3)]
    adapter = SimpleNamespace(
        measured_success_mask=lambda program: torch.tensor([True, True, False])
    )
    env = SimpleNamespace(
        num_envs=3, task_program_adapter=adapter, _demo_episode_metadata=metadata
    )
    result = DemoEpisodeResult(
        episode_index=0,
        length=5,
        completed=False,
        success=(False, True, True),
        terminated=(False, False, False),
        truncated=(False, False, False),
        terminal_reason="mixed",
        completed_by_env=(False, True, True),
        lengths=(2, 5, 5),
    )
    rows = EmbodiedEnv.select_affordance_episode_rows(
        env, result, object(), remaining=2
    )
    assert rows == (1,)
    assert env._affordance_accepted_env_ids == (1,)
    assert metadata[2]["augmentation"]["measured_success"] is False


def test_sampling_projection_maps_replanned_subset_env_ids():
    record = {
        "affordance_sample": {
            "grasp": {
                "key": "pick/grasp",
                "sampling": {"count": 4},
                "env_ids": [1, 3],
                "candidate_ids": [2, 8],
                "success": [True, True],
            }
        }
    }
    assert (
        _project_affordance_metadata(record, env_id=3)["affordance_sample"]["grasp"][
            "candidate_ids"
        ]
        == 8
    )


@pytest.mark.parametrize("unsupported", ["no_program", "rl", "no_sink", "save_failed"])
def test_collection_preflight_rejects_unsupported_hosts_before_compilation(unsupported):
    def unexpected_compile(program):
        pytest.fail("Unsupported collection host reached compilation")

    env = SimpleNamespace(
        num_envs=2,
        cfg=SimpleNamespace(
            seed=7,
            expert_trajectory=ExpertTrajectoryCfg(
                affordance_augmentation={"branches": 2}
            ),
            task_program=object(),
        ),
        dataset_manager=SimpleNamespace(save_failed_episodes=False),
        compile_task_program=unexpected_compile,
    )
    if unsupported == "no_program":
        env.cfg.task_program = None
    elif unsupported == "rl":
        env._rollout_buffer_mode = "rl"
    elif unsupported == "no_sink":
        env.dataset_manager = None
    else:
        env.dataset_manager.save_failed_episodes = True
    with pytest.raises(ValueError):
        EmbodiedEnv.prepare_affordance_collection(env)


@pytest.mark.parametrize("rows", [(), (0,), (1, 1), (True,)])
def test_host_commit_rejects_unaccepted_or_duplicate_rows_before_reset(rows):
    def unexpected_reset(**kwargs):
        pytest.fail("Unaccepted rows reached the persistence reset")

    env = SimpleNamespace(_affordance_accepted_env_ids=(1,), reset=unexpected_reset)
    with pytest.raises(ValueError):
        EmbodiedEnv.commit_demo_rows(env, rows)
