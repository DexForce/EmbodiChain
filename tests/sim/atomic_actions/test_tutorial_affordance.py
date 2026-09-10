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

"""Pure CPU tests for frozen sampling, physical-row mapping and tutorial cleanup."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import weakref

import numpy as np
import pytest
import torch

from embodichain.lab.sim.atomic_actions import TimedTrajectory
from embodichain.toolkits.graspkit import GraspCandidateBatch
from scripts.tutorials.atomic_action import tutorial_utils
from scripts.tutorials.atomic_action.affordance_utils import (
    rebase_affordance_grasps,
    sample_affordance_grasps,
    save_affordance_result,
    selected_grasps_from_result,
)


def _poses() -> torch.Tensor:
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.0, 0.2, 0.4])
    return poses


def _sampled() -> GraspCandidateBatch:
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[0, 0, 1, 3] = 0.01
    poses[0, 1, 1, 3] = 0.02
    return GraspCandidateBatch(
        poses,
        torch.tensor([[0.1, 0.2]]),
        torch.ones(1, 2, dtype=torch.bool),
        opening_widths=torch.tensor([[0.03, 0.04]]),
        grasp_ids=(("grasp-a", "grasp-b"),),
    )


def _affordance() -> SimpleNamespace:
    return SimpleNamespace(
        mesh_vertices=torch.zeros(3, 3), mesh_triangles=torch.tensor([[0, 1, 2]])
    )


def _result(*, empty: bool = False) -> SimpleNamespace:
    qpos = torch.zeros(3, 4, 2)
    qpos[0, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.2])
    qpos[2, :, 0] = torch.tensor([0.0, 0.3, 0.4, 0.5])
    mask = (
        torch.zeros(3, dtype=torch.bool) if empty else torch.tensor([True, False, True])
    )
    return SimpleNamespace(
        trajectory=TimedTrajectory.from_uniform_step(
            qpos, env_ids=torch.arange(3), step_dt=0.01
        ),
        success_mask=mask,
        valid_length=torch.tensor([3, 1, 4]),
        grasp_ids=("grasp-a", None, "grasp-b"),
        grasp_poses=_poses(),
        candidate_indices=torch.tensor([0, -1, 1]),
        rejections=({"env_id": 1, "reason": "IK_FAILED"},),
        summary={"output_count": int(mask.sum())},
    )


@pytest.mark.parametrize(
    "flag", ["--n_affordance_multi_gen", "--n-affordance-multi-gen"]
)
def test_multi_gen_parser_aliases_keep_default_disabled(flag: str) -> None:
    parser = tutorial_utils.create_tutorial_argument_parser(
        "test", features=("affordance_multi_gen",)
    )
    assert parser.parse_args([]).n_affordance_multi_gen is None
    args = parser.parse_args([flag, "3", "--affordance_output", "/tmp/result"])
    assert args.n_affordance_multi_gen == 3
    assert args.affordance_output == Path("/tmp/result") and args.affordance_seed == 13


@pytest.mark.parametrize("value", ["0", "-2", "1.5", "bad"])
def test_multi_gen_parser_rejects_nonpositive_counts(value: str) -> None:
    parser = tutorial_utils.create_tutorial_argument_parser(
        "test", features=("affordance_multi_gen",)
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--n_affordance_multi_gen", value])


def test_explicit_multi_count_drives_real_simulator_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock()
    monkeypatch.setattr(tutorial_utils, "SimulationManager", factory)
    args = Namespace(
        num_envs=99,
        n_affordance_multi_gen=3,
        device="cpu",
        renderer="hybrid",
        viser=False,
    )
    tutorial_utils.create_tutorial_simulation(args)
    assert args.num_envs == 3
    assert factory.call_args.args[0].num_envs == 3


def test_factory_optional_candidate_limit_preserves_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = MagicMock()
    monkeypatch.setattr(tutorial_utils, "AntipodalGraspPoseGenerator", factory)
    tutorial_utils.create_parallel_jaw_grasp_pose_generator(
        n_sample=10, force_refresh=False
    )
    assert factory.call_args.kwargs["algorithm_cfg"].max_candidates == 50
    tutorial_utils.create_parallel_jaw_grasp_pose_generator(
        n_sample=10, force_refresh=False, max_candidates=7
    )
    assert factory.call_args.kwargs["algorithm_cfg"].max_candidates == 7


def test_frozen_sampling_calls_backend_once_for_row_zero_and_keeps_raw_ids() -> None:
    service = MagicMock()
    service.get_grasp_candidates.return_value = _sampled()
    global_rng = torch.random.get_rng_state().clone()
    result = sample_affordance_grasps(
        service,
        _affordance(),
        object_poses=_poses(),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
        seed=37,
    )
    assert service.get_grasp_candidates.call_count == 1
    assert service.get_grasp_candidates.call_args.kwargs["obj_poses"].shape == (1, 4, 4)
    assert (
        service.get_grasp_candidates.call_args.kwargs["generator"].initial_seed() == 37
    )
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert result.grasp_ids == (("grasp-a", "grasp-b"),) * 3
    torch.testing.assert_close(
        result.poses[:, :, 0, 3], _poses()[:, None, 0, 3].expand(-1, 2)
    )
    torch.testing.assert_close(
        result.poses[:, :, 1, 3], torch.tensor([[0.01, 0.02]]).expand(3, -1)
    )


def test_frozen_sampling_rejects_different_object_local_approaches() -> None:
    with pytest.raises(ValueError, match="object-local approach"):
        sample_affordance_grasps(
            MagicMock(),
            _affordance(),
            object_poses=_poses(),
            approach_direction=torch.tensor(
                [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
            ),
        )


def test_empty_sampling_is_kept_empty_for_every_physical_row() -> None:
    service = MagicMock()
    service.get_grasp_candidates.return_value = GraspCandidateBatch(
        torch.empty(1, 0, 4, 4), torch.empty(1, 0), torch.empty(1, 0, dtype=torch.bool)
    )
    result = sample_affordance_grasps(
        service,
        _affordance(),
        object_poses=_poses(),
        approach_direction=torch.tensor([0.0, 0.0, -1.0]),
    )
    assert result.poses.shape == (3, 0, 4, 4)


def test_push_rebases_each_selected_raw_grasp_to_actual_handle_and_keeps_slots() -> (
    None
):
    result = _result()
    old = _poses()
    result.grasp_poses = old.clone()
    result.grasp_poses[:, 2, 3] = 0.1
    new = old.clone()
    new[:, 1, 3] = torch.tensor(
        [0.08, 0.0, 0.15]
    )  # Actual row-dependent pull distance.
    selected = selected_grasps_from_result(result, old, new)
    assert selected.valid_mask[:, 0].tolist() == [True, False, True]
    assert selected.grasp_ids == (("grasp-a",), ("inactive:1",), ("grasp-b",))
    torch.testing.assert_close(
        selected.poses[[0, 2], 0, 1, 3], torch.tensor([0.08, 0.15])
    )
    torch.testing.assert_close(
        selected.poses[[0, 2], 0, 2, 3], torch.tensor([0.1, 0.1])
    )


def test_rebase_preserves_failure_reasons_without_reviving_invalid_grasps() -> None:
    raw = _sampled()
    raw = GraspCandidateBatch(
        raw.poses, raw.costs, torch.tensor([[True, False]]), grasp_ids=raw.grasp_ids
    )
    result = rebase_affordance_grasps(raw, _poses()[:1], _poses()[:1])
    assert result.valid_mask.tolist() == [[True, False]]
    assert result.rejection_reasons[0][1] == "INELIGIBLE"


@pytest.mark.parametrize("empty", [False, True])
def test_save_keeps_only_compact_successes_with_original_physical_env_ids(
    tmp_path: Path, empty: bool
) -> None:
    result = _result(empty=empty)
    report = save_affordance_result(
        result, output_dir=tmp_path, name="slide_pull", physical_envs=3
    )
    assert not report["physical_validation"]
    assert report["output_count"] == (0 if empty else 2)
    assert report["compact_env_ids"] == ([] if empty else [0, 2])
    assert json.loads((tmp_path / "slide_pull.json").read_text()) == report
    with np.load(tmp_path / "slide_pull.npz", allow_pickle=False) as saved:
        assert saved["qpos"].shape == ((0, 0, 2) if empty else (2, 4, 2))
        if not empty:
            assert saved["env_ids"].tolist() == [0, 2]
            assert saved["grasp_ids"].tolist() == ["grasp-a", "grasp-b"]
            assert saved["dt"][0, 3] == 0 and not saved["valid_mask"][0, 3]


def test_save_none_only_reports_and_existing_stage_files_are_protected(
    tmp_path: Path,
) -> None:
    report = save_affordance_result(
        _result(), output_dir=None, name="pickup", physical_envs=3
    )
    assert report["output_count"] == 2
    assert not list(tmp_path.iterdir())
    save_affordance_result(
        _result(), output_dir=tmp_path, name="pickup", physical_envs=3
    )
    with pytest.raises(FileExistsError):
        save_affordance_result(
            _result(), output_dir=tmp_path, name="pickup", physical_envs=3
        )


def test_tutorial_backend_exception_releases_native_locals_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owners = []

    class Owner:
        pass

    def main() -> None:
        owner = Owner()
        owners.append(weakref.ref(owner))
        raise ValueError("original planner error")

    sim = SimpleNamespace(
        _is_constructed=True,
        is_window_recording=lambda: False,
        wait_window_record_saves=lambda: None,
        destroy=lambda **kwargs: None,
    )
    manager = MagicMock()
    manager.get_instance.return_value = sim

    def flush() -> None:
        assert owners[0]() is None
        raise RuntimeError("cleanup error")

    manager.flush_cleanup_queue.side_effect = flush
    monkeypatch.setattr(tutorial_utils, "SimulationManager", manager)
    with pytest.raises(ValueError, match="original planner error"):
        tutorial_utils.run_tutorial(main)
    manager.flush_cleanup_queue.assert_called_once()
