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

"""Protect paired held-out simulation report aggregation."""

from __future__ import annotations

import json
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from embodichain_models.dynamics.lewm import LeWMCEMCfg
from examples.learning.lewm._official_cube import (
    _collapse_robot,
    _welded_body_exclusions,
)
from examples.learning.lewm._evaluation import _check_contract, merge_sim_reports
from examples.learning.lewm._diagnostics import rank_summary
from examples.learning.lewm._reachable import select_pair


def test_render_contract_must_be_known_and_match() -> None:
    stable = {"render": {"renderer": "hybrid", "offscreen_denoising": "off"}}
    legacy = {"render": {"renderer": "hybrid", "offscreen_denoising": "dlss"}}
    assert _check_contract(stable, stable)
    with pytest.raises(ValueError, match="render contracts"):
        _check_contract(stable, legacy)
    with pytest.raises(ValueError, match="render contracts"):
        _check_contract({}, {})
    assert not _check_contract(stable, legacy, allow_render_mismatch=True)
    with pytest.raises(ValueError, match="action_dim"):
        _check_contract(
            {**stable, "action_dim": 6},
            {**legacy, "action_dim": 7},
            allow_render_mismatch=True,
        )
    with pytest.raises(ValueError, match="action_block"):
        _check_contract(stable | {"action_block": 5}, stable)


def test_future_goal_selection_requires_real_motion_and_admissible_history() -> None:
    positions = np.array([[0, 0], [0.02, 0], [0.06, 0], [0.15, 0], [0.19, 0]])
    valid = np.array([True, False, True, True, True])
    assert select_pair(positions, valid, 2, 1, 0.1, 0.005) == 2
    assert select_pair(np.zeros_like(positions), valid, 2, 1, 0.1, 0.005) is None


def test_merge_requires_exact_seed_coverage_and_matching_pairs(tmp_path) -> None:
    def row(seed, success, distance):
        return {
            "seed": seed,
            "success": success,
            "distance": distance,
            "planning_seconds": [0.5],
            "cycle_seconds": [0.6],
            "initial_cube": [seed, 0],
            "initial_qpos": [1],
            "initial_goal": [2],
        }

    paths = []
    for index in range(2):
        seed = 300_000 + index
        report = {
            "split": "test",
            "weights_sha256": "same",
            "cem": {"horizon": 5},
            "step_dt": 0.04,
            "success_threshold": 0.1,
            "device": "cuda:0",
            "gpu": "test",
            "torch": "test",
            "gpu_peak_bytes": 1,
            "start_index": index,
            "episodes": 1,
            "records": {
                "lewm": [row(seed, index == 0, 0.12)],
                "random": [row(seed, False, 0.14)],
            },
        }
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(report))
        paths.append(str(path))
    merged = merge_sim_reports(paths, 2)
    assert merged["beats_random"]
    assert not merged["acceptance_eligible"]
    assert not merged["passed"]
    assert merged["summary"]["lewm"]["success_rate"] == 0.5
    assert merged["summary"]["random"]["mean_final_distance"] == pytest.approx(0.14)
    with pytest.raises(ValueError, match="each expected seed"):
        merge_sim_reports([paths[0], paths[0]], 2)
    report = json.loads((tmp_path / "1.json").read_text())
    report["records"]["random"][0]["initial_goal"] = [3]
    (tmp_path / "1.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="different initial states"):
        merge_sim_reports(paths, 2)


def test_merge_accepts_unseen_seed_range_and_rejects_mixed_rendering(tmp_path) -> None:
    rows = []
    for kind in ("lewm", "random"):
        rows.append(
            {
                "seed": 300_100,
                "success": False,
                "distance": 0.15,
                "planning_seconds": [],
                "cycle_seconds": [],
                "initial_cube": [0],
                "initial_qpos": [0],
                "initial_goal": [1],
            }
        )
    report = {
        "split": "test",
        "weights_sha256": "model",
        "cem": {},
        "step_dt": 0.04,
        "success_threshold": 0.1,
        "render": {"offscreen_denoising": "off"},
        "render_contract_verified": True,
        "goal_render_frames": 32,
        "device": "cuda:0",
        "gpu": "test",
        "torch": "test",
        "gpu_peak_bytes": 1,
        "start_index": 100,
        "episodes": 1,
        "records": dict(zip(("lewm", "random"), ([rows[0]], [rows[1]]))),
    }
    path = tmp_path / "shard.json"
    path.write_text(json.dumps(report))
    merged = merge_sim_reports([str(path)], episodes=1, start_index=100)
    assert merged["start_index"] == 100
    with pytest.raises(ValueError, match="each expected seed"):
        merge_sim_reports([str(path)], episodes=1)
    other = dict(report, render={"offscreen_denoising": "dlss"})
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different model or evaluation contracts"):
        merge_sim_reports([str(path), str(other_path)], episodes=2, start_index=100)
    other = dict(report, goal_render_frames=1)
    other_path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different model or evaluation contracts"):
        merge_sim_reports([str(path), str(other_path)], episodes=2, start_index=100)
    other = dict(report, cem={"sampling_distribution": "training"})
    other_path.write_text(json.dumps(other))
    with pytest.raises(ValueError, match="different model or evaluation contracts"):
        merge_sim_reports([str(path), str(other_path)], episodes=2, start_index=100)


def test_candidate_ranking_reports_physical_selection_regret() -> None:
    summary = rank_summary(
        np.array([0.0, 2.0, 1.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.30, 0.20, 0.10]),
    )
    assert summary["physically_informative"]
    assert summary["selected_candidate"] == 0
    assert summary["actual_best_candidate"] == 2
    assert summary["selection_regret_m"] == pytest.approx(0.20)
    assert summary["observed_vs_physical_spearman"] == pytest.approx(1.0)


def test_candidate_ranking_does_not_interpret_stationary_cube_noise() -> None:
    summary = rank_summary(
        np.array([0.0, 1.0, 2.0]),
        np.array([2.0, 1.0, 0.0]),
        np.array([0.15000, 0.15001, 0.14999]),
    )
    assert not summary["physically_informative"]
    assert summary["predicted_vs_physical_spearman"] is None
    assert summary["observed_vs_physical_spearman"] is None


def test_acceptance_requires_full_budget_and_verified_test_rendering(tmp_path) -> None:
    def rows(is_lewm):
        return [
            {
                "seed": 300_100 + index,
                "success": is_lewm and index == 0,
                "distance": 0.15,
                "planning_seconds": [],
                "cycle_seconds": [],
                "initial_cube": [0],
                "initial_qpos": [0],
                "initial_goal": [1],
            }
            for index in range(50)
        ]

    report = {
        "split": "test",
        "weights_sha256": "model",
        "cem": LeWMCEMCfg().to_dict(),
        "step_dt": 0.04,
        "success_threshold": 0.1,
        "render": {"offscreen_denoising": "off"},
        "render_contract_verified": True,
        "device": "cuda:0",
        "gpu": "test",
        "torch": "test",
        "gpu_peak_bytes": 1,
        "start_index": 100,
        "episodes": 50,
        "records": {"lewm": rows(True), "random": rows(False)},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert merge_sim_reports([str(path)], episodes=50, start_index=100)["passed"]
    report["cem"].pop("sampling_distribution")
    report["cem"].pop("action_selection")
    path.write_text(json.dumps(report))
    assert merge_sim_reports([str(path)], episodes=50, start_index=100)["passed"]
    for key, value in (
        ("num_samples", 4),
        ("sampling_distribution", "training"),
        ("action_selection", "mean"),
    ):
        report["cem"] = LeWMCEMCfg().to_dict() | {key: value}
        path.write_text(json.dumps(report))
        assert not merge_sim_reports([str(path)], episodes=50, start_index=100)[
            "passed"
        ]
    report["cem"] = LeWMCEMCfg().to_dict()
    report["render_contract_verified"] = False
    path.write_text(json.dumps(report))
    assert not merge_sim_reports([str(path)], episodes=50, start_index=100)["passed"]


def test_robot_export_retains_massive_fixed_pad_and_folds_massless_site() -> None:
    robot = ET.fromstring("""<robot name="force-observed">
      <link name="world"/>
      <link name="ur5e/base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="base_fixed" type="fixed"><parent link="world"/><child link="ur5e/base"/></joint>
      <link name="hinged"><inertial><mass value="2"/><inertia ixx=".2" iyy=".2" izz=".2" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="hinge" type="revolute"><parent link="ur5e/base"/><child link="hinged"/><axis xyz="0 0 1"/></joint>
      <link name="pad"><inertial><mass value=".0035"/><inertia ixx="1e-7" iyy="2e-7" izz="3e-7" ixy="0" ixz="0" iyz="0"/></inertial></link>
      <joint name="pad_fixed" type="fixed"><parent link="hinged"/><child link="pad"/><origin xyz=".1 0 0"/></joint>
      <link name="pinch_site"/>
      <joint name="site_fixed" type="fixed"><parent link="pad"/><child link="pinch_site"/><origin xyz="0 .02 0"/></joint>
    </robot>""")
    source_bodies = [
        {"name": name, "mass": mass, "urdf_frame_in_body": [0, 0, 0]}
        for name, mass in [("ur5e/base", 1), ("hinged", 2), ("pad", 0.0035)]
    ]
    exported, metadata = _collapse_robot(robot, source_bodies)
    bodies = {body["name"]: body for body in metadata["bodies"]}
    assert set(bodies) == {"ur5e/base", "hinged", "pad"}
    assert bodies["pad"]["mass"] == 0.0035
    assert bodies["hinged"]["mass"] == 2
    np.testing.assert_allclose(
        bodies["pad"]["inertia_matrix"], np.diag([1e-7, 2e-7, 3e-7]), atol=1e-20
    )
    assert metadata["total_mass"] == pytest.approx(3.0035)
    site = next(
        frame for frame in metadata["source_frames"] if frame["name"] == "pinch_site"
    )
    assert site["link"] == "pad"
    np.testing.assert_allclose(np.array(site["transform"])[:3, 3], [0, 0.02, 0])
    assert "pinch_site" not in {link.get("name") for link in exported.findall("link")}


def test_fixed_link_retention_keeps_source_weld_group_collision_rules() -> None:
    bodies = [
        {"name": name, "source_weld_id": weld, "source_parent_weld_id": parent}
        for name, weld, parent in [
            ("world_base", 0, 0),
            ("a", 1, 0),
            ("a_pad", 1, 0),
            ("b", 2, 1),
            ("b_pad", 2, 1),
            ("c", 3, 2),
        ]
    ]
    actual = {
        tuple(pair) for pair in _welded_body_exclusions(bodies, filter_parent=True)
    }
    assert actual == {
        ("a", "a_pad"),
        ("b", "b_pad"),
        ("a", "b"),
        ("a", "b_pad"),
        ("a_pad", "b"),
        ("a_pad", "b_pad"),
        ("b", "c"),
        ("b_pad", "c"),
    }
    assert {
        tuple(pair) for pair in _welded_body_exclusions(bodies, filter_parent=False)
    } == {
        ("a", "a_pad"),
        ("b", "b_pad"),
    }
