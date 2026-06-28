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

"""Tests for the Franka pick-place NMG benchmark helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from embodichain.data import get_data_path
from scripts.benchmark.robotics.nmg.franka_pick_place import (
    ContactStats,
    DEFAULT_FRANKA_TCP_YAW,
    DEFAULT_FRANKA_TCP_Z,
    DEFAULT_GRASP_AXIS,
    DEFAULT_GRASP_DEPTH_OFFSET,
    DEFAULT_GRIPPER_CLOSE_MARGIN,
    FRANKA_GRIPPER_CLOSING_AXIS_INDEX,
    OBJECT_SUPPORT_MARGIN,
    TABLE_COLLIDER_INIT_POS,
    TABLE_COLLIDER_SIZE,
    TABLE_INIT_POS,
    TABLE_TOP_Z,
    compute_attached_object_pose,
    create_support_table,
    estimate_gripper_opening_range,
    estimate_object_grasp_width,
    evaluate_grasp_preflight,
    expand_mode_selection,
    expand_planner_selection,
    franka_hand_opening_from_finger_qpos,
    grasp_axis_alignment_cost,
    horizontal_bbox_axis,
    initialize_pre_pick_robot_pose,
    load_franka_hand_opening_spec,
    empty_replay_diagnostics,
    make_leaderboard_rows,
    make_failed_trial_row,
    make_skipped_rows,
    physical_gpu_memory_skip_reason,
    _record_contact_stats,
    object_body_scale,
    object_initial_position,
    object_supported_z,
    parse_args,
    rerank_grasp_costs_by_axis,
    resolve_object_body_scale,
    resolve_hand_open_close_qpos,
    simulation_requires_cuda,
    write_markdown_report,
)


class _FakeSim:
    def __init__(self):
        self.cfg = None

    def add_rigid_object(self, cfg):
        self.cfg = cfg
        return cfg


class _FakeObject:
    def __init__(self):
        self.live_pose = torch.eye(4).unsqueeze(0)
        self.live_pose[0, :3, 3] = torch.tensor([0.2, 0.1, -0.3])

    def get_local_pose(self, to_matrix: bool = False):
        assert to_matrix
        return self.live_pose.clone()


class _FakeRobotForPrePick:
    device = torch.device("cpu")

    def __init__(self):
        self.ik_pose = None
        self.qpos_set = []

    def get_qpos(self, name=None):
        if name == "main_arm":
            return torch.zeros(1, 7)
        return torch.zeros(1, 9)

    def compute_fk(self, qpos, name, to_matrix):
        pose = torch.eye(4).unsqueeze(0)
        return pose

    def compute_ik(self, pose, joint_seed, name):
        self.ik_pose = pose.clone()
        return torch.tensor([True]), torch.ones(1, 7)

    def set_qpos(self, qpos, name=None, target=False):
        self.qpos_set.append((name, target, qpos.clone()))

    def clear_dynamics(self):
        pass


def _rows() -> list[dict[str, object]]:
    return [
        {
            "planner": "ik_interpolate",
            "mode": "planner",
            "warmup": True,
            "action_success": False,
            "planner_success": False,
            "physical_success": None,
            "replay_success": None,
            "preflight_failed": False,
            "preflight_status": "ok",
            "planning_time_sec": 10.0,
            "cpu_delta_mb": 0.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "final_tcp_pos_error": 1.0,
            "final_tcp_rot_error": 1.0,
            "joint_path_length": 10.0,
            "max_joint_step": 1.0,
            "object_grasp_width_m": 0.05,
            "gripper_min_opening_m": 0.0,
            "gripper_max_opening_m": 0.08,
            **empty_replay_diagnostics(),
        },
        {
            "planner": "ik_interpolate",
            "mode": "planner",
            "warmup": False,
            "action_success": True,
            "planner_success": True,
            "physical_success": None,
            "replay_success": None,
            "preflight_failed": False,
            "preflight_status": "ok",
            "planning_time_sec": 1.0,
            "cpu_delta_mb": 1.0,
            "gpu_delta_mb": 0.0,
            "peak_gpu_mb": 0.0,
            "final_tcp_pos_error": 0.001,
            "final_tcp_rot_error": 0.01,
            "joint_path_length": 0.5,
            "max_joint_step": 0.1,
            "object_grasp_width_m": 0.05,
            "gripper_min_opening_m": 0.0,
            "gripper_max_opening_m": 0.08,
            **empty_replay_diagnostics(),
        },
        {
            "planner": "neural",
            "mode": "planner",
            "warmup": False,
            "action_success": False,
            "planner_success": False,
            "physical_success": None,
            "replay_success": None,
            "preflight_failed": True,
            "preflight_status": "too_small",
            "planning_time_sec": 0.2,
            "cpu_delta_mb": 0.5,
            "gpu_delta_mb": 2.0,
            "peak_gpu_mb": 10.0,
            "final_tcp_pos_error": 0.04,
            "final_tcp_rot_error": 0.2,
            "joint_path_length": 0.7,
            "max_joint_step": 0.2,
            "object_grasp_width_m": 0.02,
            "gripper_min_opening_m": 0.08,
            "gripper_max_opening_m": 0.16,
            **empty_replay_diagnostics(),
        },
        {
            "planner": "neural_refine",
            "mode": "physical",
            "warmup": False,
            "action_success": True,
            "planner_success": True,
            "physical_success": True,
            "replay_success": True,
            "preflight_failed": False,
            "preflight_status": "ok",
            "planning_time_sec": 0.3,
            "cpu_delta_mb": 0.5,
            "gpu_delta_mb": 2.0,
            "peak_gpu_mb": 10.0,
            "final_tcp_pos_error": 0.002,
            "final_tcp_rot_error": 0.02,
            "joint_path_length": 0.8,
            "max_joint_step": 0.2,
            "object_grasp_width_m": 0.05,
            "gripper_min_opening_m": 0.0,
            "gripper_max_opening_m": 0.08,
            "actual_min_hand_opening_m": 0.03,
            "actual_close_hand_opening_m": 0.031,
            "actual_final_hand_opening_m": 0.08,
            "close_contact_count": 2,
            "path_contact_count": 0,
            "hold_contact_count": 0,
            "max_contact_count": 2,
            "max_contact_impulse": 0.4,
            "max_total_contact_impulse": 0.7,
            "min_contact_distance_m": -0.001,
            "close_tcp_object_delta_m": [0.0, 0.0, -0.02],
            "planned_grasp_object_delta_m": [0.0, 0.0, -0.02],
            "close_leftfinger_object_delta_m": [0.0, 0.02, 0.0],
            "close_rightfinger_object_delta_m": [0.0, -0.02, 0.0],
            "close_tcp_x_axis": [1.0, 0.0, 0.0],
            "close_tcp_y_axis": [0.0, 1.0, 0.0],
            "close_tcp_z_axis": [0.0, 0.0, 1.0],
        },
    ]


def test_cli_defaults_and_selection_expansion():
    args = parse_args([])

    assert args.mode == "planner"
    assert args.object == "sugar_box"
    assert args.planner == "all"
    assert args.headless is True
    assert args.object_replay_mode == "physics"
    assert args.object_scale is None
    assert args.grasp_axis == DEFAULT_GRASP_AXIS
    assert args.tcp_z == pytest.approx(DEFAULT_FRANKA_TCP_Z)
    assert args.tcp_yaw == pytest.approx(DEFAULT_FRANKA_TCP_YAW)
    assert args.grasp_depth_offset == pytest.approx(DEFAULT_GRASP_DEPTH_OFFSET)
    assert args.n_deviated_approach_directions == 1
    assert args.gripper_close_margin == pytest.approx(DEFAULT_GRIPPER_CLOSE_MARGIN)
    assert expand_planner_selection("all") == [
        "ik_interpolate",
        "neural",
        "neural_refine",
    ]
    assert expand_planner_selection("neural") == ["neural"]
    assert expand_mode_selection("both") == ["planner", "physical"]


def test_cli_accepts_object_scale_and_short_axis_grasp():
    args = parse_args(
        [
            "--object_scale",
            "0.9",
            "0.9",
            "0.9",
            "--grasp_axis",
            "short",
        ]
    )

    assert args.object_scale == pytest.approx([0.9, 0.9, 0.9])
    assert args.grasp_axis == "short"


def test_object_body_scale_does_not_downscale_cup_by_default():
    assert object_body_scale("cup") == pytest.approx((1.0, 1.0, 1.0))


def test_resolve_object_body_scale_uses_cli_override():
    assert resolve_object_body_scale("sugar_box", [0.9, 0.8, 1.1]) == pytest.approx(
        (0.9, 0.8, 1.1)
    )


def test_cube_initial_position_places_bottom_above_support_plane():
    x, y, z = object_initial_position("cube")

    assert (x, y) == pytest.approx((0.31, 0.0))
    assert z == pytest.approx(TABLE_TOP_Z + 0.045 / 2.0 + OBJECT_SUPPORT_MARGIN)


def test_object_supported_z_respects_scale_override():
    z = object_supported_z("cube", [1.0, 1.0, 2.0])

    assert z == pytest.approx(TABLE_TOP_Z + 0.045 + OBJECT_SUPPORT_MARGIN)


def test_sugar_box_scale_uses_preset_value_by_default():
    assert object_body_scale("sugar_box") == pytest.approx((1.0, 1.0, 1.0))


def test_support_table_is_static_and_below_interaction_plane():
    sim = _FakeSim()

    cfg = create_support_table(sim)

    assert cfg.body_type == "static"
    assert cfg.uid == "support_table_collider"
    assert cfg.shape.size == list(TABLE_COLLIDER_SIZE)
    assert cfg.init_pos == TABLE_COLLIDER_INIT_POS
    assert TABLE_TOP_Z == pytest.approx(0.0)


def test_sugar_box_initial_position_places_bottom_above_support_plane():
    x, y, z = object_initial_position("sugar_box")

    assert (x, y) == pytest.approx((0.31, 0.0))
    assert z > TABLE_TOP_Z
    assert z == pytest.approx(object_supported_z("sugar_box"), abs=1e-6)
    assert z - TABLE_TOP_Z > OBJECT_SUPPORT_MARGIN


def test_resolve_hand_open_close_qpos_clamps_to_joint_limits():
    limits = torch.tensor([[0.0, 0.04], [0.0, 0.04]], dtype=torch.float32)

    hand_open, hand_close = resolve_hand_open_close_qpos(
        limits,
        open_qpos=0.08,
        close_qpos=-0.01,
    )

    assert hand_open.tolist() == pytest.approx([0.04, 0.04])
    assert hand_close.tolist() == pytest.approx([0.0, 0.0])


def test_estimate_franka_opening_range_uses_collision_surfaces():
    limits = torch.tensor([[0.0, 0.04], [0.0, 0.04]], dtype=torch.float32)
    expected_min_opening = 0.08 - 2.0 * 0.0005448426818475127

    min_opening, max_opening = estimate_gripper_opening_range(
        get_data_path("Franka/Panda/PandaWithHand.urdf"),
        limits,
    )

    assert min_opening == pytest.approx(expected_min_opening)
    assert max_opening == pytest.approx(expected_min_opening + 0.08)


def test_franka_hand_opening_from_finger_qpos_reports_physical_jaw_gap():
    spec = load_franka_hand_opening_spec(
        get_data_path("Franka/Panda/PandaWithHand.urdf")
    )
    finger_qpos = torch.tensor([[0.0, 0.0], [0.04, 0.04]], dtype=torch.float32)
    expected_min_opening = 0.08 - 2.0 * 0.0005448426818475127

    opening = franka_hand_opening_from_finger_qpos(finger_qpos, spec)

    assert opening.tolist() == pytest.approx(
        [expected_min_opening, expected_min_opening + 0.08]
    )


def test_estimate_object_grasp_width_uses_narrowest_horizontal_extent():
    vertices = torch.tensor(
        [
            [-0.1, -0.03, 0.0],
            [0.1, 0.04, 0.0],
            [0.0, 0.02, 0.1],
        ],
        dtype=torch.float32,
    )

    assert estimate_object_grasp_width(vertices) == pytest.approx(0.07)


def test_horizontal_bbox_axis_selects_short_and_long_axes():
    vertices = torch.tensor(
        [
            [-0.1, -0.03, 0.0],
            [0.1, 0.03, 0.0],
            [0.0, 0.03, 0.1],
        ],
        dtype=torch.float32,
    )

    assert horizontal_bbox_axis(vertices, "short").tolist() == pytest.approx(
        [0.0, 1.0, 0.0]
    )
    assert horizontal_bbox_axis(vertices, "long").tolist() == pytest.approx(
        [1.0, 0.0, 0.0]
    )


def test_grasp_axis_rerank_supports_explicit_generator_x_axis():
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    poses[0, :3, 0] = torch.tensor([1.0, 0.0, 0.0])
    poses[1, :3, 0] = torch.tensor([0.0, 1.0, 0.0])
    costs = torch.zeros(2, dtype=torch.float32)

    reranked = rerank_grasp_costs_by_axis(
        costs,
        poses,
        torch.tensor([1.0, 0.0, 0.0]),
        axis_index=0,
    )

    assert grasp_axis_alignment_cost(
        poses[:, :3, 0],
        torch.tensor([1.0, 0.0, 0.0]),
    ).tolist() == pytest.approx([0.0, 1.0])
    assert reranked[0] < reranked[1]


def test_grasp_axis_rerank_defaults_to_franka_closing_axis():
    poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    poses[0, :3, 0] = torch.tensor([1.0, 0.0, 0.0])
    poses[0, :3, 1] = torch.tensor([0.0, 1.0, 0.0])
    poses[1, :3, 0] = torch.tensor([0.0, 1.0, 0.0])
    poses[1, :3, 1] = torch.tensor([1.0, 0.0, 0.0])
    costs = torch.zeros(2, dtype=torch.float32)

    reranked = rerank_grasp_costs_by_axis(
        costs,
        poses,
        torch.tensor([0.0, 1.0, 0.0]),
    )

    assert FRANKA_GRIPPER_CLOSING_AXIS_INDEX == 1
    assert reranked[0] < reranked[1]


def test_compute_attached_object_pose_inverts_object_to_eef_transform():
    eef_pose = torch.eye(4)
    eef_pose[:3, 3] = torch.tensor([0.4, -0.2, 0.3])
    object_to_eef = torch.eye(4)
    object_to_eef[:3, 3] = torch.tensor([0.1, 0.0, 0.02])

    object_pose = compute_attached_object_pose(eef_pose, object_to_eef)

    assert object_pose[:3, 3].tolist() == pytest.approx([0.3, -0.2, 0.28])


def test_contact_stats_aggregation_tracks_impulse_and_distance_extrema():
    max_count, max_impulse, max_total_impulse, min_distance = _record_contact_stats(
        ContactStats(count=2, max_impulse=0.3, total_impulse=0.5, min_distance=-0.001),
        max_count=0,
        max_impulse=None,
        max_total_impulse=None,
        min_distance=None,
    )

    max_count, max_impulse, max_total_impulse, min_distance = _record_contact_stats(
        ContactStats(count=1, max_impulse=0.2, total_impulse=0.7, min_distance=0.002),
        max_count=max_count,
        max_impulse=max_impulse,
        max_total_impulse=max_total_impulse,
        min_distance=min_distance,
    )

    assert max_count == 2
    assert max_impulse == pytest.approx(0.3)
    assert max_total_impulse == pytest.approx(0.7)
    assert min_distance == pytest.approx(-0.001)


def test_evaluate_grasp_preflight_marks_too_small_object():
    preflight = evaluate_grasp_preflight(
        object_grasp_width_m=0.03,
        gripper_min_opening_m=0.08,
        gripper_max_opening_m=0.16,
    )

    assert preflight.failed
    assert preflight.status == "too_small"
    assert "narrower" in preflight.reason


def test_adaptive_franka_close_clamps_objects_below_minimum_opening():
    from scripts.benchmark.robotics.nmg.franka_pick_place import (
        resolve_adaptive_hand_close_qpos,
    )

    limits = torch.tensor([[0.0, 0.04], [0.0, 0.04]], dtype=torch.float32)

    hand_close = resolve_adaptive_hand_close_qpos(
        limits,
        object_width_m=0.045,
        margin_m=0.0,
        franka_urdf_path=get_data_path("Franka/Panda/PandaWithHand.urdf"),
    )

    assert hand_close.tolist() == pytest.approx([0.0, 0.0])


def test_simulation_cuda_preflight_and_skipped_rows():
    args = parse_args(["--device", "cpu", "--renderer", "fast-rt"])

    assert simulation_requires_cuda(args)
    rows = make_skipped_rows(
        ["ik_interpolate", "neural"],
        ["planner"],
        object_name="sugar_box",
        reason="cuda missing",
    )

    assert len(rows) == 2
    assert {row["planner"] for row in rows} == {"ik_interpolate", "neural"}
    assert all(row["preflight_status"] == "skipped" for row in rows)
    assert all(row["preflight_failed"] for row in rows)


def test_initialize_pre_pick_uses_reference_pose_before_live_physics_drift():
    robot = _FakeRobotForPrePick()
    obj = _FakeObject()
    reference_pose = torch.eye(4)
    reference_pose[:3, 3] = torch.tensor([0.31, 0.0, 0.025])

    initialize_pre_pick_robot_pose(
        robot,
        obj,
        torch.tensor([0.04, 0.04]),
        pre_pick_z=0.36,
        reference_pose=reference_pose,
    )

    assert robot.ik_pose is not None
    assert robot.ik_pose[0, :3, 3].tolist() == pytest.approx([0.31, 0.0, 0.36])


def test_physical_gpu_memory_preflight_skips_before_dexsim_oom(monkeypatch):
    from scripts.benchmark.robotics.nmg import franka_pick_place

    monkeypatch.setattr(franka_pick_place, "free_gpu_memory_mb", lambda gpu_id: 512.0)
    args = parse_args(["--mode", "physical", "--min_physical_gpu_free_mb", "1024"])

    reason = physical_gpu_memory_skip_reason(args, ["physical"])

    assert reason is not None
    assert "free GPU memory" in reason


def test_failed_trial_row_records_setup_failure():
    row = make_failed_trial_row(
        planner="ik_interpolate",
        mode="planner",
        object_name="sugar_box",
        repeat_id=0,
        warmup=False,
        reason="pre-pick ik failed",
    )

    assert row["planner_success"] is False
    assert row["preflight_status"] == "setup_failed"
    assert row["preflight_failed"] is True
    assert row["preflight_reason"] == "pre-pick ik failed"


def test_write_markdown_report_has_exactly_three_tables(tmp_path):
    report_path = tmp_path / "report.md"

    written = write_markdown_report(_rows(), str(report_path))

    text = written.read_text(encoding="utf-8")
    assert written == report_path
    assert text.count("## Time & Memory") == 1
    assert text.count("## Success & Other Metrics") == 1
    assert text.count("## Leaderboard") == 1
    assert text.count("\n| ---") == 3
    assert "## Notes" not in text


def test_leaderboard_includes_all_planner_mode_pairs():
    rows = make_leaderboard_rows(_rows())
    pairs = {(row["planner"], row["mode"]) for row in rows}

    assert pairs == {
        ("ik_interpolate", "planner"),
        ("neural", "planner"),
        ("neural_refine", "physical"),
    }
    assert rows[0]["overall_success_rate"] == "100.0%"


def test_benchmark_source_does_not_contain_invalid_hacks():
    source_path = Path("scripts/benchmark/robotics/nmg/franka_pick_place.py")
    text = source_path.read_text(encoding="utf-8")

    forbidden = [
        "semantic_held_object",
        "patched_finger_origins",
        "prepare_franka_grasp_urdf",
        "cup_scale",
        "set_local_pose(torch.bmm",
    ]
    for pattern in forbidden:
        assert pattern not in text
