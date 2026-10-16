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

"""Physical outcomes must differ from planning and command completion."""

from __future__ import annotations

import math

import pytest
import torch

from scripts.benchmark.atomic_action.evaluation import evaluate_skill
from scripts.benchmark.atomic_action.reference_benchmark import SCENES

pytestmark = pytest.mark.no_sim


def _rotation(axis: str, angle: float) -> torch.Tensor:
    """Construct an exact test rotation without importing simulation math."""
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        return torch.tensor([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    if axis == "y":
        return torch.tensor([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]])
    return torch.tensor([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def _example(skill: str) -> tuple[dict, list[dict], dict]:
    """Build measured traces with known stage outcomes for each skill."""
    trace = {"qpos": torch.zeros(13, 1, 4)}
    pose = torch.eye(4).repeat(13, 1, 1, 1)
    pose[:, :, 2, 3] = 0.05
    trace.update(
        arm=pose.clone(),
        object=pose.clone(),
        link=pose.clone(),
        joint=torch.zeros(13, 1),
    )
    metadata = {
        "hand_joint_ids": {"eef": [2, 3]},
        "hand_open": {"eef": [0.0, 0.0]},
        "hand_close": {"eef": [0.02, 0.02]},
        "arm_joint_ids": [0, 1],
        "support_z": 0.0,
    }
    phases = []

    def phase(action: str, name: str, start: int, stop: int) -> None:
        phases.append({"skill": action, "name": name, "start": start, "stop": stop})

    trace["qpos"][2:, :, 2:] = 0.02
    if skill in {"pickup", "place", "pour", "axis_align"}:
        action = "axis_align" if skill == "axis_align" else "pick_up"
        phase(action, "close", 0, 2)
        phase(action, "manipulate" if skill == "axis_align" else "lift", 2, 4)
        for name in ("arm", "object"):
            trace[name][4:, :, 2, 3] += 0.10
        if skill == "axis_align":
            metadata.update(internal_axis=[1, 0, 0], target_axis=[0, 0, 1])
            for name in ("arm", "object"):
                trace[name][4:, :, :3, :3] = _rotation("y", -math.pi / 2)
        if skill == "place":
            phase("place", "approach", 4, 6)
            phase("place", "release", 6, 8)
            phase("place", "retract", 8, 10)
            for name in ("arm", "object"):
                trace[name][6:, :, 0, 3] = 0.2
                trace[name][6:, :, 1, 3] = 0.3
            trace["object"][8:, :, 2, 3] = 0.05
            trace["qpos"][8:, :, 2:] = 0
            metadata["object_target_pose"] = trace["object"][-1].tolist()
        if skill == "pour":
            phase("pour", "pour", 4, 8)
            metadata.update(internal_axis=[1, 0, 0], pour_angle=math.pi / 4)
            for name in ("arm", "object"):
                trace[name][6, :, :3, :3] = _rotation("x", math.pi / 4)
    elif skill in {"slide", "twist", "open_door"}:
        phase(skill, "close", 0, 2)
        phase(
            skill, {"slide": "pull", "twist": "twist", "open_door": "open"}[skill], 2, 6
        )
        phase(skill, "release" if skill == "open_door" else "open", 6, 8)
        trace["qpos"][8:, :, 2:] = 0
        if skill == "twist":
            metadata.update(axis_world=[0, 1, 0], twist_angle=-math.pi / 4)
            trace["arm"][6:, :, :3, :3] = _rotation("y", -math.pi / 4)
        else:
            metadata["joint_target"] = 0.18 if skill == "slide" else math.pi / 3
            trace["joint"][6] = metadata["joint_target"]
    elif skill == "press":
        metadata.update(joint_target=0.006, press_distance=0.03)
        trace["joint"][5] = 0.0055  # Returns to zero after pressure is released.
    elif skill == "move_joint":
        phase("move_joints", "move_joints", 0, 4)
        phase("move_joints", "move_joints", 4, 8)
        trace["qpos"][4, :, :2] = 0.3
        phases[0]["target_qpos"] = [[0.3, 0.3, 0, 0]]
        phases[1]["target_qpos"] = [[0, 0, 0, 0]]
    elif skill == "handover":
        metadata.update(
            hand_joint_ids={"left_eef": [2], "right_eef": [3]},
            hand_open={"left_eef": [0.0], "right_eef": [0.0]},
            hand_close={"left_eef": [0.02], "right_eef": [0.02]},
            target_position=[0, 0, 0.15],
        )
        trace["left_arm"] = trace["arm"].clone()
        trace["right_arm"] = trace["arm"].clone()
        for name, start, stop in (
            ("pickup_close", 0, 2),
            ("pickup_transport", 2, 4),
            ("receive_close", 4, 6),
            ("handover_release", 6, 8),
            ("place", 8, 10),
        ):
            phase("hand_over", name, start, stop)
        for name in ("object", "left_arm", "right_arm"):
            trace[name][4:, :, 2, 3] += 0.1
        trace["qpos"][8:, :, 2] = 0
        trace["qpos"][11:, :, 3] = 0
    return trace, phases, metadata


@pytest.mark.parametrize("skill", SCENES)
def test_skill_specific_physical_stages_accept_actual_success(skill: str) -> None:
    trace, phases, metadata = _example(skill)
    result = evaluate_skill(skill, trace, phases, metadata, [True])[0]
    assert result["success"], result
    assert result["stages"]


@pytest.mark.parametrize("skill", SCENES)
def test_planning_success_does_not_credit_stationary_scene(skill: str) -> None:
    trace, phases, metadata = _example(skill)
    for name in trace:
        trace[name][:] = trace[name][0].clone()
    result = evaluate_skill(skill, trace, phases, metadata, [True])[0]
    assert result["planning_success"]
    assert not result["success"]
    assert result["failure_stage"]


@pytest.mark.parametrize("skill", SCENES)
def test_all_failed_plans_keep_failure_holds_without_requiring_success_phases(
    skill: str,
) -> None:
    result = evaluate_skill(
        skill, {"qpos": torch.zeros(4, 2, 8)}, [], {}, [False, False]
    )
    assert len(result) == 2
    assert all(row["failure_stage"] == "planning" for row in result)
    assert all(row["failure_reason"] == "planner_reported_failure" for row in result)


def test_mixed_planning_failure_is_row_local() -> None:
    trace, phases, metadata = _example("pickup")
    trace = {
        key: value.repeat(1, 2, *([1] * (value.ndim - 2)))
        for key, value in trace.items()
    }
    results = evaluate_skill("pickup", trace, phases, metadata, [False, True])
    assert results[0]["failure_stage"] == "planning"
    assert results[1]["success"]


def test_pickup_dropped_after_lift_fails_closing_stage() -> None:
    trace, phases, metadata = _example("pickup")
    trace["object"][-1, :, 2, 3] = 0.05
    result = evaluate_skill("pickup", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "held"
    assert result["failure_reason"] == "object_dropped"


@pytest.mark.parametrize("axis,angle", [("x", -math.pi / 4), ("y", math.pi / 4)])
def test_pour_requires_signed_rotation_about_object_axis(
    axis: str, angle: float
) -> None:
    trace, phases, metadata = _example("pour")
    for name in ("object", "arm"):
        trace[name][6, :, :3, :3] = _rotation(axis, angle)
    result = evaluate_skill("pour", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "poured"


def test_press_uses_peak_actual_stroke_and_not_command_distance() -> None:
    trace, phases, metadata = _example("press")
    assert evaluate_skill("press", trace, phases, metadata, [True])[0]["success"]
    trace["joint"][5] = 0.003
    result = evaluate_skill("press", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "pressed"
    assert result["metrics"]["stroke_ratio"] == pytest.approx(0.5)


def test_place_must_stabilize_after_release() -> None:
    trace, phases, metadata = _example("place")
    trace["object"][-2, :, 2, 3] += 0.02
    result = evaluate_skill("place", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "stable"


@pytest.mark.parametrize("offset,success", [(0.014, True), (0.016, False)])
def test_place_accepts_small_landing_offsets_within_fifteen_mm(
    offset: float, success: bool
) -> None:
    trace, phases, metadata = _example("place")
    trace["object"][-2:, :, 0, 3] += offset
    result = evaluate_skill("place", trace, phases, metadata, [True])[0]
    assert result["success"] == success
    assert result["metrics"]["placement_xy_error_m"] == pytest.approx(offset)
    assert result["metrics"]["placement_xy_tolerance_m"] == pytest.approx(0.015)
    assert result["failure_stage"] == (None if success else "placed")


def test_place_landing_allowance_does_not_relax_settling_drift() -> None:
    trace, phases, metadata = _example("place")
    trace["object"][-2, :, 0, 3] -= 0.012
    result = evaluate_skill("place", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "stable"


def test_handover_fails_if_object_falls_after_source_releases() -> None:
    trace, phases, metadata = _example("handover")
    trace["object"][8, :, 2, 3] = 0.05
    result = evaluate_skill("handover", trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "handed_over"


@pytest.mark.parametrize("skill", ["pickup", "place", "pour", "axis_align", "handover"])
def test_relative_grasp_pose_changes_do_not_fail_successful_skills(skill: str) -> None:
    trace, phases, metadata = _example(skill)
    if skill == "handover":
        trace["object"][8, :, 0, 3] += 0.02
        trace["object"][8, :, :3, :3] = _rotation("z", math.pi / 3)
    elif skill == "pour":
        trace["object"][8, :, :3, :3] = _rotation("z", math.pi / 3)
    else:
        trace["object"][4, :, 0, 3] += 0.02
        if skill != "axis_align":
            trace["object"][4, :, :3, :3] = _rotation("z", math.pi / 3)
    assert evaluate_skill(skill, trace, phases, metadata, [True])[0]["success"]


@pytest.mark.parametrize(
    "closed_command", [0.036, 0.5], ids=("pgi-metres", "robotiq-radians")
)
def test_release_threshold_uses_configured_gripper_stroke(
    closed_command: float,
) -> None:
    trace, phases, metadata = _example("handover")
    trace["qpos"][:, :, 2:] *= closed_command / 0.02
    metadata["hand_close"] = {
        "left_eef": [closed_command],
        "right_eef": [closed_command],
    }
    trace["qpos"][8:, :, 2] = 0.03 * closed_command
    assert evaluate_skill("handover", trace, phases, metadata, [True])[0]["success"]
    trace["qpos"][8:, :, 2] = 0.20 * closed_command
    assert (
        evaluate_skill("handover", trace, phases, metadata, [True])[0]["failure_stage"]
        == "handed_over"
    )


@pytest.mark.parametrize("skill", ["place", "pour"])
def test_prerequisite_grasp_failure_is_reported_separately(skill: str) -> None:
    trace, phases, metadata = _example(skill)
    trace["qpos"][2, :, 2:] = 0
    result = evaluate_skill(skill, trace, phases, metadata, [True])[0]
    assert result["failure_stage"] == "prerequisite_pickup"
