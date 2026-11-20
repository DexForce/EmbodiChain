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

"""Frozen task projection and independent, physics-free reaching acceptance."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.planners.utils import PlanResult
from embodichain.extensions.benchmarks.motion_generation.artifacts import (
    TrialJsonlWriter,
    _case_to_dict,
)
from embodichain.extensions.benchmarks.motion_generation.config import (
    PlannerSpecCfg,
    SuiteCfg,
    TrackCfg,
)
from embodichain.extensions.benchmarks.motion_generation.models import (
    BenchmarkCase,
    TrialPhase,
)
from embodichain.extensions.benchmarks.motion_generation.planners.base import (
    PlannerAdapter,
)
from embodichain.extensions.benchmarks.motion_generation.runner import BenchmarkRunner
from embodichain.extensions.benchmarks.motion_generation.scenarios.affordance_reaching import (
    AffordanceReachingScenario,
    _project_case,
    _task_targets,
)


class CartesianRobot:
    device = torch.device("cpu")

    def compute_fk(self, qpos, name, to_matrix, env_ids=None):
        poses = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        poses[..., :3, 3] = qpos
        return poses

    compute_batch_fk = compute_fk

    def compute_ik(self, pose, joint_seed, name, env_ids=None):
        return torch.ones(pose.shape[0], dtype=torch.bool), pose[:, :3, 3].clone()

    def get_qpos_limits(self, name, env_ids=None):
        return torch.tensor([[[-2.0, 2.0]] * 3])

    def get_qvel_limits(self, name):
        return torch.full((1, 3), 2.0)


def source_case():
    robot = CartesianRobot()
    references = torch.tensor([[[0.2, 0.0, 0.16]]])
    return BenchmarkCase(
        suite_version="old_atomic",
        track="skills",
        scenario_id="move_held_object",
        case_id="transport",
        seed=43001,
        batch_size=1,
        num_waypoints=1,
        path_shape="pick_up_then_held_object_transport",
        start_state_bin="pre_pick",
        start_qpos=torch.tensor([[0.0, 0.0, 0.3]]),
        target_waypoints=robot.compute_fk(references, "arm", True),
        reference_qpos=references,
        skill_id="move_held_object",
        primary_success="physical_skill_success",
        case_parameters={
            "grasp_pose": torch.eye(4)[None].tolist(),
            "approach_direction": [0, 0, -1],
            "pre_grasp_distance_m": 0.15,
            "lift_height_m": 0.16,
        },
    )


def reaching_track():
    return TrackCfg(
        id="reaching",
        scenario="affordance_reaching",
        config={
            "source_manifest": "frozen.json",
            "source_cases_sha256": "a" * 64,
            "source_track": "skills",
            "batch_size": 1,
        },
    )


def test_transport_projection_restores_full_task_without_mutating_atomic():
    source = source_case()
    original = deepcopy(_case_to_dict(source))
    suite = SuiteCfg(suite_version="reaching_v1")
    result = _project_case(source, suite, reaching_track(), CartesianRobot(), "arm")
    expected = torch.tensor(
        [[[0.0, 0.0, 0.15], [0.0, 0.0, 0.0], [0.0, 0.0, 0.16], [0.2, 0.0, 0.16]]]
    )
    torch.testing.assert_close(result.target_waypoints[..., :3, 3], expected)
    assert result.num_waypoints == 4
    assert result.primary_success == "geometry_valid"
    assert result.case_parameters["affordance_stage_names"] == [
        "pre_grasp",
        "grasp",
        "lift",
        "transport",
    ]
    assert _case_to_dict(source) == original
    assert (
        result.case_id != source.case_id
        and result.suite_version != source.suite_version
    )


@pytest.mark.parametrize(
    "skill,count", [("pick_up", 3), ("pour", 5), ("axis_align", 4)]
)
def test_existing_full_sequences_are_not_prefixed_twice(skill, count):
    source = source_case()
    targets = torch.eye(4).expand(1, count, 4, 4).clone()
    targets[0, :, 0, 3] = torch.arange(count) * 0.1
    source = replace(
        source, skill_id=skill, target_waypoints=targets, num_waypoints=count
    )
    projected, stages, added = _task_targets(source)
    torch.testing.assert_close(projected, targets, rtol=0, atol=0)
    assert added == 0 and len(stages) == count


@pytest.mark.parametrize(
    "path,planner_success,expected",
    [
        ([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], True, True),
        ([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0]], True, False),
        ([[0.0, 0.0, 0.0], [-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], True, False),
        ([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]], False, True),
    ],
)
def test_external_reaching_requires_all_ordered_goals_not_planner_flag(
    path, planner_success, expected
):
    robot, scenario = CartesianRobot(), AffordanceReachingScenario()
    refs = torch.tensor([[[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]])
    case = replace(
        source_case(),
        start_qpos=torch.zeros(1, 3),
        num_waypoints=2,
        target_waypoints=robot.compute_fk(refs, "arm", True),
        reference_qpos=refs,
        primary_success="geometry_valid",
    )
    result = PlanResult(
        success=torch.tensor([planner_success]),
        positions=torch.tensor([path]),
        dt=torch.full((1, len(path)), 0.01),
    )
    outcome = scenario.evaluate_case(
        result,
        case,
        robot,
        "arm",
        SuiteCfg(),
        planning_time_ms=0,
        timing_is_solved=False,
    ).outcomes[0]
    assert outcome.geometry_valid is expected
    assert outcome.planning_success is planner_success
    assert outcome.execution_success is None and outcome.physical_skill_success is None
    assert outcome.timed_motion_valid is None


def test_scenario_never_calls_physics_objects_or_hand_actions():
    # Any physical API access would fail: only the kinematic reset surface exists.
    calls = []
    robot = SimpleNamespace(
        set_qpos=lambda *args, **kw: calls.append(kw), clear_dynamics=lambda: None
    )
    simulation = SimpleNamespace(sync_render_state=lambda: None)
    scenario, case = AffordanceReachingScenario(), source_case()
    scenario.create_runtime_entities(simulation, SuiteCfg(), reaching_track())
    scenario.reset_case(simulation, robot, case, "arm")
    sentinel = object()
    assert (
        scenario.plan_case(SimpleNamespace(plan=lambda c: sentinel), case) is sentinel
    )
    assert calls == [{"name": "arm", "target": False}, {"name": "arm", "target": True}]


def test_frozen_source_hash_is_checked_before_projection(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"cases_sha256": "b" * 64, "cases": []}))
    track = reaching_track()
    track.config["source_manifest"] = str(path)
    with pytest.raises(ValueError, match="hash mismatch"):
        AffordanceReachingScenario().generate_cases(
            SuiteCfg(), track, CartesianRobot(), "arm", 1
        )


def test_reaching_config_rejects_physical_knobs():
    track = reaching_track()
    track.config["physics"] = {"relaxed": True}
    with pytest.raises(ValueError, match="exactly"):
        AffordanceReachingScenario().batch_sizes(SuiteCfg(), track)


@pytest.mark.parametrize("missing_capability", ["batched", "empty_world", None])
def test_reaching_runner_checks_scenario_capabilities(tmp_path, missing_capability):
    class CapabilityProbe(PlannerAdapter):
        capabilities = frozenset({"eef_waypoint", "batched", "empty_world"}) - {
            missing_capability
        }

        def availability(self) -> tuple[bool, str | None]:
            return False, "stop after capability checks"

        def build(self) -> None:
            pytest.fail("capability probe must not build a planner")

        def plan(self, case: BenchmarkCase) -> PlanResult:
            pytest.fail("capability probe must not plan")

    suite, provider = SuiteCfg(), AffordanceReachingScenario()
    case = _project_case(
        source_case(), suite, reaching_track(), CartesianRobot(), "arm"
    )
    batch_size = 8
    case = replace(
        case,
        batch_size=batch_size,
        start_qpos=case.start_qpos.expand(batch_size, -1).clone(),
        target_waypoints=case.target_waypoints.expand(batch_size, -1, -1, -1).clone(),
        reference_qpos=case.reference_qpos.expand(batch_size, -1, -1).clone(),
    )
    spec = PlannerSpecCfg(id="probe", adapter="capability_probe", enabled=True)
    runner = BenchmarkRunner(
        suite,
        [spec],
        device="cpu",
        output_root=tmp_path,
        adapters={"capability_probe": CapabilityProbe},
    )
    runner._run_adapter(
        TrialJsonlWriter(tmp_path / "trials.jsonl"),
        SimpleNamespace(),
        CartesianRobot(),
        spec,
        [case],
        provider.required_capabilities,
        provider,
    )
    assert len(runner.records) == 1
    record = runner.records[0]
    assert record.phase is TrialPhase.AVAILABILITY
    assert record.failure_code == (
        "unsupported_capability" if missing_capability else "runtime_unavailable"
    )
    if missing_capability:
        assert missing_capability in record.failure_message
