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

"""Tests for spatial domain sampling, feasibility and generalization summaries."""

from __future__ import annotations

import math
from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from scripts.benchmark.motion_generation import scenarios as _scenarios  # noqa: F401
from scripts.benchmark.motion_generation.atomic_skill import (
    AtomicSkillCaseProvider,
    InfeasibleCaseError,
)
from scripts.benchmark.motion_generation.config import (
    DomainCfg,
    TrackCfg,
    load_suite,
)
from scripts.benchmark.motion_generation.contracts import EvaluationDomain
from scripts.benchmark.motion_generation.generalization import (
    aggregate_generalization,
    spatial_breakdown,
)
from scripts.benchmark.motion_generation.models import (
    AlgorithmRole,
    BenchmarkCase,
    CaseOutcome,
    PlannerMetadata,
    TrialPhase,
    TrialRecord,
)
from scripts.benchmark.motion_generation.scenarios import atomic_task
from scripts.benchmark.motion_generation.scenarios.atomic_task import (
    AtomicTaskScenario,
    create_atomic_skill_provider,
)
from scripts.benchmark.motion_generation.spatial import (
    OBJECT_POSITION,
    ROBOT_START_OFFSET,
    TARGET_OFFSET,
    SpatialParameter,
    parse_spatial_domain,
    resolve_parameter_sizes,
    sample_spatial_parameters,
)

CPU = torch.device("cpu")
_FAR = {"radius_m": [0.55, 0.65], "azimuth_deg": [-30.0, 30.0]}


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _domain(parameters: dict, declared=(OBJECT_POSITION,), **fields):
    return parse_spatial_domain(
        {"samples": 4, "seed": 3, **fields, "parameters": parameters}, declared
    )


def test_cylinder_samples_stay_in_the_sector_and_keep_the_nominal_height():
    domain = _domain({"object.position": _FAR})

    for draw in range(64):
        values, record = sample_spatial_parameters(domain, (OBJECT_POSITION,), draw)
        x, y = values["object_position_in_base_m"]
        assert 0.55 <= math.hypot(x, y) <= 0.65
        assert -30.0 <= math.degrees(math.atan2(y, x)) <= 30.0
        assert record["object.position"]["height_m"] is None
        assert record["object.position"]["layer"] == "scene_layout"

    assert sample_spatial_parameters(
        domain, (OBJECT_POSITION,), 7
    ) == sample_spatial_parameters(domain, (OBJECT_POSITION,), 7)
    with_height = _domain({"object.position": {**_FAR, "height_m": [0.1, 0.2]}})
    values, _ = sample_spatial_parameters(with_height, (OBJECT_POSITION,), 0)
    assert 0.1 <= values["object_position_in_base_m"][2] <= 0.2


def test_adding_a_parameter_keeps_the_other_parameters_samples():
    declared = (OBJECT_POSITION, TARGET_OFFSET)
    alone = _domain({"object.position": _FAR}, declared)
    both = _domain(
        {
            "object.position": _FAR,
            "target.offset": {"low": [0.0, 0.05, 0.0], "high": [0.0, 0.10, 0.0]},
        },
        declared,
    )

    for draw in range(8):
        position, _ = sample_spatial_parameters(alone, declared, draw)
        combined, _ = sample_spatial_parameters(both, declared, draw)
        assert combined["object_position_in_base_m"] == (
            position["object_position_in_base_m"]
        )
        offset = combined["target_object_offset_m"]
        assert offset[0] == 0.0 and 0.05 <= offset[1] <= 0.10


def test_per_joint_parameters_take_the_arm_joint_count():
    angle = SpatialParameter(
        "door.open_angle", "task_goal", "uniform", "open_angle_rad", unit="rad"
    )
    declared = resolve_parameter_sizes((ROBOT_START_OFFSET, angle), 7)
    domain = _domain(
        {
            "robot.start_offset": {"low": [-0.1] * 7, "high": [0.1] * 7},
            "door.open_angle": {"low": 1.2, "high": 1.4},
        },
        declared,
    )

    values, _ = sample_spatial_parameters(domain, declared, 0)

    assert len(values["start_qpos_offset_rad"]) == 7
    assert all(-0.1 <= value <= 0.1 for value in values["start_qpos_offset_rad"])
    assert isinstance(values["open_angle_rad"], float)
    assert 1.2 <= values["open_angle_rad"] <= 1.4


@pytest.mark.parametrize(
    ("perturbations", "message"),
    [
        ({"samples": 2, "parameters": {}, "radius": 1}, "Unknown spatial domain"),
        ({"samples": 0, "parameters": {"object.position": _FAR}}, "positive integer"),
        (
            {"samples": 4, "max_attempts": 3, "parameters": {"object.position": _FAR}},
            "at least 'samples'",
        ),
        ({"samples": 2, "parameters": {"cube.position": _FAR}}, "not declared"),
        (
            {"samples": 2, "parameters": {"object.position": {"low": 0, "high": 1}}},
            "unknown cylinder fields",
        ),
        (
            {
                "samples": 2,
                "parameters": {
                    "object.position": {"radius_m": [0.6, 0.5], "azimuth_deg": [0, 1]}
                },
            },
            "low <= high",
        ),
        (
            {
                "samples": 2,
                "parameters": {
                    "object.position": {"radius_m": [0.5, 0.6], "azimuth_deg": [0, 400]}
                },
            },
            "at most 360",
        ),
        (
            {
                "samples": 2,
                "parameters": {"target.offset": {"low": [0.0], "high": [0.1]}},
            },
            "3 value",
        ),
    ],
)
def test_domain_configs_are_validated_against_the_declared_parameters(
    perturbations, message
):
    with pytest.raises(ValueError, match=message):
        parse_spatial_domain(perturbations, (OBJECT_POSITION, TARGET_OFFSET))


def test_domains_without_perturbations_sample_nothing():
    assert parse_spatial_domain({}, (OBJECT_POSITION,)) is None


def test_single_arm_providers_declare_their_spatial_parameters():
    declared = {
        skill: [p.name for p in create_atomic_skill_provider(skill).spatial_parameters]
        for skill in ("pick_up", "place", "push_object", "open_door")
    }

    assert declared == {
        "pick_up": ["object.position"],
        "place": ["object.position", "target.offset"],
        "push_object": ["object.position", "robot.start_offset", "target.offset"],
        "open_door": [
            "articulation.position",
            "robot.start_offset",
            "door.open_angle",
        ],
    }


def test_spatial_suite_domains_parse_for_every_skill_in_their_track():
    suite = load_suite("atomic_franka_pgi_spatial")
    tracks = suite.evaluation_tracks()

    assert [track.id for track in tracks] == [
        "object-skills@nominal",
        "object-skills@near_nominal",
        "object-skills@far_sector",
        "door@nominal",
        "door@shifted_microwave",
        "door@wide_opening",
    ]
    for track in tracks:
        for skill in track.config["skills"]:
            declared = resolve_parameter_sizes(
                create_atomic_skill_provider(skill["id"]).spatial_parameters, 7
            )
            domain = parse_spatial_domain(track.domain.perturbations, declared)
            assert (domain is None) == (track.domain.kind == "nominal")


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #


def _scenario_with_base(base: torch.Tensor) -> AtomicTaskScenario:
    scenario = AtomicTaskScenario()
    scenario.robot = Mock(device=CPU)
    scenario.robot.cfg.solver_cfg = {"arm": Mock(root_link_name="base")}
    scenario.robot.get_link_pose.return_value = base
    scenario.control_part = "arm"
    return scenario


def test_base_frame_positions_follow_the_arm_root_pose():
    # The base is turned 180 degrees about z and stands at (0.1, 0, 0.02).
    base = torch.eye(4).repeat(2, 1, 1)
    base[:, :2, :2] = torch.tensor([[-1.0, 0.0], [0.0, -1.0]])
    base[:, :3, 3] = torch.tensor([0.1, 0.0, 0.02])
    scenario = _scenario_with_base(base)
    nominal = torch.tensor([[-0.4, 0.0, 0.05], [-0.4, 0.0, 0.07]])

    kept = scenario.position_from_base_frame([0.5, 0.1], nominal, name="p")
    lifted = scenario.position_from_base_frame([0.5, 0.1, 0.3], nominal, name="p")

    torch.testing.assert_close(
        kept, torch.tensor([[-0.4, -0.1, 0.05], [-0.4, -0.1, 0.07]])
    )
    torch.testing.assert_close(lifted[:, 2], torch.tensor([0.32, 0.32]))
    scenario.robot.get_link_pose.assert_called_with(link_name="base", to_matrix=True)
    with pytest.raises(ValueError, match=r"\[x, y\]"):
        scenario.position_from_base_frame([0.5], nominal, name="p")


def test_robot_start_offset_shifts_the_start_and_respects_the_margin():
    scenario = AtomicTaskScenario()
    scenario.robot = Mock(device=CPU)
    scenario.robot.get_qpos.return_value = torch.zeros(1, 7)
    scenario.robot.get_qpos_limits.return_value = torch.tensor([[[-1.0, 1.0]] * 7])
    scenario.control_part = "arm"
    scenario.set_robot_start = Mock()

    start = scenario.randomize_robot_start(
        {"start_qpos_offset_rad": [0.1] * 7}, seed=1, stream=2
    )

    torch.testing.assert_close(start, torch.full((1, 7), 0.1))
    with pytest.raises(InfeasibleCaseError) as error:
        scenario.randomize_robot_start(
            {"start_qpos_offset_rad": [0.99] * 7}, seed=1, stream=2
        )
    assert error.value.reason == "start_joint_margin"


def _bench_case(case_id: str, seed: int) -> BenchmarkCase:
    return BenchmarkCase(
        suite_version="spatial_v1",
        track="pick@far",
        scenario_id="pick_up",
        case_id=case_id,
        seed=seed,
        batch_size=1,
        num_waypoints=1,
        path_shape="direct",
        start_state_bin="pre_action",
        start_qpos=torch.zeros(1, 7),
        target_waypoints=torch.eye(4).reshape(1, 1, 4, 4),
        reference_qpos=torch.zeros(1, 1, 7),
        skill_id="pick_up",
        primary_success="task_success",
        case_parameters={"seed": seed},
    )


class _EveryThirdDrawUnreachable(AtomicSkillCaseProvider):
    """Accept a draw unless its seed is a multiple of three."""

    skill_id = "pick_up"
    spatial_parameters = (OBJECT_POSITION,)

    def __init__(self, configs: list) -> None:
        self.configs = configs

    def generate_case(self, scenario, suite, track, config, *, seed, batch_size):
        self.configs.append(dict(config))
        if seed % 3 == 0:
            raise InfeasibleCaseError("unreachable", reason="ik_unreachable")
        return _bench_case(f"case-{seed}", seed)

    def build_invocation(self, scenario, case, adapter):
        raise NotImplementedError


def _spatial_track(**perturbations) -> TrackCfg:
    return TrackCfg(
        id="pick@far",
        scenario="atomic_task",
        domain=DomainCfg(
            id="far",
            version="v1",
            kind="held_out",
            perturbations={
                "seed": 0,
                "parameters": {"object.position": _FAR},
                **perturbations,
            },
        ),
    )


def _sampling_scenario(monkeypatch, configs: list) -> AtomicTaskScenario:
    monkeypatch.setattr(
        atomic_task,
        "create_atomic_skill_provider",
        lambda skill_id: _EveryThirdDrawUnreachable(configs),
    )
    scenario = AtomicTaskScenario()
    scenario.robot = Mock(device=CPU)
    scenario.robot.get_joint_ids.return_value = list(range(7))
    scenario.control_part = "arm"
    return scenario


def test_spatial_domain_redraws_infeasible_cases_and_records_each_sample(
    monkeypatch,
):
    configs: list = []
    scenario = _sampling_scenario(monkeypatch, configs)
    track = _spatial_track(samples=3)

    generated = scenario._generate_spatial_cases(
        Mock(),
        track,
        "pick_up",
        {"name": "cube_top", "object": "cube"},
        track.domain.perturbations,
        batch_size=1,
        skill_index=0,
        case_index=0,
    )

    cases = [case for _, case in generated]
    # Seeds 0 and 3 are unreachable, so draws 1, 2 and 4 are kept.
    assert [case.seed for case in cases] == [1, 2, 4]
    assert [case.case_parameters["spatial_sample"]["draw"] for case in cases] == [
        1,
        2,
        4,
    ]
    sample = cases[0].case_parameters["spatial_sample"]
    assert sample["domain"] == "far"
    assert sample["frame"] == "robot_base"
    assert configs[1]["object_position_in_base_m"] == (
        sample["parameters"]["object.position"]["position_in_base_m"]
    )
    assert cases[0].case_parameters["spatial_feasibility"] == {
        "template": "pick_up:cube_top",
        "draws": 5,
        "infeasible": 2,
        "infeasible_by_reason": {"ik_unreachable": 2},
    }


def test_spatial_domain_rejects_a_mostly_infeasible_region(monkeypatch):
    scenario = _sampling_scenario(monkeypatch, [])
    track = _spatial_track(samples=3, max_attempts=3)

    with pytest.raises(ValueError, match="kept 2 of 3"):
        scenario._generate_spatial_cases(
            Mock(),
            track,
            "pick_up",
            {"name": "cube_top", "object": "cube"},
            track.domain.perturbations,
            batch_size=1,
            skill_index=0,
            case_index=0,
        )


def test_other_generation_errors_are_not_treated_as_infeasible(monkeypatch):
    scenario = _sampling_scenario(monkeypatch, [])
    monkeypatch.setattr(
        _EveryThirdDrawUnreachable,
        "generate_case",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bug")),
    )
    track = _spatial_track(samples=1)

    with pytest.raises(RuntimeError, match="bug"):
        scenario._generate_spatial_cases(
            Mock(),
            track,
            "pick_up",
            {"name": "cube_top", "object": "cube"},
            track.domain.perturbations,
            batch_size=1,
            skill_index=0,
            case_index=0,
        )


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _outcome(success: bool) -> CaseOutcome:
    return CaseOutcome(
        env_index=0,
        planning_success=True,
        finite=True,
        ordered_waypoints_reached=True,
        motion_valid=True,
        completed_waypoint_ratio=1.0,
        final_translation_err_mm=0.0,
        final_rotation_err_deg=0.0,
        waypoint_translation_err_mm_mean=0.0,
        waypoint_translation_err_mm_p95=0.0,
        waypoint_translation_err_mm_max=0.0,
        waypoint_rotation_err_deg_mean=0.0,
        waypoint_rotation_err_deg_p95=0.0,
        waypoint_rotation_err_deg_max=0.0,
        joint_limit_violation=False,
        max_normalized_joint_violation=0.0,
        joint_path_length_rad=0.0,
        cartesian_path_length_m=0.0,
        path_efficiency=1.0,
        execution_success=True,
        task_success=success,
    )


def _record(case: BenchmarkCase, success: bool) -> TrialRecord:
    return TrialRecord(
        suite_version=case.suite_version,
        track=case.track,
        scenario_id=case.scenario_id,
        case_id=case.case_id,
        algorithm_id="baseline",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        model_revision="test",
        planner_config_hash="abc",
        seed=case.seed,
        repeat=0,
        batch_size=case.batch_size,
        waypoint_count=case.num_waypoints,
        path_shape=case.path_shape,
        start_state_bin=case.start_state_bin,
        phase=TrialPhase.MEASURED,
        robot_id=case.robot_id,
        skill_id=case.skill_id,
        primary_success=case.primary_success,
        domain=case.domain,
        outcomes=(_outcome(success),),
    )


def _population(kind: str, outcomes: list[bool], radii=None):
    domain = EvaluationDomain(id=kind, version="v1", kind=kind)
    cases, records = [], []
    for index, success in enumerate(outcomes):
        parameters = {}
        if radii is not None:
            parameters = {
                "spatial_sample": {
                    "domain": kind,
                    "draw": index,
                    "frame": "robot_base",
                    "parameters": {"object.position": {"radius_m": radii[index]}},
                },
                "spatial_feasibility": {
                    "template": "pick_up:cube_top",
                    "draws": 8,
                    "infeasible": 4,
                    "infeasible_by_reason": {"ik_unreachable": 4},
                },
            }
        case = replace(
            _bench_case(f"{kind}-{index}", index),
            track=f"pick@{kind}",
            track_group="pick",
            domain=domain,
            case_parameters=parameters,
        )
        cases.append(case)
        records.append(_record(case, success))
    return cases, records


def _metadata() -> PlannerMetadata:
    return PlannerMetadata(
        algorithm_id="baseline",
        algorithm_role=AlgorithmRole.PRIMARY_BASELINE,
        adapter="ik_interpolate",
        config_hash="abc",
        capabilities=frozenset(),
    )


def test_robustness_and_held_out_are_summarized_separately():
    nominal = _population("nominal", [True, True])
    robust = _population("robustness", [True, False])
    held_out = _population("held_out", [False, False, False, True], [0.5] * 4)
    cases = nominal[0] + robust[0] + held_out[0]
    records = nominal[1] + robust[1] + held_out[1]

    rows = aggregate_generalization(records, [_metadata()], cases, 1)

    by_kind = {row["domain_kind"]: row for row in rows}
    row = by_kind["nominal"]
    assert row["robustness_worst_success_rate"] == pytest.approx(0.5)
    assert row["robustness_retention"] == pytest.approx(0.5)
    assert row["held_out_worst_success_rate"] == pytest.approx(0.25)
    assert row["held_out_retention"] == pytest.approx(0.25)
    # The pooled fields keep their meaning across both kinds.
    assert row["domain_worst_success_rate"] == pytest.approx(0.25)
    assert by_kind["held_out"]["spatial_draws"] == 8
    assert by_kind["held_out"]["spatial_infeasible_rate"] == pytest.approx(0.5)
    assert by_kind["nominal"]["spatial_draws"] is None


def test_spatial_breakdown_reports_success_by_base_frame_radius():
    cases, records = _population(
        "held_out", [True, False, True, True], [0.52, 0.54, 0.58, 0.61]
    )

    rows = spatial_breakdown(records, [_metadata()], cases, radius_bin_m=0.05)

    assert [
        (round(row["radius_low_m"], 2), row["cases"], row["success_rate"])
        for row in rows
    ] == [(0.5, 2, 0.5), (0.55, 1, 1.0), (0.6, 1, 1.0)]
    assert {row["parameter"] for row in rows} == {"object.position"}
