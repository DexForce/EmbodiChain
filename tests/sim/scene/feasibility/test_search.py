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

"""Joint regions, actual-evidence gates and independent statistical validation."""

from __future__ import annotations

from dataclasses import replace
import json
import numpy as np
import pytest

from embodichain.lab.sim.scene.feasibility import (
    CandidateEvaluation,
    FeasibilityCheck,
    PoseSearchVariable,
    TaskPhase,
    TaskOperationMode,
    TaskFeasibilityRequest,
    SceneRobotFeasibilityCfg,
    prepare_feasibility,
    search_feasibility,
    sample_feasibility,
    binomial_interval,
    save_feasibility_atlas,
    load_feasibility_atlas,
    FeasibilityBudget,
    FeasibilityBudgetExhausted,
)

IDENTITY = np.eye(4).tolist()


def request():
    return TaskFeasibilityRequest(
        "move-cube:semantic-v1",
        "fixed-arm/tool-v1",
        "policy-v1",
        "validator-v1",
        "scene-v1",
        {"cube": IDENTITY, "arm": IDENTITY},
        (
            PoseSearchVariable(
                "cube_xy", "scene", IDENTITY, (0, 0, 0), (1, 1, 0), {"cube": IDENTITY}
            ),
            PoseSearchVariable(
                "mount",
                "robot",
                IDENTITY,
                (0, 0, 0),
                (1, 0, np.pi / 2),
                {"arm": IDENTITY},
            ),
        ),
        (TaskOperationMode("runtime", (TaskPhase("move"),)),),
    )


class TwoIslands:
    """Independent analytic fixture: only two distant joint-coordinate boxes work."""

    def prepare(self, req, budget):
        self.req = req

    def evaluate(self, candidate, budget):
        x, y, _, base, _, _ = candidate.parameters
        passed = (x < 0.4 and y < 0.4 and base < 0.4) or (
            x > 0.6 and y > 0.6 and base > 0.6
        )
        checks = tuple(
            FeasibilityCheck(stage, "passed" if passed else "failed")
            for stage in ("semantics", "geometry", "kinematics", "motion", "execution")
        )
        return CandidateEvaluation(
            candidate.candidate_id,
            checks,
            passed,
            passed,
            "measured-state",
            evidence={"raw_parameters": candidate.parameters},
        )


def test_joint_disconnected_regions_and_fixed_count_holdout(tmp_path):
    cfg = SceneRobotFeasibilityCfg(
        max_candidates=300, validation_samples=3, local_radius=0.12
    )
    prepared = prepare_feasibility(request(), cfg)
    atlas = search_feasibility(prepared, TwoIslands())
    assert len(atlas.regions) >= 2
    assert all(r.validation_complete for r in atlas.regions)
    references = {cid for r in atlas.regions for cid in r.reference_candidate_ids}
    represented = [
        o.candidate.parameters[0]
        for o in atlas.observations
        if o.candidate.candidate_id in references
    ]
    assert any(x < 0.4 for x in represented) and any(x > 0.6 for x in represented)
    search_ids = {
        o.candidate.candidate_id for o in atlas.observations if o.region_id is None
    }
    heldout = [o for o in atlas.observations if o.region_id is not None]
    assert len(heldout) == len(atlas.regions) * 3
    assert not search_ids.intersection(o.candidate.candidate_id for o in heldout)
    for region in atlas.regions:
        assert not (region.lower[0] < 0.4 and region.upper[0] > 0.6)
        trials = [o for o in heldout if o.region_id == region.region_id]
        assert region.validation_successes == sum(o.evaluation.accepted for o in trials)
    path = tmp_path / "atlas.json"
    save_feasibility_atlas(atlas, path)
    restored = load_feasibility_atlas(path)
    assert restored.atlas_id == atlas.atlas_id
    old_id = atlas.atlas_id
    prepared.cfg.seed = 99
    assert atlas.cfg.seed == 0 and atlas.atlas_id == old_id
    data = json.loads(path.read_text())
    data["atlas"]["regions"][0]["validation_successes"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        load_feasibility_atlas(path)


def test_sampler_preserves_joint_region_and_requires_both_prechecks():
    atlas = search_feasibility(
        prepare_feasibility(
            request(), SceneRobotFeasibilityCfg(max_candidates=60, validation_samples=1)
        ),
        TwoIslands(),
    )
    assert atlas.regions
    assert (
        sample_feasibility(
            atlas,
            count=2,
            seed=8,
            max_attempts=5,
            precheck=lambda c: (FeasibilityCheck("geometry", "passed"),),
        )
        == ()
    )
    samples = sample_feasibility(
        atlas,
        count=5,
        seed=8,
        precheck=lambda c: (
            FeasibilityCheck("geometry", "passed"),
            FeasibilityCheck("semantics", "passed"),
        ),
    )
    assert len(samples) == 5
    for candidate in samples:
        assert any(
            r.mode_id == candidate.mode_id
            and all(
                a <= v <= b for a, v, b in zip(r.lower, candidate.parameters, r.upper)
            )
            for r in atlas.regions
        )


def test_measured_gate_does_not_accept_projected_completion():
    checks = tuple(
        FeasibilityCheck(s, "passed")
        for s in ("semantics", "geometry", "kinematics", "motion", "execution")
    )
    assert not CandidateEvaluation("candidate", checks, True, False, "state").accepted
    assert not CandidateEvaluation("candidate", checks, True, True).accepted
    with pytest.raises(ValueError, match="cannot be committed"):
        CandidateEvaluation("candidate", checks, True, False, "state", committed=True)


def test_exact_interval_at_zero_and_all_successes():
    assert binomial_interval(0, 0) is None
    # Closed forms for two-sided 95% exact binomial extremes.
    assert binomial_interval(8, 8) == pytest.approx((0.025 ** (1 / 8), 1.0))
    assert binomial_interval(0, 8) == pytest.approx((0.0, 1 - 0.025 ** (1 / 8)))


def test_cancellation_and_interrupted_validation_do_not_publish_interval():
    class Clock:
        now = 0.0

        def __call__(self):
            return self.now

    clock = Clock()
    cfg = SceneRobotFeasibilityCfg(
        max_candidates=100, validation_samples=3, max_wall_time_s=10.0
    )
    budget = FeasibilityBudget(cfg, clock=clock)

    class SlowHoldout(TwoIslands):
        def evaluate(self, candidate, budget):
            if budget.validation:
                clock.now += 6.0
                budget.check()
            return super().evaluate(candidate, budget)

    atlas = search_feasibility(
        prepare_feasibility(request(), cfg), SlowHoldout(), budget=budget
    )
    assert atlas.stop_reason == "wall_time_exhausted"
    assert all(
        not r.validation_complete and r.confidence_interval is None
        for r in atlas.regions
    )


def test_group_mounting_is_preserved_after_yaw_and_translation():
    req = request()
    offset = np.eye(4)
    offset[0, 3] = 0.4
    variable = PoseSearchVariable(
        "pair",
        "robot",
        IDENTITY,
        (0, 0, 0),
        (1, 0, np.pi / 2),
        {"left": IDENTITY, "right": offset},
    )
    req = replace(
        req, entity_poses={"left": IDENTITY, "right": offset}, variables=(variable,)
    )
    candidate = prepare_feasibility(req).candidate(
        (1, 0, 1), mode_id="runtime", ordinal=0
    )
    left, right = (np.array(candidate.deployment.poses[k]) for k in ("left", "right"))
    assert np.linalg.inv(left) @ right == pytest.approx(offset)
    assert right[:3, 3] == pytest.approx((1, 0.4, 0))


def test_each_mode_keeps_global_exploration_and_interruption_is_not_a_trial():
    modes = tuple(TaskOperationMode(f"m{i}", (TaskPhase("move"),)) for i in range(5))
    req = replace(request(), modes=modes)

    class Pass:
        def prepare(self, req, budget):
            pass

        def evaluate(self, c, budget):
            if budget.validation:
                return CandidateEvaluation(
                    c.candidate_id, (FeasibilityCheck("execution", "budget_exhausted"),)
                )
            return CandidateEvaluation(
                c.candidate_id,
                tuple(
                    FeasibilityCheck(s, "passed")
                    for s in (
                        "semantics",
                        "geometry",
                        "kinematics",
                        "motion",
                        "execution",
                    )
                ),
                True,
                True,
                "state",
                clearance_m=np.float32(0.1),
            )

    atlas = search_feasibility(
        prepare_feasibility(
            req,
            SceneRobotFeasibilityCfg(
                max_candidates=100, local_radius=0.001, validation_samples=1
            ),
        ),
        Pass(),
    )
    for mode in modes:
        x = [
            o.candidate.parameters[0]
            for o in atlas.observations
            if o.region_id is None and o.candidate.mode_id == mode.mode_id
        ]
        assert max(x) - min(x) > 0.15
    assert atlas.stop_reason == "evaluator_budget_exhausted"
    assert not atlas.regions[0].validation_complete
    assert (
        atlas.regions[0].validation_completed == 0
        and atlas.regions[0].confidence_interval is None
    )
    assert type(atlas.observations[0].evaluation.clearance_m) is float


def test_runtime_budget_scope_counts_nested_calls_and_resets_after_exit():
    import torch
    from embodichain.lab.sim.motion._budget import (
        _planning_call,
        _planning_budget_scope,
    )

    cfg = SceneRobotFeasibilityCfg(max_ik_calls=4)
    budget = FeasibilityBudget(cfg)

    class Solver:
        @_planning_call("ik", batched=True)
        def solve(self, pose):
            return pose

    solver = Solver()
    with pytest.raises(FeasibilityBudgetExhausted, match="ik_budget"):
        with _planning_budget_scope(budget):
            solver.solve(torch.eye(4).expand(1, 2, 4, 4))
            solver.solve(torch.eye(4).expand(1, 1, 4, 4))
    assert budget.counts["ik"] == 2
    # Outside a scope, calls do not consume a previous host's quota.
    solver.solve(torch.eye(4).expand(1, 8, 4, 4))
    assert budget.counts["ik"] == 2
    # The reserved validation allocation remains available after search stops.
    budget.validation = True
    with _planning_budget_scope(budget):
        solver.solve(torch.eye(4).expand(1, 2, 4, 4))
    assert budget.counts["ik"] == 4


def test_slice_report_keeps_untrusted_task_text_inert(tmp_path):
    from embodichain.lab.sim.scene.feasibility.visualization import (
        write_feasibility_report,
    )

    malicious = "</script><script>alert(1)</script>"
    req = replace(request(), task_id=malicious)
    atlas = search_feasibility(
        prepare_feasibility(
            req, SceneRobotFeasibilityCfg(max_candidates=50, validation_samples=1)
        ),
        TwoIslands(),
    )
    path = tmp_path / "report.html"
    write_feasibility_report(atlas, path)
    html = path.read_text()
    assert malicious not in html
    payload = html.split('<script type="application/json" id="data">', 1)[1].split(
        "</script>", 1
    )[0]
    assert json.loads(payload)["task"] == malicious


@pytest.mark.gpu
def test_cuda_atlas_loads_and_resamples_without_cuda_allocation(tmp_path, monkeypatch):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = SceneRobotFeasibilityCfg(
        max_candidates=100, validation_samples=1, device="cuda:0"
    )
    atlas = search_feasibility(prepare_feasibility(request(), cfg), TwoIslands())
    path = tmp_path / "cuda.json"
    save_feasibility_atlas(atlas, path)
    original = torch.tensor

    def cpu_only(*args, **kwargs):
        assert not str(kwargs.get("device", "cpu")).startswith("cuda")
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_only)
    loaded = load_feasibility_atlas(path)
    assert loaded.atlas_id == atlas.atlas_id
    values = sample_feasibility(
        loaded,
        count=1,
        seed=10,
        precheck=lambda c: (
            FeasibilityCheck("semantics", "passed"),
            FeasibilityCheck("geometry", "passed"),
        ),
    )
    assert len(values) == 1


def test_proposal_prefix_is_independent_of_numeric_batch_boundaries():
    first = search_feasibility(
        prepare_feasibility(
            request(), SceneRobotFeasibilityCfg(max_candidates=60, batch_size=1)
        ),
        TwoIslands(),
        validate_regions=False,
    )
    second = search_feasibility(
        prepare_feasibility(
            request(), SceneRobotFeasibilityCfg(max_candidates=60, batch_size=17)
        ),
        TwoIslands(),
        validate_regions=False,
    )
    assert [o.candidate.candidate_id for o in first.observations] == [
        o.candidate.candidate_id for o in second.observations
    ]
    assert all(r.validation_planned == 0 for r in first.regions)


def test_workspace_prior_owns_selected_samples_and_bounds_cached_fk_batches():
    import torch
    from types import SimpleNamespace
    from embodichain.lab.sim.scene.feasibility import (
        RobotWorkspaceProposalSource,
        PhaseTarget,
    )

    req = replace(
        request(),
        modes=(
            TaskOperationMode(
                "runtime",
                (TaskPhase("grasp", (PhaseTarget("tool_control", IDENTITY, "cube"),)),),
            ),
        ),
    )
    cache = SimpleNamespace(
        qpos=torch.tensor([[0.9, 0.9], [0.2, 0.3]]),
        num_samples=2,
        sample_indices=lambda count, **kwargs: torch.ones(count, dtype=torch.int64),
    )
    batches = []

    def fk(qpos, **kwargs):
        batches.append(qpos.shape[1])
        output = torch.eye(4).repeat(1, qpos.shape[1], 1, 1)
        output[0, :, :2, 3] = qpos[0, :, :2]
        return output

    robot = SimpleNamespace(
        uid="arm",
        device="cpu",
        cfg=SimpleNamespace(
            workspace_cfg={
                "tool_control": SimpleNamespace(
                    strategy="point_uniform", min_score=None
                )
            }
        ),
        get_qpos=lambda: torch.zeros(1, 2),
        get_local_pose=lambda **kwargs: torch.eye(4)[None],
        get_workspace=lambda name: cache,
        compute_batch_fk=fk,
    )
    cfg = SceneRobotFeasibilityCfg(batch_size=1)
    source = RobotWorkspaceProposalSource(
        robot,
        req,
        FeasibilityBudget(cfg),
        phase_id="grasp",
        control_part="tool_control",
        variable_name="cube_xy",
        sample_count=2,
        uniform_fraction=0.0,
    )
    assert batches == [1, 1]
    cache.qpos.fill_(99)
    values = source.propose(
        (0.5, 0.5, 0.0, 0.1, 0.0, 0.0), "runtime", 0, FeasibilityBudget(cfg)
    )
    assert values[:2] == pytest.approx((0.3, 0.3))
    assert source.metrics["workspace_points"] == 2 and source.metrics["guided"] == 1
    cache.qpos.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite selected"):
        RobotWorkspaceProposalSource(
            robot,
            req,
            FeasibilityBudget(cfg),
            phase_id="grasp",
            control_part="tool_control",
            variable_name="cube_xy",
            sample_count=2,
        )
    cache.qpos.fill_(0.2)

    def broken_fk(qpos, **kwargs):
        output = fk(qpos, **kwargs)
        output[:, :, 0, 0] = 2.0
        return output

    robot.compute_batch_fk = broken_fk
    with pytest.raises(ValueError, match="rigid"):
        RobotWorkspaceProposalSource(
            robot,
            req,
            FeasibilityBudget(cfg),
            phase_id="grasp",
            control_part="tool_control",
            variable_name="cube_xy",
            sample_count=2,
        )


def test_in_progress_budget_exhaustion_is_checkpointed_as_an_incomplete_attempt():
    class Interrupted:
        def prepare(self, request, budget):
            pass

        def evaluate(self, candidate, budget):
            raise FeasibilityBudgetExhausted("recording_budget_exhausted")

    checkpoint = []
    atlas = search_feasibility(
        prepare_feasibility(request()), Interrupted(), on_observation=checkpoint.append
    )
    assert len(atlas.observations) == len(checkpoint) == 1
    assert not atlas.observations[0].completed and not atlas.regions
    assert (
        atlas.observations[0].evaluation.checks[0].reason
        == "recording_budget_exhausted"
    )
