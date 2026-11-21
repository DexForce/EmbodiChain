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
"""Check bounded semantic decisions and host ownership of trajectory generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from embodichain.gen_sim.real2sim.agent_refinement import run_agent_refinement
from embodichain.gen_sim.real2sim.annotation import (
    bound_trajectory_phases,
    validate_program_annotation,
)
from embodichain.data_pipeline.real2sim.adaptation.inputs import RobotModel


@pytest.fixture
def recording(tmp_path: Path) -> tuple:
    asset = tmp_path / "cup.glb"
    asset.write_bytes(b"fixed scene asset")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot/>")
    inputs = SimpleNamespace(
        demonstration=SimpleNamespace(
            qpos=np.zeros((8, 2)),
            timestamps=np.arange(8) / 30,
            joint_names=("a", "b"),
            time_unit="s",
        ),
        robot=RobotModel(urdf, ("a", "b"), ("arm",), ("tcp",), np.eye(4), (), 0.0),
        cameras={
            "front": SimpleNamespace(
                video=tmp_path / "real.mp4", sample_frames=np.arange(8)
            )
        },
        instruction="Move the cup",
        primary_camera="front",
    )
    scene = SimpleNamespace(
        description={"objects": [{"id": "cup", "path": str(asset)}]},
        vertices={"cup": np.ones((3, 3))},
    )
    return inputs, scene


def _annotation() -> dict:
    return dict(
        task_program=dict(
            program_id="move_cup",
            instruction="Move the cup",
            targets={
                "destination": dict(
                    kind="pose",
                    frame="world",
                    position=[0.4, 0, 0.1],
                    quaternion_xyzw=[0, 0, 0, 1],
                )
            },
            program=dict(
                kind="sequence",
                items=[
                    dict(
                        kind="segment",
                        name="acquire",
                        steps=dict(kind="invoke", call=dict(kind="pick", object="cup")),
                    ),
                    dict(
                        kind="segment",
                        name="release",
                        steps=dict(
                            kind="invoke",
                            call=dict(
                                kind="place",
                                object="cup",
                                at=dict(kind="target_ref", target="destination"),
                            ),
                        ),
                    ),
                ],
            ),
        ),
        trajectory_bindings=[
            dict(
                name="approach",
                segment="acquire",
                start=0,
                stop=4,
                kind="free",
                reference="object",
                acting_arm="arm",
                instruction="Approach and grasp the cup",
            ),
            dict(
                name="place",
                segment="release",
                start=4,
                stop=8,
                kind="hold",
                reference="target",
                acting_arm="arm",
                instruction="Carry and place the cup",
            ),
        ],
    )


@pytest.mark.parametrize("tamper", [None, "trajectory", "rollout", "missing_receipt"])
def test_prepared_entry_preserves_commands_and_reuses_completed_evidence(
    recording: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper
) -> None:
    from embodichain.gen_sim.real2sim import execution
    from embodichain.gen_sim.real2sim._artifacts import write_json

    inputs, scene = recording
    source, registered = tmp_path / "source", tmp_path / "registered"
    source.mkdir()
    registered.mkdir()
    write_json(source / "robot.json", {"urdf": inputs.robot.urdf})
    np.savez(
        source / "trajectory_original.npz",
        qpos=inputs.demonstration.qpos,
        timestamps=inputs.demonstration.timestamps,
        joint_names=inputs.demonstration.joint_names,
    )
    write_json(registered / "scene.json", scene.description)
    np.savez(registered / "scene_vertices.npz", **scene.vertices)
    native = registered / "native_inputs/native_scene_export"
    native.mkdir(parents=True)
    write_json(native / "scene_config.json", {})
    source_bytes = (source / "trajectory_original.npz").read_bytes()
    # Substitute only unavailable external boundaries, not the migrated workflow.
    monkeypatch.setattr(execution, "load_prepared_source", lambda *_: inputs)
    monkeypatch.setattr(execution.os, "setpriority", lambda *_: None)
    monkeypatch.setattr(execution.os, "sched_setaffinity", lambda *_: None)

    def infer(stage: str, context: dict, directory: Path) -> dict:
        if stage == "annotate":
            return _annotation()
        assert Path(context["rollout"]).is_dir()
        return dict(success=True, failure_phase_ids=[], corrections=[])

    monkeypatch.setattr(execution, "CodexTrajectoryAgent", lambda *_: infer)
    commands = []

    def simulate(command: list[str], **kwargs: object) -> None:
        commands.append(command)
        assert command[1:3] == ["-m", "embodichain.gen_sim.real2sim.replay"]
        package, output = Path(command[3]), Path(command[4])
        with np.load(package / "trajectory_refined.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(data["qpos"], inputs.demonstration.qpos)
            np.testing.assert_array_equal(
                data["timestamps"], inputs.demonstration.timestamps
            )
        assert (package / "native_scene_export").resolve() == native
        (output / "diagnostics").mkdir(parents=True)
        (output / "videos").mkdir()
        np.savez(output / "diagnostics/physics_rollout.npz", timestamps=[0.0])

    monkeypatch.setattr(execution.subprocess, "run", simulate)
    run = tmp_path / "run"
    result = execution.run_prepared_real2sim(
        source, registered, run, instruction=inputs.instruction, max_revisions=0
    )
    assert result["status"] == "completed"
    robot = json.loads((run / "robot_model.json").read_text())
    np.testing.assert_array_equal(robot["flange_to_tcp"], inputs.robot.flange_to_tcp)
    assert robot["urdf"] == str(inputs.robot.urdf)
    assert (run / "annotation/task_program.json").is_file()
    rollout = Path(result["latest_rollout"])
    assert (run / "videos").resolve() == rollout / "videos"

    trajectory = Path(result["latest_trajectory"])
    if tamper == "trajectory":
        with np.load(trajectory, allow_pickle=False) as data:
            modified = dict(data)
        modified["qpos"][3, 0] = 0.75
        np.savez_compressed(trajectory, **modified)
    elif tamper == "rollout":
        np.savez(rollout / "diagnostics/physics_rollout.npz", timestamps=[1.0])
    elif tamper == "missing_receipt":
        (trajectory.parent / "replay_receipt.json").unlink(missing_ok=True)
    continued = tmp_path / "continued"
    if tamper:
        with pytest.raises(
            ValueError, match="replay receipt|trajectory changed|rollout changed"
        ):
            execution.run_prepared_real2sim(
                source,
                registered,
                continued,
                instruction=inputs.instruction,
                max_revisions=0,
                continue_from=run,
            )
        assert len(commands) == 1
        assert not continued.exists()
        return
    resumed = execution.run_prepared_real2sim(
        source,
        registered,
        continued,
        instruction=inputs.instruction,
        max_revisions=0,
        continue_from=run,
    )
    assert resumed["status"] == "completed"
    assert len(commands) == 1
    assert (continued / "videos").resolve() == rollout / "videos"
    assert (continued / "iterations/000/replay_reuse.json").is_file()
    assert (source / "trajectory_original.npz").read_bytes() == source_bytes
    again = execution.run_prepared_real2sim(
        source,
        registered,
        tmp_path / "continued_again",
        instruction=inputs.instruction,
        max_revisions=0,
        continue_from=continued,
    )
    assert again["status"] == "completed"
    assert len(commands) == 1


def test_numerical_revision_replays_before_success(
    recording: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs, scene = recording
    calls, commands = [], []
    candidate = inputs.demonstration.qpos.copy()
    candidate[5:7, 0] = 0.05
    edits = [
        dict(
            phase_ids=["place"], side="arm", anchor_sample=5, translation_m=[0.01, 0, 0]
        )
    ]

    def compile_edits(model, current, original, times, phases, corrections):
        assert corrections == edits
        np.testing.assert_array_equal(original, inputs.demonstration.qpos)
        return candidate.copy(), {"method": "test numerical compiler"}

    monkeypatch.setattr(
        "embodichain.gen_sim.real2sim.agent_refinement.apply_trajectory_corrections",
        compile_edits,
    )

    def agent(stage: str, context: dict, directory: Path) -> dict:
        calls.append(stage)
        directory.mkdir(parents=True)
        if stage == "annotate":
            return _annotation()
        assert Path(context["rollout"]).is_dir()
        if len(commands) == 1:
            return dict(
                success=False,
                failure_phase_ids=["place"],
                explanation="Cup missed the target in replay",
                corrections=edits,
            )
        np.testing.assert_array_equal(commands[-1], candidate)
        return dict(
            success=True,
            failure_phase_ids=[],
            corrections=[],
            explanation="Reviewed second native execution",
        )

    def replay(qpos: np.ndarray, directory: Path) -> Path:
        calls.append("replay")
        commands.append(qpos.copy())
        result = directory / "native_render"
        result.mkdir()
        return result

    result = run_agent_refinement(
        inputs,
        scene,
        tmp_path / "run",
        agent=agent,
        replay=replay,
    )
    assert calls == ["annotate", "replay", "review", "replay", "review"]
    np.testing.assert_array_equal(commands[0], inputs.demonstration.qpos)
    np.testing.assert_array_equal(inputs.demonstration.qpos, np.zeros((8, 2)))
    assert result["success"] and result["status"] == "completed"
    assert len(result["history"]) == 2
    annotation_dir = tmp_path / "run/annotation"
    program = json.loads((annotation_dir / "task_program.json").read_text())
    assert "integration" not in program
    assert program["program"]["items"][0]["steps"]["call"]["kind"] == "pick"
    first_review = json.loads((tmp_path / "run/iterations/000/review.json").read_text())
    assert first_review["failure_segment_ids"] == ["release"]
    assert not json.loads((annotation_dir / "scope.json").read_text())[
        "execution_ready"
    ]


def test_annotation_uses_native_program_and_half_open_source_bindings() -> None:
    from embodichain.agents.mllm.task_program import decode_mllm_task_program
    from embodichain.lab.task_program.language.schema import TaskProgramIntegrationCfg

    annotation = validate_program_annotation(_annotation(), 8, {"cup"}, {"arm"})
    config = decode_mllm_task_program(
        json.dumps(annotation["task_program"]),
        integration=TaskProgramIntegrationCfg("robot", "scene", "runtime"),
    )
    assert [node.name for node in config.program.items] == ["acquire", "release"]
    phases = bound_trajectory_phases(annotation)
    assert [(p["start_frame"], p["end_frame"]) for p in phases] == [(0, 3), (4, 7)]
    assert [p["segment"] for p in phases] == ["acquire", "release"]


@pytest.mark.parametrize("fault", ["call", "object", "segment", "clock", "resource"])
def test_invalid_program_or_binding_cannot_enter_replay(fault: str) -> None:
    value = _annotation()
    call = value["task_program"]["program"]["items"][0]["steps"]["call"]
    if fault == "call":
        call["kind"] = "teleport"
    elif fault == "object":
        call["object"] = "missing"
    elif fault == "segment":
        value["trajectory_bindings"][0]["segment"] = "missing"
    elif fault == "clock":
        value["trajectory_bindings"][1]["start"] = 5
    else:
        call["resources"] = {"primary": "unbound_robot"}
    with pytest.raises(ValueError):
        validate_program_annotation(value, 8, {"cup"}, {"arm"})


def test_agent_cannot_change_fixed_scene(recording: tuple, tmp_path: Path) -> None:
    inputs, scene = recording

    def agent(stage: str, context: dict, directory: Path) -> dict:
        Path(scene.description["objects"][0]["path"]).write_bytes(b"moved object")
        return _annotation()

    def replay(qpos: np.ndarray, directory: Path) -> Path:
        pytest.fail("Changed scene must never be replayed")

    output = tmp_path / "run"
    with pytest.raises(ValueError, match="Fixed input changed"):
        run_agent_refinement(inputs, scene, output, agent=agent, replay=replay)
    assert json.loads((output / "status.json").read_text())["success"] is False


@pytest.mark.parametrize("max_revisions", [0, None])
def test_correction_budget_bounds_executions_and_final_review(
    recording: tuple,
    tmp_path: Path,
    max_revisions: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, scene = recording
    commands, remaining = [], []
    # The ordinary caller gets two corrections, not an open-ended retry loop.
    expected_budget = 2 if max_revisions is None else max_revisions

    def compile_edits(model, current, original, times, phases, corrections):
        candidate = current.copy()
        candidate[5:7, 0] += 0.01
        return candidate, {}

    monkeypatch.setattr(
        "embodichain.gen_sim.real2sim.agent_refinement.apply_trajectory_corrections",
        compile_edits,
    )

    def agent(stage: str, context: dict, directory: Path) -> dict:
        if stage == "annotate":
            return _annotation()
        remaining.append(context["revisions_remaining"])
        return dict(
            success=False,
            failure_phase_ids=["place"],
            corrections=(
                [
                    dict(
                        phase_ids=["place"],
                        side="arm",
                        anchor_sample=5,
                        translation_m=[0.01, 0, 0],
                    )
                ]
                if context["revisions_remaining"]
                else []
            ),
        )

    def replay(qpos: np.ndarray, directory: Path) -> Path:
        commands.append(qpos.copy())
        return directory

    output = tmp_path / "run"
    result = run_agent_refinement(
        inputs,
        scene,
        output,
        agent=agent,
        replay=replay,
        **({} if max_revisions is None else {"max_revisions": max_revisions}),
    )
    assert result["status"] == "revision_budget_exhausted"
    assert not result["success"]
    assert len(commands) == expected_budget + 1
    assert remaining == list(range(expected_budget, -1, -1))
    assert result["revisions"] == expected_budget
    final = output / "iterations" / f"{expected_budget:03d}" / "review.json"
    assert json.loads(final.read_text())["corrections"] == []
    with pytest.raises(FileExistsError, match="fresh run"):
        run_agent_refinement(inputs, scene, output, agent=agent, replay=replay)


def test_agent_cannot_submit_a_trajectory_file(
    recording: tuple, tmp_path: Path
) -> None:
    inputs, scene = recording
    executions = []

    def agent(stage: str, context: dict, directory: Path) -> dict:
        if stage == "annotate":
            return _annotation()
        directory.mkdir(parents=True)
        candidate = directory / "candidate.npz"
        np.savez(candidate, qpos=inputs.demonstration.qpos, joint_names=["b", "a"])
        return dict(
            success=False,
            failure_phase_ids=["place"],
            changed_phase_ids=["place"],
            candidate_path=str(candidate),
        )

    def replay(qpos: np.ndarray, directory: Path) -> Path:
        executions.append(qpos)
        return directory

    with pytest.raises(ValueError, match="only diagnosis and structured corrections"):
        run_agent_refinement(
            inputs, scene, tmp_path / "run", agent=agent, replay=replay
        )
    assert len(executions) == 1


@pytest.mark.parametrize("recover", [False, True])
def test_rejected_parameters_have_one_retry_without_executing_rejected_commands(
    recording: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recover: bool
) -> None:
    inputs, scene = recording
    executions, attempts, reviews = [], [], []

    def compiler(model, current, original, times, phases, corrections):
        attempts.append(corrections)
        if not recover or len(attempts) == 1:
            raise ValueError("IK failed at sample 5")
        candidate = current.copy()
        candidate[5:7, 0] += 0.01
        return candidate, {}

    monkeypatch.setattr(
        "embodichain.gen_sim.real2sim.agent_refinement.apply_trajectory_corrections",
        compiler,
    )

    def agent(stage: str, context: dict, directory: Path) -> dict:
        if stage == "annotate":
            return _annotation()
        reviews.append(context)
        if len(reviews) == 2:
            assert context["correction_failure"]["error"] == "IK failed at sample 5"
            assert context["correction_failure"]["corrections"] == [{}]
            assert len(executions) == 1
        if len(executions) == 2:
            assert "correction_failure" not in context
            return dict(success=True, failure_phase_ids=[], corrections=[])
        return dict(success=False, failure_phase_ids=["place"], corrections=[{}])

    def replay(qpos: np.ndarray, directory: Path) -> Path:
        executions.append(qpos.copy())
        return directory

    output = tmp_path / "run"
    result = run_agent_refinement(
        inputs, scene, output, agent=agent, replay=replay, max_revisions=1
    )
    assert len(attempts) == 2
    assert len(executions) == 1 + int(recover)
    assert result["revisions"] == int(recover)
    assert result["success"] is recover
    assert result["status"] == ("completed" if recover else "correction_rejected")
    assert (output / "iterations/000/rejection.json").is_file()
    assert (output / "iterations/000/correction_retry/review.json").is_file()
    if not recover:
        assert (output / "iterations/000/correction_retry/rejection.json").is_file()
        assert "sample 5" in result["error"]
