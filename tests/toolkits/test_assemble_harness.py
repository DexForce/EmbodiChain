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

"""Provider-free contracts for description-driven assembly and scene regeneration."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

pytest.importorskip("fcl")
pytest.importorskip("manifold3d")

from scripts.tools.assemble import generate
from scripts.tools.assemble._protocol import load_config
from scripts.tools.assemble._validation import PlacementValidator
from scripts.tools.assemble.visualize import run_cycles
from scripts.tools.assemble._json_io import read_json


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "configs" / "case.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "base_description": "A flat support",
                "assemble_description": "A block",
                "action_description": "Place the block on the support",
                "output_dir": "../output",
                "validation": {"collision": "exact"},
                "codex": {"max_turns": 8},
            }
        )
    )
    return path


@pytest.fixture
def geometry(tmp_path: Path) -> dict:
    assets = {}
    for role, size, z in (
        ("base", [0.2, 0.1, 0.02], 0.01),
        ("assemble", [0.05, 0.03, 0.08], 0.04),
    ):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation([0, 0, z])
        path = tmp_path / f"{role}.obj"
        mesh.export(path)
        assets[role] = {"path": str(path)}
    return {"assets": assets, "metadata": {}}


def action(name: str, **kwargs: object) -> dict:
    return {
        "action": name,
        "reason": "Test action",
        "design": None,
        "source": None,
        "T_base_assemble": None,
        "candidate_id": None,
    } | kwargs


def design(**changes: object) -> dict:
    return {
        "base_description": "A 20 x 10 cm support, 2 cm thick, grounded at Z=0.",
        "assemble_description": "A 5 x 3 x 8 cm block with its bottom at local Z=0.",
        "action_description": "Place the block upright on the support.",
        "base_frame": "Z up, bottom at zero, X is width.",
        "orientation": "upright",
        "assemble_frame": "Z goes from the block bottom to its top.",
        "contact_description": "Block bottom rests on the support top.",
        "assemble_axis": [0, 0, 1],
        "target_axis": [0, 0, 1],
        "axis_reason": "Keep the block upright by aligning its height axis with base +Z.",
    } | changes


def proposal(z: float = 0.024) -> list:
    pose = np.eye(4)
    pose[2, 3] = z
    return pose.tolist()


def test_config_resolves_relative_output_and_preserves_descriptions(
    config_path: Path,
) -> None:
    config = load_config(config_path)
    assert config["output_dir"] == str(config_path.parent.parent / "output")
    assert config["action_description"] == "Place the block on the support"
    assert config["validation"]["probe"] > config["validation"]["contact_tolerance"]


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"base_description": ""},
        {"codex": {"max_turns": True}},
        {"validation": {"contact_gap": 0.01}},
        {"validation": {"assemble_axis": [0, 0, 0]}},
        {"validation": {"assemble_axis": [0, 0, 1]}},
        {"validation": {"collision": "bogus"}},
        {"geometry": {"max_faces": 1e999}},
        {"output_dir": ".."},
    ],
)
def test_config_rejects_invalid_settings(config_path: Path, change: dict) -> None:
    config_path.write_text(json.dumps(json.loads(config_path.read_text()) | change))
    with pytest.raises(ValueError):
        load_config(config_path)


def test_settles_to_actual_support_and_rejects_floating_and_containment(
    config_path: Path, geometry: dict, tmp_path: Path
) -> None:
    validator = PlacementValidator(
        geometry, load_config(config_path)["validation"], tmp_path
    )
    candidate = validator.evaluate(proposal())
    assert candidate["validation"]["accepted"]
    assert candidate["T_base_assemble"][2][3] == pytest.approx(0.0201, abs=2e-8)
    assert not validator.validate(proposal(0.1))["accepted"]
    assert not validator.evaluate(proposal(-0.02))["validation"]["accepted"]


def test_rotated_pose_respects_configured_axis(
    config_path: Path, geometry: dict, tmp_path: Path
) -> None:
    config = load_config(config_path)
    config["validation"].update(assemble_axis=[0, 0, 1], target_axis=[0, 0, -1])
    validator = PlacementValidator(geometry, config["validation"], tmp_path)
    assert not validator.evaluate(proposal())["validation"]["accepted"]
    pose = np.diag([1, -1, -1, 1]).astype(float)
    pose[2, 3] = 0.104  # Inverted 8 cm block starts 4 mm above the 2 cm support.
    assert validator.evaluate(pose)["validation"]["accepted"]


def test_full_action_loop_saves_independently_verified_pose(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_generate", lambda *args: geometry)
    decisions = iter(
        [
            action("plan", design=design()),
            action("generate", source="def build(): pass"),
            action("evaluate", T_base_assemble=proposal()),
            action("finish", candidate_id=3),
        ]
    )
    result = generate.run_harness(
        load_config(config_path), decide=lambda *args: next(decisions)
    )
    assert result["success"]
    assert result["trace"][-1]["observation"]["independently_revalidated"]
    assert result["assets"]["base"]["sha256"]
    assert (
        read_json(Path(result["run_directory"]) / "result.json")["T_base_assemble"]
        == result["T_base_assemble"]
    )


def test_regeneration_invalidates_previous_candidates(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_generate", lambda *args: geometry)
    decisions = iter(
        [
            action("plan", design=design()),
            action("generate", source="first"),
            action("evaluate", T_base_assemble=proposal()),
            action("generate", source="second"),
            action("finish", candidate_id=3),
            action("evaluate", T_base_assemble=proposal()),
            action("finish", candidate_id=6),
        ]
    )
    result = generate.run_harness(
        load_config(config_path), decide=lambda *args: next(decisions)
    )
    assert result["success"]
    assert "error" in result["trace"][4]["observation"]
    assert [x["candidate_id"] for x in result["candidates"]] == [6]


def test_timing_includes_model_build_retries_pose_checks_and_final_validation(
    config_path: Path,
    geometry: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    from scripts.tools.assemble import _validation

    clock = [0.0]
    monkeypatch.setattr(
        generate, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )

    def build(source: str, *args: object) -> dict:
        if source == "broken":
            clock[0] += 5
            raise ValueError("Invalid mesh")
        clock[0] += 7
        return geometry

    class TimedValidator:
        def __init__(self, *args: object) -> None:
            clock[0] += 3  # Collision preprocessing, repeated at final validation.
            self.hull_counts = None

        def evaluate(self, pose: list) -> dict:
            clock[0] += 4
            return {
                "T_base_assemble": pose,
                "validation": {"accepted": pose[2][3] < 0.1},
            }

        def validate(self, pose: list) -> dict:
            clock[0] += 6
            return {"accepted": True}

    monkeypatch.setattr(generate, "_generate", build)
    monkeypatch.setattr(_validation, "PlacementValidator", TimedValidator)
    decisions = iter(
        [
            (2, action("plan", design=design())),
            (10, action("generate", source="broken")),
            (20, action("generate", source="valid")),
            (30, action("evaluate", T_base_assemble=proposal(0.5))),
            (40, action("evaluate", T_base_assemble=proposal())),
            (50, action("finish", candidate_id=5)),
        ]
    )

    def decide(*args: object) -> dict:
        duration, choice = next(decisions)
        clock[0] += duration
        return choice

    handoff = config_path.parent / "handoff.json"
    result = generate.run_harness(
        load_config(config_path), decide=decide, result_file=handoff
    )
    expected = {"planning": 2, "objects": 45, "relative_pose": 137, "total": 184}
    assert result["success"]
    assert result["timing_seconds"] == expected
    assert result["trace"][1]["timing"] == {"stage": "objects", "seconds": 15}
    assert read_json(handoff)["timing_seconds"] == expected
    assert (
        read_json(Path(result["run_directory"]) / "result.json")["timing_seconds"]
        == expected
    )
    assert (
        read_json(Path(result["config"]["output_dir"]) / "latest.json")[
            "timing_seconds"
        ]
        == expected
    )
    output = capsys.readouterr().out
    assert "Object generation: 15.00s this turn" in output
    assert "Relative pose generation: 59.00s this turn; 137.00s cumulative" in output
    assert "Cycle 1 timing (complete): objects=45.00s, relative pose=137.00s" in output


@pytest.mark.parametrize("error_type", [TimeoutError, KeyboardInterrupt])
@pytest.mark.parametrize("failed_stage", ["planning", "objects", "relative_pose"])
def test_timing_persists_failed_model_calls_and_interruptions(
    config_path: Path,
    geometry: dict,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    error_type: type[BaseException],
    failed_stage: str,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        generate, "time", SimpleNamespace(perf_counter=lambda: clock[0])
    )
    monkeypatch.setattr(generate, "_generate", lambda *args: geometry)
    failure_turn = {"planning": 1, "objects": 2, "relative_pose": 3}[failed_stage]

    def decide(prompt: str, directory: Path, turn: int, config: dict) -> dict:
        if turn == failure_turn:
            clock[0] += 7
            raise error_type("request stopped")
        return (
            action("plan", design=design())
            if turn == 1
            else action("generate", source="build")
        )

    config = load_config(config_path)
    if error_type is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            generate.run_harness(config, decide=decide)
    else:
        assert not generate.run_harness(config, decide=decide)["success"]
    result = read_json(Path(config["output_dir"]) / "latest.json")
    assert result["status"] == (
        "interrupted" if error_type is KeyboardInterrupt else "failed"
    )
    assert result["timing_seconds"][failed_stage] == 7
    assert result["timing_seconds"]["total"] == 7
    assert f"Cycle 1 timing ({result['status']})" in capsys.readouterr().out


def test_final_reload_rejects_mesh_modified_after_evaluation(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_generate", lambda *args: geometry)

    def decide(prompt: str, directory: Path, turn: int, config: dict) -> dict:
        if turn == 1:
            return action("plan", design=design())
        if turn == 2:
            return action("generate", source="build")
        if turn == 3:
            return action("evaluate", T_base_assemble=proposal())
        if turn == 4:
            trimesh.creation.box([1, 1, 1]).export(geometry["assets"]["base"]["path"])
            return action("finish", candidate_id=3)
        return action("fail")

    result = generate.run_harness(load_config(config_path), decide=decide)
    assert not result["success"]
    assert result["T_base_assemble"] is None
    assert (
        "Independent final validation failed"
        in result["trace"][3]["observation"]["error"]
    )


def test_failed_run_overwrites_latest_success(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(generate, "_generate", lambda *args: geometry)
    config = load_config(config_path)
    decisions = iter(
        [
            action("plan", design=design()),
            action("generate", source="build"),
            action("evaluate", T_base_assemble=proposal()),
            action("finish", candidate_id=3),
        ]
    )
    first = generate.run_harness(config, decide=lambda *args: next(decisions))
    second = generate.run_harness(config, decide=lambda *args: action("fail"))
    assert first["success"] and not second["success"]
    latest = read_json(Path(config["output_dir"]) / "latest.json")
    assert latest["T_base_assemble"] is None
    assert latest["run_directory"] != first["run_directory"]


def test_blender_worker_exports_hollow_mesh_without_axis_remapping(
    config_path: Path, tmp_path: Path
) -> None:
    source = """from scripts.tools.assemble.blender_helpers import box, cylinder, difference

def build():
    base = box([.2, .1, .02], [0,0,.01])
    mug = difference(cylinder(.04,.09,[0,0,.045]), cylinder(.035,.09,[0,0,.051]))
    return {"base": base, "assemble": mug, "metadata": {"mouth_axis": [0,0,1]}}
"""
    report = generate._generate(
        source, tmp_path / "generated", load_config(config_path)["geometry"]
    )
    from scripts.tools.assemble._geometry import load_mesh

    mesh = load_mesh(Path(report["assets"]["assemble"]["path"]))
    assert mesh.is_volume
    # The +Z mouth stays open: a downward center ray first hits the 6 mm floor,
    # while a ray through the wall hits the 90 mm rim in the exported frame.
    hits, rays, _ = mesh.ray.intersects_location(
        [[0, 0, 0.1], [0.0375, 0, 0.1]], [[0, 0, -1], [0, 0, -1]]
    )
    assert hits[rays == 0, 2].max() == pytest.approx(0.006, abs=1e-6)
    assert hits[rays == 1, 2].max() == pytest.approx(0.09, abs=1e-6)
    assert (tmp_path / "generated" / "assets.blend").is_file()


def test_blender_worker_reduces_face_count_preserving_cavity_and_asset_frame(
    config_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    source = """from scripts.tools.assemble.blender_helpers import box, cylinder, difference

def build():
    base = box([.2, .1, .02], [.1, -.05, .01])
    base.rotation_euler.z = .3
    mug = difference(cylinder(.04,.09,[0,0,.045]), cylinder(.035,.09,[0,0,.051]))
    for obj in (base, mug):
        modifier = obj.modifiers.new('Dense flat subdivisions', 'SUBSURF')
        modifier.subdivision_type = 'SIMPLE'
        modifier.levels = 3
    return {"base": base, "assemble": mug, "metadata": {}}
"""
    from scripts.tools.assemble._geometry import load_mesh
    from scripts.tools.assemble._protocol import run_process

    directory = tmp_path / "simplified"
    settings = load_config(config_path)["geometry"] | {"max_faces": 500}
    report = generate._generate(source, directory, settings)
    for asset in report["assets"].values():
        assert asset["mesh_processing"]["decimated"]
        assert asset["mesh_processing"]["input_triangles"] > settings["max_faces"]
        assert 0 < asset["faces"] <= settings["max_faces"]
        assert load_mesh(Path(asset["path"])).is_volume
    base = load_mesh(Path(report["assets"]["base"]["path"]))
    reference = trimesh.creation.box([0.2, 0.1, 0.02])
    reference.apply_transform(trimesh.transformations.rotation_matrix(0.3, [0, 0, 1]))
    reference.apply_translation([0.1, -0.05, 0.01])
    np.testing.assert_allclose(base.bounds, reference.bounds, atol=1e-6)
    assert base.volume == pytest.approx(reference.volume, rel=1e-4)
    mug = load_mesh(Path(report["assets"]["assemble"]["path"]))
    hits, rays, _ = mug.ray.intersects_location(
        [[0, 0, 0.1], [0.0375, 0, 0.1]], [[0, 0, -1], [0, 0, -1]]
    )
    assert hits[rays == 0, 2].max() == pytest.approx(0.006, abs=1e-5)
    assert hits[rays == 1, 2].max() == pytest.approx(0.09, abs=1e-5)
    assert "base mesh simplified:" in capsys.readouterr().out
    # Reopen the saved Blender scene in isolation; the simplified geometry and
    # baked transforms must agree with the actual exported triangle meshes.
    check = """import bpy, json, numpy as np, sys
from pathlib import Path
from mathutils import Matrix
root = Path(sys.argv[1])
report = json.loads((root / 'geometry.json').read_text())
bpy.ops.wm.open_mainfile(filepath=str(root / 'assets.blend'))
objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
assert len(objects) == 2
for role, z in [('base', .02), ('assemble', .09)]:
    obj = min(objects, key=lambda obj: abs(obj.dimensions.z - z))
    assert obj.matrix_world == Matrix.Identity(4)
    obj.data.calc_loop_triangles()
    assert len(obj.data.loop_triangles) == report['assets'][role]['mesh_processing']['exported_triangles']
    vertices = np.array([vertex.co[:] for vertex in obj.data.vertices])
    np.testing.assert_allclose([vertices.min(axis=0), vertices.max(axis=0)], report['assets'][role]['bounds'], atol=1e-6)
"""
    import sys

    run_process(
        [sys.executable, "-c", check, str(directory)],
        directory,
        directory / "scene_check",
        30,
    )


def test_enter_generates_before_replacing_the_scene_and_eof_stops() -> None:
    events = []
    keys = iter(["unexpected\n", "\n", ""])

    def generate_cycle(cycle: int) -> dict:
        events.append(("generate", cycle))
        return {"success": True, "cycle": cycle}

    count = run_cycles(
        generate_cycle,
        lambda result: events.append(("show", result["cycle"])),
        lambda: next(keys),
    )
    assert count == 2
    assert events == [
        ("generate", 1),
        ("show", 1),
        ("generate", 2),
        ("show", 2),
    ]


def test_blender_boolean_seams_export_as_closed_solids(
    config_path: Path, tmp_path: Path
) -> None:
    # These intersecting cylinders produced sub-micron Boolean edges that became
    # open cracks when OBJ loading simply discarded degenerate triangles.
    source = """from scripts.tools.assemble.blender_helpers import box, cylinder, rod, union

def build():
    parts = [cylinder(.075,.014,[0,0,.007]), cylinder(.012,.282,[0,0,.153],48)]
    joints = [
        ([0,0,.105],[.105,0,.151],.008),
        ([.105,0,.145],[.105,0,.245],.008),
        ([0,0,.150],[-.067,.020,.205],.008),
        ([-.067,.020,.205],[-.080,.024,.226],.007),
        ([0,0,.195],[-.038,-.061,.247],.008),
        ([-.038,-.061,.247],[-.044,-.071,.267],.007),
        ([0,0,.232],[.014,.054,.278],.0075),
        ([0,0,.260],[-.030,.027,.296],.007),
    ]
    parts.extend(rod(a,b,r,40) for a,b,r in joints)
    return {"base": union(parts), "assemble": box([.03,.02,.05],[0,0,.025]), "metadata": {}}
"""
    report = generate._generate(
        source, tmp_path / "boolean_seams", load_config(config_path)["geometry"]
    )
    from scripts.tools.assemble._geometry import load_mesh

    mesh = load_mesh(Path(report["assets"]["base"]["path"]))
    assert mesh.is_watertight and mesh.is_volume


def test_failed_rebuild_cannot_finish_an_old_candidate(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def build(source: str, *args: object) -> dict:
        if source == "broken":
            raise RuntimeError("Blender could not build the replacement")
        return geometry

    monkeypatch.setattr(generate, "_generate", build)
    decisions = iter(
        [
            action("plan", design=design()),
            action("generate", source="first"),
            action("evaluate", T_base_assemble=proposal()),
            action("generate", source="broken"),
            action("finish", candidate_id=3),
            action("fail"),
        ]
    )
    result = generate.run_harness(
        load_config(config_path), decide=lambda *args: next(decisions)
    )
    assert not result["success"]
    assert result["assets"] is None and result["candidates"] == []
    assert "No generated assets" in result["trace"][4]["observation"]["error"]


def test_failed_generation_can_retry_on_enter() -> None:
    events = []
    results = iter(
        [
            {
                "success": False,
                "reason": "bad geometry",
                "run_directory": "/tmp/failed",
            },
            {"success": True},
        ]
    )
    count = run_cycles(
        lambda _: next(results),
        lambda _: events.append("show"),
        lambda: "\n",
        limit=2,
    )
    assert count == 2
    assert events == ["show"]


def test_planning_expands_short_intent_before_any_build(
    config_path: Path, geometry: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.assemble._prompt import build_prompt

    config = load_config(config_path)
    config.update(
        base_description="支架", assemble_description="杯子", action_description="倒扣"
    )
    first_prompt = build_prompt(config, 1, [], None, [])
    assert "action=plan" in first_prompt and "倒扣" in first_prompt
    assert "Helper implementation" not in first_prompt
    attempts = []
    monkeypatch.setattr(
        generate, "_generate", lambda *args: attempts.append(args) or geometry
    )
    decisions = iter(
        [
            action("generate", source="too early"),
            action("plan", design=design()),
            action("generate", source="valid"),
            action("evaluate", T_base_assemble=proposal()),
            action("finish", candidate_id=4),
        ]
    )
    result = generate.run_harness(config, decide=lambda *args: next(decisions))
    assert result["success"] and len(attempts) == 1
    assert "valid plan" in result["trace"][0]["observation"]["error"]
    assert config["validation"]["assemble_axis"] is None
    assert result["resolved_validation"]["assemble_axis"] == [0, 0, 1]
    assert (Path(result["run_directory"]) / "design_plan.json").is_file()


def test_automatic_axes_are_normalized_and_enforced(
    config_path: Path, geometry: dict, tmp_path: Path
) -> None:
    from scripts.tools.assemble._planning import resolve_design

    config = load_config(config_path)
    plan, validation = resolve_design(
        design(assemble_axis=[0, 0, 3], target_axis=[0, 0, -2], orientation="inverted"),
        config["validation"],
    )
    assert plan["assemble_axis"] == [0, 0, 1] and validation["target_axis"] == [
        0,
        0,
        -1,
    ]
    validator = PlacementValidator(geometry, validation, tmp_path)
    assert not validator.evaluate(proposal())["validation"]["accepted"]
    pose = np.diag([1, -1, -1, 1]).astype(float)
    pose[2, 3] = 0.104
    assert validator.evaluate(pose)["validation"]["accepted"]


@pytest.mark.parametrize(
    "axes", [[0, 0, 0], [True, 0, 1], [0, float("nan"), 1], [1, 2]]
)
def test_planning_rejects_invalid_axis(config_path: Path, axes: list) -> None:
    from scripts.tools.assemble._planning import resolve_design

    with pytest.raises(ValueError):
        resolve_design(
            design(assemble_axis=axes), load_config(config_path)["validation"]
        )


def test_plan_cannot_override_explicit_user_axes(config_path: Path) -> None:
    from scripts.tools.assemble._planning import resolve_design

    settings = load_config(config_path)["validation"] | {
        "assemble_axis": [0, 0, 1],
        "target_axis": [0, 0, -1],
    }
    with pytest.raises(ValueError, match="explicit validation.target_axis"):
        resolve_design(design(), settings)


def test_frozen_plan_cannot_be_relaxed_to_accept_a_bad_pose(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decisions = iter(
        [
            action(
                "plan", design=design(target_axis=[0, 0, -1], orientation="inverted")
            ),
            action("plan", design=design()),
            action("fail"),
        ]
    )
    result = generate.run_harness(
        load_config(config_path), decide=lambda *args: next(decisions)
    )
    assert "frozen" in result["trace"][1]["observation"]["error"]
    assert result["resolved_validation"]["target_axis"] == [0, 0, -1]


def test_generation_failure_keeps_the_previous_scene_until_a_replacement_is_ready() -> (
    None
):
    events = []

    def next_result(cycle: int) -> dict:
        events.append(("generate", cycle))
        return {"success": cycle != 2, "reason": "deliberate failure", "cycle": cycle}

    run_cycles(
        next_result,
        lambda result: events.append(("replace", result["cycle"])),
        lambda: "\n",
        limit=3,
    )
    assert events == [
        ("generate", 1),
        ("replace", 1),
        ("generate", 2),
        ("generate", 3),
        ("replace", 3),
    ]


def test_generation_process_keeps_pumping_and_streams_output(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    import sys
    from scripts.tools.assemble._generation_process import wait_process

    ticks = []
    code = wait_process(
        [
            sys.executable,
            "-c",
            "import time; print('building',flush=True); time.sleep(.25); print('done')",
        ],
        tmp_path,
        5,
        lambda: ticks.append(True),
    )
    assert code == 0 and len(ticks) >= 2
    assert "building" in capsys.readouterr().out
    assert "done" in (tmp_path / "generation.log").read_text()


def test_generation_timeout_stops_children_in_separate_process_groups(
    tmp_path: Path,
) -> None:
    import sys
    import psutil
    from scripts.tools.assemble._generation_process import wait_process

    script = """import subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    pid_file = tmp_path / "child.pid"
    with pytest.raises(TimeoutError):
        wait_process(
            [sys.executable, "-c", script, str(pid_file)], tmp_path, 1, lambda: None
        )
    pid = int(pid_file.read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_subprocess_result_is_read_from_unique_handoff_not_stale_latest(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.assemble import _generation_process as jobs
    from scripts.tools.assemble._json_io import write_json

    output = Path(load_config(config_path)["output_dir"])
    output.mkdir()
    write_json(output / "latest.json", {"success": True, "stale": True})

    def worker(
        command: list[str], directory: Path, timeout: float, pump: object
    ) -> int:
        assert "--config" in command and "--cycle" in command
        destination = Path(command[command.index("--result-file") + 1])
        write_json(destination, {"success": False, "reason": "fresh failure"})
        return 1

    monkeypatch.setattr(jobs, "wait_process", worker)
    assert jobs.run_generation(config_path, 2, 5, lambda: None) == {
        "success": False,
        "reason": "fresh failure",
    }


def test_inverted_plan_cannot_claim_that_an_upward_mouth_is_downward(
    config_path: Path,
) -> None:
    from scripts.tools.assemble._planning import resolve_design

    with pytest.raises(ValueError, match="inverted plan requires"):
        resolve_design(
            design(orientation="inverted", target_axis=[0, 0, 1]),
            load_config(config_path)["validation"],
        )


def test_timeout_marks_live_checkpoints_failed(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.tools.assemble import _generation_process as jobs
    from scripts.tools.assemble._json_io import write_json

    output = Path(load_config(config_path)["output_dir"])
    actual_run = output / "actual_run"
    actual_run.mkdir(parents=True)
    snapshot = {
        "success": False,
        "status": "running",
        "run_directory": str(actual_run),
        "T_base_assemble": None,
    }

    def worker(
        command: list[str], directory: Path, timeout: float, pump: object
    ) -> int:
        write_json(directory / "result.json", snapshot)
        write_json(actual_run / "result.json", snapshot)
        write_json(output / "latest.json", snapshot)
        raise TimeoutError("deliberate timeout")

    monkeypatch.setattr(jobs, "wait_process", worker)
    result = jobs.run_generation(config_path, 1, 5, lambda: None)
    assert result["status"] == "failed"
    assert read_json(output / "latest.json")["status"] == "failed"
    assert read_json(actual_run / "result.json")["status"] == "failed"
