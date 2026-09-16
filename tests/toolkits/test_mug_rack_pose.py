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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

pytest.importorskip("open3d")
pytest.importorskip("fcl")
pytest.importorskip("manifold3d")

from scripts.tools.mug_rack_pose._collision import ExactCollision, HullCollision
from scripts.tools.mug_rack_pose._geometry import (
    find_tips,
    frame_from_up,
    infer_cavity,
    load_mesh,
    placement_pose,
)
from scripts.tools.mug_rack_pose._search import main, validate_pose


@pytest.fixture
def cup() -> trimesh.Trimesh:
    """A 9 cm hollow cup with a 6 mm bottom, generated without external assets."""
    profile = [
        [0, 0],
        [0.04, 0],
        [0.04, 0.09],
        [0.035, 0.09],
        [0.035, 0.006],
        [0, 0.006],
        [0, 0],
    ]
    return trimesh.creation.revolve(profile, sections=64)


@pytest.fixture
def rack() -> trimesh.Trimesh:
    """A base with a tall central trunk and a shorter offset support post."""
    base = trimesh.creation.box(extents=[0.26, 0.20, 0.02])
    base.apply_translation([0, 0, 0.01])
    trunk = trimesh.creation.cylinder(
        radius=0.008, segment=[[0, 0, 0.01], [0, 0, 0.28]], sections=32
    )
    branch = trimesh.creation.cylinder(
        radius=0.006, segment=[[0.1, 0, 0.01], [0.1, 0, 0.2]], sections=32
    )
    return trimesh.boolean.union([base, trunk, branch], engine="manifold")


def test_cavity_inference_survives_arbitrary_local_rotation_and_translation(
    cup: trimesh.Trimesh,
) -> None:
    rotation = Rotation.from_euler("xyz", [37, -23, 71], degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [0.3, -0.2, 0.1]
    cup.apply_transform(transform)
    cavity = infer_cavity(cup)
    assert cavity.basis[:, 2] @ rotation[:, 2] > 0.99999
    assert cavity.depth == pytest.approx(0.084, abs=0.001)
    assert cavity.radius == pytest.approx(0.035, abs=0.004)


@pytest.mark.parametrize("kind", ["solid", "through_hole"])
def test_rejects_solid_cup_and_handle_like_through_hole(kind: str) -> None:
    if kind == "solid":
        mesh = trimesh.creation.cylinder(radius=0.04, height=0.09, sections=64)
    else:
        mesh = trimesh.creation.annulus(
            r_min=0.03, r_max=0.04, height=0.09, sections=64
        )
    with pytest.raises(ValueError, match="No enclosed deep cup cavity"):
        infer_cavity(mesh)


def test_detects_branch_and_rejects_central_trunk(rack: trimesh.Trimesh) -> None:
    tips, _ = find_tips(rack, np.array([0, 0, 1]), radius=0.035)
    assert len(tips) == 1
    np.testing.assert_allclose(tips[0], [0.1, 0, 0.2], atol=1e-6)


def test_contact_and_world_frame_composition(
    cup: trimesh.Trimesh, rack: trimesh.Trimesh
) -> None:
    cavity = infer_cavity(cup)
    up = np.array([0, 0, 1])
    tip = np.array([0.1, 0, 0.2])
    gap, probe = 0.00005, 0.0005
    pose = placement_pose(cavity, tip, up, yaw=0.3, gap=gap)
    check = validate_pose(
        pose, tip, cavity, rack, cup, ExactCollision(rack, cup), up, gap, probe, 0.003
    )
    assert check["accepted"]
    assert check["intersection_volume_m3"] == 0
    assert check["down_probe_intersection_m3"] > check["volume_tolerance_m3"]
    assert check["up_probe_intersection_m3"] == 0
    world_rack = np.eye(4)
    world_rack[:3, :3] = Rotation.from_euler("xyz", [0.2, 0.4, -0.1]).as_matrix()
    world_rack[:3, 3] = [1, 2, 3]
    world_floor = world_rack @ pose @ np.r_[cavity.floor, 1]
    expected = world_rack @ np.r_[tip + gap * up, 1]
    np.testing.assert_allclose(world_floor, expected, atol=1e-9)


def test_containment_detected_by_solids_even_without_triangle_intersection() -> None:
    outer = trimesh.creation.box(extents=[1, 1, 1])
    inner = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    exact = ExactCollision(outer, inner)
    assert not exact.surface_collision(np.eye(4))
    assert exact.intersection_volume(np.eye(4)) == pytest.approx(0.001)
    assert HullCollision([outer], [inner]).collides(np.eye(4))


def test_seating_with_both_asset_frames_rotated_and_offset(
    cup: trimesh.Trimesh,
    rack: trimesh.Trimesh,
) -> None:
    rack_transform = np.eye(4)
    rack_transform[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.7]).as_matrix()
    rack_transform[:3, 3] = [0.2, -0.1, 0.3]
    mug_transform = np.eye(4)
    mug_transform[:3, :3] = Rotation.from_euler("xyz", [-0.8, 0.6, 1.2]).as_matrix()
    mug_transform[:3, 3] = [-0.2, 0.2, -0.1]
    rack.apply_transform(rack_transform)
    cup.apply_transform(mug_transform)
    tip = (rack_transform @ [0.1, 0, 0.2, 1])[:3]
    up = rack_transform[:3, 2]
    cavity = infer_cavity(cup)
    pose = placement_pose(cavity, tip, up, yaw=0, gap=0.00005)
    check = validate_pose(
        pose,
        tip,
        cavity,
        rack,
        cup,
        ExactCollision(rack, cup),
        up,
        0.00005,
        0.0005,
        0.003,
    )
    assert check["accepted"]


def test_hull_overlap_is_refined_and_cli_writes_meter_pose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cup: trimesh.Trimesh,
    rack: trimesh.Trimesh,
) -> None:
    from scripts.tools.mug_rack_pose import _search as generate

    # Deliberately over-approximate the complete cup with one solid convex hull.
    # The cavity disappears, but original-mesh verification must still succeed.
    monkeypatch.setattr(generate, "decompose", lambda mesh, *args: [mesh.convex_hull])
    mug_path, rack_path = tmp_path / "cup.obj", tmp_path / "rack.obj"
    cup.apply_scale(1000)
    cup.export(mug_path)
    rack.export(rack_path)
    output = tmp_path / "result"
    status = main(
        [
            "--rack",
            str(rack_path),
            "--mug",
            str(mug_path),
            "--output",
            str(output),
            "--mug-scale",
            "0.001",
            "--yaw-samples",
            "4",
        ]
    )
    assert status == 0
    pose = np.load(output / "relative_pose.npy")
    assert pose.shape == (4, 4)
    assert pose[2, 3] == pytest.approx(0.20605, abs=1e-6)
    report = json.loads((output / "result.json").read_text())
    assert report["solutions"][0]["visacd_overlap"] is True
    assert report["solutions"][0]["validation"]["accepted"] is True
    assert (output / "assembly.glb").is_file()


def test_no_solution_emits_diagnostics_without_a_pose(
    tmp_path: Path, cup: trimesh.Trimesh
) -> None:
    solid = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    rack_path, mug_path = tmp_path / "solid.obj", tmp_path / "cup.obj"
    solid.export(rack_path)
    cup.export(mug_path)
    output = tmp_path / "result"
    status = main(
        [
            "--rack",
            str(rack_path),
            "--mug",
            str(mug_path),
            "--output",
            str(output),
            "--collision",
            "exact",
            "--tip",
            "0",
            "0",
            "0.1",
            "--yaw-samples",
            "2",
        ]
    )
    assert status == 2
    assert json.loads((output / "result.json").read_text())["success"] is False
    assert not (output / "relative_pose.npy").exists()


def test_load_welds_obj_seams_but_does_not_fill_an_open_surface(
    tmp_path: Path, cup: trimesh.Trimesh
) -> None:
    path = tmp_path / "cup.obj"
    unmerged = cup.copy()
    unmerged.unmerge_vertices()
    unmerged.export(path)
    assert load_mesh(path).is_volume
    cup.update_faces(np.arange(len(cup.faces) - 1))
    cup.export(path)
    with pytest.raises(ValueError, match="not watertight"):
        load_mesh(path)


@pytest.mark.parametrize("axis", [[0, 0, 0], [float("nan"), 0, 1]])
def test_invalid_frame_axis_is_rejected(axis: list[float]) -> None:
    with pytest.raises(ValueError, match="nonzero"):
        frame_from_up(np.array(axis))


@pytest.fixture
def json_job(tmp_path: Path, cup: trimesh.Trimesh, rack: trimesh.Trimesh) -> Path:
    cup.export(tmp_path / "cup.obj")
    rack.export(tmp_path / "rack.obj")
    job = {
        "rack": {"path": "rack.obj"},
        "mug": {"path": "cup.obj"},
        "output_json": "output/result.json",
        "search": {"collision": "exact", "yaw_samples": 4},
        "codex": {"max_turns": 2},
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    return path


def _decision(action: str, candidate_id: int | None = None) -> dict:
    return {
        "action": action,
        "candidate_id": candidate_id,
        "reason": "测试选择",
        "yaw_samples": None,
        "mug_up": None,
        "tips": None,
    }


def test_json_paths_use_input_directory_not_process_cwd(
    json_job: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.tools.mug_rack_pose._json_io import load_request

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    request = load_request(json_job)
    assert request["rack"]["path"] == str(tmp_path / "rack.obj")
    assert request["output_json"] == str(tmp_path / "output/result.json")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"rack": {"path": "rack.obj", "scle": 1}}, "Unknown rack"),
        ({"rack": {"path": "rack.obj", "scale": True}}, "finite and positive"),
        ({"codex": {"max_turns": 1}}, "max_turns"),
        ({"output_json": "rack.obj"}, "must differ"),
        ({"mug": {"path": "cup.obj", "up": [0, 0, 0]}}, "cannot be zero"),
    ],
)
def test_invalid_json_job_is_rejected(
    json_job: Path, change: dict, message: str
) -> None:
    from scripts.tools.mug_rack_pose._json_io import load_request

    value = json.loads(json_job.read_text())
    value.update(change)
    json_job.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        load_request(json_job)


def test_codex_decision_loop_passes_geometry_observations_and_writes_selected_pose(
    json_job: Path,
) -> None:
    from scripts.tools.mug_rack_pose.generate import run_harness
    from scripts.tools.mug_rack_pose._json_io import load_request
    from scripts.tools.mug_rack_pose.visualize import load_placement

    calls = []

    def decide(prompt: str, *_args: object) -> dict:
        calls.append(prompt)
        if len(calls) == 1:
            return _decision("search")
        assert '"candidate_id": 0' in prompt
        assert '"intersection_volume_m3": 0.0' in prompt
        return _decision("finish", 0)

    request = load_request(json_job)
    result = run_harness(request, decide=decide)
    assert len(calls) == 2
    assert result["success"] is True
    assert result["validation"]["accepted"] is True
    saved = json.loads(Path(request["output_json"]).read_text())
    assert saved["T_rack_mug"][2][3] == pytest.approx(0.20605, abs=1e-6)
    inputs, pose = load_placement(Path(request["output_json"]))
    assert inputs["mug"] == str(json_job.parent / "cup.obj")
    np.testing.assert_allclose(pose, result["T_rack_mug"])


def test_independent_validation_rejects_a_forged_candidate(
    json_job: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.tools.mug_rack_pose import generate
    from scripts.tools.mug_rack_pose._json_io import load_request

    original = generate._search

    def corrupt(*args: object) -> dict:
        report = original(*args)
        # The old accepted flag stays True, but the pose is moved off the support.
        report["solutions"][0]["T_rack_mug"][0][3] += 0.5
        return report

    monkeypatch.setattr(generate, "_search", corrupt)
    actions = iter([_decision("search"), _decision("finish", 0)])
    result = generate.run_harness(
        load_request(json_job), decide=lambda *_: next(actions)
    )
    assert result["success"] is False
    assert result["T_rack_mug"] is None
    assert (
        "independent geometric validation"
        in result["trace"][-1]["observation"]["error"]
    )


def test_codex_failure_replaces_stale_success_with_json_error(json_job: Path) -> None:
    from scripts.tools.mug_rack_pose.generate import run_harness
    from scripts.tools.mug_rack_pose._json_io import load_request

    request = load_request(json_job)
    path = Path(request["output_json"])
    path.parent.mkdir()
    path.write_text(json.dumps({"success": True, "T_rack_mug": np.eye(4).tolist()}))

    def fail(*_args: object) -> dict:
        raise RuntimeError("Codex unavailable")

    run_harness(request, decide=fail)
    saved = json.loads(path.read_text())
    assert saved["status"] == "error"
    assert saved["success"] is False
    assert saved["T_rack_mug"] is None


def test_viewer_rejects_unsuccessful_or_nonrigid_result(tmp_path: Path) -> None:
    from scripts.tools.mug_rack_pose.visualize import load_placement

    path = tmp_path / "result.json"
    path.write_text(json.dumps({"schema": "codex-mug-rack-pose/v1", "success": False}))
    with pytest.raises(ValueError, match="successful"):
        load_placement(path)
    reflection = np.eye(4)
    reflection[0, 0] = -1
    path.write_text(
        json.dumps(
            {
                "schema": "codex-mug-rack-pose/v1",
                "success": True,
                "validation": {"accepted": True},
                "T_rack_mug": reflection.tolist(),
            }
        )
    )
    with pytest.raises(ValueError, match="right-handed"):
        load_placement(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "1e999"])
def test_json_reader_rejects_nonfinite_numbers(tmp_path: Path, literal: str) -> None:
    from scripts.tools.mug_rack_pose._json_io import read_json

    path = tmp_path / "invalid.json"
    path.write_text('{"value": ' + literal + "}")
    with pytest.raises(ValueError, match="Non-finite"):
        read_json(path)
