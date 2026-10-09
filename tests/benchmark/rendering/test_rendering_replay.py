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

"""Frozen visual replay identity, cadence, kinematics, and topology contracts."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import write_json
from scripts.benchmark.rendering.replay import (
    frame_at,
    load_deformation_replay,
    load_replay,
    prepare_replay,
    validate_replay,
)
from scripts.benchmark.rendering.scene_cases import fixed_views
from scripts.benchmark.rendering.scenes import build_scene, load_scene


@pytest.fixture(scope="module")
def rigid_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("rigid_replay")
    scene = build_scene("calibration", root / "scene")
    return scene, prepare_replay(scene, root / "sequence")


@pytest.fixture(scope="module")
def robot_fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    pytest.importorskip("pytorch_kinematics")
    pytest.importorskip("trimesh")
    urdf = (
        Path.home() / ".cache/embodichain_data/extract/Franka/Panda/PandaWithHand.urdf"
    )
    if not urdf.is_file():
        pytest.skip(
            "CPU robot fixture requires the locally cached, source-hashed Franka asset"
        )
    root = tmp_path_factory.mktemp("robot_replay")
    scene = build_scene("franka_workstation", root / "scene", franka_urdf=urdf)
    return scene, prepare_replay(scene, root / "sequence", franka_urdf=urdf), urdf


def test_twenty_views_are_deterministic_independent_and_pinhole(
    rigid_fixture: tuple[Path, Path],
) -> None:
    scene = load_scene(rigid_fixture[0])
    original = deepcopy(scene)
    views = fixed_views(scene)
    assert len(views) == 20
    assert views == fixed_views(scene)
    assert len({tuple(view["eye"]) for view in views}) == 20
    assert [view["view_id"] for view in views] == list(range(20))
    assert all(view["camera_model"] == scene["camera_model"] for view in views)
    assert all(view["target"] == scene["cameras"][0]["target"] for view in views)
    views[0]["camera_model"]["near_m"] = 9
    views[0]["target"][0] += 5
    assert scene == original
    assert views[1]["camera_model"] == scene["camera_model"]


def test_rigid_replay_has_exact_cadence_and_nontrivial_object_motion(
    rigid_fixture: tuple[Path, Path],
) -> None:
    scene_path, replay_path = rigid_fixture
    replay = load_replay(replay_path, scene_path)
    assert replay["kind"] == "scripted_kinematic_visual_replay"
    assert replay["physics_steps"] == 0
    assert replay["physical_validation"] == "not_performed"
    assert replay["frame_count"] == 100
    assert [frame["frame_id"] for frame in replay["frames"]] == list(range(100))
    assert [frame["time_s"] for frame in replay["frames"]] == [
        index / 30 for index in range(100)
    ]
    positions = np.asarray(
        [frame["objects"]["blue_block"] for frame in replay["frames"]]
    )[:, :3, 3]
    assert np.max(positions[:, 2]) - positions[0, 2] > 0.17
    assert np.linalg.norm(positions[-1] - positions[0]) > 0.3
    assert positions[-1] == pytest.approx([0.05, 0.18, 0.089])
    assert all(
        frame["cameras"] == replay["frames"][0]["cameras"] for frame in replay["frames"]
    )


def test_preparation_is_deterministic_and_does_not_replace_frozen_output(
    rigid_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    scene, first = rigid_fixture
    second = prepare_replay(scene, tmp_path / "second.json")
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_replay(scene, second)


def test_source_hash_change_is_rejected(
    rigid_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    scene_path, replay_path = rigid_fixture
    changed = load_scene(scene_path)
    changed["objects"][0]["world_from_object"][0][3] += 0.01
    path = write_json(tmp_path / "changed.json", changed)
    with pytest.raises(ValueError, match="source hash mismatch"):
        load_replay(replay_path, path)


def test_robot_source_hash_is_verified_before_kinematic_generation(
    rigid_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    scene = load_scene(rigid_fixture[0])
    scene["robot_source"] = {"urdf_sha256": "0" * 64}
    scene_path = write_json(tmp_path / "scene.json", scene)
    source = tmp_path / "wrong.urdf"
    source.write_text('<robot name="wrong"/>')
    with pytest.raises(ValueError, match="URDF hash mismatch"):
        prepare_replay(scene_path, tmp_path / "replay.json", franka_urdf=source)
    assert not (tmp_path / "replay.json").exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("order", "ordered"),
        ("timestamp", "timestamps"),
        ("object", "objects identities"),
        ("external_camera", "remain fixed"),
        ("scale", "orthonormal"),
        ("physical_claim", "physical validation"),
    ],
)
def test_malformed_frame_state_is_rejected(
    rigid_fixture: tuple[Path, Path], mutation: str, message: str
) -> None:
    replay = load_replay(rigid_fixture[1])
    frame = replay["frames"][10]
    if mutation == "order":
        frame["frame_id"] = 11
    elif mutation == "timestamp":
        frame["time_s"] += 0.001
    elif mutation == "object":
        frame["objects"].pop("blue_block")
    elif mutation == "external_camera":
        frame["cameras"]["external_front"]["eye"][0] += 0.01
    elif mutation == "scale":
        frame["objects"]["blue_block"][0][0] = 2
    else:
        replay["physical_validation"] = "passed"
    with pytest.raises(ValueError, match=message):
        validate_replay(replay)


def test_frame_lookup_is_exact_and_independent(
    rigid_fixture: tuple[Path, Path],
) -> None:
    replay = load_replay(rigid_fixture[1])
    frame = frame_at(replay, 99)
    frame["objects"]["blue_block"][0][3] += 8
    assert frame != replay["frames"][99]
    for invalid in (-1, 100, 0.5, True):
        with pytest.raises(IndexError):
            frame_at(replay, invalid)


def test_rigid_loading_needs_no_simulator_or_numeric_dependencies(
    rigid_fixture: tuple[Path, Path],
) -> None:
    scene, replay = rigid_fixture
    code = "from pathlib import Path; from scripts.benchmark.rendering.replay import load_replay; load_replay(Path(__import__('sys').argv[1]), Path(__import__('sys').argv[2]))"
    subprocess.run(
        [sys.executable, "-S", "-c", code, str(replay), str(scene)],
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[3],
    )


def test_robot_links_match_shared_fk_and_camera_mount_is_constant(
    robot_fixture: tuple[Path, Path, Path],
) -> None:
    import pytorch_kinematics as pk
    import torch

    scene_path, replay_path, urdf = robot_fixture
    scene, replay = load_scene(scene_path), load_replay(replay_path, scene_path)
    chain = pk.build_chain_from_urdf(urdf.read_bytes()).to(
        dtype=torch.float64, device="cpu"
    )
    names = chain.get_joint_parameter_names()
    root_pose = np.asarray(scene["robot_source"]["world_from_robot"])
    parent = scene["robot_source"]["wrist_parent"]
    eye_offsets = []
    for frame in replay["frames"]:
        hand_inverse = np.linalg.inv(frame["robot_links"][parent])
        eye_offsets.append(
            (hand_inverse @ np.r_[frame["cameras"]["wrist"]["eye"], 1])[:3]
        )
        for name in ("external_front", "external_side"):
            assert frame["cameras"][name] == replay["lineage"]["frozen_cameras"][name]
    np.testing.assert_allclose(
        eye_offsets, np.repeat(np.asarray(eye_offsets[:1]), 100, axis=0), atol=2e-6
    )
    assert (
        np.ptp([frame["cameras"]["wrist"]["eye"][0] for frame in replay["frames"]])
        > 0.2
    )
    for frame_id in (0, 49, 99):
        frame = frame_at(replay, frame_id)
        fk = chain.forward_kinematics(
            torch.tensor(
                [[frame["joint_positions"][name] for name in names]],
                dtype=torch.float64,
            )
        )
        for name, transform in fk.items():
            np.testing.assert_allclose(
                frame["robot_links"][name],
                root_pose @ transform.get_matrix()[0].numpy(),
                atol=1e-9,
            )
        for uid, binding in replay["lineage"]["robot_visual_bindings"].items():
            np.testing.assert_allclose(
                frame["objects"][uid],
                np.asarray(frame["robot_links"][binding["link"]])
                @ binding["link_from_object"],
                atol=1e-9,
            )


def test_grasp_attachment_lift_and_release_keep_real_joint_limits(
    robot_fixture: tuple[Path, Path, Path],
) -> None:
    scene, path, _ = robot_fixture
    replay = load_replay(path, scene)
    grasp = replay["lineage"]["grasp"]
    assert grasp["object_uid"] == "blue_block"
    assert replay["lineage"]["placement_target_uid"] == "tray_base"
    assert grasp["jaw_opening_m"] == 0.07
    assert grasp["contact_validation"] == "not_performed"
    for frame in replay["frames"][grasp["attach_frame"] : grasp["release_frame"] + 1]:
        hand = np.asarray(frame["robot_links"][grasp["parent_link"]])
        np.testing.assert_allclose(
            np.linalg.inv(hand) @ frame["objects"][grasp["object_uid"]],
            grasp["hand_from_object"],
            atol=1e-8,
        )
        assert frame["joint_positions"]["fr3_finger_joint1"] == pytest.approx(0.035)
    for frame in replay["frames"]:
        for name, qpos in frame["joint_positions"].items():
            lower, upper = replay["lineage"]["joint_limits"][name]
            assert lower <= qpos <= upper
    last = np.asarray(replay["frames"][-1]["objects"]["blue_block"])
    np.testing.assert_allclose(last[:3, 3], [0.05, 0.18, 0.089], atol=2e-5)
    assert replay["source"]["maximum_fk_position_error_m"] < 1e-5
    assert replay["source"]["maximum_fk_rotation_error_rad"] < 1e-5


def test_changed_wrist_attachment_is_rejected(
    robot_fixture: tuple[Path, Path, Path],
) -> None:
    scene, path, _ = robot_fixture
    replay = load_replay(path, scene)
    replay["frames"][40]["cameras"]["wrist"]["eye"][0] += 0.001
    with pytest.raises(ValueError, match="attachment changed"):
        validate_replay(replay)


def test_missing_external_deformation_is_unsupported(
    rigid_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    for absent in (None, tmp_path / "missing.json"):
        status = load_deformation_replay(absent, rigid_fixture[0])
        assert status == {
            "status": "unsupported",
            "run_status": "not_run",
            "reason": "missing_external_frozen_deformation_sequence",
        }
    with pytest.raises(ValueError, match="external frozen deformation"):
        load_deformation_replay(rigid_fixture[1], rigid_fixture[0])
    with pytest.raises(ValueError, match="regular JSON file"):
        load_deformation_replay(tmp_path, rigid_fixture[0])


@pytest.fixture
def deformation_fixture(tmp_path: Path) -> tuple[Path, Path]:
    scene_path = build_scene("calibration", tmp_path / "scene")
    scene = load_scene(scene_path)
    mesh_path = scene_path.parent / "sheet.npz"
    vertices = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]
    normals = [[0.0, 0.0, 1.0]] * 3
    np.savez(
        mesh_path,
        vertices=vertices,
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        normals=normals,
    )
    scene["objects"][0]["geometry"] = {
        "kind": "mesh",
        "path": mesh_path.name,
        "sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "vertices": 3,
        "triangles": 1,
    }
    write_json(scene_path, scene)
    path = prepare_replay(scene_path, tmp_path / "sequence", frames=20)
    replay = load_replay(path, scene_path)
    replay["kind"] = "external_frozen_deformation_replay"
    replay["physical_validation"] = "not_assessed"
    replay["source"][
        "method"
    ] = "external unit-test frozen vertex fixture; no simulated soft body"
    replay["deformation_assets"] = {
        "sheet_v1": {
            "object_uid": "floor",
            "geometry_sha256": scene["objects"][0]["geometry"]["sha256"],
            "vertex_count": 3,
        }
    }
    for frame in replay["frames"]:
        frame["deformations"] = {
            "sheet_v1": {"vertices": deepcopy(vertices), "normals": deepcopy(normals)}
        }
        frame["deformations"]["sheet_v1"]["vertices"][2][2] = frame["frame_id"] * 0.001
    write_json(path, replay)
    return scene_path, path


def test_external_deformation_loads_only_against_immutable_topology(
    deformation_fixture: tuple[Path, Path],
) -> None:
    scene_path, path = deformation_fixture
    replay = load_deformation_replay(path, scene_path)
    assert replay["frame_count"] == 20
    assert replay["frames"][-1]["deformations"]["sheet_v1"]["vertices"][2][2] == 0.019
    with pytest.raises(ValueError, match="requires its frozen scene"):
        load_replay(path)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("hash", "hash mismatch"),
        ("missing", "every topology asset"),
        ("vertices", "shape or finite"),
        ("normals", "unit vectors"),
        ("topology", "topology is immutable"),
    ],
)
def test_invalid_deformation_data_is_rejected(
    deformation_fixture: tuple[Path, Path], mutation: str, message: str
) -> None:
    scene, path = deformation_fixture
    replay = json.loads(path.read_text())
    state = replay["frames"][0]["deformations"]["sheet_v1"]
    if mutation == "hash":
        replay["deformation_assets"]["sheet_v1"]["geometry_sha256"] = "0" * 64
    elif mutation == "missing":
        replay["frames"][0]["deformations"] = {}
    elif mutation == "vertices":
        state["vertices"].pop()
    elif mutation == "normals":
        state["normals"][0] = [0.0, 0.0, 0.0]
    else:
        state["triangles"] = [[0, 2, 1]]
    write_json(path, replay)
    with pytest.raises(ValueError, match=message):
        load_deformation_replay(path, scene)
