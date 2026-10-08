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

"""Evidence and coordinate contracts for identical cross-renderer scenes."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from pathlib import PurePosixPath

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.benchmark.rendering.scenes import (
    build_scene,
    camera_intrinsics,
    camera_world_pose,
    load_scene,
    scene_fingerprint,
    validate_scene,
)
from scripts.benchmark.rendering.backends.scene_isaaclab import camera_prim_path
from scripts.benchmark.rendering.backends.scene_embodichain import (
    material_display_color,
)
from scripts.benchmark.rendering.preview import compare_geometry


def test_geometry_check_excludes_invalid_depth_and_keeps_instance_mismatch() -> None:
    first = {
        "seg": np.array([[1, 0], [2, 2]]),
        "depth": np.array([[1.0, np.inf], [2.0, 2.0]]),
    }
    second = {
        "seg": np.array([[1, 0], [2, 3]]),
        "depth": np.array([[1.1, np.inf], [2.1, 2.0]]),
    }
    result = compare_geometry(first, second)
    assert result["pixel_instance_agreement_ratio"] == 0.75
    assert result["foreground_instance_agreement_ratio"] == pytest.approx(2 / 3)
    assert result["common_depth_pixels"] == 2
    assert result["depth_median_absolute_difference_m"] == pytest.approx(0.1)


def test_linear_material_is_encoded_for_native_display_srgb_boundary() -> None:
    # The 18% linear grey card is 0.461356 in display-sRGB, not 0.18.
    assert material_display_color([0.0, 0.18, 1.0]) == pytest.approx(
        [0.0, 0.4613561295, 1.0]
    )


def test_vectorized_camera_has_one_parent_per_exposure() -> None:
    # Isaac Camera derives its instance count from matching parent prims.
    # Sharing one parent across three camera prims makes initialization fail.
    paths = [camera_prim_path(env, camera) for env in range(2) for camera in range(3)]
    assert len({PurePosixPath(path).parent for path in paths}) == 6


def test_frozen_scene_is_portable_and_sensitive_to_camera_changes(
    tmp_path: Path,
) -> None:
    first = load_scene(build_scene("calibration", tmp_path / "first"))
    second = load_scene(build_scene("calibration", tmp_path / "second"))
    assert scene_fingerprint(first) == scene_fingerprint(second)
    second["cameras"][0]["eye"][0] += 0.1
    assert scene_fingerprint(first) != scene_fingerprint(second)


@pytest.mark.parametrize("field", ["uid", "instance_id"])
def test_duplicate_object_identity_is_rejected(tmp_path: Path, field: str) -> None:
    scene = load_scene(build_scene("calibration", tmp_path / "scene"))
    scene["objects"][1][field] = scene["objects"][0][field]
    with pytest.raises(ValueError, match="Duplicate object"):
        validate_scene(scene)


def test_geometry_evidence_cannot_silently_change(tmp_path: Path) -> None:
    path = build_scene("calibration", tmp_path / "scene")
    scene = load_scene(path)
    mesh = path.parent / "mesh.npz"
    mesh.write_bytes(b"frozen mesh evidence")
    scene["objects"][0]["geometry"] = {
        "kind": "mesh",
        "path": mesh.name,
        "sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(scene))
    assert load_scene(path)["objects"][0]["geometry"]["kind"] == "mesh"
    mesh.write_bytes(b"different geometry")
    with pytest.raises(ValueError, match="Geometry hash mismatch"):
        load_scene(path)


def test_look_at_pose_uses_xyzw_and_preserves_camera_up() -> None:
    camera = {"eye": [1.2, -0.7, 0.9], "target": [0.2, 0.4, 0.1], "up": [0.3, 0.2, 1.0]}
    position, quaternion = camera_world_pose(camera)
    rotation = Rotation.from_quat(quaternion)
    expected_forward = np.subtract(camera["target"], camera["eye"])
    expected_forward /= np.linalg.norm(expected_forward)
    assert position == camera["eye"]
    np.testing.assert_allclose(rotation.apply([0, 0, -1]), expected_forward, atol=1e-12)
    assert np.dot(rotation.apply([0, 1, 0]), camera["up"]) > 0
    assert rotation.as_matrix().T @ rotation.as_matrix() == pytest.approx(np.eye(3))


def test_invalid_look_at_does_not_reach_renderer(tmp_path: Path) -> None:
    scene = load_scene(build_scene("calibration", tmp_path / "scene"))
    scene["cameras"][0]["target"] = copy.deepcopy(scene["cameras"][0]["eye"])
    with pytest.raises(ValueError, match="nondegenerate"):
        validate_scene(scene)


def test_non_square_output_preserves_square_pixels(tmp_path: Path) -> None:
    scene = load_scene(build_scene("calibration", tmp_path / "scene"))
    k = camera_intrinsics(scene, 1920, 1080)
    assert k[0][0] == k[1][1]
    assert [k[0][2], k[1][2]] == [960.0, 540.0]
    assert k[2] == [0.0, 0.0, 1.0]


def test_preview_prepare_and_help_do_not_require_a_simulator(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[3]
    output = tmp_path / "preview"
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.benchmark",
            "rendering-preview",
            "--scene",
            "calibration",
            "--prepare-only",
            "--output",
            str(output),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    review = json.loads((output / "review.json").read_text())
    assert review["approval_status"] == "pending_user_review"
    assert review["benchmark_executed"] is False
    assert not list(output.rglob("metrics.json"))
