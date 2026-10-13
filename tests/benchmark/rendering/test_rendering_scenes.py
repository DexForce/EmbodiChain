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
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from pathlib import PurePosixPath
from types import SimpleNamespace

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
from scripts.benchmark.rendering import preview
from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.rendering.color import linear_to_srgb, linear_u8_to_srgb_u8


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
    assert "render_profile" not in review
    assert not (output / "render_profile.json").exists()


def test_custom_preview_profile_is_copied_hashed_and_reused_without_approval(
    tmp_path: Path,
) -> None:
    profile = {
        "name": "fixture-probe",
        "frame_time_s": 1 / 30,
        "isaaclab": {"carb_settings": {"/rtx/post/tonemap/op": 0}},
        "lighting_controls": {
            "isaaclab": {"directional_intensity": 3, "environment_intensity": 0.15}
        },
    }
    source = tmp_path / "input-profile.json"
    source.write_text(json.dumps(profile, indent=4) + "\n")
    output = tmp_path / "preview"
    preview.main(
        [
            "--scene",
            "calibration",
            "--prepare-only",
            "--output",
            str(output),
            "--render-profile",
            str(source),
        ]
    )
    frozen = output / "render_profile.json"
    assert frozen.read_bytes() == source.read_bytes()
    review = json.loads((output / "review.json").read_text())
    assert review["render_profile"] == {
        "path": "render_profile.json",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "profile_sha256": stable_hash(profile),
    }
    scene_hash = review["scenes"][0]["sha256"]
    assert review["approval_status"] == "pending_user_review"
    assert review["benchmark_executed"] is False
    # Reusing the prepared geometry also reuses its explicitly frozen profile.
    preview.main(
        [
            "--scene",
            "calibration",
            "--prepare-only",
            "--reuse-scenes",
            "--output",
            str(output),
        ]
    )
    assert (
        json.loads((output / "review.json").read_text())["scenes"][0]["sha256"]
        == scene_hash
    )
    source.write_text(json.dumps({**profile, "name": "different"}))
    with pytest.raises(FileExistsError, match="different render profile"):
        preview.main(
            [
                "--scene",
                "calibration",
                "--prepare-only",
                "--reuse-scenes",
                "--output",
                str(output),
                "--render-profile",
                str(source),
            ]
        )
    assert json.loads(frozen.read_text()) == profile


def test_preview_launcher_passes_only_frozen_profile_to_both_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = {"name": "fixture-probe", "frame_time_s": 1 / 30}
    source = tmp_path / "probe.json"
    source.write_text(json.dumps(profile))
    isaac_root = tmp_path / "isaac"
    (isaac_root / "source").mkdir(parents=True)
    commands = []

    def execute(command: list[str], *args: object, **kwargs: object) -> dict[str, str]:
        commands.append(command)
        return {"status": "completed"}

    monkeypatch.setattr(preview, "execute_worker", execute)
    monkeypatch.setattr(preview, "write_comparison", lambda *args: [])
    output = tmp_path / "preview"
    preview.main(
        [
            "--scene",
            "calibration",
            "--output",
            str(output),
            "--render-profile",
            str(source),
            "--isaaclab-root",
            str(isaac_root),
            "--isaaclab-python",
            sys.executable,
        ]
    )
    assert {command[command.index("--worker") + 1] for command in commands} == {
        "embodichain",
        "isaaclab",
    }
    for command in commands:
        assert command[command.index("--render-profile") + 1] == str(
            output / "render_profile.json"
        )
        assert (
            command[command.index("--render-profile-sha256") + 1]
            == hashlib.sha256(source.read_bytes()).hexdigest()
        )


@pytest.mark.parametrize("backend", ["embodichain", "isaaclab"])
@pytest.mark.parametrize("custom", [False, True])
def test_preview_worker_profile_forwarding_preserves_default_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, custom: bool
) -> None:
    import importlib

    scene_path = build_scene("calibration", tmp_path / "scene")
    received = []
    profile = {"name": "fixture-probe", "frame_time_s": 1 / 30}
    profile_path = tmp_path / "render_profile.json"
    profile_path.write_text(json.dumps(profile))

    class FixtureScene:
        def __init__(self, manifest: Path, **kwargs: object) -> None:
            received.append(kwargs)
            self.metadata = {
                "camera_names": [
                    item["name"] for item in load_scene(manifest)["cameras"]
                ]
            }

        def capture(self) -> dict[str, np.ndarray]:
            return {
                "rgb": np.arange(2 * 6 * 8 * 3, dtype=np.uint8).reshape(2, 6, 8, 3),
                "depth": np.ones((2, 6, 8), dtype=np.float32),
                "seg": np.ones((2, 6, 8), dtype=np.int32),
            }

        def close(self) -> None:
            pass

    module = importlib.import_module(
        f"scripts.benchmark.rendering.backends.scene_{backend}"
    )
    monkeypatch.setattr(
        module,
        "EmbodiChainScene" if backend == "embodichain" else "IsaacLabScene",
        FixtureScene,
    )
    args = argparse.Namespace(
        output=tmp_path / "worker",
        worker=backend,
        manifest=scene_path,
        width=8,
        height=6,
        gpu_id=0,
        settle_frames=1,
        isaaclab_root=tmp_path,
        render_profile=profile_path if custom else None,
        render_profile_sha256=(
            hashlib.sha256(profile_path.read_bytes()).hexdigest() if custom else None
        ),
    )
    preview._worker(args)
    record = json.loads((args.output / "preview.json").read_text())
    assert record["status"] == "completed"
    assert record["approval_status"] == "pending_user_review"
    assert record["benchmark_executed"] is False
    if custom:
        assert received[0]["profile"] == profile
        assert record["render_profile"]["profile_sha256"] == stable_hash(profile)
    else:
        assert "profile" not in received[0]
        assert "render_profile" not in record


def test_profile_checksum_mismatch_stops_before_adapter_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering.backends import scene_isaaclab

    called = []
    monkeypatch.setattr(
        scene_isaaclab, "IsaacLabScene", lambda *args, **kwargs: called.append(True)
    )
    profile = tmp_path / "profile.json"
    profile.write_text('{"name":"changed"}')
    args = argparse.Namespace(
        output=tmp_path / "worker",
        worker="isaaclab",
        render_profile=profile,
        render_profile_sha256="not-the-file-hash",
    )
    with pytest.raises(SystemExit):
        preview._worker(args)
    assert not called
    record = json.loads((args.output / "preview.json").read_text())
    assert record["status"] == "failed"
    assert "checksum mismatch" in record["error"]


def test_shared_color_transfer_preserves_device_and_native_helper_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from scripts.benchmark.rendering.backends.scene_embodichain import (
        _linear_u8_to_srgb_u8,
    )

    assert _linear_u8_to_srgb_u8 is linear_u8_to_srgb_u8
    source = torch.tensor([0, 1, 38, 128, 255], dtype=torch.uint8)
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda *args: pytest.fail("Unexpected host transfer")
    )
    result = linear_u8_to_srgb_u8(source, torch)
    assert result.device == source.device
    assert torch.equal(result, torch.tensor([0, 13, 108, 188, 255], dtype=torch.uint8))
    grey = linear_to_srgb(torch.tensor([0.0, 0.18, 1.0]), torch)
    assert torch.allclose(grey, torch.tensor([0.0, 0.46135613, 1.0]), atol=1e-6)


def test_isaac_clamp_conversion_is_explicit_verified_and_has_no_host_readback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from scripts.benchmark.rendering.backends.scene_isaaclab import (
        IsaacLabScene,
        _clamp_rgb_conversion,
    )

    observed = {
        "/rtx/post/tonemap/op": 0,
        "/rtx/post/colorcorr/enabled": False,
        "/rtx/post/colorgrad/enabled": False,
        "/rtx/post/tonemap/ocio/enabled": False,
    }
    assert _clamp_rgb_conversion({}, observed) is False
    assert _clamp_rgb_conversion({"/rtx/post/tonemap/op": 6}, observed) is False
    adapter = IsaacLabScene.__new__(IsaacLabScene)
    adapter._clamp_rgb_requested = False
    adapter.torch = torch
    source = torch.tensor([[[38, 38, 38]]], dtype=torch.uint8)
    assert adapter._canonical_rgb(source) is source
    adapter._clamp_rgb_requested = True
    adapter._post_camera_overrides = {"/rtx/post/tonemap/op": 0}
    adapter.sim = SimpleNamespace(get_setting=observed.get)
    adapter.metadata = {}
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda *args: pytest.fail("Unexpected host transfer")
    )
    converted = adapter._canonical_rgb(source)
    assert torch.equal(converted, torch.full_like(source, 108))
    assert torch.equal(source, torch.full_like(source, 38))
    assert adapter.metadata["rgb_conversion_verified_settings"] == observed
    for path in observed:
        changed = dict(observed)
        changed[path] = 6 if path.endswith("/op") else True
        with pytest.raises(RuntimeError, match="requires observed"):
            _clamp_rgb_conversion(adapter._post_camera_overrides, changed)
