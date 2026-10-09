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

"""Real metric behavior, independent truth requirements and temporal rejection."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import stable_hash
from scripts.benchmark.rendering import quality
from scripts.benchmark.rendering.quality import (
    default_protocol,
    evaluate_frame,
    evaluate_sequence,
)
from scripts.benchmark.rendering.reference import (
    DEPTH_SEMANTICS,
    SEGMENTATION_SEMANTICS,
)
from scripts.benchmark.rendering.quality_protocol import (
    protocol_for_modalities,
    protocol_requirements,
)


def _frame(size: int = 16, index: int = 0) -> dict[str, Any]:
    """Analytical test plane; synthetic fixtures never claim render provenance."""
    return {
        "rgb": np.full((size, size, 3), 0.5, dtype=np.float32),
        "depth": np.ones((size, size), dtype=np.float32),
        "seg": np.ones((size, size), dtype=np.int32),
        "metadata": {
            "rgb_encoding": "srgb_float",
            "depth_semantics": DEPTH_SEMANTICS,
            "segmentation_semantics": SEGMENTATION_SEMANTICS,
            "scene_sha256": "analytical-test-scene",
            "profile_sha256": "test-profile",
            "instance_ids": [1, 2],
            "backend": "embodichain",
            "frame_id": index,
            "time_s": index / 60,
            "camera": {
                "convention": "opencv",
                "intrinsics": [[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]],
                "world_from_camera": np.eye(4).tolist(),
            },
        },
    }


def _no_lpips(*args: Any) -> Any:
    raise FileNotFoundError("test: pretrained weights are absent")


@pytest.fixture(autouse=True)
def no_model_download(monkeypatch: pytest.MonkeyPatch) -> None:
    # Most cases exercise geometry/temporal math; real learned LPIPS is tested
    # separately when the optional package and official cached weights exist.
    monkeypatch.setattr(quality, "_lpips", _no_lpips)


def test_real_ssim_and_depth_error_have_known_values() -> None:
    reference = _frame()
    candidate = copy.deepcopy(reference)
    candidate["rgb"][:] = 0.25
    candidate["depth"] += 0.03
    result = evaluate_frame(candidate, reference)
    # Constant-image SSIM reduces to the luminance term with K1=.01, L=1.
    expected_ssim = (2 * 0.25 * 0.5 + 0.01**2) / (0.25**2 + 0.5**2 + 0.01**2)
    assert result["metrics"]["ssim"]["value"] == pytest.approx(expected_ssim)
    assert result["metrics"]["depth_rmse_m"]["value"] == pytest.approx(0.03, abs=1e-7)
    assert result["metrics"]["depth_rmse_m"]["status"] == "failed"


def test_missing_depth_cannot_hide_behind_small_common_pixel_error() -> None:
    reference = _frame()
    candidate = copy.deepcopy(reference)
    candidate["depth"][:8] = np.inf
    result = evaluate_frame(candidate, reference)
    assert result["metrics"]["depth_rmse_m"]["value"] == 0
    assert result["metrics"]["depth_valid_coverage"]["value"] == 0.5
    assert result["metrics"]["depth_valid_coverage"]["status"] == "failed"


def test_canonical_ids_are_not_matched_or_hidden_by_background() -> None:
    reference = _frame()
    reference["seg"][:] = 0
    reference["depth"][:] = np.inf
    reference["seg"][4:8, 4:8] = 1
    reference["depth"][4:8, 4:8] = 1
    candidate = copy.deepcopy(reference)
    candidate["seg"][4:8, 4:8] = 2
    result = evaluate_frame(candidate, reference)
    assert result["metrics"]["instance_iou"]["value"] == 0
    assert result["metrics"]["instance_iou"]["denominator"] == 2
    assert result["metrics"]["silhouette_iou"]["value"] == 1
    assert result["metrics"]["instance_iou"]["background_included"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("depth_semantics", "distance_to_camera"),
        ("segmentation_semantics", "renderer_native_ids"),
        ("instance_ids", [42]),
    ],
)
def test_wrong_geometry_semantics_or_ids_are_rejected(key: str, value: Any) -> None:
    candidate, reference = _frame(), _frame()
    candidate["metadata"][key] = value
    result = evaluate_frame(candidate, reference)
    assert result["checks"]["geometry_semantics"]["status"] == "failed"
    assert result["metrics"]["depth_rmse_m"]["value"] is None
    assert result["quality_status"] == "not_qualified"


def test_unknown_rgb_encoding_is_not_inferred() -> None:
    candidate, reference = _frame(), _frame()
    del candidate["metadata"]["rgb_encoding"]
    result = evaluate_frame(candidate, reference)
    assert result["checks"]["rgb_semantics"]["status"] == "failed"
    assert result["metrics"]["ssim"]["value"] is None


def test_linear_and_display_srgb_are_compared_in_declared_common_space() -> None:
    reference, candidate = _frame(), _frame()
    reference["metadata"]["rgb_encoding"] = "linear_rec709"
    reference["rgb"][:] = 0.18
    candidate["rgb"][:] = 1.055 * 0.18 ** (1 / 2.4) - 0.055
    assert evaluate_frame(candidate, reference)["metrics"]["ssim"][
        "value"
    ] == pytest.approx(1)


def test_missing_reference_or_weights_never_qualifies(tmp_path: Path) -> None:
    result = evaluate_frame(_frame(), None, output_dir=tmp_path / "missing")
    assert result["quality_status"] == "not_qualified"
    assert result["metrics"]["ssim"]["value"] is None
    assert (tmp_path / "missing/quality_evidence.json").is_file()
    result = evaluate_frame(_frame(), _frame())
    assert result["metrics"]["ssim"]["value"] == 1
    assert result["metrics"]["lpips"]["value"] is None
    assert "weights are absent" in result["metrics"]["lpips"]["reason"]
    assert result["quality_status"] == "not_qualified"
    json.dumps(result, allow_nan=False)


def test_a_tested_backend_cannot_be_ground_truth() -> None:
    reference = _frame()
    provenance = {
        "kind": "independent_offline",
        "renderer": "isaaclab",
        "assets_verified": True,
        "implementation_sha256": "example",
        "source_scene_sha256": "analytical-test-scene",
    }
    provenance["reference_id"] = stable_hash(provenance)
    reference["reference_provenance"] = provenance
    result = evaluate_frame(_frame(), reference)
    assert result["checks"]["independent_reference"]["status"] == "failed"


def test_forged_pass_bit_does_not_override_reference_noise() -> None:
    reference = _frame()
    reference["reference_validation"] = {
        "metric": "independent_seed_display_srgb_rmse",
        "value": 0.5,
        "seeds": [1, 2],
        "passed": True,
    }
    result = evaluate_frame(_frame(), reference)
    assert result["checks"]["reference_noise_floor"]["status"] == "failed"


def test_scene_and_camera_mismatch_are_visible_failures() -> None:
    reference, candidate = _frame(), _frame()
    candidate["metadata"]["scene_sha256"] = "different"
    candidate["metadata"]["camera"]["world_from_camera"][0][3] = 1
    result = evaluate_frame(candidate, reference)
    assert result["checks"]["scene_binding"]["status"] == "failed"
    assert result["checks"]["camera_binding"]["status"] == "failed"


def test_metric_crops_and_errors_keep_checksums(tmp_path: Path) -> None:
    result = evaluate_frame(_frame(), _frame(), output_dir=tmp_path)
    assert result["crops"][0]["instance_id"] == 1
    assert any(item["kind"] == "error_maps" for item in result["artifacts"])
    for item in result["artifacts"]:
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def test_thresholds_are_frozen_before_measurement() -> None:
    protocol = default_protocol()
    protocol["frame_thresholds"]["ssim"][1] = 0
    with pytest.raises(ValueError, match="frozen"):
        evaluate_frame(_frame(), _frame(), protocol)


def test_qualification_derives_from_every_required_check() -> None:
    result = {
        "checks": {"reference": {"status": "passed"}},
        "metrics": {
            name: {"status": "passed"}
            for name in default_protocol()["frame_thresholds"]
        },
    }
    assert quality._finish(result)["quality_status"] == "qualified"
    result["metrics"]["lpips"]["status"] = "missing"
    result = quality._finish(result)
    assert result["quality_status"] == "not_qualified"
    assert result["qualification"]["missing_checks"] == ["lpips"]


def test_original_full_protocol_hash_stays_frozen() -> None:
    assert (
        stable_hash(default_protocol())
        == "fb93d8353c350a9ff42358ae936ee218f374d3d8a8b96286985888b3eaa2b4d4"
    )
    for modalities in (("rgb",), ("rgb", "depth"), ("rgb", "normals")):
        policy = protocol_for_modalities(modalities)
        assert policy["frame_thresholds"] == default_protocol()["frame_thresholds"]
        assert policy["lpips"] == default_protocol()["lpips"]


def test_rgb_policy_explicitly_excludes_unrequested_geometry(tmp_path: Path) -> None:
    frame = _frame()
    del frame["depth"], frame["seg"]
    policy = protocol_for_modalities(("rgb",))
    result = evaluate_frame(frame, frame, policy, output_dir=tmp_path)
    assert result["required_metrics"] == ["ssim", "lpips"]
    assert result["checks"]["requested_outputs"]["status"] == "passed"
    assert result["metrics"]["ssim"]["value"] == 1
    for name in set(default_protocol()["frame_thresholds"]) - {"ssim", "lpips"}:
        metric = result["metrics"][name]
        assert metric["value"] is None
        assert metric["status"] == "not_requested"
        assert metric["reason"] == "modality_not_requested"
        assert metric["required"] is False
        assert name not in result["qualification"]["missing_checks"]
    assert "geometry_semantics" not in result["required_checks"]
    assert any(item["kind"] == "error_maps" for item in result["artifacts"])


def test_rgbd_policy_requires_depth_but_does_not_require_instance_ids() -> None:
    frame = _frame()
    del frame["seg"], frame["metadata"]["instance_ids"]
    policy = protocol_for_modalities(("rgb", "depth"))
    result = evaluate_frame(frame, frame, policy)
    assert result["metrics"]["depth_rmse_m"]["value"] == 0
    assert result["metrics"]["instance_iou"]["status"] == "not_requested"
    assert result["checks"]["depth_semantics"]["status"] == "passed"
    del frame["depth"]
    result = evaluate_frame(frame, frame, policy)
    assert result["checks"]["requested_outputs"]["status"] == "missing"
    assert result["metrics"]["depth_rmse_m"]["status"] == "missing"
    assert "depth_rmse_m" in result["qualification"]["missing_checks"]
    assert result["quality_status"] == "not_qualified"


def test_normals_are_required_outputs_but_quality_is_explicitly_unassessed() -> None:
    frame = _frame()
    del frame["depth"], frame["seg"]
    policy = protocol_for_modalities(("rgb", "normals"))
    missing = evaluate_frame(frame, frame, policy)
    assert missing["checks"]["requested_outputs"]["status"] == "missing"
    frame["normals"] = np.zeros_like(frame["rgb"], dtype=np.float32)
    result = evaluate_frame(frame, frame, policy)
    assert result["checks"]["requested_outputs"]["status"] == "passed"
    assert result["assessed_modalities"] == ["rgb"]
    assert result["unassessed_modalities"] == ["normals"]
    assert result["modality_quality"]["normals"]["status"] == "not_qualified"
    frame["normals"][0, 0] = np.nan
    assert (
        evaluate_frame(frame, frame, policy)["checks"]["requested_outputs"]["status"]
        == "failed"
    )


def test_unrequested_metrics_cannot_satisfy_full_protocol() -> None:
    result = {
        **protocol_requirements("pure-render-quality/rgb-v1"),
        "checks": {
            name: {"status": "passed"}
            for name in protocol_requirements("pure-render-quality/rgb-v1")[
                "required_checks"
            ]
        },
        "metrics": {
            name: (
                {"status": "passed"}
                if name in ("ssim", "lpips")
                else {"status": "not_requested", "required": False}
            )
            for name in default_protocol()["frame_thresholds"]
        },
    }
    assert quality._finish(result)["quality_status"] == "qualified"
    result.update(protocol_requirements("pure-render-quality/v1"))
    # A caller may not reuse a narrower metric's required=False under full scope.
    result = quality._finish(result)
    assert result["quality_status"] == "not_qualified"
    assert "depth_rmse_m" in result["qualification"]["missing_checks"]


def test_declared_30_hz_sequence_does_not_rewrite_timestamps() -> None:
    frames = [_frame(index=index) for index in range(100)]
    for index, frame in enumerate(frames):
        frame["metadata"]["time_s"] = index / 30
    original_times = [frame["metadata"]["time_s"] for frame in frames]
    policy = protocol_for_modalities(("rgb", "depth", "seg"), temporal_dt_s=1 / 30)
    result = evaluate_sequence(frames, frames, policy)
    assert result["protocol_id"] == "pure-render-quality/temporal30-v1"
    assert result["evaluation_kind"] == "sequence"
    assert result["checks"]["fixed_cadence"]["status"] == "passed"
    assert result["checks"]["reference_count"]["status"] == "passed"
    assert [frame["metadata"]["time_s"] for frame in frames] == original_times
    assert (
        evaluate_sequence(frames, frames)["checks"]["fixed_cadence"]["status"]
        == "failed"
    )


def test_protocol_requirement_lookup_needs_no_site_packages() -> None:
    command = [
        sys.executable,
        "-S",
        "-c",
        "from scripts.benchmark.rendering.quality_protocol import protocol_requirements; import sys; "
        "assert protocol_requirements('pure-render-quality/rgb-v1')['required_metrics'] == ['ssim','lpips']; "
        "assert 'numpy' not in sys.modules and 'torch' not in sys.modules",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def test_static_sequence_rejects_one_brightness_and_depth_glitch() -> None:
    references = [_frame(index=index) for index in range(100)]
    candidates = copy.deepcopy(references)
    candidates[50]["rgb"] += 0.2
    candidates[50]["depth"] += 0.03
    result = evaluate_sequence(candidates, references)
    assert result["metrics"]["static_region_flicker"]["value"] == pytest.approx(0.2)
    assert result["metrics"]["static_region_flicker"]["status"] == "failed"
    assert result["metrics"]["temporal_depth_rmse_m"]["value"] == pytest.approx(
        0.03, abs=1e-7
    )
    assert result["metrics"]["temporal_depth_rmse_m"]["status"] == "failed"
    assert result["metrics"]["reprojection_rgb_mae"]["denominator"] == 99
    assert result["quality_status"] == "not_qualified"


def test_static_sequence_reports_consistency_without_independent_reference() -> None:
    candidates = [_frame(index=index) for index in range(100)]
    result = evaluate_sequence(candidates, None)
    assert result["metrics"]["static_region_flicker"]["value"] == 0
    assert result["metrics"]["temporal_depth_rmse_m"]["value"] == 0
    assert result["consistency_only"] is True
    assert result["quality_status"] == "not_qualified"


def test_short_or_reordered_sequence_cannot_qualify() -> None:
    frames = [_frame(index=1), _frame(index=0)]
    result = evaluate_sequence(frames, frames)
    assert result["checks"]["sequence_length"]["status"] == "failed"
    assert result["checks"]["frame_order"]["status"] == "failed"
    assert result["checks"]["fixed_cadence"]["status"] == "failed"


def test_reprojection_uses_reference_visibility_and_rejects_occlusion() -> None:
    previous, current = _frame(), _frame(index=1)
    current["depth"][:, :8] = 0.5
    current["seg"][:, :8] = 2
    rows, cols, ys, xs, static, projected, scale = quality._correspondences(
        previous, current, default_protocol()
    )
    assert len(rows) == 128
    assert np.all(cols >= 8)
    np.testing.assert_array_equal(cols, xs)
    np.testing.assert_allclose(projected, 1)


def test_reprojection_uses_cv_camera_translation_and_object_motion() -> None:
    previous, current = _frame(), _frame(index=1)
    # At optical depth 1 and fx=16, 1/16 m translates by exactly one pixel.
    current["metadata"]["camera"]["world_from_camera"][0][3] = 1 / 16
    rows, cols, ys, xs, *_ = quality._correspondences(
        previous, current, default_protocol()
    )
    np.testing.assert_array_equal(xs, cols - 1)
    current["metadata"]["camera"]["world_from_camera"] = np.eye(4).tolist()
    previous["metadata"]["moving_instance_ids"] = [1]
    current["metadata"]["moving_instance_ids"] = [1]
    previous["metadata"]["world_from_instances"] = {"1": np.eye(4).tolist()}
    transform = np.eye(4)
    transform[0, 3] = 1 / 16
    current["metadata"]["world_from_instances"] = {"1": transform.tolist()}
    rows, cols, ys, xs, static, *_ = quality._correspondences(
        previous, current, default_protocol()
    )
    np.testing.assert_array_equal(xs, cols + 1)
    assert not static.any()


def test_real_learned_lpips_requires_and_uses_cached_official_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("lpips")
    import torch

    checkpoint = Path(torch.hub.get_dir()) / "checkpoints/squeezenet1_1-b8a52dc0.pth"
    if not checkpoint.exists():
        pytest.skip(
            "Official pretrained LPIPS backbone is not cached; no test download"
        )
    # Undo only this file's no-model fixture to exercise the genuine implementation.
    monkeypatch.undo()
    first = _frame(size=64)
    identity = evaluate_frame(first, first)["metrics"]["lpips"]
    second = copy.deepcopy(first)
    second["rgb"][::2] = 0
    distortion = evaluate_frame(first, second)["metrics"]["lpips"]
    assert identity["value"] == pytest.approx(0, abs=1e-8)
    assert distortion["value"] > 0
    assert distortion["evidence"]["backbone_sha256"].startswith("b8a52dc0")
