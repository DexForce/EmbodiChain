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

"""Measured receipt tests using genuine tiny CPU references and fixture captures.

The capture fixtures copy reference pixels to isolate receipt verification; they
are not claims about either tested renderer. SSIM/noise are recomputed from real
arrays. Learned LPIPS inference itself is covered in test_rendering_quality.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import stable_hash, write_json
from scripts.benchmark.rendering import quality
from scripts.benchmark.rendering.calibration import (
    create_calibration_receipt,
    verify_calibration,
)
from scripts.benchmark.rendering.formal import render_profile
from scripts.benchmark.rendering.quality_protocol import default_protocol
from scripts.benchmark.rendering.reference import load_reference_frame, render_reference
from scripts.benchmark.rendering.scenes import build_scene


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _claim(
    path: Path, profile: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "qualified",
        "reference_id": reference["reference_provenance"]["reference_id"],
        "profile_sha256": stable_hash(profile),
        "evidence": [
            {
                "path": str(path),
                "sha256": _sha(path),
                "kind": "photometric_calibration_receipt",
            }
        ],
    }


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    pytest.importorskip("mitsuba")
    # NN dependency/failure handling has independent tests; here a fixture score
    # isolates source binding, actual SSIM, and real reference repeat-seed noise.
    monkeypatch.setattr(
        quality, "_lpips", lambda first, second: (0.0, {"fixture": True})
    )
    scene = build_scene("calibration", tmp_path / "scene")
    render_reference(scene, None, 16, 16, tmp_path / "reference", spp=4096, threads=2)
    profile = render_profile("realtime")
    profile["lighting_controls"] = {
        "embodichain": {
            "directional_intensity": 3.0,
            "environment_intensity": 15.0,
            "environment_color": [1.0, 1.0, 1.0],
        }
    }
    controls = profile["lighting_controls"]["embodichain"]
    postprocess = {
        "scope": "applied native WorldConfig, not renderer readback",
        "settings": {
            "tone_mapping_enabled": False,
            "tone_mapping_exposure": 1.0,
            "auto_exposure_enabled": False,
        },
    }
    pairs = []
    for index in range(2):
        ref_path = tmp_path / f"reference/frame_{index:04d}.json"
        ref = load_reference_frame(ref_path)
        run = tmp_path / f"fixture_run_{index}"
        frame_dir = run / "frames"
        frame_dir.mkdir(parents=True)
        arrays = frame_dir / "frame.npz"
        rgb = np.round(quality._display_rgb(ref) * 255).astype(np.uint8)
        np.savez_compressed(arrays, rgb=rgb)
        metadata = copy.deepcopy(ref["metadata"])
        metadata.update(
            backend="embodichain",
            profile_sha256=stable_hash(profile),
            rgb_encoding="srgb_uint8",
            env_id=0,
            camera_name=metadata["camera"]["name"],
            fixture=True,
        )
        frame = frame_dir / "frame.json"
        write_json(
            frame,
            {
                "schema_version": 1,
                "metadata": metadata,
                "arrays": {"path": arrays.name, "sha256": _sha(arrays)},
            },
        )
        write_json(run / "worker_config.json", {"profile": profile})
        index_path = run / "frames.json"
        write_json(
            index_path,
            {
                "frames": [
                    {
                        "path": "frames/frame.json",
                        "sha256": _sha(frame),
                        "arrays_path": "frames/frame.npz",
                        "arrays_sha256": _sha(arrays),
                    }
                ]
            },
        )
        write_json(
            run / "result.json",
            {
                "fixture": True,
                "status": "completed",
                "backend": "embodichain",
                "profile_sha256": stable_hash(profile),
                "frames_manifest": {"path": "frames.json", "sha256": _sha(index_path)},
                "renderer": {
                    "effective_lighting_controls": controls,
                    "postprocess_evidence": postprocess,
                },
            },
        )
        pairs.append({"capture": frame, "reference": ref_path})
    candidate = {
        "backend": "embodichain",
        "profile_sha256": stable_hash(profile),
        "effective_lighting_controls": controls,
        "postprocess_evidence": postprocess,
    }
    return {
        "profile": profile,
        "pairs": pairs,
        "candidate": candidate,
        "reference": load_reference_frame(pairs[1]["reference"]),
    }


def test_valid_receipt_recomputes_heldout_metrics_and_retains_source_proof(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    path = tmp_path / "receipt.json"
    result = create_calibration_receipt(
        path,
        backend="embodichain",
        profile=sources["profile"],
        training_pairs=sources["pairs"][:1],
        heldout_pairs=sources["pairs"][1:],
    )
    assert result["status"] == "qualified"
    assert result["measurements"]["heldout"][0]["reference_noise_rmse"] <= 0.01
    check = verify_calibration(
        _claim(path, sources["profile"], sources["reference"]),
        sources["candidate"],
        sources["reference"],
        default_protocol(),
    )
    assert check["status"] == "passed"
    kinds = {item["kind"] for item in check["evidence"]["source_evidence"]}
    assert {
        "calibration_frame_arrays",
        "calibration_worker_result",
        "calibration_worker_config",
        "calibration_frame_index",
    } <= kinds


def test_manual_status_and_arbitrary_hashed_json_do_not_qualify(tmp_path: Path) -> None:
    path = write_json(tmp_path / "arbitrary.json", {"status": "qualified"})
    profile = {"name": "fixture"}
    ref = {"reference_provenance": {"reference_id": "fixture-reference"}}
    claim = _claim(path, profile, ref)
    result = verify_calibration(
        claim,
        {"profile_sha256": stable_hash(profile), "backend": "embodichain"},
        ref,
        default_protocol(),
    )
    assert result["status"] == "failed"


def test_training_capture_cannot_be_relabelled_as_heldout(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    pair = sources["pairs"][:1]
    with pytest.raises(ValueError, match="overlap training"):
        create_calibration_receipt(
            tmp_path / "receipt.json",
            backend="embodichain",
            profile=sources["profile"],
            training_pairs=pair,
            heldout_pairs=pair,
        )


def test_postprocess_or_physical_reference_change_invalidates_receipt(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    path = tmp_path / "receipt.json"
    create_calibration_receipt(
        path,
        backend="embodichain",
        profile=sources["profile"],
        training_pairs=sources["pairs"][:1],
        heldout_pairs=sources["pairs"][1:],
    )
    claim = _claim(path, sources["profile"], sources["reference"])
    metadata = copy.deepcopy(sources["candidate"])
    metadata["postprocess_evidence"]["settings"]["tone_mapping_exposure"] = 2.0
    assert (
        verify_calibration(claim, metadata, sources["reference"], default_protocol())[
            "status"
        ]
        == "failed"
    )
    ref = copy.deepcopy(sources["reference"])
    ref["reference_provenance"]["lighting"]["environment_radiance"] = 0.3
    assert (
        verify_calibration(claim, sources["candidate"], ref, default_protocol())[
            "status"
        ]
        == "failed"
    )


def test_source_tampering_and_forged_stored_metric_are_rejected(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    path = tmp_path / "receipt.json"
    receipt = create_calibration_receipt(
        path,
        backend="embodichain",
        profile=sources["profile"],
        training_pairs=sources["pairs"][:1],
        heldout_pairs=sources["pairs"][1:],
    )
    receipt["measurements"]["heldout"][0]["metrics"]["ssim"]["value"] = 0.123
    write_json(path, receipt)
    claim = _claim(path, sources["profile"], sources["reference"])
    check = verify_calibration(
        claim, sources["candidate"], sources["reference"], default_protocol()
    )
    assert check["status"] == "failed"
    assert "recomputed" in check["reason"]
    arrays = sources["pairs"][1]["capture"].with_suffix(".npz")
    arrays.write_bytes(b"modified source")
    assert (
        verify_calibration(
            claim, sources["candidate"], sources["reference"], default_protocol()
        )["status"]
        == "failed"
    )


def test_bad_pixels_cannot_be_promoted_by_forging_qualified_status(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    frame_path = sources["pairs"][1]["capture"]
    run = frame_path.parent.parent
    arrays_path = frame_path.with_suffix(".npz")
    np.savez_compressed(arrays_path, rgb=np.zeros((16, 16, 3), dtype=np.uint8))
    frame = json.loads(frame_path.read_text())
    frame["arrays"]["sha256"] = _sha(arrays_path)
    write_json(frame_path, frame)
    index_path = run / "frames.json"
    index = json.loads(index_path.read_text())
    index["frames"][0].update(sha256=_sha(frame_path), arrays_sha256=_sha(arrays_path))
    write_json(index_path, index)
    result_path = run / "result.json"
    result = json.loads(result_path.read_text())
    result["frames_manifest"]["sha256"] = _sha(index_path)
    write_json(result_path, result)
    path = tmp_path / "receipt.json"
    receipt = create_calibration_receipt(
        path,
        backend="embodichain",
        profile=sources["profile"],
        training_pairs=sources["pairs"][:1],
        heldout_pairs=sources["pairs"][1:],
    )
    assert receipt["status"] == "not_qualified"
    receipt["status"] = "qualified"
    receipt["measurements"]["heldout"][0]["metrics"]["ssim"].update(
        value=1.0, status="passed"
    )
    write_json(path, receipt)
    check = verify_calibration(
        _claim(path, sources["profile"], sources["reference"]),
        sources["candidate"],
        sources["reference"],
        default_protocol(),
    )
    assert check["status"] == "failed"
    assert "Recomputed heldout" in check["reason"]


def test_stored_noise_pass_flag_cannot_replace_missing_independent_repeat(
    tmp_path: Path, sources: dict[str, Any]
) -> None:
    reference_path = sources["pairs"][1]["reference"]
    record = json.loads(reference_path.read_text())
    arrays_path = reference_path.parent / record["arrays"]["path"]
    with np.load(arrays_path) as archive:
        arrays = {key: archive[key].copy() for key in ("rgb", "depth", "seg")}
    np.savez_compressed(arrays_path, **arrays)
    record["arrays"]["sha256"] = _sha(arrays_path)
    record["reference_validation"]["passed"] = True
    write_json(reference_path, record)
    manifest_path = reference_path.parent / "reference.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["frames"][1]["sha256"] = _sha(reference_path)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="independently seeded RGB repeat"):
        create_calibration_receipt(
            tmp_path / "receipt.json",
            backend="embodichain",
            profile=sources["profile"],
            training_pairs=sources["pairs"][:1],
            heldout_pairs=sources["pairs"][1:],
        )
