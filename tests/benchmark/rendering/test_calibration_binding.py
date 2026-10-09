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

"""Cross-experiment imports of measured tiny CPU calibration receipts.

Source images/noise/SSIM are real CPU-reference measurements; LPIPS uses the
existing fixture score to isolate binding. Target worker captures use the
simulator-free production-schema fixture and remain image-quality unqualified.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from typing import Any

import pytest

from scripts.benchmark.core.artifacts import (
    read_json,
    read_json_object,
    stable_hash,
    write_json,
)
from scripts.benchmark.rendering import evaluate, quality, reference
from scripts.benchmark.rendering.calibration import create_calibration_receipt
from scripts.benchmark.rendering.calibration_binding import bind_calibration_receipt
from scripts.benchmark.rendering.formal import (
    RenderCase,
    render_profile,
    view_fingerprint,
)
from scripts.benchmark.rendering.quality_protocol import protocol_for_modalities
from scripts.benchmark.rendering.scenes import load_scene

ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def measured_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("measured_calibration_source")
    patch = pytest.MonkeyPatch()
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_rendering_calibration.py"))
    )
    sources = fixture["sources"].__wrapped__(root, patch)
    # Keep the genuine reference/capture pixels while using the current frozen
    # profile in every synthetic worker-provenance field.
    profile = render_profile("realtime")
    for pair in sources["pairs"]:
        frame_path = pair["capture"]
        run = frame_path.parent.parent
        frame = read_json_object(frame_path)
        frame["metadata"]["profile_sha256"] = stable_hash(profile)
        write_json(frame_path, frame)
        write_json(run / "worker_config.json", {"profile": profile})
        index = read_json_object(run / "frames.json")
        index["frames"][0]["sha256"] = _sha(frame_path)
        write_json(run / "frames.json", index)
        result = read_json_object(run / "result.json")
        result["profile_sha256"] = stable_hash(profile)
        result["frames_manifest"]["sha256"] = _sha(run / "frames.json")
        result["renderer"]["effective_lighting_controls"] = profile[
            "lighting_controls"
        ]["embodichain"]
        write_json(run / "result.json", result)
    receipt = create_calibration_receipt(
        root / "receipt.json",
        backend="embodichain",
        profile=profile,
        training_pairs=sources["pairs"][:1],
        heldout_pairs=sources["pairs"][1:],
    )
    assert receipt["status"] == "qualified"
    yield root
    patch.undo()


@pytest.fixture
def source(
    tmp_path: Path, measured_source: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    monkeypatch.setattr(quality, "_lpips", lambda a, b: (0.0, {"fixture": True}))
    destination = tmp_path / "source"
    shutil.copytree(measured_source, destination)
    return destination / "receipt.json"


def _target(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str = "realtime",
    wrong_lighting: bool = False,
    wrong_postprocess: bool = False,
    changed_reference: bool = False,
    initial_evaluation: bool = True,
) -> dict[str, Any]:
    fixture = runpy.run_path(str(Path(__file__).with_name("test_formal_worker.py")))
    fixture["runtime"].__wrapped__(monkeypatch)
    case = RenderCase(
        "R-03", "workshop", width=16, height=16, modalities=("rgb",), profile=profile
    )
    config_path = fixture["config_file"](root / "inputs", case, measured=2, warmup=0)
    config = read_json_object(config_path)
    scene_path = Path(config["scene_path"])
    scene = load_scene(scene_path)
    scene["objects"][0]["world_from_object"][0][3] += 0.1
    write_json(scene_path, scene)
    config["scene_sha256"] = stable_hash(scene)
    config["view_sha256"] = view_fingerprint(case, scene)
    config["approved_scene_receipt"]["scenes"][case.scene_id] = stable_hash(scene)
    write_json(config_path, config)
    run_id = "r00_embodichain"
    result = fixture["worker"].run_worker(config_path, root / run_id, "embodichain", 0)
    assert result["status"] == "completed", result.get("error")
    result["renderer"]["effective_lighting_controls"] = deepcopy(
        config["profile"]["lighting_controls"]["embodichain"]
    )
    result["renderer"]["postprocess_evidence"] = {
        "scope": "applied native WorldConfig, not renderer readback",
        "settings": {
            "tone_mapping_enabled": False,
            "tone_mapping_exposure": 1.0,
            "auto_exposure_enabled": False,
        },
    }
    if wrong_lighting:
        result["renderer"]["effective_lighting_controls"]["environment_intensity"] *= 2
    if wrong_postprocess:
        result["renderer"]["postprocess_evidence"]["settings"][
            "tone_mapping_exposure"
        ] = 2
    row = {**result, "run_id": run_id, "run_dir": run_id, "attempt": 0}
    write_json(root / run_id / "result.json", row)
    write_json(root / "runs.json", [row])
    write_json(
        root / "definition.json",
        {"schema_version": 1, "definition_version": "2.0", "experiment_id": "R-03"},
    )
    write_json(root / "artifact_index.json", {"schema_version": 1, "artifacts": []})
    (root / "raw.jsonl").write_text(json.dumps({"record_id": run_id, **row}) + "\n")
    write_json(root / "quality.json", {"schema_version": 1, "runs": []})
    run = evaluate._load_run(root.resolve(), row)
    aliases, refs = [], []
    if initial_evaluation:
        render = reference.render_reference
        if changed_reference:
            monkeypatch.setattr(
                reference,
                "render_reference",
                lambda *args, **kwargs: render(
                    *args,
                    **kwargs,
                    lighting={
                        "directional_irradiance": 3.0,
                        "environment_radiance": 0.3,
                    },
                ),
            )
        for descriptor in run["descriptors"]:
            ref, alias = evaluate._independent_reference(
                root.resolve(), run, descriptor, {"spp": 4096, "threads": 2, "seed": 0}
            )
            assert ref is not None
            refs.append(ref)
            aliases.append(alias)
        if changed_reference:
            monkeypatch.setattr(reference, "render_reference", render)
        path = write_json(
            root
            / "quality_evaluations/evaluation_0001/run_0000/reference_bindings.json",
            aliases,
        )
        record = {
            key: row[key]
            for key in (
                "run_id",
                "backend",
                "case_id",
                "config_sha256",
                "scene_sha256",
                "view_sha256",
                "profile_sha256",
            )
        }
        record.update(
            quality_status="not_qualified",
            artifacts=[
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": _sha(path),
                    "kind": "quality_detail",
                }
            ],
        )
        write_json(root / "quality.json", {"schema_version": 1, "runs": [record]})
    return {"row": row, "run": run, "references": refs, "aliases": aliases}


def test_import_is_byte_preserving_cross_scene_and_does_not_qualify_target_images(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    fixture = _target(target, monkeypatch)
    source_receipt = read_json_object(source)
    source_files = {
        item["path"]: (source.parent / item["path"]).read_bytes()
        for item in source_receipt["source_evidence"]
    }
    before = {
        name: (target / name).read_bytes()
        for name in ("raw.jsonl", "runs.json", "quality.json")
    }
    monkeypatch.setattr(
        reference,
        "render_reference",
        lambda *a, **k: pytest.fail("Binding cannot generate new references"),
    )
    report = bind_calibration_receipt(source, target)
    assert report["status"] == "bound", report
    assert report["target_quality_status_changed"] is False
    assert all(frame["status"] == "passed" for frame in report["frames"])
    imported = target / report["imported_receipt"]
    assert imported.read_bytes() == source.read_bytes()
    assert imported.parent.name == _sha(source)
    for relative, content in source_files.items():
        assert (imported.parent / relative).read_bytes() == content
    document = read_json_object(target / report["evidence_document"])
    record = document["records"][0]
    assert record["scene_sha256"] == fixture["row"]["scene_sha256"]
    assert (
        record["scene_sha256"]
        != read_json_object(
            source.parent / source_receipt["heldout"][0]["capture"]["path"]
        )["metadata"]["scene_sha256"]
    )
    assert record["verified_target_frames"] == len(fixture["run"]["descriptors"])
    assert len(record["photometric_calibration"]["evidence"]) == 1
    shutil.move(source.parent, tmp_path / "source_unavailable")
    candidate = evaluate._candidate(fixture["run"]["descriptors"][0])
    candidate["metadata"].update(
        {
            key: deepcopy(fixture["run"]["renderer"][key])
            for key in ("effective_lighting_controls", "postprocess_evidence")
        }
    )
    policy = protocol_for_modalities(("rgb",))
    evaluate._calibration(
        target.resolve(),
        document,
        fixture["row"],
        candidate,
        fixture["references"][0],
        policy,
    )
    result = quality.evaluate_frame(candidate, fixture["references"][0], policy)
    assert result["checks"]["photometric_calibration"]["status"] == "passed"
    assert result["quality_status"] == "not_qualified"
    assert result["metrics"]["ssim"]["status"] == "failed"
    assert all(
        (target / name).read_bytes() == content for name, content in before.items()
    )


@pytest.mark.parametrize(
    "forgery", ["status_only", "stored_metric", "source_list", "source_bytes"]
)
def test_forged_or_damaged_receipt_is_rejected_without_import(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, forgery: str
) -> None:
    target = tmp_path / "target"
    _target(target, monkeypatch, initial_evaluation=False)
    receipt = read_json_object(source)
    if forgery == "status_only":
        receipt = {"status": "qualified"}
    elif forgery == "stored_metric":
        receipt["measurements"]["heldout"][0]["metrics"]["ssim"]["value"] = 0.123
    elif forgery == "source_list":
        receipt["source_evidence"] = [
            {"path": "../outside", "sha256": "0" * 64, "kind": "manual"}
        ]
    else:
        path = source.parent / receipt["source_evidence"][0]["path"]
        path.write_bytes(b"tampered")
    write_json(source, receipt)
    report = bind_calibration_receipt(source, target)
    assert report["status"] == "rejected"
    assert report["failure_phase"] == "source_import"
    assert report["evidence_document"] is None
    assert not (target / "calibration_imports").exists()
    assert (target / report["report_path"]).is_file()


@pytest.mark.parametrize(
    "mismatch", ["profile", "lighting", "postprocess", "reference"]
)
def test_wrong_target_contract_retains_failed_scopes_without_claim(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    target = tmp_path / "target"
    _target(
        target,
        monkeypatch,
        profile="high_quality" if mismatch == "profile" else "realtime",
        wrong_lighting=mismatch == "lighting",
        wrong_postprocess=mismatch == "postprocess",
        changed_reference=mismatch == "reference",
    )
    report = bind_calibration_receipt(source, target)
    assert report["status"] == "no_valid_bindings", report
    assert report["source_verification"]["status"] == "passed"
    assert report["failed_scopes"]
    assert report["evidence_document"] is None
    assert not report["records"]
    expected = {
        "profile": "backend/profile",
        "lighting": "light controls",
        "postprocess": "postprocess",
        "reference": "physical reference",
    }[mismatch]
    assert expected in json.dumps(report["failed_scopes"])


def test_missing_initial_evaluation_is_explicit_and_import_can_be_reused(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    _target(target, monkeypatch, initial_evaluation=False)
    first = bind_calibration_receipt(source, target)
    assert first["status"] == "no_valid_bindings"
    assert "Initial offline evaluation" in first["failed_scopes"][0]["reason"]
    second = bind_calibration_receipt(source, target)
    assert second["source_verification"]["reused_import"] is True
    assert second["imported_receipt"] == first["imported_receipt"]
    assert second["report_path"] != first["report_path"]


def test_reference_alias_outside_target_is_rejected(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    fixture = _target(target, monkeypatch)
    document = read_json_object(target / "quality.json")
    artifact = document["runs"][0]["artifacts"][0]
    path = target / artifact["path"]
    aliases = fixture["aliases"]
    aliases[0]["reference_frame"] = "../outside.json"
    write_json(path, aliases)
    artifact["sha256"] = _sha(path)
    write_json(target / "quality.json", document)
    report = bind_calibration_receipt(source, target)
    assert report["status"] == "no_valid_bindings"
    assert report["evidence_document"] is None
    assert "escapes" in report["frames"][0]["reason"]


def test_help_has_no_array_or_gpu_imports() -> None:
    command = """
import sys
from scripts.benchmark.rendering.calibration_binding import main
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert not any(name in sys.modules for name in ('torch', 'numpy', 'mitsuba', 'sapien', 'isaaclab'))
"""
    subprocess.run(
        [sys.executable, "-S", "-c", command], cwd=ROOT, check=True, capture_output=True
    )


def test_nonexistent_target_is_not_created(tmp_path: Path) -> None:
    target = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="Target experiment root"):
        bind_calibration_receipt(tmp_path / "receipt.json", target)
    assert not target.exists()
