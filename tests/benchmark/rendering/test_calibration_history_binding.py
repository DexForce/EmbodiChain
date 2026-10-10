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

"""Calibration routing through a real file-validated repeated history proof.

Receipt numerics have separate measured-reference tests. These fixtures isolate
history selection/alias identity and never certify image or calibration quality.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.benchmark.rendering import calibration_binding as binding
from scripts.benchmark.rendering import evaluate, reference
from scripts.benchmark.rendering.frozen_source import (
    artifact,
    read_json,
    sha256,
    stable_hash,
    write_json,
)


def _target(root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(root / "history")
    original = fixture["expected_original"]
    scene_path = fixture["root"] / "original/scene.json"
    run = {
        "row": original,
        "case": SimpleNamespace(**fixture["config"]["case"]),
        "config": {**fixture["config"], "replay_path": "history/original/replay.json"},
        "scene_path": scene_path,
        "renderer": read_json(fixture["root"] / "capture/renderer.json"),
        "input_artifacts": [],
        "coverage": {"representative_env_ids": [0]},
    }
    monkeypatch.setattr(evaluate, "load_scene", lambda path: read_json(path))
    monkeypatch.setattr(evaluate, "load_replay", lambda path, scene: read_json(path))
    loaded = evaluate._load_history(root, run, fixture["path"])
    # The original loader supplies an older post-capture population. The binder
    # must replace it using the evaluated, actually validated history manifest.
    run["descriptors"] = []
    for descriptor in loaded["descriptors"][1:4]:
        old = deepcopy(descriptor)
        old["metadata"].pop("history")
        run["descriptors"].append(old)
    monkeypatch.setattr(evaluate, "_load_run", lambda *args: deepcopy(run))
    aliases, references = [], {}
    for descriptor in loaded["descriptors"]:
        metadata = descriptor["metadata"]
        camera = descriptor["canonical_camera"]
        identity = {
            "scene_sha256": metadata["scene_sha256"],
            "camera": {key: value for key, value in camera.items() if key != "name"},
            "object_transforms": metadata["world_from_objects"],
            "width": run["case"].width,
            "height": run["case"].height,
        }
        key = stable_hash(identity)
        directory = root / "references" / key
        frame = directory / "frame.json"
        if not frame.exists():
            ref_metadata = deepcopy(metadata)
            ref_metadata.pop("history")
            ref_metadata["camera"] = deepcopy(camera)
            value = {
                "metadata": ref_metadata,
                "reference_provenance": {
                    "reference_id": "reference-" + key,
                    "source_scene_sha256": metadata["scene_sha256"],
                },
            }
            write_json(frame, value)
            references[frame.resolve()] = value
        cache = write_json(
            directory / "cache_binding.json",
            {
                "identity": identity,
                "frame": frame.relative_to(root).as_posix(),
                "artifacts": [artifact(root, frame, "independent_reference")],
            },
        )
        aliases.append(
            {
                "status": "completed",
                "cache_key": key,
                "geometry_identity_sha256": key,
                "frame_id": metadata["frame_id"],
                "time_s": metadata["time_s"],
                "source_frame_id": metadata["frame_id"],
                "reference_id": "reference-" + key,
                "reference_frame": frame.relative_to(root).as_posix(),
                "binding_artifact": artifact(root, cache, "reference_cache_binding"),
                "history": deepcopy(metadata["history"]),
            }
        )
    monkeypatch.setattr(
        reference,
        "load_reference_frame",
        lambda path: deepcopy(references[Path(path).resolve()]),
    )
    aliases_path = write_json(root / "evaluation/reference_bindings.json", aliases)
    record = {
        key: original[key]
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
        timed_history=loaded["timed_history"],
        artifacts=[artifact(root, aliases_path, "quality_detail")],
    )
    write_json(root / "quality.json", {"schema_version": 1, "runs": [record]})
    write_json(root / "runs.json", [original])
    write_json(
        root / "definition.json", {"schema_version": 1, "definition_version": "2.0"}
    )
    (root / "raw.jsonl").write_bytes(
        (fixture["root"] / "original/raw.jsonl").read_bytes()
    )
    receipt = write_json(
        root / "receipt.json", {"fixture": "receipt verification tested separately"}
    )
    monkeypatch.setattr(
        binding,
        "_import_receipt",
        lambda *args: {
            "receipt": receipt,
            "receipt_sha256": sha256(receipt),
            "reused": False,
            "manifest": artifact(root, receipt, "fixture_receipt_manifest"),
        },
    )
    verified = []

    def verify(
        claim: Any,
        metadata: dict[str, Any],
        ref: dict[str, Any],
        policy: dict[str, Any],
    ) -> dict[str, Any]:
        verified.append(
            {
                "candidate": deepcopy(metadata),
                "reference": deepcopy(ref["metadata"]),
                "policy": policy["id"],
            }
        )
        return {"status": "passed"}

    monkeypatch.setattr(binding, "verify_calibration", verify)
    return {
        "proof": fixture,
        "run": run,
        "loaded": loaded,
        "aliases": aliases,
        "aliases_path": aliases_path,
        "record": record,
        "receipt": receipt,
        "verified": verified,
    }


def test_repeated_raw_frames_bind_the_evaluated_history_exposures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _target(tmp_path, monkeypatch)
    before = {
        name: (tmp_path / name).read_bytes()
        for name in ("raw.jsonl", "runs.json", "quality.json")
    }
    report = binding.bind_calibration_receipt(fixture["receipt"], tmp_path)
    assert report["status"] == "bound", report["failed_scopes"]
    assert len(report["frames"]) == 7  # Last warmup guard plus six measured calls.
    assert [frame["history"]["scoring_index"] for frame in report["frames"]] == list(
        range(7)
    )
    assert [frame["frame_id"] for frame in report["frames"]] == [1, 0, 1, 2, 0, 1, 2]
    assert len(fixture["verified"]) == 7
    for checked in fixture["verified"]:
        assert checked["candidate"]["history"] == checked["reference"]["history"]
        assert (
            checked["candidate"]["effective_lighting_controls"]
            == fixture["loaded"]["renderer"]["effective_lighting_controls"]
        )
        assert checked["policy"] == "pure-render-quality/rgb-timed-history30-v1"
    assert sum(record["verified_target_frames"] for record in report["records"]) == 7
    assert all(
        (tmp_path / name).read_bytes() == content for name, content in before.items()
    )


@pytest.mark.parametrize(
    "changed",
    [
        "missing_history",
        "history_sha",
        "history_schedule",
        "alias_exposure",
        "duplicate_alias",
    ],
)
def test_unbound_or_ambiguous_history_cannot_emit_calibration_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str
) -> None:
    fixture = _target(tmp_path, monkeypatch)
    quality = read_json(tmp_path / "quality.json")
    record = quality["runs"][0]
    if changed == "missing_history":
        record.pop("timed_history")
    elif changed == "history_sha":
        record["timed_history"]["sha256"] = "0" * 64
    elif changed == "history_schedule":
        record["timed_history"]["measured_frames"] -= 1
    else:
        aliases = fixture["aliases"]
        if changed == "alias_exposure":
            aliases[0]["history"]["exposure_index"] = 1
        else:
            # A second distinct selected alias is ambiguous; precision attempts
            # belong inside the selection record, not as extra exposure aliases.
            aliases.append(
                {
                    **deepcopy(aliases[0]),
                    "reference_precision": {
                        "path": "other_selection.json",
                        "sha256": "0" * 64,
                    },
                }
            )
        write_json(fixture["aliases_path"], aliases)
        record["artifacts"][0]["sha256"] = sha256(fixture["aliases_path"])
    write_json(tmp_path / "quality.json", quality)
    report = binding.bind_calibration_receipt(fixture["receipt"], tmp_path)
    assert report["status"] == "no_valid_bindings", report
    assert report["evidence_document"] is None
    assert report["failed_scopes"]


@pytest.mark.parametrize("scope", ["static", "R-02"])
def test_ordinary_reference_aliases_keep_their_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scope: str
) -> None:
    fixture = _target(tmp_path, monkeypatch)
    quality = read_json(tmp_path / "quality.json")
    quality["runs"][0].pop("timed_history")
    aliases = {}
    for old in fixture["aliases"]:
        alias = deepcopy(old)
        alias.pop("history")
        aliases[alias["frame_id"]] = alias
    write_json(fixture["aliases_path"], list(aliases.values()))
    quality["runs"][0]["artifacts"][0]["sha256"] = sha256(fixture["aliases_path"])
    write_json(tmp_path / "quality.json", quality)
    if scope == "static":
        fixture["run"]["case"].temporal_mode = "static"
    else:
        fixture["run"]["case"].experiment_id = "R-02"
    monkeypatch.setattr(
        evaluate,
        "_load_history",
        lambda *args: pytest.fail("Ordinary reference path requested a history proof"),
    )
    report = binding.bind_calibration_receipt(fixture["receipt"], tmp_path)
    assert report["status"] == "bound", report
    assert len(report["frames"]) == 3
    assert all("history" not in frame for frame in report["frames"])
