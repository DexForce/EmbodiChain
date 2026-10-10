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

"""Tiny synthetic independent-repeat fixtures; no rendering or quality claims."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.core.artifacts import (
    read_json,
    read_json_object,
    stable_hash,
    write_json,
)
from scripts.benchmark.rendering import evaluate, publication_report, reference
from scripts.benchmark.rendering import reference_precision as precision


def descriptor(root: Path, path: Path, kind: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": kind,
    }


def render_fixture(
    output: Path,
    identity: dict[str, Any],
    deviation: float,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist numerical repeats using the real reference artifact envelope."""
    settings = identity["settings"]
    width, height = identity["width"], identity["height"]
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1,
        "fixture": "synthetic numerical test only; no independent rendering claim",
        "kind": "independent_offline",
        "renderer": "mitsuba",
        "device": "cpu",
        "variant": "llvm_ad_rgb",
        "assets_verified": True,
        "source_scene_sha256": identity["scene_sha256"],
        "implementation_sha256": identity["reference_implementation_sha256"],
        "threads": settings["threads"],
        "width": width,
        "height": height,
        "integrator": {
            "type": "path",
            "max_depth": 8,
            "spp": settings["spp"],
            "seed": settings["seed"],
            "independent_repeat_count": 2,
        },
        "noise_floor_policy": {
            "metric": "independent_seed_display_srgb_rmse",
            "maximum": 0.01,
        },
        "lighting": {"fixture_radiance": 0.15},
        "camera_specs_sha256": "fixture",
    }
    provenance["reference_id"] = stable_hash(provenance)
    rgb = np.full((height, width, 3), 0.5, dtype=np.float32)
    repeat = rgb + np.float32(deviation)
    display = [
        np.where(value <= 0.0031308, 12.92 * value, 1.055 * value ** (1 / 2.4) - 0.055)
        for value in (rgb, repeat)
    ]
    noise = float(np.sqrt(np.mean((display[0] - display[1]) ** 2)))
    arrays = output / "frame_0000.npz"
    np.savez_compressed(
        arrays,
        rgb=rgb,
        rgb_repeat=repeat,
        depth=np.ones((height, width), np.float32),
        seg=np.ones((height, width), np.int32),
    )
    frame = {
        "schema_version": 1,
        "metadata": {
            "backend": "mitsuba",
            "rgb_encoding": "linear_rec709",
            "frame_id": 0,
            "frame_index": 0,
            "time_s": 0,
            **(metadata or {}),
        },
        "reference_provenance": provenance,
        "reference_validation": {
            "metric": "independent_seed_display_srgb_rmse",
            "value": noise,
            "threshold": 0.01,
            "passed": noise <= 0.01,
            "seeds": [0, 1],
            "samples_per_pixel_per_repeat": settings["spp"],
            "denominator": width * height * 3,
        },
        "arrays": {
            "path": arrays.name,
            "sha256": hashlib.sha256(arrays.read_bytes()).hexdigest(),
        },
    }
    frame_path = write_json(output / "frame_0000.json", frame)
    manifest = {
        "schema_version": 1,
        "provenance": provenance,
        "frames": [
            {
                "path": frame_path.name,
                "sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            }
        ],
    }
    write_json(output / "reference.json", manifest)
    return manifest


class Cache:
    """Exercise real persisted arrays with an acquisition callback free of candidates."""

    def __init__(self, root: Path, deviations: tuple[float, float]) -> None:
        self.root = root
        self.deviations = deviations
        self.calls: list[int] = []
        self.frame_id = 0
        self.time_s = 0.0
        self.history: dict[str, Any] | None = None
        self.mutate: Any = None
        self.policy = precision.freeze_precision_policy(
            root, root / "choices", {"spp": 4096, "seed": 0, "threads": 8}
        )

    def acquire(
        self, settings: dict[str, int]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert set(settings) == {"spp", "seed", "threads"}
        self.calls.append(settings["spp"])
        identity = {
            "schema_version": 1,
            "scene_sha256": "frozen-scene",
            "camera": {"fixture": "fixed"},
            "object_transforms": {},
            "width": 8,
            "height": 6,
            "settings": settings,
            "reference_implementation_sha256": "frozen-generator",
        }
        key = stable_hash(identity)
        directory = self.root / "references" / key
        path = directory / "cache_binding.json"
        if not directory.exists():
            render_fixture(
                directory, identity, self.deviations[settings["spp"] == 16384]
            )
            binding = {
                "identity": identity,
                "frame": (directory / "frame_0000.json")
                .relative_to(self.root)
                .as_posix(),
                "artifacts": [
                    descriptor(self.root, directory / name, "independent_reference")
                    for name in ("reference.json", "frame_0000.json", "frame_0000.npz")
                ],
            }
            write_json(path, binding)
            if self.mutate:
                self.mutate(path)
        binding = read_json_object(path)
        loaded = reference.load_reference_frame(directory / "frame_0000.json")
        alias = {
            "status": "completed",
            "cache_key": key,
            "geometry_identity_sha256": key,
            "reference_id": loaded["reference_provenance"]["reference_id"],
            "reference_frame": binding["frame"],
            "source_frame_id": 0,
            "frame_id": self.frame_id,
            "time_s": self.time_s,
            "binding_artifact": descriptor(self.root, path, "reference_cache_binding"),
        }
        loaded["metadata"].update(
            frame_id=self.frame_id, frame_index=self.frame_id, time_s=self.time_s
        )
        if self.history is not None:
            alias["history"] = deepcopy(self.history)
            loaded["metadata"]["history"] = deepcopy(self.history)
        return loaded, alias

    def select(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return precision.select_reference(
            self.root, self.root / "choices", self.policy, self.acquire
        )


def reseal_cache(path: Path) -> None:
    """Reseal fixture envelopes so tests reach semantic and numeric verification."""
    root = path.parents[2]
    frame_path = path.parent / "frame_0000.json"
    frame = read_json_object(frame_path)
    frame["arrays"]["sha256"] = hashlib.sha256(
        (path.parent / "frame_0000.npz").read_bytes()
    ).hexdigest()
    write_json(frame_path, frame)
    manifest = read_json_object(path.parent / "reference.json")
    manifest["provenance"] = frame["reference_provenance"]
    manifest["frames"][0]["sha256"] = hashlib.sha256(
        frame_path.read_bytes()
    ).hexdigest()
    write_json(path.parent / "reference.json", manifest)
    binding = read_json_object(path)
    binding["artifacts"] = [
        descriptor(root, path.parent / name, "independent_reference")
        for name in ("reference.json", "frame_0000.json", "frame_0000.npz")
    ]
    write_json(path, binding)


def quality_record(cache: Cache, alias: dict[str, Any]) -> dict[str, Any]:
    verified = precision.verify_reference_precision(
        cache.root, alias["reference_precision"]
    )
    inputs = write_json(
        cache.root / "inputs.json",
        {
            "settings": {"spp": 4096, "seed": 0, "threads": 8},
            "reference_precision_policy": cache.policy,
        },
    )
    bindings = write_json(cache.root / "aliases.json", [alias])
    context = {
        "policy": cache.policy,
        "evaluation_inputs": descriptor(
            cache.root, inputs, "reference_precision_evaluation_inputs"
        ),
        "bindings": descriptor(cache.root, bindings, "reference_precision_bindings"),
    }
    return {
        "reference_precision": context,
        "reference_id": alias["reference_id"],
        "reference_ids": [alias["reference_id"]],
        "coverage": {"evaluated_samples": 1},
        "artifacts": [*verified["artifacts"], *context.values()],
    }


def test_policy_is_explicit_immutable_and_frozen_before_acquisition(
    tmp_path: Path,
) -> None:
    value = precision.precision_policy()
    value["threshold"] = 1
    assert precision.precision_policy()["threshold"] == 0.01
    for settings in (
        {"spp": 256, "seed": 0, "threads": 8},
        {"spp": 4096, "seed": 1, "threads": 8},
        {"spp": 4096, "seed": 0, "threads": 4},
    ):
        with pytest.raises(ValueError, match="requires"):
            precision.freeze_precision_policy(tmp_path, tmp_path / "choices", settings)
    with pytest.raises(ValueError, match="Unknown"):
        precision.precision_policy("adaptive-best-image")
    cache = Cache(tmp_path, (0.0, 0.0))
    assert (
        read_json_object(tmp_path / cache.policy["path"])
        == precision.precision_policy()
    )
    assert cache.calls == []


@pytest.mark.parametrize(
    "deviations,expected,passed",
    [
        ((0.0, 0.0), [4096], True),
        ((0.1, 0.0), [4096, 16384], True),
        ((0.1, 0.05), [4096, 16384], False),
    ],
)
def test_fixed_ladder_preserves_every_attempt_and_selects_first_pass_or_cap(
    tmp_path: Path, deviations: tuple[float, float], expected: list[int], passed: bool
) -> None:
    cache = Cache(tmp_path, deviations)
    ref, alias = cache.select()
    assert cache.calls == expected
    proof = precision.verify_reference_precision(
        tmp_path, alias["reference_precision"], expected_alias=alias
    )
    assert proof["passed"] is passed
    assert proof["selected"]["spp"] == expected[-1]
    assert len(proof["selection"]["attempts"]) == len(expected)
    assert ref["reference_validation"]["passed"] is passed
    assert (
        len([item for item in ref["artifacts"] if item["kind"] == "reference_metadata"])
        == 1
    )
    record = quality_record(cache, alias)
    assert precision.precision_quality_reasons(tmp_path, record) == (
        [] if passed else ["quality_reference_precision_noise"]
    )


def test_reused_geometry_preserves_new_raw_and_history_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = Cache(tmp_path, (0.1, 0.0))
    _, first = cache.select()
    cache.frame_id, cache.time_s = 7, 7 / 30
    cache.history = {"exposure_index": 107, "scoring_index": 108}
    monkeypatch.setattr(
        precision,
        "measure_reference_noise",
        lambda *args, **kwargs: pytest.fail(
            "Already checksum-verified choice must be reused"
        ),
    )
    ref, alias = cache.select()
    assert alias["reference_precision"] == first["reference_precision"]
    assert alias["frame_id"] == ref["metadata"]["frame_id"] == 7
    assert alias["time_s"] == 7 / 30
    assert alias["history"] == ref["metadata"]["history"] == cache.history


@pytest.mark.parametrize(
    "damage",
    [
        "missing_repeat",
        "nonfinite",
        "false_noise",
        "wrong_seeds",
        "wrong_denominator",
        "corrupt_bytes",
    ],
)
def test_corrupt_or_inconsistent_noise_stops_without_escalating(
    tmp_path: Path, damage: str
) -> None:
    cache = Cache(tmp_path, (0.1, 0.0))

    def mutate(path: Path) -> None:
        arrays_path = path.parent / "frame_0000.npz"
        if damage == "corrupt_bytes":
            arrays_path.write_bytes(b"corrupt")
            return
        with np.load(arrays_path) as data:
            arrays = {key: data[key] for key in data.files}
        if damage == "missing_repeat":
            del arrays["rgb_repeat"]
        elif damage == "nonfinite":
            arrays["rgb_repeat"][0, 0, 0] = float("nan")
        np.savez_compressed(arrays_path, **arrays)
        frame_path = path.parent / "frame_0000.json"
        frame = read_json_object(frame_path)
        if damage == "false_noise":
            frame["reference_validation"].update(value=0.0, passed=True)
        elif damage == "wrong_seeds":
            frame["reference_validation"]["seeds"] = [2, 3]
        elif damage == "wrong_denominator":
            frame["reference_validation"]["denominator"] = 3
        write_json(frame_path, frame)
        reseal_cache(path)

    cache.mutate = mutate
    with pytest.raises(ValueError, match="precision stopped"):
        cache.select()
    assert cache.calls == [4096]
    failures = list((tmp_path / "choices").glob("failure_*.sha256.json"))
    assert len(failures) == 1
    failure = read_json_object(tmp_path / read_json_object(failures[0])["path"])
    assert failure["escalation_stopped"] is True


def test_missing_reference_does_not_trigger_cap(tmp_path: Path) -> None:
    cache = Cache(tmp_path, (0.0, 0.0))
    calls = []

    def acquire(settings: dict[str, int]) -> tuple[None, dict[str, Any]]:
        calls.append(settings)
        return None, {"status": "missing", "reason": "renderer unavailable"}

    with pytest.raises(ValueError, match="unavailable"):
        precision.select_reference(
            tmp_path, tmp_path / "choices", cache.policy, acquire
        )
    assert len(calls) == 1


def test_precision_cannot_change_reference_lighting_between_rungs(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path, (0.1, 0.0))

    def mutate(path: Path) -> None:
        if read_json_object(path)["identity"]["settings"]["spp"] != 16384:
            return
        frame_path = path.parent / "frame_0000.json"
        frame = read_json_object(frame_path)
        provenance = frame["reference_provenance"]
        provenance["lighting"]["fixture_radiance"] = 0.25
        del provenance["reference_id"]
        provenance["reference_id"] = stable_hash(provenance)
        write_json(frame_path, frame)
        reseal_cache(path)

    cache.mutate = mutate
    with pytest.raises(ValueError, match="change only spp"):
        cache.select()
    assert cache.calls == [4096, 16384]
    failure = read_json_object(tmp_path / "choices" / "failure_0000.json")
    assert len(failure["completed_attempts"]) == 2
    for attempt in failure["completed_attempts"]:
        assert (tmp_path / attempt["path"]).is_file()


def test_resealed_nonfirst_choice_and_dropped_sources_are_rejected(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path, (0.1, 0.0))
    _, alias = cache.select()
    record = quality_record(cache, alias)
    record["artifacts"] = record["artifacts"][1:]
    assert "quality_reference_precision" in precision.precision_quality_reasons(
        tmp_path, record
    )
    path = tmp_path / alias["reference_precision"]["path"]
    selection = read_json_object(path)
    selection["selected_index"] = 0
    write_json(path, selection)
    with pytest.raises(ValueError, match="selection rule"):
        precision.verify_reference_precision(
            tmp_path, descriptor(tmp_path, path, "reference_precision_selection")
        )


def test_report_verifier_remains_stdlib_only_and_detects_later_tampering(
    tmp_path: Path,
) -> None:
    cache = Cache(tmp_path, (0.1, 0.0))
    _, alias = cache.select()
    record = quality_record(cache, alias)
    record_path = write_json(tmp_path / "record.json", record)
    command = "import json,sys; from pathlib import Path; from scripts.benchmark.rendering.reference_precision import precision_quality_reasons; assert precision_quality_reasons(Path(sys.argv[1]),json.loads(Path(sys.argv[2]).read_text())) == []; assert 'numpy' not in sys.modules and 'torch' not in sys.modules"
    subprocess.run(
        [sys.executable, "-S", "-c", command, str(tmp_path), str(record_path)],
        check=True,
    )
    attempt = read_json_object(
        tmp_path
        / read_json_object(tmp_path / alias["reference_precision"]["path"])["attempts"][
            0
        ]["path"]
    )
    source = tmp_path / next(
        item["path"]
        for item in attempt["source_evidence"]
        if item["path"].endswith(".npz")
    )
    source.write_bytes(source.read_bytes() + b"tamper")
    assert "quality_reference_precision" in precision.precision_quality_reasons(
        tmp_path, record
    )


def test_publication_gate_cannot_accept_manual_pass_after_failed_cap(
    tmp_path: Path,
) -> None:
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_publication_report.py"))
    )
    rows = fixture["_rows"](repeats=1)
    records = fixture["_freeze"](tmp_path, rows)
    cache = Cache(tmp_path, (0.1, 0.05))
    _, alias = cache.select()
    records[0].update(quality_record(cache, alias))
    assert "quality_reference_precision_noise" in publication_report._quality_evidence(
        tmp_path, rows[0], records[0], {}
    )


def test_evaluator_selects_before_candidate_pixels_and_preserves_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = runpy.run_path(
        str(Path(__file__).with_name("test_rendering_evaluate.py"))
    )
    state = fixture["offline"].__wrapped__(monkeypatch)
    case = fixture["RenderCase"](
        "R-04", "franka_workstation", width=8, height=6, num_envs=1
    )
    rows = fixture["experiment"](tmp_path, case, monkeypatch)
    raw = (tmp_path / "raw.jsonl").read_bytes()
    order = []
    candidate = evaluate._candidate

    def load_candidate(item: dict[str, Any]) -> dict[str, Any]:
        order.append("candidate")
        assert order[:2] == [4096, 16384]
        return candidate(item)

    def render(
        scene_path: Path,
        specs: list[dict[str, Any]],
        width: int,
        height: int,
        output: Path,
        **settings: Any,
    ) -> dict[str, Any]:
        order.append(settings["spp"])
        scene = fixture["load_scene"](scene_path)
        identity = {
            "width": width,
            "height": height,
            "settings": settings,
            "scene_sha256": fixture["scene_fingerprint"](scene),
            "reference_implementation_sha256": hashlib.sha256(
                Path(reference.__file__).read_bytes()
            ).hexdigest(),
        }
        metadata = {
            "scene_sha256": identity["scene_sha256"],
            "camera": reference.camera_metadata(scene, specs[0], width, height),
            "frame_id": specs[0]["frame_id"],
            "time_s": specs[0]["time_s"],
        }
        return render_fixture(
            output, identity, 0.1 if settings["spp"] == 4096 else 0.0, metadata=metadata
        )

    monkeypatch.setattr(evaluate, "_candidate", load_candidate)
    monkeypatch.setattr(reference, "render_reference", render)
    evaluate.evaluate_experiment(
        tmp_path,
        spp=4096,
        seed=0,
        threads=8,
        reference_precision_policy=precision.PRECISION_POLICY_ID,
    )
    record = read_json_object(tmp_path / "quality.json")["runs"][0]
    assert record["checks"]["reference_precision"]["status"] == "passed"
    assert record["quality_status"] == "not_qualified"  # Calibration is still missing.
    assert precision.precision_quality_reasons(tmp_path, record) == []
    aliases = read_json(tmp_path / record["reference_precision"]["bindings"]["path"])
    assert len(aliases) == 1
    assert (tmp_path / "raw.jsonl").read_bytes() == raw
    assert read_json(tmp_path / "runs.json")[0]["metrics"] == rows[0]["metrics"]
    assert len(state["frame_calls"]) == 1


def test_cli_requires_explicit_settings_for_opt_in_and_default_stays_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="requires"):
        evaluate.evaluate_experiment(
            tmp_path, reference_precision_policy=precision.PRECISION_POLICY_ID
        )
    calls = []
    monkeypatch.setattr(
        evaluate,
        "evaluate_experiment",
        lambda *args, **kwargs: calls.append(kwargs) or tmp_path,
    )
    assert evaluate.main(["--experiment-root", str(tmp_path)]) == 0
    assert calls[-1]["reference_precision_policy"] is None
    assert (
        evaluate.main(
            [
                "--experiment-root",
                str(tmp_path),
                "--spp",
                "4096",
                "--threads",
                "8",
                "--reference-precision-policy",
                precision.PRECISION_POLICY_ID,
            ]
        )
        == 0
    )
    assert calls[-1]["reference_precision_policy"] == precision.PRECISION_POLICY_ID
