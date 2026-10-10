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

"""Small fake-buffer history acquisition tests; no simulator or CUDA is used."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import runpy
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.rendering import capture, formal_worker
from scripts.benchmark.rendering.frozen_source import read_json
from scripts.benchmark.rendering.history_worker import capture_history


class Array:
    is_cuda = True

    def __init__(self, data: np.ndarray) -> None:
        self.data, self.dtype, self.shape = data, data.dtype, data.shape


class Adapter:
    def __init__(self) -> None:
        self.counter, self.host_calls = 0, 0
        self.states = []

    def apply_state(self, state: dict[str, Any]) -> None:
        self.state = deepcopy(state)
        self.views = list(self.state["cameras"].values())

    def set_cameras(self, views: list[dict[str, Any]]) -> None:
        self.views = deepcopy(views)

    def render_device(self) -> dict[str, Array]:
        self.counter += 1
        self.states.append(self.state["frame_id"])
        value = self.state["frame_id"] * 50 + round(self.views[0]["eye"][0] * 100)
        return {"rgb": Array(np.full((1, 4, 4, 3), value, dtype=np.uint8))}

    def synchronize(self) -> None:
        pass

    def to_host(self, arrays: dict[str, Array]) -> dict[str, np.ndarray]:
        self.host_calls += 1
        return {name: array.data.copy() for name, array in arrays.items()}

    def capture_evidence(self) -> dict[str, Any]:
        return {
            "status": "recorded",
            "render_submission_id": self.counter,
            "logical_state": {
                "frame_id": self.state["frame_id"],
                "time_s": self.state["time_s"],
                "num_envs": 1,
                "camera_names": [view["name"] for view in self.views],
                "modalities": ["rgb"],
            },
        }


@pytest.mark.parametrize("delivery", ["device_ready", "host_ready"])
def test_original_loop_tap_has_no_extra_exposures_and_preserves_guard(
    tmp_path: Path, delivery: str
) -> None:
    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "contract")
    request = read_json(fixture["root"] / "request.json")
    request["source"] = {"commit": "a" * 40}
    config = fixture["config"]
    config["case"]["delivery"] = delivery
    scene = read_json(fixture["root"] / "original/scene.json")
    case = SimpleNamespace(**config["case"], case_id="fixture-moving")
    adapter = Adapter()
    phases = []
    sampler = SimpleNamespace(mark_phase=phases.append)
    output = tmp_path / "fresh"
    output.mkdir()
    original_function = capture.capture_packet
    worker_function = formal_worker.capture_packet
    manifest, events, freshness = capture_history(
        adapter,
        formal_worker,
        capture,
        request,
        config,
        scene,
        fixture["replay"],
        case,
        output,
        sampler,
    )
    assert freshness["passed"]
    assert adapter.counter == 4 + 2 + 6
    assert adapter.states == [0] * 4 + [0, 1] + [0, 1, 2, 0, 1, 2]
    assert manifest["capture_batches"] == 7
    assert manifest["performance_claim"] is False
    assert (
        capture.capture_packet is original_function
        and formal_worker.capture_packet is worker_function
    )
    frames = [
        read_json(output / item["path"])["metadata"] for item in manifest["frames"]
    ]
    assert [frame["frame_id"] for frame in frames] == [1, 0, 1, 2, 0, 1, 2]
    assert [frame["time_s"] for frame in frames] == [
        1 / 30,
        0,
        1 / 30,
        2 / 30,
        0,
        1 / 30,
        2 / 30,
    ]
    assert [frame["history"]["scoring_index"] for frame in frames] == list(range(7))
    assert all(frame["moving_instance_ids"] == [1] for frame in frames)
    with np.load(output / manifest["frames"][0]["arrays_path"]) as data:
        assert np.all(data["rgb"] == 50)  # Actual last warmup, not a new capture.
    if delivery == "device_ready":
        assert adapter.host_calls == 4 + 7
        assert (
            sum(
                event.get("additional_quality_delivery", {}).get("d2h_group_calls", 0)
                for group in events.values()
                for event in group
            )
            == 7
        )
        assert all(
            event["packet"]["d2h_group_calls"] == 0 for event in events["measured"]
        )
    else:
        assert adapter.host_calls == adapter.counter
        assert all(
            event["additional_quality_delivery"]["d2h_group_calls"] == 0
            for event in events["measured"]
        )
    assert phases == [
        "freshness_probe",
        "warmup",
        "measurement",
        "measurement_complete",
    ]


def test_dry_run_is_nonmutating_and_retains_full_source_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering import history_acquisition as launcher

    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    original_builder = runpy.run_path(
        str(Path(__file__).with_name("test_history_binding.py"))
    )["original_experiment"]
    fixture = builder(tmp_path / "proof")
    original = original_builder(fixture, tmp_path / "original")
    source = {
        "worktree": str(tmp_path / "source"),
        "commit": "a" * 40,
        "source_files": fixture["manifest"]["original"]["source_files"],
    }
    monkeypatch.setattr(launcher, "verify_source_worktree", lambda *args: source)
    monkeypatch.setattr(
        launcher,
        "original_environment",
        lambda *args: pytest.fail("dry-run attempted process environment setup"),
    )
    output = tmp_path / "not-created"
    plan = launcher.acquire_history(
        original["root"],
        original["row"]["run_id"],
        source_worktree=Path(source["worktree"]),
        source_commit=source["commit"],
        output=output,
        dry_run=True,
    )
    assert not output.exists()
    assert plan["schedule"]["measured_frames"] == 6
    assert plan["schedule"]["entry_seam"]["from_frame_id"] == 1
    assert plan["performance_claim"] is False
    with pytest.raises(ValueError, match="outside the frozen original tree"):
        launcher.acquire_history(
            original["root"],
            original["row"]["run_id"],
            source_worktree=Path(source["worktree"]),
            source_commit=source["commit"],
            output=original["root"] / "forbidden",
            dry_run=True,
        )


def test_complete_checkpoint_survives_native_close_terminating_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering import history_worker as producer
    from scripts.benchmark.rendering.frozen_source import write_json

    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    fixture = builder(tmp_path / "proof")
    request = read_json(fixture["root"] / "request.json")
    request.update(
        kind="timed_render_history_request/v1",
        source={
            "worktree": str(tmp_path / "source"),
            "commit": "a" * 40,
            "source_files": request["original"]["source_files"],
        },
        producer_artifacts=[
            row
            for row in fixture["manifest"]["producer_artifacts"]
            if row["role"] != "request"
        ],
    )
    request_path = write_json(fixture["root"] / "request.json", request)
    output = fixture["root"] / "acquired"
    adapter = Adapter()
    adapter.metadata = read_json(
        fixture["root"] / fixture["manifest"]["renderer"]["path"]
    )
    adapter.metadata["physics_enabled"] = False

    def close() -> None:
        result = read_json(output / "result.json")
        assert result["status"] == "completed"
        assert result["cleanup_status"] == "pending_native_close"
        assert result["metrics"] == {}
        history = read_json(fixture["root"] / "history.json")
        assert history["status"] == "completed"
        assert history["frames_manifest"]["path"] == "acquired/frames.json"
        assert history["acquisition"]["persistence"]["extra_render_submissions"] == 0
        raise SystemExit(0)

    adapter.close = close

    class Sampler:
        def __init__(self, gpu_id: int) -> None:
            pass

        def __enter__(self) -> Sampler:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

        def mark_phase(self, name: str) -> None:
            pass

        def summary(self) -> dict[str, Any]:
            return {"gpu": {"uuid": "GPU-fixture", "driver": "fixture-driver"}}

    monkeypatch.setattr(
        producer.frozen, "verify_source_worktree", lambda *args: request["source"]
    )
    monkeypatch.setattr(producer.frozen, "pin_source_imports", lambda *args: None)
    monkeypatch.setattr(
        producer.frozen,
        "verify_imported_sources",
        lambda *args: fixture["manifest"]["acquisition"]["imported_sources"],
    )
    monkeypatch.setattr(
        formal_worker,
        "RenderCase",
        lambda **kwargs: SimpleNamespace(**kwargs, case_id="fixture-moving"),
    )
    monkeypatch.setattr(
        formal_worker,
        "_preflight",
        lambda *args: (
            read_json(fixture["root"] / "original/scene.json"),
            fixture["replay"],
        ),
    )
    monkeypatch.setattr(formal_worker, "_create_adapter", lambda *args: adapter)
    monkeypatch.setattr(formal_worker, "ResourceSampler", Sampler)
    with pytest.raises(SystemExit) as caught:
        producer.run_worker(request_path, output, 0)
    assert caught.value.code == 0
    assert adapter.counter == 12


def test_launcher_registers_auxiliaries_before_final_proof_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.rendering import history_acquisition as launcher
    from scripts.benchmark.rendering.frozen_source import artifact, write_json

    builder = runpy.run_path(str(Path(__file__).with_name("test_history_contract.py")))[
        "build_valid_history_fixture"
    ]
    original_builder = runpy.run_path(
        str(Path(__file__).with_name("test_history_binding.py"))
    )["original_experiment"]
    fixture = builder(tmp_path / "proof")
    original = original_builder(fixture, tmp_path / "original")
    source = {
        "worktree": str(fixture["root"] / "original/source"),
        "commit": "a" * 40,
        "source_files": fixture["manifest"]["original"]["source_files"],
    }
    monkeypatch.setattr(launcher, "verify_source_worktree", lambda *args: source)
    monkeypatch.setattr(launcher, "original_environment", lambda *args: {})

    def fake_worker(root: Path, specs: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert len(specs) == 1
        shutil.copytree(fixture["root"] / "capture", root / "capture")
        request = read_json(root / "request.json")
        history = deepcopy(fixture["manifest"])
        for key in (
            "original",
            "schedule",
            "schedule_sha256",
            "history_quality_protocol",
            "history_quality_requirements",
        ):
            history[key] = request[key]
        history["producer_artifacts"] = [
            *request["producer_artifacts"],
            artifact(root, root / "request.json", "history_request", role="request"),
        ]
        write_json(root / "history.json", history)
        return [{"status": "completed"}]

    monkeypatch.setattr(launcher, "run_experiment", fake_worker)
    result = launcher.acquire_history(
        original["root"],
        original["row"]["run_id"],
        source_worktree=Path(source["worktree"]),
        source_commit=source["commit"],
        output=tmp_path / "acquisition",
    )
    assert result["status"] == "completed"
    checked = read_json(Path(result["root"]) / "history_validation.json")
    assert checked["valid"], checked["reasons"]
