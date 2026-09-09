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
import os
from pathlib import Path
import sys
import subprocess

import imageio.v2 as imageio
import numpy as np
import psutil
import pytest

from embodichain.gen_sim.agent_lab.catalog import write_json
from embodichain.gen_sim.agent_lab.delivery import finalize_run
from embodichain.gen_sim.agent_lab import _launch, session
from embodichain.gen_sim.agent_lab.usage import UsageMeter


@pytest.fixture
def run(tmp_path: Path) -> Path:
    (tmp_path / "workspace").mkdir()
    write_json(
        tmp_path / "run.json",
        {
            "task": {"task_id": "custom", "instruction": "move", "acceptance": "hold"},
            "python": sys.executable,
            "repo": str(tmp_path),
            "mode": "B",
            "objective": "cold_start",
        },
    )
    return tmp_path


def make_attempt(
    root: Path, name: str, *, completed: bool = True, video: bool = True
) -> Path:
    attempt = root / "attempts" / name
    (attempt / "code").mkdir(parents=True)
    (attempt / "code/probe.py").write_text("def solve(lab): return None\n")
    write_json(attempt / "lab.json", {"seed": 0, "scene": "/assets"})
    write_json(
        attempt / "result.json",
        {"execution_completed": completed, "task_success": None},
    )
    write_json(
        attempt / "process.json",
        {
            "returncode": 0 if completed else 1,
            "command": [
                sys.executable,
                "-m",
                "embodichain.gen_sim.agent_lab.worker",
                "--script",
                str(attempt / "code/probe.py"),
                "--config",
                str(attempt / "lab.json"),
                "--output",
                str(attempt),
            ],
        },
    )
    if video:
        with imageio.get_writer(
            str(attempt / "video.mp4"), fps=10, codec="libx264", pixelformat="yuv420p"
        ) as writer:
            for value in (0, 80, 160):
                writer.append_data(np.full((32, 32, 3), value, dtype=np.uint8))
    return attempt


def test_explicit_selection_beats_last_attempt_and_claim_is_not_verdict(
    run: Path,
) -> None:
    selected = make_attempt(run, "a_good")
    make_attempt(run, "z_last")
    write_json(
        run / "workspace/handoff.json",
        {"selected_attempt": "attempts/a_good", "task_completed": True},
    )
    result = finalize_run(run, execution={"status": "completed"})
    assert result["selection"]["attempt_id"] == selected.name
    assert result["assessment"]["task_success"] is None
    assert result["assessment"]["agent_claim"]["task_completed"]
    assert result["video"]["frames"] == 3
    assert (run / "final/report.md").is_file()
    assert (run / "final/video.mp4").read_bytes() == (
        selected / "video.mp4"
    ).read_bytes()
    assert "a_good" in (run / "final/report.md").read_text()
    assert "尚未独立验收" in (run / "final/report.md").read_text()


@pytest.mark.parametrize("status", ["failed", "timed_out", "stopped", "interrupted"])
def test_every_exit_delivers_reports_even_without_video(run: Path, status: str) -> None:
    result = finalize_run(run, execution={"status": status})
    assert result["execution"]["status"] == status
    assert result["video"] is None
    assert result["video_missing_reason"]
    assert not (run / "final/video.mp4").exists()
    assert json.loads((run / "final/result.json").read_text()) == result


def test_invalid_video_and_malformed_handoff_still_produce_delivery(run: Path) -> None:
    attempt = make_attempt(run, "broken", video=False)
    (attempt / "video.mp4").write_bytes(b"invalid video")
    (run / "workspace/handoff.json").write_text("{broken")
    result = finalize_run(run, execution={"status": "failed"})
    assert result["video"] is None
    assert len(result["issues"]) >= 2
    assert (attempt / "video.mp4").read_bytes() == b"invalid video"


def test_foreign_attempt_cannot_be_published(run: Path, tmp_path: Path) -> None:
    foreign = make_attempt(tmp_path / "other_run", "outside")
    make_attempt(run, "local")
    result = finalize_run(run, attempt=foreign, execution={"status": "completed"})
    assert result["video"] is None
    assert any("outside this run" in issue for issue in result["issues"])


def test_symlinked_foreign_video_is_rejected(run: Path, tmp_path: Path) -> None:
    foreign = make_attempt(tmp_path / "other_run", "outside")
    local = make_attempt(run, "local", video=False)
    (local / "video.mp4").symlink_to(foreign / "video.mp4")
    result = finalize_run(run, attempt=local, execution={"status": "failed"})
    assert result["video"] is None


def test_republication_is_idempotent_and_never_leaves_stale_video(run: Path) -> None:
    selected = make_attempt(run, "first")
    first = finalize_run(run, attempt=selected, execution={"status": "completed"})
    assert (
        finalize_run(run, attempt=selected, execution={"status": "completed"}) == first
    )
    old_directory = (run / "final").resolve()
    empty = make_attempt(run, "empty", video=False)
    second = finalize_run(run, attempt=empty, execution={"status": "failed"})
    assert second["video"] is None
    assert not (run / "final/video.mp4").exists()
    assert (old_directory / "video.mp4").exists()


def test_partial_attempt_is_not_a_success_video(run: Path) -> None:
    attempt = make_attempt(run, "partial", completed=False)
    result = finalize_run(run, attempt=attempt, execution={"status": "timed_out"})
    assert result["video"]["kind"] == "partial_attempt"
    assert result["assessment"]["task_success"] is None


def test_live_worker_blocks_publication(run: Path) -> None:
    attempt = make_attempt(run, "active", video=False)
    write_json(
        attempt / "process.json",
        {"pid": os.getpid(), "created_at": psutil.Process().create_time()},
    )
    with pytest.raises(RuntimeError, match="still running"):
        finalize_run(run)
    assert not (run / "final").exists()


def test_failed_atomic_publish_preserves_old_delivery(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = make_attempt(run, "first")
    finalize_run(run, attempt=first, execution={"status": "completed"})
    before = (run / "final/result.json").read_bytes()
    replacement = make_attempt(run, "replacement", video=False)
    original_replace = os.replace

    def fail_pointer(source, target):
        if Path(target) == run / "final":
            raise OSError("simulated interruption")
        return original_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_pointer)
    with pytest.raises(OSError, match="simulated interruption"):
        finalize_run(run, attempt=replacement, execution={"status": "failed"})
    assert (run / "final/result.json").read_bytes() == before


def test_explicit_selection_survives_recovery_without_repeating_flag(run: Path) -> None:
    selected = make_attempt(run, "a_chosen")
    make_attempt(run, "z_newer")
    finalize_run(run, attempt=selected)
    result = finalize_run(run)
    assert result["selection"]["attempt_id"] == "a_chosen"
    assert result["selection"]["source"] == "cli"


def test_other_codecs_are_normalized_without_changing_frame_count(run: Path) -> None:
    attempt = make_attempt(run, "mpeg", video=False)
    with imageio.get_writer(
        str(attempt / "video.mp4"), fps=10, codec="mpeg4"
    ) as writer:
        for value in (0, 80, 160):
            writer.append_data(np.full((32, 32, 3), value, dtype=np.uint8))
    first = finalize_run(run, attempt=attempt)
    assert first["video"]["codec"] == "h264"
    assert first["video"]["frames"] == 3
    assert first["video"]["source_sha256"] != first["video"]["sha256"]
    assert finalize_run(run, attempt=attempt) == first


def test_delivery_recovers_after_host_was_killed(run: Path) -> None:
    attempt = make_attempt(run, "interrupted", completed=False)
    write_json(
        run / "launch.json", {"status": "running", "pid": 99999999, "created_at": 0}
    )
    result = finalize_run(run, attempt=attempt)
    assert result["execution"]["status"] == "interrupted"
    assert result["execution"]["recovered"]
    assert result["video"]["kind"] == "partial_attempt"


@pytest.mark.parametrize(
    "outcome", ["completed", "failed", "timed_out", "stopped", "interrupted"]
)
def test_launch_always_finalizes_terminal_paths(
    run: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    def fake_process(command, *, output, **kwargs):
        if outcome in ("stopped", "interrupted"):
            write_json(
                output / "process.json", {"interrupted": True, "returncode": -15}
            )
            if outcome == "stopped":
                write_json(run / "stop.json", {"launch": str(output)})
            raise KeyboardInterrupt
        return {
            "timed_out": outcome == "timed_out",
            "returncode": 0 if outcome == "completed" else 1,
        }

    monkeypatch.setattr(_launch, "_process", fake_process)
    if outcome == "interrupted":
        with pytest.raises(KeyboardInterrupt):
            _launch._launch(
                run, minutes=1, batch=True, model="gpt-6-astra", effort="xhigh"
            )
    else:
        _launch._launch(run, minutes=1, batch=True, model="gpt-6-astra", effort="xhigh")
    result = json.loads((run / "final/result.json").read_text())
    assert result["execution"]["status"] == outcome
    assert result["video"] is None


def test_legacy_search_timeout_finalizes(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = json.loads((run / "run.json").read_text())
    manifest["task"].update(
        source_dir="/assets", image="/image.png", asset_status="exported"
    )
    write_json(run / "run.json", manifest)

    def fake_process(command, *, output, **kwargs):
        (output / "stdout.log").write_text("")
        return {"timed_out": True, "returncode": -15}

    monkeypatch.setattr(session, "_process", fake_process)
    session.solve(run, minutes=1)
    assert (
        json.loads((run / "final/result.json").read_text())["execution"]["status"]
        == "timed_out"
    )


def test_standalone_cli_produces_a_delivery_without_simulation(run: Path) -> None:
    script = run / "workspace/plain.py"
    script.write_text("raise RuntimeError('expected failure')\n")
    repo = Path(__file__).resolve().parents[3]
    manifest = json.loads((run / "run.json").read_text())
    manifest["repo"] = str(repo)
    write_json(run / "run.json", manifest)
    env = {**os.environ, "PYTHONPATH": str(repo)}
    env.pop("GENSIM_LAB_BROKER", None)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "embodichain.gen_sim.agent_lab",
            "run",
            "--run-dir",
            str(run),
            "--script",
            str(script),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stderr
    delivery = json.loads((run / "final/result.json").read_text())
    assert delivery["execution"]["status"] == "failed"
    assert delivery["video"] is None
    assert delivery["reproduction"]["command"] is not None
    assert delivery["reproduction"]["file_integrity"] == "matched"


def test_changed_frozen_code_is_not_presented_as_reproducible(run: Path) -> None:
    attempt = make_attempt(run, "changed")
    write_json(attempt / "inputs.json", {"files_sha256": {"code/probe.py": "old-hash"}})
    result = finalize_run(run, attempt=attempt)
    assert result["reproduction"]["file_integrity"] == "mismatch"
    assert result["reproduction"]["command"] is None
    assert any("changed after execution" in issue for issue in result["issues"])


def test_newer_delivery_state_does_not_hide_an_active_host(run: Path) -> None:
    write_json(
        run / "launch.json",
        {
            "status": "running",
            "pid": os.getpid(),
            "created_at": psutil.Process().create_time(),
        },
    )
    write_json(run / "delivery_state.json", {"status": "completed"})
    with pytest.raises(RuntimeError, match="still running"):
        finalize_run(run)


def test_report_uses_host_usage_not_agent_guess(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(run / "home"))
    output = run / "codex/one"
    output.mkdir(parents=True)
    write_json(
        output / "process.json",
        {"started_at": 100, "wall_seconds": 20, "returncode": 0},
    )
    (output / "stdout.log").write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 10},
            }
        )
        + "\n"
    )
    write_json(run / "workspace/handoff.json", {"token_estimate": 999999999})
    result = finalize_run(run)
    assert result["resources"]["tokens"]["total_tokens"] == 110
    assert json.loads((run / "usage.json").read_text()) == result["resources"]
    report = (run / "final/report.md").read_text()
    assert "Token 总量：110" in report
    assert "999999999" not in report


def test_publication_closes_meter_and_remains_idempotent(
    run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEX_HOME", str(run / "home"))
    meter = UsageMeter(run, "test")
    try:
        first = finalize_run(run, usage_meter=meter)
    finally:
        meter.close()
    assert first["resources"]["timing"]["status"] == "reported"
    assert first["resources"]["timing"]["total_wall_seconds"] > 0
    assert finalize_run(run) == first
