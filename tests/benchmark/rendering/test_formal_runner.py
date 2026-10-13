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

"""Formal runner selection, artifact and process contracts using CPU fixtures."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.benchmark.core.artifacts import (
    ArtifactStore,
    read_json,
    stable_hash,
    write_json,
)
from scripts.benchmark.rendering import formal_runner as runner
from scripts.benchmark.rendering.formal import (
    RenderCase,
    experiment_cases,
    validate_approval,
    view_fingerprint,
)


@pytest.fixture
def approved(tmp_path: Path) -> Path:
    root = tmp_path / "approved"
    hashes = {}
    for name in ("calibration", "workshop", "franka_workstation"):
        scene = {
            "schema_version": 1,
            "scene_id": name,
            "units": "metres",
            "up_axis": "Z",
            "physics_steps": 0,
            "camera_model": {"horizontal_fov_deg": 60.0, "near_m": 0.01, "far_m": 10.0},
            "objects": [
                {
                    "uid": "box",
                    "instance_id": 1,
                    "geometry": {"kind": "box", "size": [0.1, 0.1, 0.1]},
                }
            ],
            "cameras": [
                {
                    "name": camera,
                    "eye": [1.5, -1.5 + index * 0.2, 1.5],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                }
                for index, camera in enumerate(
                    ("external_front", "external_side", "wrist")
                )
            ],
        }
        write_json(root / name / "scene.json", scene)
        (root / name / "preview-only.txt").write_text("Not a canonical scene input")
        hashes[name] = stable_hash(scene)
    write_json(
        root / "approval.json",
        {
            "status": "approved",
            "scope": "scene_layout_and_camera_scheme",
            "scenes": hashes,
            "approved_by": "test fixture",
        },
    )
    return root


@pytest.fixture
def fixture_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = tmp_path / "IsaacLab"
    (source / "source" / "isaaclab").mkdir(parents=True)
    (source / "source" / "isaaclab_assets").mkdir()
    worker = tmp_path / "fixture_worker.py"
    worker.write_text("""
import json
from pathlib import Path
import sys
config_path, output, backend, repeat, mode = sys.argv[1:]
config = json.loads(Path(config_path).read_text())
failed = mode == 'oom' and backend == 'embodichain'
gpu = {'uuid': 'CPU-fixture-device', 'name': 'CPU fixture; no GPU execution'}
row = {
    'schema_version': 1, 'backend': backend, 'repeat': int(repeat),
    'status': 'failed' if failed else 'completed',
    'failure_reason_code': 'oom' if failed else None,
    'error': 'synthetic OOM for launcher contract' if failed else None,
    'quality_status': 'not_qualified', 'config': config['case'],
    'config_sha256': config['config_sha256'],
    'scene_sha256': config['scene_sha256'], 'view_sha256': config['view_sha256'],
    'profile_sha256': config['profile_sha256'], 'hardware_id': gpu['uuid'],
    'resources': {'gpu': gpu}, 'execution_attempted': True,
    'metrics': {'camera_frames_per_s': 12.0, 'batch_steps_per_s': 12.0,
                'boundary': 'fixture', 'completion': 'fixture'},
}
Path(output, 'result.json').write_text(json.dumps(row))
sys.exit(1 if failed else 0)
""")
    state: dict[str, Any] = {"mode": "success", "commands": [], "source": source}

    def command(
        python: Path, config: Path, output: Path, backend: str, gpu_id: int, repeat: int
    ) -> tuple[str, ...]:
        state["commands"].append((backend, repeat, read_json(config)))
        return (
            str(python.absolute()),
            str(worker),
            str(config),
            str(output),
            backend,
            str(repeat),
            state["mode"],
        )

    monkeypatch.setattr(runner, "_worker_command", command)
    monkeypatch.setattr(
        runner,
        "_device_evidence",
        lambda gpu_id: {
            "method": "CPU fixture",
            "gpu": {
                "index": gpu_id,
                "uuid": "CPU-fixture-device",
                "name": "CPU fixture; no GPU execution",
            },
        },
    )
    return state


def _args(approved: Path, output: Path, runtime: dict[str, Any]) -> list[str]:
    return [
        "--scene-root",
        str(approved),
        "--output",
        str(output),
        "--isaaclab-root",
        str(runtime["source"]),
        "--isaaclab-python",
        sys.executable,
        "--embodichain-python",
        sys.executable,
    ]


def _roots(output: Path) -> tuple[Path, list[Path]]:
    invocation = next(output.iterdir())
    index = read_json(invocation / "index.json")
    return invocation, [invocation / record["path"] for record in index["experiments"]]


def test_list_and_dry_run_are_nonmutating_without_site_packages(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    script = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'torch', 'pynvml', 'embodichain', 'dexsim', 'isaaclab', 'isaacsim'} or fullname.endswith(('.replay', '.resources', '.formal_worker')):
            raise AssertionError('Runtime imported during inspection: ' + fullname)
sys.meta_path.insert(0, Block())
from scripts.benchmark.rendering.formal_runner import main
main(sys.argv[1:])
"""
    for flag in ("--list", "--dry-run"):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                script,
                flag,
                *(["--json", "--show-excluded"] if flag == "--list" else []),
                "--experiment",
                "R04",
                "--scene-root",
                str(tmp_path / "missing"),
                "--output",
                str(output),
                "--max-cases",
                "1",
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout)
        assert plan["filesystem_mutations"] is False
        assert plan["selected_case_count"] == 1
        assert any(
            record["selection_reasons"] == ["max_cases_limit"]
            for record in plan["cases"]
        )
    assert not output.exists()


def test_list_is_compact_and_scoped_while_full_json_remains_explicit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runner.main(["--list", "--experiment", "R04"])
    text = capsys.readouterr().out
    assert "5 selected" in text
    assert len(text.splitlines()) == 7
    assert "r-03-" not in text
    runner.main(["--list", "--experiment", "R04", "--json"])
    listing = json.loads(capsys.readouterr().out)
    assert len(listing["cases"]) == 5
    assert all(record["selected"] for record in listing["cases"])
    runner.main(["--list", "--experiment", "R04", "--show-excluded"])
    assert "excluded:experiment_filter" in capsys.readouterr().out


def test_smoke_preserves_case_semantics_and_publication_repeat_policy(
    capsys: pytest.CaptureFixture[str],
) -> None:
    case = next(case for case in experiment_cases("R-03") if case.width == 1920)
    runner.main(["--dry-run", "--case-id", case.case_id, "--smoke"])
    smoke = json.loads(capsys.readouterr().out)
    selected = [record for record in smoke["cases"] if record["selected"]]
    assert selected[0]["case"] == case.to_dict()
    assert selected[0]["repeats"] == 1
    assert (smoke["warmup_frames"], smoke["measured_frames"]) == (2, 4)
    runner.main(["--dry-run", "--publication", "--repeats", "3"])
    publication = json.loads(capsys.readouterr().out)
    assert {
        record["repeats"]
        for record in publication["cases"]
        if record["experiment_id"] == "R-04"
    } == {10}
    assert {
        record["repeats"]
        for record in publication["cases"]
        if record["experiment_id"] in {"R-01", "R-02"}
    } == {1}
    assert publication["repeat_policy"]["publication_minimum"] == 10
    moving_rgb = next(
        record
        for record in publication["cases"]
        if record["case"]["temporal_mode"] == "moving"
        and record["case"]["modalities"] == ["rgb"]
    )
    assert (
        moving_rgb["quality_protocol"]["temporal_quality"]
        == "assessed_by_declared_policy"
    )
    assert (
        moving_rgb["quality_protocol"]["protocol_id"]
        == "pure-render-quality/rgb-temporal30-v1"
    )
    assert moving_rgb["quality_protocol"]["evaluation_kind"] == "sequence"
    sequence = next(
        record for record in publication["cases"] if record["experiment_id"] == "R-02"
    )
    assert (
        sequence["quality_protocol"]["protocol_id"]
        == "pure-render-quality/temporal30-v1"
    )


def test_runtime_pins_isaac_source_and_preserves_symlink_virtualenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "IsaacLab"
    first = source / "source" / "isaaclab"
    second = source / "source" / "isaaclab_tasks"
    first.mkdir(parents=True)
    second.mkdir()
    python = tmp_path / "runtime" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CONDA_PREFIX", "/wrong/conda")
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "wrong")
    monkeypatch.setenv("PYTHONPATH", "/wrong/checkout")

    environment = runner._runtime_environment(tmp_path / "repo", python, source)

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        str(tmp_path / "repo"),
        str(first),
        str(second),
    ]
    assert environment["VIRTUAL_ENV"] == str(python.absolute().parent.parent)
    assert environment["VIRTUAL_ENV"] != str(python.resolve().parent.parent)
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert not any(key.startswith("CONDA_") for key in environment)


def test_real_core_execution_retains_full_plan_oom_and_raw_evidence(
    tmp_path: Path, approved: Path, fixture_runtime: dict[str, Any]
) -> None:
    output = tmp_path / "results"
    fixture_runtime["mode"] = "oom"
    with pytest.raises(SystemExit) as error:
        runner.main(
            _args(approved, output, fixture_runtime)
            + ["--experiment", "R-04", "--max-cases", "1", "--smoke"]
        )
    assert error.value.code == 1
    invocation, roots = _roots(output)
    root = roots[0]
    rows = read_json(root / "runs.json")
    assert len(rows) == len(experiment_cases("R-04")) * 2
    assert sum(row["status"] == "completed" for row in rows) == 1
    assert (
        sum(
            row["status"] == "failed" and row.get("failure_reason_code") == "oom"
            for row in rows
        )
        == 1
    )
    assert sum(row["status"] == "not_run" for row in rows) == 8
    assert all(
        row["selection_reason"] == "max_cases_limit"
        for row in rows
        if row["status"] == "not_run"
    )
    raw = [json.loads(line) for line in (root / "raw.jsonl").read_text().splitlines()]
    assert {row["record_id"] for row in raw} == {row["run_id"] for row in rows}
    assert all(
        {key: value for key, value in raw_row.items() if key != "record_id"}
        == next(row for row in rows if row["run_id"] == raw_row["run_id"])
        for raw_row in raw
    )
    assert read_json(root / "summary.json")["selection_coverage"] == {
        "full_planned_trials": 10,
        "selected_trials": 2,
        "runnable_trials": 2,
        "excluded_trials": 8,
        "started_processes": 2,
        "completed_selected_trials": 1,
        "selected_not_run": 0,
    }
    assert read_json(invocation / "acceptance.json")["status"] == "not_run"
    assert read_json(root / "definition.json")["definition_version"] == "2.0"
    assert len(read_json(root / "definition.json")["parameter_matrix"]["case"]) == 5
    assert read_json(root / "definition.json")["budget"]["max_runs"] == 2
    protocols = read_json(root / "definition.json")["quality_protocol"]
    assert set(protocols["policies"]) == {"pure-render-quality/rgb-v1"}
    assert len(protocols["case_protocols"]) == 5
    assert all(
        assignment["protocol_sha256"]
        == stable_hash(protocols["policies"][assignment["protocol_id"]])
        for assignment in protocols["case_protocols"].values()
    )
    assert validate_approval(root / "scenes", root / "approval.json") == read_json(
        approved / "approval.json"
    )
    assert not list((root / "scenes").rglob("preview-only.txt"))
    # Recover even if the materialized runs projection lost a completed row.
    write_json(
        root / "runs.json", [row for row in rows if row["status"] != "completed"]
    )
    before = (root / "raw.jsonl").read_bytes()
    runner.main(["--report-only", str(root)])
    assert (
        sum(row["status"] == "completed" for row in read_json(root / "runs.json")) == 1
    )
    assert (root / "raw.jsonl").read_bytes() == before
    from scripts.benchmark.rendering.publication_report import rebuild_report

    rebuild_report(root)
    assert (
        read_json(root / "summary.json")["selection_coverage"]["selected_trials"] == 2
    )
    assert "Selection coverage" in (root / "report.md").read_text()


def test_backend_order_alternates_and_config_hash_ignores_output_paths(
    tmp_path: Path, approved: Path, fixture_runtime: dict[str, Any]
) -> None:
    common = [
        "--experiment",
        "R04",
        "--max-cases",
        "1",
        "--repeats",
        "3",
        "--warmup-frames",
        "0",
        "--measured-frames",
        "1",
    ]
    output = tmp_path / "first"
    runner.main(_args(approved, output, fixture_runtime) + common)
    assert [
        (backend, repeat) for backend, repeat, _ in fixture_runtime["commands"]
    ] == [
        ("embodichain", 0),
        ("isaaclab", 0),
        ("isaaclab", 1),
        ("embodichain", 1),
        ("embodichain", 2),
        ("isaaclab", 2),
    ]
    original = fixture_runtime["commands"][0][2]
    fixture_runtime["commands"].clear()
    runner.main(
        _args(approved, tmp_path / "second", fixture_runtime)
        + common
        + ["--backend", "E"]
    )
    moved = fixture_runtime["commands"][0][2]
    assert original["scene_path"] != moved["scene_path"]
    assert original["config_sha256"] == moved["config_sha256"]
    assert original["view_sha256"] == moved["view_sha256"]
    assert original["profile_sha256"] == moved["profile_sha256"]


def test_changed_approval_is_rejected_before_output_creation(
    tmp_path: Path, approved: Path, fixture_runtime: dict[str, Any]
) -> None:
    scene = read_json(approved / "calibration" / "scene.json")
    scene["cameras"][0]["eye"][0] += 0.01
    write_json(approved / "calibration" / "scene.json", scene)
    output = tmp_path / "absent"
    with pytest.raises(ValueError, match="Approved scene was changed"):
        runner.main(
            _args(approved, output, fixture_runtime)
            + ["--experiment", "R04", "--max-cases", "1"]
        )
    assert not output.exists()
    assert not fixture_runtime["commands"]


def test_replay_preparation_once_per_scene_precedes_workers(
    tmp_path: Path,
    approved: Path,
    fixture_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import replay

    prepared = []

    def prepare(scene_path: Path, output: Path, frames: int, fps: float) -> Path:
        assert not fixture_runtime["commands"]
        prepared.append((scene_path, frames, fps))
        return write_json(
            output,
            {
                "frame_count": frames,
                "fps": fps,
                "frames": [],
                "scene_sha256": stable_hash(read_json(scene_path)),
            },
        )

    monkeypatch.setattr(replay, "prepare_replay", prepare)
    first = next(case for case in experiment_cases("R-02") if not case.deformation)
    second = next(
        case
        for case in experiment_cases("R-03")
        if case.scene_id == first.scene_id and case.temporal_mode == "moving"
    )
    runner.main(
        _args(approved, tmp_path / "results", fixture_runtime)
        + ["--case-id", first.case_id, "--case-id", second.case_id, "--smoke"]
    )
    assert len(prepared) == 1 and prepared[0][1:] == (100, 30)
    paths = {record[2]["replay_path"] for record in fixture_runtime["commands"]}
    assert len(paths) == 2
    assert len({Path(path).read_bytes() for path in paths}) == 1


def test_replay_dependency_failure_retains_not_run_trials_without_workers(
    tmp_path: Path,
    approved: Path,
    fixture_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.benchmark.rendering import replay

    def missing(*args: Any, **kwargs: Any) -> None:
        raise ImportError("CPU FK dependency missing")

    monkeypatch.setattr(replay, "prepare_replay", missing)
    case = next(
        case for case in experiment_cases("R-03") if case.temporal_mode == "moving"
    )
    with pytest.raises(SystemExit) as error:
        runner.main(
            _args(approved, tmp_path / "results", fixture_runtime)
            + ["--case-id", case.case_id, "--smoke"]
        )
    assert error.value.code == 1
    assert not fixture_runtime["commands"]
    _, roots = _roots(tmp_path / "results")
    selected = [row for row in read_json(roots[0] / "runs.json") if row["selected"]]
    assert len(selected) == 2
    assert all(
        row["status"] == "not_run" and row["failure_reason_code"] == "dependency_error"
        for row in selected
    )
    assert all("CPU FK dependency missing" in row["failure_reason"] for row in selected)


def test_interrupt_preserves_unstarted_experiments_and_denominators(
    tmp_path: Path,
    approved: Path,
    fixture_runtime: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(root: Path, specs: list[Any], **kwargs: Any) -> None:
        first = specs[0]
        row = {
            "schema_version": 1,
            "experiment_id": kwargs["experiment_id"],
            "run_id": first.output.name,
            "run_dir": first.output.name,
            "case_id": first.case_id,
            "backend": first.backend,
            "repeat": first.repeat,
            "status": "interrupted",
            "error": "CPU fixture interruption",
        }
        write_json(root / "runs.json", [row])
        ArtifactStore(root).append_raw({"record_id": row["run_id"], **row})
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_experiment", interrupt)
    with pytest.raises(SystemExit) as error:
        runner.main(
            _args(approved, tmp_path / "results", fixture_runtime)
            + [
                "--case-id",
                experiment_cases("R-04")[0].case_id,
                "--case-id",
                experiment_cases("R-05")[0].case_id,
                "--smoke",
            ]
        )
    assert error.value.code == 130
    _, roots = _roots(tmp_path / "results")
    assert len(roots) == 2
    assert read_json(roots[0] / "summary.json")["status_counts"]["interrupted"] == 1
    second = read_json(roots[1] / "runs.json")
    assert all(row["status"] == "not_run" for row in second)
    assert all(
        "interrupted" in row["failure_reason"] for row in second if row["selected"]
    )
    assert (
        read_json(roots[1] / "summary.json")["selection_coverage"]["selected_not_run"]
        == 2
    )


def test_shared_view_fingerprint_binds_views_and_motion_not_environment_count(
    approved: Path,
) -> None:
    scene = read_json(approved / "franka_workstation" / "scene.json")
    one, sixteen = experiment_cases("R-04")[:2]
    assert view_fingerprint(one, scene) == view_fingerprint(sixteen, scene)
    reordered = deepcopy(scene)
    reordered["cameras"].reverse()
    assert view_fingerprint(one, scene) == view_fingerprint(one, reordered)
    three = experiment_cases("R-05")[0]
    assert view_fingerprint(one, scene) != view_fingerprint(three, scene)
    changed = deepcopy(scene)
    changed["cameras"][0]["eye"][0] += 0.1
    assert view_fingerprint(one, changed) != view_fingerprint(one, scene)
    moving = next(case for case in experiment_cases("R-02") if not case.deformation)
    with pytest.raises(ValueError, match="frozen replay"):
        view_fingerprint(moving, scene)
    assert view_fingerprint(moving, scene, {"frames": [1]}) != view_fingerprint(
        moving, scene, {"frames": [2]}
    )
    r01 = experiment_cases("R-01")[0]
    calibration = read_json(approved / "calibration" / "scene.json")
    assert len(view_fingerprint(r01, calibration)) == 64


@pytest.mark.parametrize(
    "args",
    [
        ["--smoke", "--publication"],
        ["--num-envs", "0"],
        ["--max-cases", "0"],
        ["--case-id", "unknown"],
        ["--repeats", "0"],
    ],
)
def test_invalid_selection_is_rejected_without_output(
    tmp_path: Path, args: list[str]
) -> None:
    output = tmp_path / "untouched"
    with pytest.raises(SystemExit) as error:
        runner.main(["--dry-run", "--output", str(output), *args])
    assert error.value.code == 2
    assert not output.exists()
