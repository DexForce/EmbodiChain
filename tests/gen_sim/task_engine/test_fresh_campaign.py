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

"""Provider-free coverage of the public-CLI fresh-run launcher."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest

from embodichain.gen_sim.task_engine import fresh_campaign as run

_INSTRUCTION = "拉出烤箱托盘，把里面的餐碗移到桌上的服务托盘内，再把纸杯放到餐碗右侧（碗旁），最后推回空烤箱托盘。"


def test_environment_prefers_checkout_sources_without_global_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/some/other/checkout")
    environment = run._environment()
    assert environment["PYTHONPATH"].split(run.os.pathsep) == [
        str(run._ROOT),
        str(run._ROOT / "extensions"),
        str(run._ROOT / "extensions/embodichain_tasks"),
    ]
    assert run.os.environ["PYTHONPATH"] == "/some/other/checkout"


@pytest.fixture
def checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "checkout with spaces"
    for package in (
        "embodichain",
        "extensions/embodichain_robots",
        "extensions/embodichain_tasks/embodichain_tasks",
        "extensions/embodichain_tasks/configs",
    ):
        directory = root / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
    tasks_package = root / "extensions/embodichain_tasks/embodichain_tasks"
    shutil.copyfile(
        run._ROOT / "extensions/embodichain_tasks/embodichain_tasks/__init__.py",
        tasks_package / "__init__.py",
    )
    gym_utils = root / "embodichain/lab/gym/utils/gym_utils.py"
    gym_utils.parent.mkdir(parents=True)
    gym_utils.write_text(
        "def register_manager_modules(names): pass\n", encoding="utf-8"
    )
    importer = tasks_package / "utils/importer.py"
    importer.parent.mkdir()
    importer.write_text(
        "def import_packages(name, blacklist):\n"
        "    from importlib.resources import files\n"
        "    import json\n"
        "    assert json.loads(files(name+'.configs').joinpath('marker.json').read_text()) == {'checkout': 'current'}\n",
        encoding="utf-8",
    )
    (root / "extensions/embodichain_tasks/configs/marker.json").write_text(
        '{"checkout": "current"}', encoding="utf-8"
    )
    module = root / "embodichain/gen_sim/task_engine"
    module.mkdir(parents=True)
    (module / "__main__.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "import embodichain_tasks.configs as configs\n"
        "args=sys.argv[1:]\n"
        "task=args[args.index('--task-id')+1]\n"
        "out=Path(args[args.index('--output-root')+1])/'one-run'\n"
        "out.mkdir()\n"
        "status='failed' if task=='task1171' else 'succeeded'\n"
        "(out/'run_manifest.json').write_text(json.dumps({'status':status}))\n"
        "print(json.dumps({'args':args,'cwd':os.getcwd(),'pythonpath':os.environ['PYTHONPATH'],'config_origin':configs.__file__}))\n"
        "sys.exit(2 if status=='failed' else 0)\n",
        encoding="utf-8",
    )
    for task in (f"task{number}" for number in range(1171, 1191)):
        scene = root / "gym_project/task100/new3" / task / "scene_export"
        scene.mkdir(parents=True)
        (scene / "scene_config.json").write_text("{}", encoding="utf-8")

    def git(*arguments: str) -> str:
        if arguments[0] == "ls-files":
            return "\0".join(
                str(p.relative_to(root)) for p in (root / "embodichain").rglob("*.py")
            )
        return "test-head" if arguments[0] == "rev-parse" else "test-branch"

    monkeypatch.setattr(run, "_ROOT", root)
    monkeypatch.setattr(run, "_git", git)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/some/other/checkout")
    return root


@pytest.mark.parametrize("relocated", [False, True])
def test_source_config_resources_follow_the_checkout_without_installing(
    checkout: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relocated: bool
) -> None:
    root = checkout
    if relocated:
        root = tmp_path / "another checkout"
        shutil.copytree(checkout, root)
    monkeypatch.setattr(run, "_ROOT", root)
    environment = {
        "PYTHONPATH": run.os.pathsep.join(
            str(root / part)
            for part in (".", "extensions", "extensions/embodichain_tasks")
        )
    }
    origins = run._preflight(environment)
    assert Path(origins["embodichain_tasks.configs"]) == (
        root / "extensions/embodichain_tasks/configs/__init__.py"
    )
    assert all(Path(origin).is_relative_to(root) for origin in origins.values())
    assert not (root / "outputs").exists()


def test_wheel_layout_keeps_its_bundled_config_resources(checkout: Path) -> None:
    installed = checkout / "wheel_site"
    shutil.copytree(
        checkout / "extensions/embodichain_tasks/embodichain_tasks",
        installed / "embodichain_tasks",
    )
    shutil.copytree(
        checkout / "extensions/embodichain_tasks/configs",
        installed / "embodichain_tasks/configs",
    )
    origins = run._preflight(
        {
            "PYTHONPATH": run.os.pathsep.join(
                map(str, (checkout, installed, checkout / "extensions"))
            )
        }
    )
    assert Path(origins["embodichain_tasks.configs"]) == (
        installed / "embodichain_tasks/configs/__init__.py"
    )


def test_preview_keeps_user_instruction_and_public_command(
    checkout: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        run, "_execute", Mock(side_effect=AssertionError("must not run"))
    )
    assert run.main(["--tasks", "task1185", "--robot-profile", "dual_ur10"]) == 0
    plan = json.loads(capsys.readouterr().out)
    command = plan["cases"][0]["command"]
    assert command[:6] == [
        sys.executable,
        "-u",
        "-B",
        "-m",
        "embodichain.gen_sim.task_engine",
        "run-all",
    ]
    assert command[6:] == [
        "--mode",
        "scene",
        "--task-id",
        "task1185",
        "--instruction",
        _INSTRUCTION,
        "--scene",
        "gym_project/task100/new3/task1185",
        "--output-root",
        plan["cases"][0]["output_root"],
        "--robot-profile",
        "dual_ur10",
        "--model",
        "mimo-v2.6-pro",
        "--base-seed",
        "0",
    ]
    assert plan["cwd"] == str(checkout)
    assert Path(plan["cases"][0]["output_root"]).is_relative_to(
        checkout / "outputs/pipeline"
    )
    assert plan["cases"][0]["status"] == "not_started"
    assert not (checkout / "outputs").exists()
    manifest = json.loads(run._MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["tasks"]) == {f"task{n}" for n in range(1171, 1191)}
    assert "1. 拿起量杯" in manifest["tasks"]["task1182"]
    assert "\\n" not in manifest["tasks"]["task1182"]


def test_sequential_cli_execution_preserves_failures_and_isolates_batches(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The batch starts before midnight in Shanghai; its tasks start afterwards.
    start = datetime(2026, 10, 30, 15, 59, 59, 123456, tzinfo=timezone.utc)
    moments = iter(start + timedelta(seconds=i) for i in range(10))
    clock = Mock()
    clock.now.side_effect = lambda tz: next(moments).astimezone(tz)
    monkeypatch.setattr(run, "datetime", clock)
    arguments = [
        "--tasks",
        "task1171",
        "task1185",
        "--robot-profile",
        "dual_ur10",
        "--execute",
    ]
    assert run.main(arguments) == 1
    assert run.main(arguments) == 1
    plans = sorted((checkout / "outputs/pipeline").glob("*/plan.json"))
    assert len(plans) == 2
    assert plans[0].parent.name == "20261030_235959"
    assert json.loads(plans[0].read_text())["batch_started_at"] == (
        "2026-10-30T23:59:59.123456+08:00"
    )
    for path in plans:
        plan = json.loads(path.read_text())
        assert plan["status"] == "finished"
        for case in plan["cases"]:
            assert Path(case["output_root"]) == (
                path.parent / case["robot_profile"] / case["task_id"]
            )
        failed, succeeded = plan["cases"]
        assert failed["returncode"] == 2 and not failed["workflow_succeeded"]
        assert succeeded["returncode"] == 0 and succeeded["workflow_succeeded"]
        observed = json.loads(Path(succeeded["log"]).read_text())
        assert observed["args"][0] == "run-all"
        assert observed["cwd"] == str(checkout)
        assert "/some/other/checkout" not in observed["pythonpath"]
        assert observed["pythonpath"].split(run.os.pathsep) == [
            str(checkout),
            str(checkout / "extensions"),
            str(checkout / "extensions/embodichain_tasks"),
        ]
        assert Path(plan["package_origins"]["embodichain_tasks.configs"]) == (
            checkout / "extensions/embodichain_tasks/configs/__init__.py"
        )
        assert (
            observed["config_origin"]
            == plan["package_origins"]["embodichain_tasks.configs"]
        )
        assert not (path.parent / ".packages").exists()
        assert not (path.parent / "packages.log").exists()
        assert all(
            Path(origin).is_relative_to(checkout)
            for origin in plan["package_origins"].values()
        )
        receipt = json.loads(
            (Path(succeeded["output_root"]) / "invocation.json").read_text()
        )
        assert receipt["inputs_unchanged"] is True
        assert receipt["command"] == succeeded["command"]


def test_zero_exit_without_manifest_is_not_reported_as_success(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_execute", lambda *_: 0)
    assert (
        run.main(["--tasks", "task1185", "--robot-profile", "dual_franka", "--execute"])
        == 1
    )
    path = next((checkout / "outputs/pipeline").glob("*/plan.json"))
    case = json.loads(path.read_text())["cases"][0]
    assert case["workflow_status"] == "unavailable"
    assert case["workflow_succeeded"] is False


@pytest.mark.parametrize(
    ("status", "has_video"),
    [("succeeded", True), ("failed", True), ("failed", False)],
)
def test_finished_case_displays_original_artifacts_and_final_json(
    checkout: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    has_video: bool,
) -> None:
    expected: dict = {}
    logs: list[Path] = []

    def execute(command: list[str], log: Path, environment: dict[str, str]) -> int:
        output = log.parent / "one-run"
        output.mkdir()
        videos = (
            [str(output / "videos/button.mp4"), str(output / "videos/audience.mp4")]
            if has_video
            else []
        )
        html = str(output / "semantic_graph.html") if has_video else None
        expected.update(
            run_id="one-run",
            status=status,
            failure_class=None if status == "succeeded" else "action_execution",
            elapsed_seconds=12.345,
            output_dir=str(output),
            manifest=str(output / "run_manifest.json"),
            final_bundle=str(output / "final/bundle"),
            video_paths=videos,
            semantic_graph_html=html,
        )
        (output / "run_manifest.json").write_text(
            json.dumps({"status": status, "failure_class": expected["failure_class"]}),
            encoding="utf-8",
        )
        notices = (
            [f"[Task Engine] Video saved: {path}\n" for path in videos]
            if videos
            else ["[Task Engine] No video files generated for this run.\n"]
        )
        if html:
            notices.append(f"[Task Engine] Semantic graph HTML: {html}\n")
        log.write_text(
            'verbose startup diagnostics\n{\n  "debug": true\n}\n'
            + "".join(notices)
            + "HTTP request diagnostics\n"
            + json.dumps(expected, ensure_ascii=False, indent=2)
            + "\nverbose shutdown diagnostics\n",
            encoding="utf-8",
        )
        logs.append(log)
        return 0 if status == "succeeded" else 2

    monkeypatch.setattr(run, "_execute", execute)
    assert run.main(
        ["--tasks", "task1173", "--robot-profile", "dual_ur10", "--execute"]
    ) == (0 if status == "succeeded" else 1)
    terminal = capsys.readouterr().out
    for path in expected["video_paths"]:
        assert f"[Task Engine] Video saved: {path}\n" in terminal
    if expected["semantic_graph_html"]:
        assert (
            f"[Task Engine] Semantic graph HTML: {expected['semantic_graph_html']}\n"
            in terminal
        )
    else:
        assert "[Task Engine] No video files generated for this run.\n" in terminal
        assert "[Task Engine] Semantic graph HTML:" not in terminal
    result, _ = json.JSONDecoder().raw_decode(terminal[terminal.index("\n{\n") + 1 :])
    assert result == expected
    assert "verbose" not in terminal and "HTTP request diagnostics" not in terminal
    assert "verbose startup diagnostics" in logs[0].read_text(encoding="utf-8")


def test_scene_drift_stops_remaining_tasks_without_qualifying_current_run(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = run._execute

    def mutate(command: list[str], log: Path, environment: dict[str, str]) -> int:
        code = original(command, log, environment)
        scene = (
            checkout
            / "gym_project/task100/new3/task1171/scene_export/scene_config.json"
        )
        scene.write_text('{"changed":true}', encoding="utf-8")
        return code

    monkeypatch.setattr(run, "_execute", mutate)
    assert (
        run.main(
            [
                "--tasks",
                "task1185",
                "task1171",
                "--robot-profile",
                "dual_ur10",
                "--execute",
            ]
        )
        == 1
    )
    path = next((checkout / "outputs/pipeline").glob("*/plan.json"))
    plan = json.loads(path.read_text())
    assert plan["status"] == "aborted"
    assert plan["cases"][0]["inputs_unchanged"] is False
    assert plan["cases"][0]["workflow_succeeded"] is False
    assert plan["cases"][1]["status"] == "not_started"


@pytest.mark.parametrize("origin", [None, "/other/checkout/embodichain/__init__.py"])
def test_missing_or_external_package_resolution_is_rejected(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, origin: str | None
) -> None:
    monkeypatch.setattr(
        run.subprocess,
        "check_output",
        lambda *a, **k: json.dumps({"embodichain": origin}),
    )
    with pytest.raises(RuntimeError, match="outside this checkout"):
        run._preflight(run._environment())


def test_package_import_failure_stops_before_creating_any_case(
    checkout: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (
        checkout / "extensions/embodichain_tasks/embodichain_tasks/__init__.py"
    ).write_text("raise KeyError('joint_names')\n", encoding="utf-8")
    execute = Mock(side_effect=AssertionError("must not launch task CLI"))
    monkeypatch.setattr(run, "_execute", execute)
    assert run.main(["--tasks", "task1185", "--execute"]) == 1
    assert "Package import preflight failed" in capsys.readouterr().err
    execute.assert_not_called()
    path = next((checkout / "outputs/pipeline").glob("*/plan.json"))
    plan = json.loads(path.read_text())
    assert plan["status"] == "aborted"
    assert all(case["status"] == "not_started" for case in plan["cases"])
    assert not list(path.parent.glob("dual_*"))


def test_launch_error_leaves_an_aborted_not_running_receipt(
    checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run, "_execute", Mock(side_effect=OSError("launch failed")))
    assert (
        run.main(["--tasks", "task1185", "--robot-profile", "dual_ur10", "--execute"])
        == 1
    )
    path = next((checkout / "outputs/pipeline").glob("*/plan.json"))
    plan = json.loads(path.read_text())
    assert plan["status"] == "aborted"
    assert plan["cases"][0]["status"] == "aborted"
    assert plan["cases"][0]["workflow_succeeded"] is False


def test_interrupt_requests_cli_cleanup_and_waits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt, 2]
    process.poll.return_value = None
    monkeypatch.setattr(run.subprocess, "Popen", Mock(return_value=process))
    assert run._execute([sys.executable], tmp_path / "cli.log", {}) == 130
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert process.wait.call_count == 2


def test_case_timeout_is_not_success_and_only_our_process_group_is_stopped(
    tmp_path, monkeypatch
):
    process = Mock(pid=12345)
    process.wait.side_effect = [
        subprocess.TimeoutExpired("cli", 1800),
        subprocess.TimeoutExpired("cleanup", 60),
        -9,
    ]
    process.poll.return_value = None
    monkeypatch.setattr(run.subprocess, "Popen", Mock(return_value=process))
    terminate = Mock()
    monkeypatch.setattr(run.os, "killpg", terminate)
    assert run._execute([sys.executable], tmp_path / "cli.log", {}) == 124
    assert process.wait.call_args_list[0].kwargs == {"timeout": 1800}
    process.send_signal.assert_called_once_with(signal.SIGINT)
    terminate.assert_called_once_with(12345, signal.SIGKILL)


def test_default_preview_covers_both_robots_without_historical_overrides(
    checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["case_count"] == 40
    expected = [
        (robot, f"task{number}")
        for robot in ("dual_franka", "dual_ur10")
        for number in range(1171, 1191)
    ]
    assert [
        (case["robot_profile"], case["task_id"]) for case in plan["cases"]
    ] == expected
    for case in plan["cases"]:
        assert not {
            "--config",
            "--bundle",
            "--native-pick-exploration",
            "--simulation-exploration",
        } & set(case["command"])
    assert not (checkout / "outputs").exists()


def test_planned_commands_are_accepted_by_current_public_cli(
    checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from embodichain.gen_sim.task_engine.cli import build_parser

    assert run.main(["--tasks", "task1185", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    parser = build_parser()
    for case in plan["cases"]:
        parsed = parser.parse_args(case["command"][5:])
        assert parsed.command == "run-all"
        assert parsed.config is None
        assert parsed.native_pick_exploration is False
        assert parsed.robot_profile == case["robot_profile"]


def test_unsupported_exploration_is_not_silently_remapped(
    checkout: Path,
) -> None:
    with pytest.raises(SystemExit) as error:
        run.main(["--tasks", "task1185", "--simulation-exploration"])
    assert error.value.code == 2
    assert not (checkout / "outputs").exists()


def test_output_cannot_be_written_into_frozen_scenes(checkout: Path) -> None:
    with pytest.raises(SystemExit) as error:
        run.main(["--output-root", "gym_project/task100/new3/results"])
    assert error.value.code == 2
    assert not (checkout / "gym_project/task100/new3/results").exists()
