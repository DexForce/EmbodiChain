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

from importlib.metadata import PackagePath
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.cli import main as cli
from embodichain.cli import run_example as launcher

pytestmark = pytest.mark.no_sim
_REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def examples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "examples"
    (root / "nested").mkdir(parents=True)
    (root / "nested/demo.py").write_text(
        "raise AssertionError('discovery executed code')\n"
    )
    (root / "nested/helper.py").write_text("VALUE = 41\n")
    (root / "nested/__init__.py").write_text("")
    (root / "nested/__main__.py").write_text(
        "raise AssertionError('discovery executed code')\n"
    )
    monkeypatch.setattr(launcher, "_examples_root", lambda: root)
    return root


@pytest.mark.parametrize("arguments", [[], ["--help"], ["--list"]])
def test_discovery_never_executes_scripts(
    examples: Path, monkeypatch, capsys, arguments: list[str]
) -> None:
    execute = Mock(side_effect=AssertionError("Discovery started a process"))
    monkeypatch.setattr(launcher.os, "execv", execute)
    if arguments == ["--help"]:
        with pytest.raises(SystemExit) as exc:
            launcher.main(arguments)
        assert exc.value.code == 0
    else:
        launcher.main(arguments)
    output = capsys.readouterr().out
    assert (
        ("nested/demo" in output)
        if arguments == ["--list"]
        else ("examples/" in output)
    )
    execute.assert_not_called()


@pytest.mark.parametrize(
    "selector, relative",
    [
        ("nested/demo", "nested/demo.py"),
        ("nested/demo.py", "nested/demo.py"),
        ("examples/nested/demo.py", "nested/demo.py"),
        ("nested", "nested/__main__.py"),
    ],
)
@pytest.mark.parametrize(
    "arguments, forwarded",
    [
        (["--help"], ["--help"]),
        (
            ["--variants", "8", "--output-dir", "path with spaces"],
            ["--variants", "8", "--output-dir", "path with spaces"],
        ),
        (["--", "--headless"], ["--headless"]),
    ],
)
def test_launches_any_script_with_arguments_unchanged(
    examples: Path,
    monkeypatch,
    selector: str,
    relative: str,
    arguments: list[str],
    forwarded: list[str],
) -> None:
    execute = Mock()
    monkeypatch.setattr(launcher.os, "execv", execute)
    cli.main(["run-example", selector, *arguments])
    execute.assert_called_once_with(
        sys.executable,
        [
            sys.executable,
            str(Path(launcher.__file__).with_name("_example_runner.py").resolve()),
            str(examples),
            str(examples / relative),
            *forwarded,
        ],
    )


@pytest.mark.parametrize(
    "selector",
    ["missing", "../outside.py", "nested/../../outside.py", "nested/helper.txt"],
)
def test_invalid_path_fails_before_starting_a_process(
    examples: Path, monkeypatch, capsys, selector: str
) -> None:
    execute = Mock()
    monkeypatch.setattr(launcher.os, "execv", execute)
    with pytest.raises(SystemExit) as exc:
        launcher.main([selector])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    execute.assert_not_called()


def test_symlink_outside_the_tree_is_neither_listed_nor_executed(
    examples: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("raise AssertionError('outside')\n")
    (examples / "escape.py").symlink_to(outside)
    assert "escape" not in launcher._example_names(examples)
    with pytest.raises(ValueError, match="inside"):
        launcher._example_path(examples, "escape")


def test_new_script_is_discovered_without_registration(examples: Path) -> None:
    (examples / "another_family").mkdir()
    (examples / "another_family/new_demo.py").write_text("print('new')\n")
    assert "another_family/new_demo" in launcher._example_names(examples)
    assert "nested" in launcher._example_names(examples)
    assert "nested/__init__" not in launcher._example_names(examples)


def test_installed_root_comes_from_distribution_record(
    tmp_path: Path, monkeypatch
) -> None:
    core = tmp_path / "site-packages/embodichain/cli/run_example.py"
    core.parent.mkdir(parents=True)
    root = tmp_path / "share/embodichain/examples"
    root.mkdir(parents=True)
    marker = PackagePath("../../../share/embodichain/examples/__init__.py")
    distribution = SimpleNamespace(
        files=[marker], locate_file=lambda entry: root / "__init__.py"
    )
    monkeypatch.setattr(launcher, "__file__", str(core))
    monkeypatch.setattr(launcher.metadata, "distribution", lambda name: distribution)
    assert launcher._examples_root() == root


def test_relocated_target_install_finds_colocated_data(
    tmp_path: Path, monkeypatch
) -> None:
    installed = tmp_path / "target"
    root = installed / "share/embodichain/examples"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    monkeypatch.setattr(
        launcher, "__file__", str(installed / "embodichain/cli/run_example.py")
    )
    monkeypatch.setattr(
        launcher.metadata,
        "distribution",
        Mock(side_effect=AssertionError("relocated data is already present")),
    )
    assert launcher._examples_root() == root


def test_missing_example_data_reports_installation_error(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        launcher, "__file__", str(tmp_path / "embodichain/cli/run_example.py")
    )
    monkeypatch.setattr(
        launcher.metadata, "distribution", lambda name: SimpleNamespace(files=[])
    )
    with pytest.raises(SystemExit) as exc:
        launcher.main(["--list"])
    assert exc.value.code == 2
    assert "Reinstall EmbodiChain" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["run-atomic-skill", "run-atomic-action"])
def test_existing_atomic_command_forwards_to_ordinary_example(
    monkeypatch, command: str
) -> None:
    run = Mock()
    monkeypatch.setattr(launcher, "main", run)
    cli.main([command, "--skill", "place", "--variants", "2"])
    run.assert_called_once_with(
        ["sim/atomic_skill", "--skill", "place", "--variants", "2"]
    )


def _command(root: Path, *arguments: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "from pathlib import Path; import sys; "
            "sys.path.insert(0, sys.argv.pop(1)); "
            "from embodichain.cli import run_example; "
            "root = Path(sys.argv.pop(1)); "
            "run_example._examples_root = lambda: root; "
            "run_example.main(sys.argv[1:])"
        ),
        str(_REPOSITORY),
        str(root),
        *arguments,
    ]


def test_fresh_process_preserves_imports_arguments_and_callers_directory(
    examples: Path, tmp_path: Path
) -> None:
    (examples / "nested/demo.py").write_text(
        "from .helper import VALUE\n"
        "from helper import VALUE as sibling\n"
        "from examples.nested.helper import VALUE as absolute\n"
        "import json, pathlib, sys\n"
        "pathlib.Path('result.json').write_text(json.dumps([VALUE, sibling, absolute, sys.argv[1:], str(pathlib.Path.cwd())]))\n"
    )
    caller = tmp_path / "caller"
    caller.mkdir()
    # An unrelated package in the caller must not capture the selected tree.
    (caller / "examples").mkdir()
    (caller / "examples/__init__.py").write_text(
        "raise AssertionError('wrong examples')\n"
    )
    result = subprocess.run(
        _command(examples, "nested/demo", "--value", "path with spaces"),
        cwd=caller,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((caller / "result.json").read_text()) == [
        41,
        41,
        41,
        ["--value", "path with spaces"],
        str(caller),
    ]


def test_script_exit_status_propagates(examples: Path, tmp_path: Path) -> None:
    (examples / "nested/demo.py").write_text("raise SystemExit(7)\n")
    result = subprocess.run(
        _command(examples, "nested/demo"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 7


def test_module_entry_supports_multiprocessing_spawn(
    examples: Path, tmp_path: Path
) -> None:
    (examples / "__init__.py").write_text("")
    (examples / "nested/demo.py").write_text(
        "from multiprocessing import get_context\n"
        "from .helper import VALUE\n"
        "def worker(queue):\n    queue.put(VALUE)\n"
        "if __name__ == '__main__':\n"
        "    ctx = get_context('spawn')\n    queue = ctx.Queue()\n"
        "    process = ctx.Process(target=worker, args=(queue,))\n"
        "    process.start()\n    assert queue.get(timeout=10) == VALUE\n"
        "    process.join(timeout=10)\n    assert process.exitcode == 0\n"
    )
    result = subprocess.run(
        _command(examples, "nested/demo"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery")
def test_interrupt_reaches_script_cleanup(examples: Path, tmp_path: Path) -> None:
    (examples / "nested/demo.py").write_text(
        "from pathlib import Path\nimport time\n"
        "try:\n    Path('ready').touch()\n    time.sleep(30)\n"
        "finally:\n    Path('cleaned').touch()\n"
    )
    process = subprocess.Popen(
        _command(examples, "nested/demo"),
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while (
            not (tmp_path / "ready").exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        process.send_signal(signal.SIGINT)
        process.communicate(timeout=10)
        assert (tmp_path / "cleaned").exists()
        assert process.returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("filename", ["demo.v2.py", "demo.py"])
def test_explicit_file_is_executed_even_with_package_name_collision(
    examples: Path, tmp_path: Path, filename: str
) -> None:
    (examples / "demo").mkdir()
    (examples / "demo/__init__.py").write_text("")
    (examples / "demo/__main__.py").write_text("raise AssertionError('wrong file')\n")
    (examples / filename).write_text("print('selected file')\n")
    result = subprocess.run(
        _command(examples, filename),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "selected file"


@pytest.mark.parametrize("from_script_directory", [False, True])
def test_sibling_helper_does_not_shadow_installed_dependency(
    examples: Path, tmp_path: Path, from_script_directory: bool
) -> None:
    dependencies = tmp_path / "dependencies"
    (dependencies / "fixture_sdk").mkdir(parents=True)
    (dependencies / "fixture_sdk/__init__.py").write_text("")
    (dependencies / "fixture_sdk/api.py").write_text("VALUE = 73\n")
    (examples / "nested/fixture_sdk.py").write_text(
        "raise AssertionError('SDK shadowed')\n"
    )
    (examples / "nested/demo.py").write_text(
        "from fixture_sdk.api import VALUE\nfrom helper import VALUE as sibling\nassert (VALUE, sibling) == (73, 41)\n"
    )
    result = subprocess.run(
        _command(examples, "nested/demo"),
        cwd=examples / "nested" if from_script_directory else tmp_path,
        env={**os.environ, "PYTHONPATH": str(dependencies)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_parent_module_collision_does_not_execute_another_file(
    examples: Path, tmp_path: Path
) -> None:
    (examples / "demo").mkdir()
    (examples / "demo.py").write_text("raise AssertionError('unselected parent')\n")
    (examples / "demo/selected.py").write_text("print('selected file')\n")
    result = subprocess.run(
        _command(examples, "demo/selected.py"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "selected file"


def test_nested_namespace_directories_need_no_initializers(
    examples: Path, tmp_path: Path
) -> None:
    folder = examples / "family/subfamily/topic"
    folder.mkdir(parents=True)
    (folder / "helper.py").write_text("VALUE = 53\n")
    (folder / "demo.py").write_text("from .helper import VALUE\nprint(VALUE)\n")
    result = subprocess.run(
        _command(examples, "family/subfamily/topic/demo"),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "53"


def test_legacy_example_selector_resolves_to_canonical_files(tmp_path: Path) -> None:
    directory = tmp_path / "sim/atomic_skill"
    directory.mkdir(parents=True)
    (directory / "__main__.py").write_text("")
    (directory / "place.py").write_text("")
    assert (
        launcher._example_path(tmp_path, "sim/atomic_action")
        == directory / "__main__.py"
    )
    assert (
        launcher._example_path(tmp_path, "sim/atomic_action/place")
        == directory / "place.py"
    )
