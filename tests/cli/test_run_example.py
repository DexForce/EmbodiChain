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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.cli import main as cli
from embodichain.cli import run_example as launcher

pytestmark = pytest.mark.no_sim


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    entry = SimpleNamespace(
        name="fixture",
        load=lambda: {
            "fixture": {"target": "fixture.example:main", "description": "Test example"}
        },
    )

    def entry_points(*, group: str):
        assert group == "embodichain.examples"
        return [entry]

    monkeypatch.setattr(launcher.metadata, "entry_points", entry_points)


@pytest.mark.parametrize("arguments", [[], ["--help"], ["--list"]])
def test_discovery_does_not_import_example_implementations(
    catalog, monkeypatch: pytest.MonkeyPatch, capsys, arguments: list[str]
) -> None:
    load = Mock(side_effect=AssertionError("Discovery imported an example"))
    monkeypatch.setattr(launcher, "import_module", load)
    if arguments == ["--help"]:
        with pytest.raises(SystemExit) as exc:
            launcher.main(arguments)
        assert exc.value.code == 0
    else:
        launcher.main(arguments)
    assert "fixture" in capsys.readouterr().out
    load.assert_not_called()


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
def test_selected_example_receives_its_arguments(
    catalog, monkeypatch: pytest.MonkeyPatch, arguments: list[str], forwarded: list[str]
) -> None:
    handler = Mock()
    load = Mock(return_value=SimpleNamespace(main=handler))
    monkeypatch.setattr(launcher, "import_module", load)
    cli.main(["run-example", "fixture", *arguments])
    load.assert_called_once_with("fixture.example")
    handler.assert_called_once_with(forwarded)


def test_unknown_example_fails_before_import(catalog, monkeypatch, capsys) -> None:
    load = Mock()
    monkeypatch.setattr(launcher, "import_module", load)
    with pytest.raises(SystemExit) as exc:
        launcher.main(["missing"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    load.assert_not_called()


def test_catalogs_reject_duplicate_names(catalog, monkeypatch) -> None:
    entry = launcher.metadata.entry_points(group="embodichain.examples")[0]
    monkeypatch.setattr(launcher.metadata, "entry_points", lambda **_: [entry, entry])
    with pytest.raises(ValueError, match="Duplicate example name"):
        launcher.main(["--list"])


@pytest.mark.parametrize(
    "target", ["missing_separator", "module:", "module:handler.extra"]
)
def test_catalogs_reject_invalid_targets(monkeypatch, target: str) -> None:
    entry = SimpleNamespace(
        name="invalid", load=lambda: {"fixture": {"target": target}}
    )
    monkeypatch.setattr(launcher.metadata, "entry_points", lambda **_: [entry])
    with pytest.raises(ValueError, match="module:callable"):
        launcher.main(["--list"])


def test_existing_atomic_command_forwards_to_the_example(monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(launcher, "main", run)
    cli.main(["run-atomic-action", "--skill", "place", "--variants", "2"])
    run.assert_called_once_with(
        ["atomic-action", "--skill", "place", "--variants", "2"]
    )
