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

"""Tests for the read-only public API documentation checker."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "docs" / "scripts" / "check_api_docs.py"


def _load_checker_module():
    spec = importlib.util.spec_from_file_location("check_api_docs", _SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_api_docs"] = module
    spec.loader.exec_module(module)
    return module


_checker = _load_checker_module()
ApiDocsError = _checker.ApiDocsError
MissingExport = _checker.MissingExport
PackageRoot = _checker.PackageRoot
PublicModule = _checker.PublicModule
check_api_docs = _checker.check_api_docs
collect_documented_exports = _checker.collect_documented_exports
discover_public_modules = _checker.discover_public_modules
find_missing_exports = _checker.find_missing_exports
format_json_report = _checker.format_json_report
format_text_report = _checker.format_text_report


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _file_snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_discover_public_modules_uses_static_all(tmp_path: Path) -> None:
    package_path = tmp_path / "sample"
    _write(package_path / "__init__.py", '__all__ = ["Alpha", "helper"]\n')
    _write(package_path / "feature" / "__init__.py", '__all__ = ["Feature"]\n')
    _write(package_path / "module.py", '__all__ = ["NotPackageLevel"]\n')
    _write(package_path / "_private" / "__init__.py", '__all__ = ["Hidden"]\n')

    modules = discover_public_modules((PackageRoot("sample", package_path),))

    assert [(module.name, module.exports) for module in modules] == [
        ("sample", ("Alpha", "helper")),
        ("sample.feature", ("Feature",)),
        ("sample.module", ("NotPackageLevel",)),
    ]


def test_discover_public_modules_collects_branch_scoped_static_all(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "sample"
    _write(
        package_path / "__init__.py",
        """try:
    __all__ = ["Primary", "Shared"]
except ImportError:
    __all__: list[str] = ["Fallback", "Shared"]

def local_scope() -> None:
    __all__ = ["NotAModuleExport"]
""",
    )

    modules = discover_public_modules((PackageRoot("sample", package_path),))

    assert [(module.name, module.exports) for module in modules] == [
        ("sample", ("Primary", "Shared", "Fallback")),
    ]


def test_discover_public_modules_rejects_dynamic_all(tmp_path: Path) -> None:
    package_path = tmp_path / "sample"
    _write(package_path / "__init__.py", "__all__ = build_exports()\n")

    with pytest.raises(ApiDocsError, match="static list of strings"):
        discover_public_modules((PackageRoot("sample", package_path),))


def test_collect_documented_exports_parses_all_api_pages(tmp_path: Path) -> None:
    api_root = tmp_path / "api_reference"
    public_modules = (
        PublicModule(
            "sample",
            ("Alpha", "Beta", "Gamma", "Orphan"),
            tmp_path / "sample",
        ),
        PublicModule("other", ("Delta", "Epsilon"), tmp_path / "other"),
    )
    _write(
        api_root / "a_other.rst",
        """.. currentmodule:: other

.. autoclass:: Epsilon
""",
    )
    _write(
        api_root / "z_sample.rst",
        """.. automodule:: sample

   .. autosummary::

      Alpha

.. currentmodule:: sample

.. autoclass:: Beta

.. automodule:: other
   :members:
   :exclude-members: Epsilon
""",
    )
    _write(
        api_root / "public_api.rst",
        """.. currentmodule:: sample

.. autosummary::

        Gamma
""",
    )
    _write(
        api_root / "_autosummary" / "orphan.rst",
        """.. currentmodule:: sample

.. autodata:: Orphan
""",
    )

    documented = collect_documented_exports(api_root, public_modules)

    assert documented == {
        "sample.Alpha",
        "sample.Beta",
        "sample.Gamma",
        "other.Delta",
        "other.Epsilon",
    }


def test_find_missing_exports_reports_public_import_path(tmp_path: Path) -> None:
    modules = (PublicModule("sample", ("Alpha", "Beta"), tmp_path / "sample.py"),)

    missing = find_missing_exports(modules, {"sample.Alpha"})

    assert missing == (MissingExport("sample", "Beta", tmp_path / "sample.py"),)
    assert missing[0].qualified_name == "sample.Beta"


def test_check_api_docs_does_not_modify_files(tmp_path: Path) -> None:
    package_path = tmp_path / "sample"
    api_root = tmp_path / "api_reference"
    _write(package_path / "__init__.py", '__all__ = ["Alpha", "Beta"]\n')
    _write(
        api_root / "sample.rst",
        """.. currentmodule:: sample

.. autoclass:: Alpha
""",
    )
    before = _file_snapshot(tmp_path)

    result = check_api_docs(
        package_roots=(PackageRoot("sample", package_path),),
        api_reference_root=api_root,
    )

    assert result.total_exports == 2
    assert [item.qualified_name for item in result.missing] == ["sample.Beta"]
    assert _file_snapshot(tmp_path) == before


def test_reports_support_humans_and_agent_skill(tmp_path: Path) -> None:
    missing = MissingExport("sample", "Beta", tmp_path / "sample.py")
    result = _checker.CheckResult(total_exports=2, missing=(missing,))

    payload = json.loads(format_json_report(result))

    assert payload["documented_exports"] == 1
    assert payload["missing_count"] == 1
    assert payload["missing"][0]["qualified_name"] == "sample.Beta"
    assert "$update-api-docs" in format_text_report(result)


def test_main_writes_missing_json_to_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = MissingExport("sample", "Beta", tmp_path / "sample.py")
    result = _checker.CheckResult(total_exports=2, missing=(missing,))
    monkeypatch.setattr(_checker, "check_api_docs", lambda: result)

    exit_code = _checker.main(["--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert json.loads(captured.out)["missing_count"] == 1
    assert captured.err == ""
