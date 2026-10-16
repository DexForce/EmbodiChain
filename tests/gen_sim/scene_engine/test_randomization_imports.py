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

"""Check public imports in fresh processes without depending on test order."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

_PACKAGE = "embodichain.gen_sim.scene_engine.randomization"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _run_script(script: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.mark.parametrize(
    "statement",
    [
        f"import {_PACKAGE}",
        f"from {_PACKAGE}.cfg import SceneRandomizationCfg, load_randomization_config",
    ],
)
def test_configuration_imports_do_not_load_geometry(statement: str) -> None:
    _run_script(f"""
import sys
{statement}
assert {_PACKAGE + '.baseline'!r} not in sys.modules
assert {_PACKAGE + '._geometry'!r} not in sys.modules
assert 'trimesh' not in sys.modules
""")


def test_public_api_exports_are_available_and_unknown_names_raise() -> None:
    _run_script(f"""
import {_PACKAGE} as randomization
from {_PACKAGE} import (
    SceneRandomizationCfg,
    load_randomization_config,
    PreparedSceneBaseline,
    prepare_scene_baseline,
)
assert randomization.SceneRandomizationCfg is SceneRandomizationCfg
assert randomization.load_randomization_config is load_randomization_config
assert randomization.PreparedSceneBaseline is PreparedSceneBaseline
assert randomization.prepare_scene_baseline is prepare_scene_baseline
try:
    randomization.unknown_randomization_attribute
except AttributeError as exc:
    assert 'unknown_randomization_attribute' in str(exc)
else:
    raise AssertionError('Unknown package attributes must raise AttributeError.')
""")


def test_cli_help_does_not_load_baseline_or_geometry() -> None:
    result = _run_script(f"""
import sys
from embodichain.cli.main import main
try:
    main(['scene-randomization', '--help'])
except SystemExit as exc:
    assert exc.code == 0
else:
    raise AssertionError('CLI help must exit successfully.')
assert {_PACKAGE + '.baseline'!r} not in sys.modules
assert {_PACKAGE + '._geometry'!r} not in sys.modules
""")
    assert "--config" in result.stdout
    assert "--output_root" in result.stdout
    assert "--rebuild" in result.stdout
