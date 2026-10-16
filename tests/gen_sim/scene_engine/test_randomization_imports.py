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
    ("statement", "loads_configuration"),
    [
        (f"import {_PACKAGE}", False),
        (
            f"from {_PACKAGE}.cfg import SceneRandomizationCfg, load_randomization_config",
            True,
        ),
    ],
)
def test_configuration_imports_do_not_load_geometry(
    statement: str, loads_configuration: bool
) -> None:
    _run_script(f"""
import sys
{statement}
assert ({_PACKAGE + '.cfg'!r} in sys.modules) is {loads_configuration!r}
assert {_PACKAGE + '.baseline'!r} not in sys.modules
assert {_PACKAGE + '._geometry'!r} not in sys.modules
assert {_PACKAGE + '.generation'!r} not in sys.modules
assert 'trimesh' not in sys.modules
""")


def test_public_api_exports_are_available_from_owning_modules() -> None:
    _run_script(f"""
from {_PACKAGE}.cfg import (
    SceneRandomizationCfg,
    load_randomization_config,
)
from {_PACKAGE}.baseline import (
    PreparedSceneBaseline,
    prepare_scene_baseline,
)
from {_PACKAGE}.generation import (
    SceneRandomizationResult,
    randomize_scene,
)
for symbols, owner in (
    ((SceneRandomizationCfg, load_randomization_config), {_PACKAGE + '.cfg'!r}),
    ((PreparedSceneBaseline, prepare_scene_baseline), {_PACKAGE + '.baseline'!r}),
    ((SceneRandomizationResult, randomize_scene), {_PACKAGE + '.generation'!r}),
):
    for symbol in symbols:
        assert symbol.__module__ == owner
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
assert {_PACKAGE + '.cfg'!r} not in sys.modules
assert {_PACKAGE + '.baseline'!r} not in sys.modules
assert {_PACKAGE + '._geometry'!r} not in sys.modules
assert {_PACKAGE + '.generation'!r} not in sys.modules
""")
    assert "--config" in result.stdout
    assert "--output_root" in result.stdout
    assert "--rebuild" in result.stdout
