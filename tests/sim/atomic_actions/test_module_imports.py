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

"""Regression tests for atomic-action module imports and file entry points."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import embodichain.lab.sim.atomic_actions as atomic_actions

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ATOMIC_ACTIONS_DIRECTORY = (
    REPOSITORY_ROOT / "embodichain" / "lab" / "sim" / "atomic_actions"
)
PRIMITIVES_DIRECTORY = ATOMIC_ACTIONS_DIRECTORY / "primitives"
TUTORIAL_DIRECTORY = REPOSITORY_ROOT / "scripts" / "tutorials" / "atomic_action"

PUBLIC_PRIMITIVE_SCRIPTS = tuple(
    path
    for path in sorted(PRIMITIVES_DIRECTORY.glob("*.py"))
    if not path.name.startswith("_")
)
TUTORIAL_SCRIPTS = tuple(sorted(TUTORIAL_DIRECTORY.glob("*.py")))

RUN_PUBLIC_PRIMITIVES_CODE = """
import runpy
import sys

for module_path in sys.argv[1:]:
    runpy.run_path(module_path, run_name="__main__")
"""

IMPORT_TUTORIALS_CODE = """
import runpy
import sys

for module_path in sys.argv[1:]:
    runpy.run_path(module_path, run_name="__atomic_action_import_check__")
"""


def _run_import_check(
    code: str, paths: tuple[Path, ...]
) -> subprocess.CompletedProcess[str]:
    """Run multiple module files in one isolated Python process."""
    return subprocess.run(
        [sys.executable, "-c", code, *(str(path) for path in paths)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_all_public_primitive_modules_can_run_as_files() -> None:
    """Public primitive files should resolve package imports when run directly."""
    result = _run_import_check(RUN_PUBLIC_PRIMITIVES_CODE, PUBLIC_PRIMITIVE_SCRIPTS)

    assert result.returncode == 0, result.stderr


def test_all_atomic_action_tutorials_import_without_running_main() -> None:
    """Tutorial modules should import without starting their simulations."""
    result = _run_import_check(IMPORT_TUTORIALS_CODE, TUTORIAL_SCRIPTS)

    assert result.returncode == 0, result.stderr


def test_atomic_action_public_exports_exist() -> None:
    """Every declared public export should resolve on the package."""
    missing = [
        name for name in atomic_actions.__all__ if not hasattr(atomic_actions, name)
    ]

    assert missing == []


def test_atomic_action_star_import_resolves_all_public_exports() -> None:
    """Star import should not fail on a stale public export."""
    namespace: dict[str, object] = {}

    exec("from embodichain.lab.sim.atomic_actions import *", namespace)

    assert set(atomic_actions.__all__).issubset(namespace)


def test_atomic_action_modules_do_not_depend_on_semantic_skills() -> None:
    """The atomic-action core must not import the higher semantic-skill layer."""
    forbidden_dependencies = (
        "embodichain.lab.semantic_skills",
        "from ..skills",
        "from .skills",
        "RobotSkillProfile",
    )
    offenders = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in sorted(ATOMIC_ACTIONS_DIRECTORY.rglob("*.py"))
        if any(value in path.read_text() for value in forbidden_dependencies)
    ]

    assert offenders == []
