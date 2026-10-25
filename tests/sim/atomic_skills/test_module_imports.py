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

"""Regression tests for atomic-skill module imports and file entry points."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

import embodichain.lab.sim.atomic_skills as atomic_skills

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
ATOMIC_SKILLS_DIRECTORY = (
    REPOSITORY_ROOT / "embodichain" / "lab" / "sim" / "atomic_skills"
)
PRIMITIVES_DIRECTORY = ATOMIC_SKILLS_DIRECTORY / "primitives"
TUTORIAL_DIRECTORY = REPOSITORY_ROOT / "scripts" / "tutorials" / "atomic_skill"

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
    runpy.run_path(module_path, run_name="__atomic_skill_import_check__")
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


def test_all_atomic_skill_tutorials_import_without_running_main() -> None:
    """Tutorial modules should import without starting their simulations."""
    result = _run_import_check(IMPORT_TUTORIALS_CODE, TUTORIAL_SCRIPTS)

    assert result.returncode == 0, result.stderr


def test_atomic_skill_star_import_resolves_all_public_exports() -> None:
    """Star import should not fail on a stale public export."""
    namespace: dict[str, object] = {}

    exec("from embodichain.lab.sim.atomic_skills import *", namespace)

    assert set(atomic_skills.__all__).issubset(namespace)


def test_atomic_skill_modules_do_not_depend_on_semantics() -> None:
    """The Atomic Skill core must not import the Task Program semantic layer."""
    forbidden_dependencies = (
        "embodichain.lab.task_program.semantics",
        "from ..skills",
        "from .skills",
        "RobotSkillProfile",
    )
    offenders = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in sorted(ATOMIC_SKILLS_DIRECTORY.rglob("*.py"))
        if any(value in path.read_text() for value in forbidden_dependencies)
    ]

    assert offenders == []


@pytest.mark.parametrize("legacy_first", [False, True])
def test_legacy_imports_share_modules_types_and_registry(legacy_first: bool) -> None:
    """Import order must not split classes or the installed skill catalog."""
    code = """
import importlib
import sys
from types import SimpleNamespace
legacy_name = "embodichain.lab.sim.atomic_actions"
canonical_name = "embodichain.lab.sim.atomic_skills"
first, second = ((legacy_name, canonical_name) if sys.argv[1] == "True"
                 else (canonical_name, legacy_name))
importlib.import_module(first)
importlib.import_module(second)
legacy = importlib.import_module(legacy_name)
canonical = importlib.import_module(canonical_name)
assert legacy.AtomicAction is canonical.AtomicSkill
assert legacy.AtomicActionEngine is canonical.AtomicSkillEngine
assert legacy.ActionInvocation is canonical.SkillInvocation
for suffix in ("core", "plans", "primitives", "primitives.pick_up", "_plan_expansion"):
    assert importlib.import_module(legacy_name + "." + suffix) is importlib.import_module(canonical_name + "." + suffix)
assert legacy.primitives.PickUp is canonical.PickUp
assert legacy.core.AtomicAction is canonical.AtomicSkill
engine = canonical.AtomicSkillEngine(
    SimpleNamespace(robot=SimpleNamespace(device="cpu"), device="cpu"),
    load_builtins=False,
)
engine.register(legacy.PickUp())
assert engine.skill_implementations["pick_up"].__class__ is canonical.PickUp
assert tuple(engine.skill_catalog) == ("pick_up",)
assert engine.actions.keys() == engine.skill_implementations.keys()
assert engine.skills.keys() == engine.skill_catalog.keys()
namespace = {}
exec("from " + legacy_name + " import *", namespace)
assert namespace["AtomicActionEngine"] is canonical.AtomicSkillEngine
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(legacy_first)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "suffix", ["planned", "planning_failed", "timeout", "retry", "completed"]
)
def test_legacy_event_values_resolve_to_canonical_events(suffix: str) -> None:
    from embodichain.lab.sim.atomic_skills import ExecutionEventKind

    event = ExecutionEventKind("action_" + suffix)
    assert event.value == "skill_" + suffix
    assert ExecutionEventKind["ACTION_" + suffix.upper()] is event
