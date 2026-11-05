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

"""Task interpretation and shared simulation keep their ownership boundary."""

from __future__ import annotations

import ast
from importlib.util import resolve_name
from pathlib import Path
import subprocess
import sys

_REPOSITORY = Path(__file__).parents[3]


def test_task_interpretation_imports_without_runtime_or_simulator(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import embodichain.gen_sim.task_engine as engine; "
            "assert callable(engine.task_spec_from_candidate); "
            "assert not any(name == 'embodichain.lab' or name.startswith('embodichain.lab.') "
            "or name == 'dexsim' or name.startswith('dexsim.') for name in sys.modules)",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_shared_runtime_does_not_import_task_engine():
    forbidden = "embodichain.gen_sim.task_engine"
    violations = []
    for folder in ("compute", "lab/sim", "lab/task_program", "toolkits/graspkit"):
        for path in (_REPOSITORY / "embodichain" / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = []
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        package = ".".join(path.relative_to(_REPOSITORY).parts[:-1])
                        module = resolve_name("." * node.level + module, package)
                    names = [
                        module,
                        *(module + "." + alias.name for alias in node.names),
                    ]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.Call) and node.args:
                    function = (
                        node.func.attr
                        if isinstance(node.func, ast.Attribute)
                        else node.func.id if isinstance(node.func, ast.Name) else ""
                    )
                    if (
                        function in {"import_module", "__import__"}
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                    ):
                        names = [node.args[0].value]
                if any(
                    name == forbidden or name.startswith(forbidden + ".")
                    for name in names
                ):
                    violations.append((str(path.relative_to(_REPOSITORY)), node.lineno))
    assert violations == []


def test_task_engine_does_not_implement_skills_planners_or_contact_sensors():
    implementations = {
        "AtomicAction",
        "AtomicActionEngine",
        "MotionGenerator",
        "BasePlanner",
        "CuroboPlanner",
        "ContactSensor",
        "EmbodiedEnv",
    }
    violations = []
    for path in (_REPOSITORY / "embodichain/gen_sim/task_engine").rglob("*.py"):
        tree = ast.parse(path.read_text())
        aliases = {name: name for name in implementations}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases.update({a.asname or a.name: a.name for a in node.names})
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                value = base.value if isinstance(base, ast.Subscript) else base
                name = (
                    value.attr
                    if isinstance(value, ast.Attribute)
                    else value.id if isinstance(value, ast.Name) else ""
                )
                if aliases.get(name, name) in implementations:
                    violations.append((path.name, node.name))
    assert violations == []
