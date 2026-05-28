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

import ast
import subprocess
import sys
from pathlib import Path

from embodichain.toolkits.scaffold.spec import TaskSpec

_MARKER_PREFIX = "# --- embodichain-new-task:"


def check_collisions(spec: TaskSpec, paths: list[Path]) -> None:
    """Raise if any output path exists and ``force`` is False."""
    existing = [p for p in paths if p.exists()]
    if existing and not spec.force:
        lines = "\n  ".join(str(p) for p in existing)
        raise FileExistsError(
            f"Output path(s) already exist (use --force to overwrite):\n  {lines}"
        )


def patch_tasks_init(spec: TaskSpec) -> None:
    """Append import and ``__all__`` entry to tasks ``__init__.py``."""
    init_path = spec.tasks_init_path()
    if not init_path.exists():
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_path.write_text(
            "# Auto-generated tasks package.\nfrom __future__ import annotations\n\n"
            "__all__ = []\n",
            encoding="utf-8",
        )

    content = init_path.read_text(encoding="utf-8")
    marker = f"{_MARKER_PREFIX} {spec.gym_id}"
    import_line = spec.tasks_init_import_line()

    if import_line in content:
        content = _upsert_all(content, spec.task_class)
        init_path.write_text(content, encoding="utf-8")
        return

    if import_line not in content:
        if not content.endswith("\n"):
            content += "\n"
        content += f"\n{marker}\n{import_line}\n"

    content = _upsert_all(content, spec.task_class)
    init_path.write_text(content, encoding="utf-8")


def _upsert_all(content: str, class_name: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        if f'"{class_name}"' not in content:
            if "__all__" in content:
                content = content.replace(
                    "__all__ = [",
                    f'__all__ = [\n    "{class_name}",',
                    1,
                )
            else:
                content += f'\n__all__ = ["{class_name}"]\n'
        return content

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        elts = node.value.elts
                        names = []
                        for elt in elts:
                            if isinstance(elt, ast.Constant) and isinstance(
                                elt.value, str
                            ):
                                names.append(elt.value)
                        if class_name not in names:
                            names.append(class_name)
                            new_list = ", ".join(repr(n) for n in names)
                            lines = content.splitlines()
                            start = node.lineno - 1
                            end = node.end_lineno or node.lineno
                            lines[start:end] = [f"__all__ = [{new_list}]"]
                            return "\n".join(lines) + (
                                "\n" if content.endswith("\n") else ""
                            )
    if "__all__" not in content:
        content += f'\n__all__ = ["{class_name}"]\n'
    return content


def run_black(paths: list[Path]) -> None:
    py_files = [str(p) for p in paths if p.suffix == ".py" and p.exists()]
    if not py_files:
        return
    subprocess.run(
        [sys.executable, "-m", "black", *py_files],
        check=False,
    )


def init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=False)
    subprocess.run(["git", "add", "-f", "."], cwd=root, check=False)
    subprocess.run(
        ["git", "commit", "-q", "-m", "Initial commit from embodichain-new-task"],
        cwd=root,
        check=False,
    )
