# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Render the official task catalog from the checkout's configuration files."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["setup"]

_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = "embodichain.extensions.tasks"
_CONFIGS = _ROOT / "embodichain/extensions/tasks/configs"
_COLUMNS = (
    "Category",
    "Task key",
    "Deployment",
    "Environment ID",
    "Physics",
    "Capability",
)


def _row(values: Iterable[str]) -> nodes.row:
    return nodes.row(
        "", *(nodes.entry("", nodes.paragraph(text=value)) for value in values)
    )


class _TaskDeployments(SphinxDirective):
    """Build a table from the same static catalog used by the task CLI."""

    def run(self) -> list[nodes.Node]:
        """Return a deployment table and track its configuration dependencies."""
        from embodichain.cli._task_catalog import _capability_label, _load_catalog

        self.env.note_dependency(_ROOT / "embodichain/cli/_task_catalog.py")
        self.env.note_dependency(_CONFIGS)
        for path in _CONFIGS.rglob("*"):
            if path.is_dir() or path.suffix in {".json", ".yaml", ".yml"}:
                self.env.note_dependency(path)
        tasks = _load_catalog({_PACKAGE: _CONFIGS / "tasks"})

        table = nodes.table()
        group = nodes.tgroup(cols=len(_COLUMNS))
        table += group
        for width in (18, 14, 17, 22, 10, 19):
            group += nodes.colspec(colwidth=width)
        group += nodes.thead("", _row(_COLUMNS))
        body = nodes.tbody()
        group += body
        for task in tasks:
            category = "/".join(task.task_path[:-1]) or "—"
            for deployment in task.deployments:
                body += _row(
                    (
                        category,
                        task.key,
                        deployment.name,
                        deployment.env_id,
                        deployment.physics or "—",
                        _capability_label(deployment),
                    )
                )
        return [table]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the deployment index directive with Sphinx.

    Args:
        app: Sphinx application that renders the documentation.

    Returns:
        Extension metadata for parallel reads and writes.
    """
    app.add_directive("task-deployments", _TaskDeployments)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
