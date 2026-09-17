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

"""Build and embed version-local architecture resources in Sphinx."""

from __future__ import annotations

import html
import json
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING, Any

from build_architecture_assets import ROOT, build_assets

if TYPE_CHECKING:
    from sphinx.application import Sphinx

__all__ = ["render_entry", "setup"]
ENTRY = "overview/architecture/index"
MARKER = "<!-- architecture-explorer -->"


def _prepare(app: Sphinx) -> None:
    # Text builders do not automatically exclude HTML static paths from discovery.
    app.config.exclude_patterns = [
        *app.config.exclude_patterns,
        "_static/architecture/**",
    ]
    build_assets(
        ROOT,
        Path(app.srcdir) / "_static/architecture",
        html=app.builder.format == "html",
    )


def render_entry(app: Sphinx, docname: str) -> str:
    """Create searchable Markdown and an HTML-only interactive entry.

    Args:
        app: Active Sphinx application.
        docname: Architecture entry document name.

    Returns:
        Version-relative links, iframe for HTML, and the generated text overview.
    """
    bundle = Path(app.srcdir) / "_static/architecture"
    for filename in ("architecture.json", "summary.md"):
        app.env.note_dependency(str(bundle / filename))
    data = json.loads((bundle / "architecture.json").read_text(encoding="utf-8"))
    summary = (bundle / "summary.md").read_text(encoding="utf-8")
    summary = summary.replace("# Architecture overview", "## Module reference", 1)
    base = (
        f"https://github.com/{data['repository']}/blob/{data['revision']}/docs/source/"
    )
    for node in data["nodes"]:
        for doc in node["documentation"]:
            for extension in (".md", ".rst"):
                summary = summary.replace(
                    f"[{doc['label']}]({base}{doc['docname']}{extension})",
                    f"{{doc}}`{doc['label']} </{doc['docname']}>`",
                )
    if app.builder.format != "html":
        return summary
    origin = posixpath.dirname(app.builder.get_target_uri(docname)) or "."
    target = posixpath.relpath("_static/architecture/index.html", origin)
    url = html.escape(target + "?docsRoot=../../&theme=light", quote=True)
    interactive = (
        "```{raw} html\n"
        f'<p><a class="architecture-fullscreen" href="{url}">Open full-screen explorer</a></p>\n'
        f'<iframe class="architecture-frame" title="EmbodiChain architecture explorer" src="{url}" '
        'style="width:100%;height:880px;border:1px solid #b8c1ca;border-radius:4px" loading="lazy"></iframe>\n'
        "```\n\n"
        "The module reference below is searchable and remains available without JavaScript. "
        "Use **Share view** inside the explorer to copy a full-screen link with the current selection and filters.\n\n"
    )
    return interactive + summary


def _source_read(app: Sphinx, docname: str, source: list[str]) -> None:
    if docname == ENTRY:
        source[0] = source[0].replace(MARKER, render_entry(app, docname))


def setup(app: Sphinx) -> dict[str, Any]:
    """Register build preparation and the architecture entry renderer.

    Args:
        app: Sphinx application loading this extension.

    Returns:
        Extension metadata and parallel-read safety declaration.
    """
    app.connect("builder-inited", _prepare)
    app.connect("source-read", _source_read)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
