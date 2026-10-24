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

from copy import deepcopy
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest
from PIL import Image

from embodichain.gen_sim.task_engine import _graph_localization as translation
from embodichain.gen_sim.task_engine._semantic_graph_presentation import (
    _graph_presentation,
)
from embodichain.gen_sim.task_engine.semantic_graph_html import (
    render_semantic_task_graph_html,
    write_semantic_task_graph_html,
)
from .test_semantic_graph_validation import _plan
from .test_graph_localization import translated


class Payload(HTMLParser):
    def __init__(self):
        super().__init__()
        self.selected = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.selected = dict(attrs).get("id") == "graph-data"

    def handle_endtag(self, tag):
        if tag == "script":
            self.selected = False

    def handle_data(self, text):
        if self.selected:
            self.parts.append(text)


def payload(html):
    parsed = Payload()
    parsed.feed(html)
    return json.loads("".join(parsed.parts))


def test_offline_viewer_uses_current_task_and_preserves_literal_template_tokens(
    monkeypatch,
):
    plan = _plan()
    graph = deepcopy(plan.graph)
    graph["instruction"] = "__VIEWER_SCRIPT__ </script><img onerror=bad()>"
    before = deepcopy(graph)
    monkeypatch.setattr(
        translation,
        "_default_translation_caller",
        lambda **kwargs: pytest.fail("Pure HTML rendering must stay offline"),
    )
    html = render_semantic_task_graph_html(graph)
    assert payload(html)["graph"] == before
    assert graph == before
    assert 'id="i-Workflow"' in html and "data:font/woff2;base64," in html
    assert 'data-language="zh"' in html and 'data-language="en"' in html
    assert "connect-src 'none'" in html and "font-src data:" in html
    assert "task1181" not in html and "20261024_054454" not in html
    assert "<img onerror=bad()>" not in html
    assert "__GRAPH_DATA__" not in html and "__VIEWER_STYLE__" not in html


def test_thumbnails_use_exact_source_uid_and_do_not_invent_missing_previews(tmp_path):
    project, bundle = tmp_path / "project", tmp_path / "bundle"
    config = project / "scene_export/scene_config.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    images = project / "scene_understanding/object_images"
    images.mkdir(parents=True)
    Image.new("RGB", (24, 24), "blue").save(images / "donor_rgba.png")
    bundle.mkdir()
    objects = [{"uid": "a", "name": "one"}, {"uid": "b", "name": "two"}]
    (bundle / "scene_manifest.json").write_text(json.dumps({"objects": objects}))
    (bundle / "static_scene_manifest.json").write_text(
        json.dumps(
            {
                "source": {"config_path": str(config)},
                "objects": [
                    {"uid": "a", "source_uid": "donor"},
                    {"uid": "b", "source_uid": "absent"},
                ],
            }
        )
    )
    graph = {
        "task_id": "sample",
        "nodes": [{"call": {"object": "a"}}, {"call": {"arguments": {"object": "b"}}}],
    }
    rows = _graph_presentation(graph, bundle_root=bundle)["assets"]
    assert rows[0]["cutout"].startswith("data:image/png;base64,")
    assert rows[1]["cutout"] is None and rows[1]["render"] is None
    outside = tmp_path / "foreign.png"
    Image.new("RGB", (24, 24), "red").save(outside)
    (images / "absent_rgba.png").symlink_to(outside)
    assert _graph_presentation(graph, bundle_root=bundle)["assets"][1]["cutout"] is None


def test_write_loads_only_matching_display_translation_and_keeps_graph_unchanged(
    tmp_path, monkeypatch
):
    plan = _plan()
    bundle, output = tmp_path / "bundle", tmp_path / "run"
    bundle.mkdir()
    output.mkdir()
    presentation = _graph_presentation(plan.graph, bundle_root=bundle)
    translation._ensure_localizations(
        output / translation._FILENAME, plan.graph, presentation, caller=translated
    )
    monkeypatch.setattr(
        translation,
        "_default_translation_caller",
        lambda **kwargs: pytest.fail("Offline writer invoked a provider"),
    )
    before = deepcopy(plan.graph)
    path = write_semantic_task_graph_html(
        plan.graph, output / "semantic_graph.html", bundle_root=bundle
    )
    data = payload(path.read_text())
    assert (
        data["presentation"]["localizations"]["instruction"]["en"]
        == "Display translation"
    )
    assert data["graph"] == before and plan.graph == before
    changed = deepcopy(plan.graph)
    changed["instruction"] = "different source"
    path = write_semantic_task_graph_html(changed, path, bundle_root=bundle)
    assert "localizations" not in payload(path.read_text())["presentation"]


def test_visualize_cli_never_generates_translations(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine import cli
    from .test_semantic_graph_validation import _contract

    bundle = tmp_path / "bundle"
    plan = _contract(bundle)
    graph_path = bundle / "semantic_task_graph.json"
    graph_path.write_text(json.dumps(plan.graph))
    monkeypatch.setattr(
        translation,
        "_ensure_localizations",
        lambda *args, **kwargs: pytest.fail(
            "visualize must not call a translation provider"
        ),
    )
    output = tmp_path / "offline.html"
    assert (
        cli.main(["visualize", "--graph", str(graph_path), "--output", str(output)])
        == 0
    )
    assert payload(output.read_text())["graph"] == plan.graph
    assert not (tmp_path / translation._FILENAME).exists()
