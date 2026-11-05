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


def translated(**kwargs):
    source = json.loads(kwargs["prompt"])
    return {
        "title": {"zh": "任务图", "en": "Task graph"},
        "instruction": {"zh": source["instruction"], "en": "Display translation"},
        "assets": {
            uid: {key: {"zh": text, "en": text} for key, text in asset.items()}
            for uid, asset in source["assets"].items()
        },
    }


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


@pytest.mark.parametrize("spec", [None, {"schema_version": "task_spec.absent/v1"}])
def test_offline_viewer_uses_current_task_and_preserves_literal_template_tokens(
    monkeypatch,
    spec,
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
    html = render_semantic_task_graph_html(graph, task_spec=spec)
    assert payload(html)["graph"] == before
    assert payload(html)["task_spec"] is None
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


def test_viewer_recomputes_modern_and_archived_coverage_without_runtime(tmp_path):
    from embodichain.gen_sim.task_engine.contracts import canonical_hash
    from embodichain.gen_sim.task_engine.semantic_graph_validation import (
        validate_graph_coverage,
    )

    plan = _plan()
    coverage = validate_graph_coverage(
        plan.graph, plan.task_spec, plan.candidate, plan.role_bindings, plan.lineage
    )
    modern = dict(
        schema_version="semantic_graph_coverage/v1",
        coverage=coverage,
        source=dict(candidate=plan.candidate, role_bindings=plan.role_bindings),
    )
    legacy = dict(
        modern,
        schema_version="semantic_graph_validation/v1",
        checker_revision=1,
        graph_digest=canonical_hash(plan.graph),
        task_spec_digest=canonical_hash(plan.task_spec.to_dict()),
    )
    for validation in (modern, legacy):
        before = deepcopy(validation)
        rendered = render_semantic_task_graph_html(
            plan.graph, task_spec=plan.task_spec, validation=validation
        )
        assert payload(rendered)["validation"] == validation == before
        damaged = deepcopy(validation)
        damaged["coverage"]["issues"].append({"code": "stale", "owner": "graph"})
        with pytest.raises(ValueError, match="stale"):
            render_semantic_task_graph_html(
                plan.graph, task_spec=plan.task_spec, validation=damaged
            )


def test_report_rendering_keeps_original_facts_and_checks_graph_identity(tmp_path):
    from embodichain.gen_sim.task_engine.reporting import validate_execution_report

    plan = _plan()
    report = dict(
        schema_version="task_program_execution_report/v1",
        status="failed",
        task_id=plan.graph["task_id"],
        semantic_call_count=len(plan.graph["nodes"]),
        integration_fingerprint=plan.graph["integration_fingerprint"],
        record_dir=str(tmp_path),
        environments=[
            dict(
                env_id=0,
                success=False,
                terminal_reason="original_failed",
                semantic_success={},
            )
        ],
        runtime_result={"segments": []},
        failure=None,
    )
    report = validate_execution_report(report)
    path = tmp_path / "execution_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    original = path.read_bytes()
    rendered = render_semantic_task_graph_html(plan.graph, report)
    assert payload(rendered)["report"] == report
    assert path.read_bytes() == original
    report["integration_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        render_semantic_task_graph_html(plan.graph, report)


def test_graph_display_never_modifies_a_sealed_bundle(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    identity = b'{"schema_version": "immutable"}'
    (bundle / "integration_fingerprint.json").write_bytes(identity)
    with pytest.raises(ValueError, match="outside a sealed bundle"):
        write_semantic_task_graph_html(_plan().graph, bundle / "semantic_graph.html")
    assert (bundle / "integration_fingerprint.json").read_bytes() == identity
    assert not (bundle / "semantic_graph.html").exists()
