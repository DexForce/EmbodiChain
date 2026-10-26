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
import json
from pathlib import Path

import pytest

from embodichain.gen_sim.task_engine import _graph_localization as translation
from embodichain.gen_sim.task_engine.semantic_graph_html import _finalize_run_html
from .test_semantic_graph_validation import _contract


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


def test_translation_cache_is_display_only_and_reused_before_credentials(
    tmp_path, monkeypatch
):
    graph = {"instruction": "放置纸杯", "task_id": "sample"}
    presentation = {"assets": [{"uid": "cup", "name": "paper cup", "description": ""}]}
    before = deepcopy((graph, presentation))
    calls = []

    def caller(**kwargs):
        calls.append(kwargs)
        return translated(**kwargs)

    path = tmp_path / translation._FILENAME
    first = translation._ensure_localizations(path, graph, presentation, caller=caller)
    assert first["status"] == "translated"
    assert (graph, presentation) == before
    monkeypatch.setattr(
        translation,
        "_default_translation_caller",
        lambda **kwargs: pytest.fail("Cached translations must not invoke a provider"),
    )
    assert translation._ensure_localizations(path, graph, presentation) == first
    assert len(calls) == 1
    presentation["assets"][0]["name"] = "different cup"
    assert translation._read_localizations(path, graph, presentation) is None
    presentation = before[1]
    graph["instruction"] = "different instruction"
    assert translation._read_localizations(path, graph, presentation) is None


@pytest.mark.parametrize(
    "failure", ["timeout", "extra_status", "new_uid", "missing_language", "oversized"]
)
def test_translation_failure_stays_unavailable_and_is_not_retried(tmp_path, failure):
    graph = {"instruction": "source", "task_id": "sample"}
    presentation = {"assets": []}
    count = 0

    def caller(**kwargs):
        nonlocal count
        count += 1
        if failure == "timeout":
            raise TimeoutError("provider detail must not be published")
        result = translated(**kwargs)
        if failure == "extra_status":
            result["status"] = "satisfied"
        elif failure == "new_uid":
            result["assets"]["foreign"] = {}
        elif failure == "missing_language":
            result["instruction"].pop("en")
        else:
            result["title"]["en"] = "x" * 161
        return result

    path = tmp_path / translation._FILENAME
    first = translation._ensure_localizations(path, graph, presentation, caller=caller)
    assert first["status"] == "unavailable"
    assert "localizations" not in first
    assert "provider detail" not in path.read_text()
    assert (
        translation._ensure_localizations(path, graph, presentation, caller=caller)
        == first
    )
    assert count == 1


@pytest.mark.parametrize("failure", [False, True])
def test_two_finalizers_attempt_translation_once_without_changing_bundle(
    tmp_path, monkeypatch, failure
):
    bundle, output = tmp_path / "bundle", tmp_path / "run"
    plan = _contract(bundle)
    (bundle / "semantic_task_graph.json").write_text(json.dumps(plan.graph))
    before = {
        str(path.relative_to(bundle)): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    output.mkdir()
    calls = []

    def caller(**kwargs):
        calls.append(kwargs)
        if failure:
            raise TimeoutError("offline")
        return translated(**kwargs)

    monkeypatch.setattr(translation, "_default_translation_caller", caller)
    import embodichain.gen_sim.task_engine.interpretation as interpretation

    monkeypatch.setattr(
        interpretation, "_instruction_model", lambda model: model or "test-model"
    )
    for _ in range(2):
        result = _finalize_run_html(
            output,
            bundle=bundle,
            html_path=output / "semantic_graph.html",
            summary={"task_id": plan.graph["task_id"], "elapsed_seconds": 0},
            translate=True,
            model="selected-model",
        )
        assert result == output / "semantic_graph.html"
    assert len(calls) == 1 and calls[0]["model"] == "selected-model"
    assert (output / translation._FILENAME).is_file()
    assert not (bundle / translation._FILENAME).exists()
    assert before == {
        str(path.relative_to(bundle)): path.read_bytes()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    saved = json.loads((output / translation._FILENAME).read_text())
    assert saved["status"] == ("unavailable" if failure else "translated")


def test_display_cache_cannot_follow_foreign_symlink(tmp_path):
    graph, presentation = {"instruction": "source"}, {"assets": []}
    foreign = tmp_path / "foreign.json"
    translation._ensure_localizations(foreign, graph, presentation, caller=translated)
    local = tmp_path / "run" / translation._FILENAME
    local.parent.mkdir()
    local.symlink_to(foreign)
    assert translation._read_localizations(local, graph, presentation) is None


def test_cli_finalizer_forwards_the_explicit_display_model(tmp_path, monkeypatch):
    from embodichain.gen_sim.task_engine.cli import _finish_html
    import embodichain.gen_sim.task_engine.semantic_graph_html as html

    captured = {}
    output = tmp_path / "semantic_graph.html"

    def finalize(*args, **kwargs):
        captured.update(kwargs)
        return output

    monkeypatch.setattr(html, "_finalize_run_html", finalize)
    result = _finish_html(
        tmp_path,
        tmp_path / "bundle",
        {"run_id": "test-run", "status": "succeeded", "elapsed_seconds": 1},
        task_id="test-task",
        created_at="2026-10-24T00:00:00+00:00",
        model="explicit-model",
    )
    assert result == str(output)
    assert captured["translate"] is True
    assert captured["model"] == "explicit-model"
