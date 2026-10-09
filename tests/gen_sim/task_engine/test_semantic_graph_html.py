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
from dataclasses import replace
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from embodichain.gen_sim.task_engine.semantic_graph_html import (
    render_semantic_task_graph_html,
)
from embodichain.gen_sim.task_engine._bundle_runner import _build_execution_report
from embodichain.gen_sim.task_engine import cli
from .test_semantic_graph_validation import _plan, _contract
from embodichain.utils.utility import load_config


class _Scripts(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.scripts = []
        self.current = None
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.current = [dict(attrs), ""]

    def handle_data(self, data):
        if self.current is not None:
            self.current[1] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.scripts.append(self.current)
            self.current = None


def _report(graph, runtime=None):
    return _build_execution_report(
        graph,
        runtime,
        row_success=[True],
        terminal_reasons=["completed"],
        failure=None,
        trajectory_root=Path("/local/test"),
    )


def test_html_is_offline_and_embedded_data_cannot_terminate_script() -> None:
    graph = deepcopy(_plan().graph)
    graph["instruction"] = '</script><script>alert("unsafe")</script>&\u2028end'
    content = render_semantic_task_graph_html(graph)
    scripts = _Scripts(content).scripts
    assert len(scripts) == 2
    assert json.loads(scripts[0][1])["graph"] == graph
    assert "</script>" not in scripts[0][1]
    assert "fetch(" not in scripts[1][1]
    assert "connect-src 'none'" in content
    assert "__GRAPH_DATA__" not in content
    node = shutil.which("node")
    if node:
        subprocess.run(
            [node, "-e", "new Function(require('fs').readFileSync(0,'utf8'))"],
            input=scripts[1][1],
            text=True,
            check=True,
            capture_output=True,
        )


def test_html_companions_are_identity_checked(tmp_path) -> None:
    plan = _contract(tmp_path)
    envelope = load_config(tmp_path / "semantic_graph_validation.json")
    content = render_semantic_task_graph_html(
        plan.graph, task_spec=plan.task_spec, validation=envelope
    )
    assert json.loads(_Scripts(content).scripts[0][1])["validation"] == envelope
    with pytest.raises(ValueError, match="identity"):
        render_semantic_task_graph_html(
            plan.graph, task_spec=replace(plan.task_spec, task_id="foreign")
        )
    envelope.pop("source")
    with pytest.raises(ValueError, match="Malformed"):
        render_semantic_task_graph_html(
            plan.graph, task_spec=plan.task_spec, validation=envelope
        )


@pytest.mark.parametrize("runtime", [[1], {"segments": [3]}, {"segments": "wrong"}])
def test_nested_report_errors_are_explicit(runtime) -> None:
    plan = _plan()
    report = _report(plan.graph)
    report["runtime_result"] = runtime
    with pytest.raises((ValueError, TypeError), match="runtime|segment"):
        render_semantic_task_graph_html(plan.graph, report)


def test_unknown_segments_are_not_associated_by_index_and_stale_verdict_rejected() -> (
    None
):
    plan = _plan()
    report = _report(plan.graph, {"segments": [{"name": "other", "successes": [True]}]})
    data = json.loads(
        _Scripts(render_semantic_task_graph_html(plan.graph, report)).scripts[0][1]
    )
    assert data["call_evidence"] == {}
    report["runtime_result"]["task_spec_evaluation"] = {
        "schema_version": "task_spec_evaluation/v1",
        "graph_digest": "foreign",
    }
    with pytest.raises(ValueError, match="identity"):
        render_semantic_task_graph_html(plan.graph, report, task_spec=plan.task_spec)


@pytest.mark.parametrize(
    "damage", ["missing_goal", "duplicate_goal", "unknown_goal", "false_summary"]
)
def test_runtime_goal_overlay_requires_complete_owned_clauses(damage: str) -> None:
    from .test_task_spec_runtime import _evidence, _evaluate

    data = _evidence()
    plan = data[0]
    evaluation = _evaluate(data)
    runtime = deepcopy(data[2])
    runtime["task_spec_evaluation"] = evaluation
    report = _report(plan.graph, runtime)
    render_semantic_task_graph_html(plan.graph, report, task_spec=plan.task_spec)
    row = report["runtime_result"]["task_spec_evaluation"]["environments"][0]
    if damage == "missing_goal":
        row.pop("goal")
    elif damage == "duplicate_goal":
        row["goal"].append(deepcopy(row["goal"][0]))
    elif damage == "unknown_goal":
        row["goal"][0]["predicate_id"] = "foreign"
    else:
        row["goal"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="clauses|summary"):
        render_semantic_task_graph_html(plan.graph, report, task_spec=plan.task_spec)


def test_small_fit_can_zoom_out_without_jumping_to_larger_scale() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    script = _Scripts(render_semantic_task_graph_html(_plan().graph)).scripts[1][1]
    functions = "\n".join(
        line
        for line in script.splitlines()
        if line.startswith(("function fit(", "function zoom("))
    )
    program = (
        "const view={x:0,y:0,scale:1},worldBounds={x:-100,y:-50,width:4000,height:600};const svg={getBoundingClientRect:()=>({width:400,height:600})};const transform=()=>{};"
        + functions
        + ";fit();const before=view.scale;zoom(1/1.2);if(!(view.scale<before))process.exit(1);"
    )
    subprocess.run([node, "-e", program], check=True, text=True, capture_output=True)


@pytest.mark.parametrize(
    "active,successes,expected",
    [
        ([True, False], [False, True], "failed"),
        (None, [True, True], "unknown"),
        ([True], [True, True], "unknown"),
        ([True, True], [True, True], "succeeded"),
    ],
)
def test_call_status_preserves_failures_and_requires_exact_active_evidence(
    active, successes, expected
) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable for pure status-function unit test")
    script = _Scripts(render_semantic_task_graph_html(_plan().graph)).scripts[1][1]
    function = next(
        line for line in script.splitlines() if line.startswith("function callStatus(")
    )
    program = (
        "const report={environments:[{env_id:0},{env_id:1}]}; const envRows=()=>report.environments; const data="
        + json.dumps(
            {
                "call_evidence": {
                    "call": {
                        "metadata": {"runtime": {"env_ids": [0, 1]}},
                        "active": active,
                        "successes": successes,
                    }
                }
            }
        )
        + ";"
        + function
        + ";process.stdout.write(callStatus({id:'call'}));"
    )
    result = subprocess.run(
        [node, "-e", program], check=True, text=True, capture_output=True
    )
    assert result.stdout == expected


def test_cli_html_and_explicit_missing_companion(tmp_path) -> None:
    graph = tmp_path / "semantic_task_graph.json"
    graph.write_text(json.dumps(_plan().graph), encoding="utf-8")
    output = tmp_path / "semantic_graph.html"
    assert (
        cli.main(
            [
                "visualize",
                "--graph",
                str(graph),
                "--output",
                str(output),
                "--view",
                "groups",
            ]
        )
        == 0
    )
    assert output.is_file()
    with pytest.raises(SystemExit):
        cli.main(
            [
                "visualize",
                "--graph",
                str(graph),
                "--output",
                str(output),
                "--task-spec",
                str(tmp_path / "missing.json"),
            ]
        )
