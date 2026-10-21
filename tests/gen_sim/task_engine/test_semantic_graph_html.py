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
import base64
from dataclasses import replace
from html.parser import HTMLParser
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from embodichain.gen_sim.task_engine.semantic_graph_html import (
    _finalize_run_html,
    render_semantic_task_graph_html,
    write_semantic_task_graph_html,
)
from embodichain.gen_sim.task_engine._semantic_graph_presentation import (
    _graph_presentation,
    _thumbnail,
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


def test_writer_embeds_bounded_local_images_with_provenance(tmp_path: Path) -> None:
    image_module = pytest.importorskip("PIL.Image")
    plan = _plan()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    project = tmp_path / "project"
    cutout = project / "scene_understanding/object_images/a_rgba.png"
    render = project / "scene_generation/debug/vlm_views/a.png"
    for path in (cutout, render):
        path.parent.mkdir(parents=True, exist_ok=True)
        image_module.new("RGBA", (900, 600), (0, 100, 200, 255)).save(path)
    objects = [{"uid": "a", "name": "Cup", "geometry": {"shape": {"fpath": "a.glb"}}}]
    (bundle / "scene_manifest.json").write_text(
        json.dumps({"objects": objects, "robot_profile": "dual_franka"})
    )
    (bundle / "static_scene_manifest.json").write_text(
        json.dumps(
            {
                "source": {
                    "config_path": str(project / "scene_export/scene_config.json")
                },
                "objects": objects,
            }
        )
    )
    output = tmp_path / "run/semantic_graph.html"
    report = _report(plan.graph)
    before = deepcopy(report)
    write_semantic_task_graph_html(
        plan.graph,
        output,
        report,
        bundle_root=bundle,
        run_summary={
            "task_id": plan.graph["task_id"],
            "elapsed_seconds": 12.345,
            "elapsed_scope": "test workflow",
        },
    )
    data = json.loads(_Scripts(output.read_text()).scripts[0][1])
    assert report == before
    assert data["report"] == report
    assert data["graph"] == plan.graph
    assert data["presentation"]["wall_seconds"] == 12.345
    asset = data["presentation"]["assets"][0]
    assert asset["cutout_source"] == cutout.as_posix()
    assert asset["render_source"] == render.as_posix()
    for key, size in (("cutout", 192), ("render", 768)):
        with image_module.open(
            BytesIO(base64.b64decode(asset[key].split(",")[1]))
        ) as image:
            assert max(image.size) <= size
    assert "DESIGN PREVIEW" not in output.read_text()


@pytest.mark.parametrize("elapsed", [None, -1, float("nan"), float("inf"), True, "12"])
def test_wall_time_is_never_inferred_from_invalid_or_missing_data(elapsed) -> None:
    graph = _plan().graph
    data = _graph_presentation(
        graph, run_summary={"task_id": graph["task_id"], "elapsed_seconds": elapsed}
    )
    assert data["wall_seconds"] is None
    assert (
        _graph_presentation(
            graph, run_summary={"task_id": "other", "elapsed_seconds": 12}
        )["wall_seconds"]
        is None
    )


def test_missing_corrupt_and_external_images_stay_unavailable(tmp_path: Path) -> None:
    plan = _plan()
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not a PNG")
    link = project / "linked.png"
    link.symlink_to(outside)
    assert _thumbnail(link, project, render=False) is None
    assert _thumbnail(outside, tmp_path, render=False) is None
    (project / "scene_manifest.json").write_text("invalid JSON")
    (project / "static_scene_manifest.json").write_text(json.dumps({"objects": None}))
    assert _graph_presentation(plan.graph, bundle_root=project)["assets"] == []
    (project / "scene_manifest.json").write_text(
        json.dumps({"objects": [{"uid": "a"}]})
    )
    asset = _graph_presentation(plan.graph, bundle_root=project)["assets"][0]
    assert asset["render"] is None and asset["cutout"] is None


def test_completed_run_finalizer_generates_selected_html_without_changing_evidence(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    bundle = run / "final/bundle"
    plan = _contract(bundle)
    (bundle / "semantic_task_graph.json").write_text(json.dumps(plan.graph))
    selected = run / "attempts/scene_0001/action_attempts/action_0002"
    selected.mkdir(parents=True)
    report = _report(plan.graph)
    (selected / "execution_report.json").write_text(json.dumps(report))
    (run / "final/selection.json").write_text(
        json.dumps({"scene_attempt": 1, "action_attempt": 2})
    )
    before = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    summary = {
        "task_id": plan.graph["task_id"],
        "run_id": "run",
        "elapsed_seconds": 42.25,
        "elapsed_scope": "workflow wall clock",
    }
    _finalize_run_html(run, bundle=bundle, summary=summary)
    data = json.loads(
        _Scripts((selected / "semantic_graph.html").read_text()).scripts[0][1]
    )
    assert data["presentation"]["wall_seconds"] == 42.25
    assert data["report"] == report
    assert (
        json.loads((run / "run_summary.json").read_text())["elapsed_seconds"] == 42.25
    )
    assert all(path.read_bytes() == value for path, value in before.items())


def test_finalizer_does_not_rewrite_an_external_prepared_bundle(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    historical = tmp_path / "old_bundle/semantic_graph.html"
    historical.parent.mkdir()
    historical.write_text("historical viewer")
    _finalize_run_html(
        run,
        bundle=historical.parent,
        html_path=historical,
        summary={"elapsed_seconds": 1},
    )
    assert historical.read_text() == "historical viewer"


@pytest.mark.parametrize("damage", ["report", "contract"])
def test_finalizer_preserves_existing_html_on_identity_failure(
    tmp_path: Path, capsys, damage: str
) -> None:
    run = tmp_path / "run"
    bundle = run / "bundle"
    plan = _contract(bundle)
    (bundle / "semantic_task_graph.json").write_text(json.dumps(plan.graph))
    report = _report(plan.graph)
    if damage == "report":
        report["integration_fingerprint"] = "foreign"
    else:
        (bundle / "components/scene.yaml").write_text("stale: true\n")
    (run / "execution_report.json").write_text(json.dumps(report))
    html = run / "semantic_graph.html"
    html.write_text("existing viewer")
    _finalize_run_html(
        run, bundle=bundle, html_path=html, summary={"elapsed_seconds": 1}
    )
    assert html.read_text() == "existing viewer"
    assert "presentation unavailable" in capsys.readouterr().err


def test_channels_and_parallel_node_layout_are_not_conflated() -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    script = _Scripts(render_semantic_task_graph_html(_plan().graph)).scripts[1][1]
    functions = "\n".join(
        line
        for line in script.splitlines()
        if line.startswith(
            ("function lane(", "function groupLane(", "function layout(")
        )
    )
    program = (
        """
const mode='groups',positions=new Map(),laneY={},laneHeight={};let worldBounds;
const calls=new Map([
['a',{call:{resources:{primary:'left'}}}],
['b',{call:{resources:{primary:'left'}}}],
['c',{call:{resources:{primary:'right'}}}]
]);
const graph={task_groups:[{id:'a',node_ids:['a'],depends_on:[]},{id:'b',node_ids:['b'],depends_on:[]},{id:'c',node_ids:['c'],depends_on:[]}]};
const groups=new Map(graph.task_groups.map(g=>[g.id,g]));
"""
        + functions
        + """
layout();
if(positions.get('group:b').y+103>=laneY.auto)throw Error('Parallel nodes overlap AUTO');
const resources=[{}, {primary:'auto'}, {primary:'left'}, {primary:'right'}, {source:'left',destination:'right'}];
process.stdout.write(JSON.stringify(resources.map(resources=>lane({call:{resources}}))));
"""
    )
    result = subprocess.run(
        [node, "-e", program], text=True, capture_output=True, check=True
    )
    assert json.loads(result.stdout) == ["auto", "auto", "left", "right", "coordinated"]


@pytest.mark.parametrize("prepared", [False, True])
def test_successful_cli_automatically_refreshes_html_and_persists_real_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, prepared: bool
) -> None:
    plan = _plan()
    source_bundle = tmp_path / "prepared"
    if prepared:
        _contract(source_bundle)
        (source_bundle / "semantic_task_graph.json").write_text(json.dumps(plan.graph))
    ticks = iter([100.0, 112.5])
    monkeypatch.setattr(cli, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(cli, "SceneAdapter", lambda **_kwargs: object())
    monkeypatch.setattr(cli, "_environment_successes", lambda *_args, **_kwargs: [True])

    def execute(bundle, output, **_kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        report = _report(plan.graph)
        (output / "execution_report.json").write_text(json.dumps(report))
        (output / "semantic_graph.html").write_text("initial viewer")
        return report

    class Workflow:
        def __init__(self, **_kwargs):
            pass

        def run(self, request, **_kwargs):
            output = Path(request["output_dir"])
            bundle = output / "final/bundle"
            _contract(bundle)
            (bundle / "semantic_task_graph.json").write_text(json.dumps(plan.graph))
            execute(bundle, output)
            return SimpleNamespace(
                status="succeeded",
                succeeded=True,
                failure_class=None,
                output_dir=output,
                manifest_path=output / "run_manifest.json",
                final_bundle=bundle,
            )

    monkeypatch.setattr(cli, "TaskEngineWorkflow", Workflow)
    monkeypatch.setattr(cli, "SubprocessActionExecutor", lambda: execute)
    args = (
        ["run", "--bundle", str(source_bundle)]
        if prepared
        else [
            "run-all",
            "--mode",
            "image",
            "--task-id",
            plan.graph["task_id"],
            "--instruction",
            "upright object",
            "--image",
            str(tmp_path / "input.png"),
        ]
    )
    assert cli.main([*args, "--output-root", str(tmp_path / "history")]) == 0
    payload = json.loads(capsys.readouterr().out)
    summary = json.loads((Path(payload["output_dir"]) / "run_summary.json").read_text())
    data = json.loads(
        _Scripts(Path(payload["semantic_graph_html"]).read_text()).scripts[0][1]
    )
    assert payload["elapsed_seconds"] == summary["elapsed_seconds"] == 12.5
    assert data["presentation"]["wall_seconds"] == 12.5
    assert data["presentation"]["run_id"] == payload["run_id"]
    assert data["report"] == _report(plan.graph)
