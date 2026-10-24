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

"""Offline, single-canvas exploration of the real semantic graph and contracts."""

from __future__ import annotations

from collections.abc import Mapping
import base64
import json
from pathlib import Path
import re
import sys
from typing import Any

from embodichain.gen_sim.task_spec import TaskSpec, validate_task_spec

from .contracts import canonical_hash
from ._semantic_graph_presentation import _graph_presentation
from .reporting import validate_execution_report
from .semantic_graph import validate_semantic_task_graph
from .semantic_graph_validation import (
    GRAPH_VALIDATION_SCHEMA,
    validate_graph_coverage,
    _validate_runtime_evaluation,
)

__all__ = ["render_semantic_task_graph_html", "write_semantic_task_graph_html"]


def _json_script(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    for character, escaped in (
        ("<", r"\u003c"),
        (">", r"\u003e"),
        ("&", r"\u0026"),
        ("\u2028", r"\u2028"),
        ("\u2029", r"\u2029"),
    ):
        encoded = encoded.replace(character, escaped)
    return encoded


def _viewer_document(data: Mapping[str, Any]) -> str:
    root = Path(__file__).parent
    assets = root / "_graph_assets"
    fonts = (assets / "fonts.css").read_text(encoding="utf-8")
    for token, name in (
        ("__PLEX_REGULAR__", "plex-sc-regular.woff2"),
        ("__PLEX_MEDIUM__", "plex-sc-medium.woff2"),
        ("__MANROPE__", "manrope.ttf"),
    ):
        fonts = fonts.replace(
            token, base64.b64encode((assets / name).read_bytes()).decode("ascii")
        )
    values = {
        "__GRAPH_DATA__": _json_script(data),
        "__VIEWER_STYLE__": fonts
        + (root / "_semantic_graph_viewer.css").read_text(encoding="utf-8"),
        "__VIEWER_SCRIPT__": (root / "_semantic_graph_viewer.js").read_text(
            encoding="utf-8"
        ),
        "__FONT_LICENSES__": _json_script(
            (assets / "LICENSE.txt").read_text(encoding="utf-8")
        ),
    }
    template = (root / "_semantic_graph_viewer.html").read_text(encoding="utf-8")
    pattern = re.compile(r"__[A-Z][A-Z_]*__")
    tokens = pattern.findall(template)
    if set(tokens) != set(values) or any(tokens.count(token) != 1 for token in values):
        raise ValueError("Semantic graph viewer has invalid template slots.")
    # Substitute once so token-like source instructions stay literal evidence.
    return pattern.sub(lambda match: values[match.group()], template)


def render_semantic_task_graph_html(
    graph: Mapping[str, Any],
    execution_report: Mapping[str, Any] | None = None,
    *,
    task_spec: TaskSpec | Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    presentation: Mapping[str, Any] | None = None,
) -> str:
    """Render self-contained HTML without altering graph or recorded evidence.

    Args:
        graph: Existing v38 graph topology and call data.
        execution_report: Optional identity-matched original execution report.
        task_spec: Optional normative companion; absent stays absent.
        validation: Optional recomputable static coverage envelope.
        presentation: Optional display metadata, never a success certificate.

    Returns:
        Standalone HTML with embedded data, styles, SVG and local interactions.

    Raises:
        ValueError: A report or companion identity does not match this graph.
    """
    selected = validate_semantic_task_graph(graph)
    spec = None if task_spec is None else TaskSpec.from_dict(task_spec)
    if spec is not None:
        validate_task_spec(spec)
    if spec is not None and spec.task_id != selected["task_id"]:
        raise ValueError("HTML TaskSpec task identity does not match graph.")
    envelope = None if validation is None else dict(validation)
    if envelope is not None:
        if (
            spec is None
            or envelope.get("schema_version") != GRAPH_VALIDATION_SCHEMA
            or envelope.get("checker_revision") != 1
            or envelope.get("graph_digest") != canonical_hash(selected)
            or envelope.get("task_spec_digest") != canonical_hash(spec.to_dict())
        ):
            raise ValueError(
                "HTML coverage companion identity does not match graph/TaskSpec."
            )
        source = envelope.get("source")
        if (
            not isinstance(source, Mapping)
            or set(source) != {"candidate", "role_bindings"}
            or not isinstance(envelope.get("coverage"), Mapping)
            or "lineage" not in envelope["coverage"]
        ):
            raise ValueError("Malformed HTML coverage companion.")
        coverage = validate_graph_coverage(
            selected,
            spec,
            source["candidate"],
            source["role_bindings"],
            envelope["coverage"]["lineage"],
        )
        if coverage != envelope["coverage"]:
            raise ValueError("HTML coverage companion is stale.")
    report = (
        None
        if execution_report is None
        else validate_execution_report(execution_report)
    )
    calls: dict[str, dict[str, Any]] = {}
    evaluation = None
    if report is not None:
        if (
            report["task_id"] != selected["task_id"]
            or report["integration_fingerprint"] != selected["integration_fingerprint"]
            or report["semantic_call_count"] != len(selected["nodes"])
        ):
            raise ValueError(
                "Execution report identity does not match this semantic graph."
            )
        env_ids = [row["env_id"] for row in report["environments"]]
        if len(set(env_ids)) != len(env_ids):
            raise ValueError(
                "Execution report contains duplicate environment identities."
            )
        nodes = {node["id"] for node in selected["nodes"]}
        runtime = report["runtime_result"] or {}
        if not isinstance(runtime, Mapping) or not isinstance(
            runtime.get("segments", []), list
        ):
            raise ValueError("Malformed execution report runtime evidence.")
        for segment in runtime.get("segments", []):
            if not isinstance(segment, Mapping):
                raise ValueError("Malformed execution report segment evidence.")
            name = segment.get("name")
            if name not in nodes:
                continue
            if name in calls:
                raise ValueError(
                    "Execution report contains duplicate segment evidence."
                )
            calls[name] = segment
        evaluation = runtime.get("task_spec_evaluation")
        if evaluation is not None:
            if spec is None:
                raise ValueError("HTML TaskSpec runtime evaluation identity differs.")
            _validate_runtime_evaluation(
                evaluation, spec, selected, env_ids, contract=envelope
            )
    data = {
        "graph": selected,
        "task_spec": None if spec is None else spec.to_dict(),
        "validation": envelope,
        "report": report,
        "call_evidence": calls,
        "task_spec_evaluation": evaluation,
        "identities": {
            "graph_digest": canonical_hash(selected),
            "integration_fingerprint": selected["integration_fingerprint"],
        },
        "presentation": {
            **_graph_presentation(selected, execution_report=report),
            **dict(presentation or {}),
        },
    }
    return _viewer_document(data)


def write_semantic_task_graph_html(
    graph: Mapping[str, Any],
    output: str | Path,
    execution_report: Mapping[str, Any] | None = None,
    *,
    task_spec: TaskSpec | Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    bundle_root: str | Path | None = None,
    run_summary: Mapping[str, Any] | None = None,
) -> Path:
    """Write an offline graph viewer.

    Args:
        graph: Canonical graph source.
        output: New HTML destination, not a historical JSON/YAML artifact.
        execution_report: Optional original report for separate runtime overlays.
        task_spec: Optional normative task declaration.
        validation: Optional identity-bound static coverage.
        bundle_root: Bundle owning optional scene metadata and asset-image sources.
            Defaults to the output directory. Missing images stay unavailable.
        run_summary: Optional task-matched wall-clock summary, separate from evidence.

    Returns:
        Absolute HTML path.
    """
    destination = Path(output).expanduser().resolve()
    selected = validate_semantic_task_graph(graph)
    content = render_semantic_task_graph_html(
        selected,
        execution_report,
        task_spec=task_spec,
        validation=validation,
        presentation=_graph_presentation(
            selected,
            bundle_root=(
                Path(bundle_root).expanduser().resolve()
                if bundle_root is not None
                else destination.parent
            ),
            output=destination,
            execution_report=execution_report,
            run_summary=run_summary,
        ),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def _finalize_run_html(
    output_dir: Path,
    *,
    bundle: Path | None,
    summary: Mapping[str, Any],
    html_path: str | Path | None = None,
    translate: bool = False,
    model: str | None = None,
) -> Path | None:
    """Publish display-only timing and refresh only this run's selected viewer."""
    if not output_dir.is_dir():
        return
    try:
        (output_dir / "run_summary.json").write_text(
            json.dumps(
                {"schema_version": "task_engine_run_summary/v1", **summary},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if html_path is None:
            selection_path = output_dir / "final/selection.json"
            if selection_path.is_file():
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                if not isinstance(selection, dict):
                    raise ValueError("Malformed run selection for HTML presentation.")
                scene, action = selection.get("scene_attempt"), selection.get(
                    "action_attempt"
                )
                if (
                    type(scene) is not int
                    or scene < 1
                    or (action is not None and (type(action) is not int or action < 1))
                ):
                    raise ValueError("Invalid run selection for HTML presentation.")
                if (
                    type(scene) is int
                    and type(action) is int
                    and min(scene, action) > 0
                ):
                    scene_root = output_dir / "attempts" / f"scene_{scene:04d}"
                    html_path = (
                        scene_root
                        / "action_attempts"
                        / f"action_{action:04d}"
                        / "semantic_graph.html"
                    )
                    bundle = bundle or scene_root / "bundle"
            if html_path is None and bundle is not None:
                html_path = bundle / "semantic_graph.html"
            if html_path is None:
                reports = sorted(
                    output_dir.glob(
                        "attempts/scene_*/action_attempts/action_*/execution_report.json"
                    )
                )
                if reports:
                    html_path = reports[-1].parent / "semantic_graph.html"
                    bundle = reports[-1].parent.parent.parent / "bundle"
        if bundle is None or html_path is None:
            return None
        destination = Path(html_path).resolve()
        if not destination.is_relative_to(output_dir.resolve()):
            return
        if not all(
            (bundle / name).is_file()
            for name in (
                "semantic_task_graph.json",
                "task_spec.json",
                "semantic_graph_validation.json",
            )
        ):
            return
        from .semantic_graph_validation import verify_graph_contract

        graph = json.loads(
            (bundle / "semantic_task_graph.json").read_text(encoding="utf-8")
        )
        contract = verify_graph_contract(bundle, graph)
        report_path = destination.parent / "execution_report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else None
        )
        if translate:
            from ._graph_localization import _FILENAME, _ensure_localizations

            _ensure_localizations(
                destination.parent / _FILENAME,
                graph,
                _graph_presentation(graph, bundle_root=bundle),
                model=model,
            )
        write_semantic_task_graph_html(
            graph,
            destination,
            report,
            task_spec=json.loads(
                (bundle / "task_spec.json").read_text(encoding="utf-8")
            ),
            validation=contract,
            bundle_root=bundle,
            run_summary=summary,
        )
        return destination
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"[Task Engine] Semantic graph presentation unavailable: {exc}",
            file=sys.stderr,
        )
