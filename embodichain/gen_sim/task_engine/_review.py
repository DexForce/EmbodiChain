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

"""Render a plan snapshot with an optional identity-checked native attempt."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from errno import EEXIST
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..task_spec import canonical_hash
from ._image_run import _image_preview

if TYPE_CHECKING:
    from ..services import GenSimServices

__all__: list[str] = []


def _attempt(plan: dict[str, Any], root: Path, nodes: set[str]) -> dict[str, Any]:
    manifest = json.loads((root / "run_manifest.json").read_text())
    feedback = json.loads((root / "attempt_feedback.json").read_text())
    report = json.loads((root / "execution_report.json").read_text())
    keys = (
        "spec_digest",
        "graph_digest",
        "plan_digest",
        "bundle_digest",
        "scene_digest",
        "program_digest",
        "policy_sha256",
        "run_id",
        "attempt_id",
    )
    if (
        feedback.get("schema_version") != "embodichain.attempt-feedback/v1"
        or any(not feedback.get(k) or manifest.get(k) != feedback[k] for k in keys)
        or manifest["spec_digest"] != plan["spec_digest"]
        or manifest["graph_digest"] != plan["semantic_graph_sha256"]
        or manifest["plan_digest"] != plan["plan_digest"]
        or feedback["execution_report_digest"] != canonical_hash(report)
        or feedback["execution"] != report["status"]
    ):
        raise ValueError("Execution evidence does not match this plan and attempt")
    calls = {}
    for segment in report.get("segments", []):
        name = segment["name"]
        if name not in nodes or name in calls:
            raise ValueError("Execution segment cannot be paired with one graph call")
        runtime = segment.get("metadata", {}).get("runtime") or {}
        issues = [
            {
                "kind": e["kind"],
                "message": e.get("message"),
                "timestamp": e.get("timestamp"),
                "env_mask": e.get("env_mask"),
            }
            for e in runtime.get("events", [])
            if "failed" in e["kind"]
        ]
        calls[name] = {
            k: segment.get(k)
            for k in (
                "success",
                "failure_reason",
                "start_step",
                "end_step",
                "successes",
                "active",
                "failure_reasons",
                "start_steps",
                "end_steps",
            )
        }
        # DemoSegmentResult row arrays use global environment indices, even
        # when the runtime invocation itself only contains a subset of rows.
        env_ids = (
            list(range(len(segment["active"])))
            if segment.get("active")
            else runtime.get("env_ids", [])
        )
        calls[name].update(issues=issues, env_ids=env_ids)
    return {
        "status": feedback["execution"],
        "physical": feedback["physical_verdict"],
        "verdict": manifest.get("task_spec_verdict"),
        "step": feedback.get("capture_step"),
        "missing": feedback.get("missing_evidence", []),
        "calls": calls,
        "identity": {k: feedback[k] for k in keys},
    }


def _producer_pairing(
    graph: dict[str, Any], spec: dict[str, Any], coverage: dict[str, Any]
) -> dict[str, Any]:
    group = {n["id"]: n["task_instance_id"] for n in graph["nodes"]}
    # Authoring coverage stores a node ID; native coverage stores a node/group pair.
    producers = {
        identifier: producer if isinstance(producer, str) else producer[0]
        for identifier, producer in coverage["producers"].items()
    }
    audited = coverage.get("goal_producers", {})
    milestone_calls = {
        m["id"]: producers[m["id"]]
        for m in spec["objectives"]["milestones"]
        if m["id"] in producers
    }
    goal_calls = {
        g["id"]: audited[g["id"]] if g["id"] in audited else [producers[g["id"]]]
        for g in spec["objectives"]["goal"]
        if g["id"] in audited or g["id"] in producers
    }
    return {
        "milestone_groups": {
            identifier: group[node] for identifier, node in milestone_calls.items()
        },
        "goal_groups": {
            identifier: list(dict.fromkeys(group[node] for node in nodes))
            for identifier, nodes in goal_calls.items()
        },
        "milestone_calls": milestone_calls,
        "goal_calls": goal_calls,
    }


def _read_preview_boxes(
    path: Path, image_sha256: str, roles: set[str]
) -> dict[str, list[float] | None]:
    packet = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if (
        not isinstance(packet, dict)
        or set(packet) != {"image_sha256", "boxes"}
        or not isinstance(packet["boxes"], dict)
    ):
        raise ValueError("Preview box input requires image_sha256 and a boxes object")
    if packet["image_sha256"] != image_sha256:
        raise ValueError("Preview boxes belong to a different source image")
    for role, box in packet["boxes"].items():
        if role not in roles:
            raise ValueError(f"Unknown preview box role: {role!r}")
        if box is not None and (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in box)
            or not (box[0] < box[2] and box[1] < box[3])
        ):
            raise ValueError(
                f"Expected a normalized bbox [x0, y0, x1, y1] for {role!r}"
            )
    return packet["boxes"]


def _request_preview_box_packet(
    spec: dict[str, Any], image: Path, *, services: GenSimServices | None = None
) -> dict[str, Any]:
    """Locate visible frozen roles for display, not for scene binding."""
    from ..services.compat import structured_call

    image_sha256 = spec["inputs"]["image"]["sha256"]
    if sha256(image.read_bytes()).hexdigest() != image_sha256:
        raise ValueError("Preview boxes require the frozen source image")
    configuration = spec["configuration"]
    roles = {
        role["id"]: {
            "description": configuration.get("descriptions", {}).get(
                role["id"], role["id"]
            ),
            "kind": role["kind"],
            "part_of": role.get("properties", {}).get("part_of"),
            "cardinality": role.get("cardinality", "one"),
        }
        for role in configuration["roles"]
    }
    if not roles:
        return {"image_sha256": image_sha256, "boxes": {}}
    bbox_schema = {
        "anyOf": [
            {"type": "null"},
            {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0, "maximum": 1000},
            },
        ]
    }
    raw = structured_call(
        role="task_image_grounding",
        services=services,
        image_path=image,
        schema={
            "type": "object",
            "required": list(roles),
            "properties": dict.fromkeys(roles, bbox_schema),
            "additionalProperties": False,
        },
        system_prompt=(
            "Locate each listed role in the source image for a visual preview. "
            "Return one tight [x0,y0,x1,y1] box on a 0..1000 grid per clearly identifiable "
            "visible object or part, using the exact role ID. Use null for hidden or ambiguous "
            "roles, abstract locations, and future placements. Do not box shadows or reflections. "
            "A part needs its own distinguishable visible pixels, not just its owner's box. "
            "A description identifies the role but does not prove that it is visible. "
            "Never guess a box."
        ),
        prompt=json.dumps(
            {"instruction": spec["inputs"]["instruction"], "roles": roles},
            ensure_ascii=False,
        ),
    )
    if sha256(image.read_bytes()).hexdigest() != image_sha256:
        raise ValueError("Source image changed during preview localization")
    if not isinstance(raw, Mapping) or raw.keys() != roles.keys():
        raise ValueError("Preview localization requires one entry per frozen role")
    boxes = {}
    for role, bbox in raw.items():
        if bbox is not None and (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(type(value) is not int or not 0 <= value <= 1000 for value in bbox)
            or not (bbox[0] < bbox[2] and bbox[1] < bbox[3])
        ):
            raise ValueError(f"Invalid preview bbox for {role!r}")
        boxes[role] = None if bbox is None else [value / 1000 for value in bbox]
    return {"image_sha256": image_sha256, "boxes": boxes}


def _hybrid_preview(result, image, attempt, *, preview_boxes=None):
    """Render verified hybrid contracts without template candidate metadata."""
    import base64
    from io import BytesIO
    from PIL import Image
    from .semantic_graph_html import _graph_data, _viewer_document

    graph = result["semantic_task_graph"]
    spec = result["task_spec_v1"]
    grounding = result.get("image_grounding", {})
    role_ids = {r["id"] for r in spec["configuration"]["roles"]}
    source_boxes = {
        obj["id"]: obj["bbox"]
        for obj in grounding.get("objects", [])
        if obj["provenance"] == "observed"
        and obj["bbox"] is not None
        and obj["id"] in role_ids
        and grounding.get("bindings", {}).get(obj["id"]) == [obj["id"]]
    }
    data = _graph_data(graph, task_spec=result["task_spec"])
    with Image.open(image) as picture:
        picture = picture.convert("RGB")
        picture.thumbnail((640, 480))
        buffer = BytesIO()
        picture.save(buffer, "JPEG")
    data["review"] = {
        "source_image": "data:image/jpeg;base64,"
        + base64.b64encode(buffer.getvalue()).decode(),
        "spec": spec,
        "pending": sum(
            o["provenance"] != "observed" for o in grounding.get("objects", [])
        ),
        "attempt": attempt,
        "pairing": {
            "role_entities": {
                r["id"]: [r["id"]] for r in spec["configuration"]["roles"]
            },
            **_producer_pairing(graph, spec, result["coverage"]),
        },
    }
    if source_boxes:
        data["review"]["source_boxes"] = source_boxes
    if preview_boxes:
        data["review"]["preview_boxes"] = preview_boxes
    annotations = (
        result.get("semantic_graph_validation", {}).get("source", {}).get("annotations")
    )
    if annotations is not None:
        data["review"]["annotations"] = annotations
    return _viewer_document(data)


def write_review(
    plan_dir: Path,
    output: Path,
    run_dir: Path | None = None,
    *,
    preview_boxes: Path | None = None,
    auto_preview_boxes: bool = False,
    services: GenSimServices | None = None,
) -> None:
    from ..workflow import _source_image, load_task_plan

    if preview_boxes is not None and auto_preview_boxes:
        raise ValueError("Choose either manual or automatic preview boxes")
    plan = load_task_plan(plan_dir)
    root = Path(plan["root"])
    destination = output.expanduser().resolve()
    if destination.is_relative_to(root) or any(
        (p / "integration_fingerprint.json").is_file() for p in destination.parents
    ):
        raise ValueError(
            "Select a fresh HTML file outside the published plan or bundle"
        )
    if destination.exists():
        raise FileExistsError(EEXIST, "Review output already exists", destination)
    hybrid = plan["schema_version"].startswith("embodichain.hybrid-")
    keys = (
        ["semantic_task_graph", "task_spec", "task_spec_v1", "coverage"]
        if hybrid
        else [
            "semantic_task_graph",
            "semantic_graph_validation",
            "task_spec",
            "task_candidates",
            "image_grounding",
        ]
    )
    if hybrid and "image_grounding.json" in plan["artifacts"]:
        keys.append("image_grounding")
    if plan["schema_version"] == "embodichain.hybrid-task-plan/v1":
        keys.append("semantic_graph_validation")
    if not hybrid and "task_document" in plan:
        keys.extend(["task_spec_v1", "plan_coverage"])
    result = {k: json.loads((root / (k + ".json")).read_text()) for k in keys}
    if "qualifier_coverage" in plan:
        result["semantic_graph_validation"]["qualifier_coverage"] = plan[
            "qualifier_coverage"
        ]
    attempt = None
    if run_dir is not None:
        if "task_document" not in plan:
            raise ValueError("Native attempt review requires a version-2 plan")
        try:
            attempt = _attempt(
                plan, run_dir, {n["id"] for n in result["semantic_task_graph"]["nodes"]}
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            attempt = {"error": str(exc)}
    summary = json.loads((root / "run_manifest.json").read_text())
    image = _source_image(root, plan)
    boxes = None
    if preview_boxes is not None or auto_preview_boxes:
        spec = result.get("task_spec_v1")
        if spec is None:
            raise ValueError("Preview boxes require a frozen TaskSpec")
        if preview_boxes is not None:
            boxes = _read_preview_boxes(
                preview_boxes,
                plan["image_sha256"],
                {r["id"] for r in spec["configuration"]["roles"]},
            )
        else:
            boxes = _request_preview_box_packet(spec, image, services=services)["boxes"]
    html = (
        _hybrid_preview(result, image, attempt, preview_boxes=boxes)
        if hybrid
        else _image_preview(
            result, image, summary, attempt=attempt, preview_boxes=boxes
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(html)


def main(
    argv: list[str] | None = None, *, prog: str = "embodichain gensim task view"
) -> None:
    """Generate a fresh standalone HTML review using the published source plan."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run", type=Path)
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument(
        "--preview-boxes",
        type=Path,
        metavar="FILE",
        help="Render display-only boxes from a JSON file with image_sha256 and role-keyed boxes",
    )
    preview.add_argument(
        "--auto-preview-boxes",
        action="store_true",
        help="Use task_image_grounding to locate visible roles for this preview",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    write_review(
        args.plan,
        args.output,
        args.run,
        preview_boxes=args.preview_boxes,
        auto_preview_boxes=args.auto_preview_boxes,
    )
    print(args.output.resolve())
