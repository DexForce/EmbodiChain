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

"""CLI for image-to-semantic-graph planning without simulation."""

from __future__ import annotations

import argparse
import base64
from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import math
from pathlib import Path
import shutil
import uuid
from time import perf_counter
from typing import Any

from .image_planning import (
    generate_semantic_task_graph_from_image,
    SemanticPlanningError,
)
from .orchestration.workflow_artifacts import WorkflowRunArtifacts, write_run_json
from .semantic_graph_html import render_semantic_task_graph_html

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Generate JSON declarations and an offline graph viewer from a photograph.

    Args:
        argv: CLI arguments, excluding the executable; defaults to sys.argv.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain task-engine",
        description="Plan image-grounded pick/place Semantic Calls without simulation.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Fresh output directory; existing directories are never overwritten.",
    )
    parser.add_argument("--task-id", default="image_task")
    args = parser.parse_args(argv)
    started = perf_counter()
    with WorkflowRunArtifacts(args.output) as artifacts:
        try:
            result = generate_semantic_task_graph_from_image(
                args.image, args.instruction, task_id=args.task_id
            )
        except SemanticPlanningError as error:
            failure = args.output.with_name(
                args.output.name + ".failure-" + uuid.uuid4().hex + ".json"
            )
            write_run_json(
                failure,
                {"error": str(error), "artifacts": error.artifacts},
                overwrite=False,
            )
            raise
        for name, value in result.items():
            write_run_json(artifacts.output_dir / f"{name}.json", value)
        source_copy = artifacts.output_dir / f"source_image{args.image.suffix.lower()}"
        shutil.copyfile(args.image, source_copy)
        if (
            hashlib.sha256(source_copy.read_bytes()).hexdigest()
            != result["image_grounding"]["image_sha256"]
        ):
            raise ValueError("Input image changed before preview publication.")
        summary = {
            "schema_version": "task_engine.image_semantic_run/v1",
            "task_id": args.task_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "elapsed_scope": "instruction interpretation and image semantic planning",
            "stage": "semantic_graph",
            "status": "graph_ready",
            "semantic_status": result["semantic_graph_validation"]["coverage"][
                "semantic_status"
            ],
            "visual_status": result["semantic_graph_validation"]["coverage"][
                "visual_status"
            ],
            "pending_bindings": len(result["binding_requirements"]),
            "execution_ready": False,
            "runtime_evidence_status": "not_run",
            "image_sha256": result["image_grounding"]["image_sha256"],
            "integration_status": "unbound",
        }
        (artifacts.output_dir / "semantic_task_graph.html").write_text(
            _image_preview(result, source_copy, summary), encoding="utf-8"
        )
        write_run_json(artifacts.output_dir / "run_manifest.json", summary)
        artifacts.commit()
    print(f"Semantic graph: {artifacts.output_dir / 'semantic_task_graph.json'}")
    print(f"Offline viewer: {artifacts.output_dir / 'semantic_task_graph.html'}")
    print("Stage: semantic_graph; runtime integration unbound; simulation not run.")


def _image_preview(
    result: dict[str, Any], image_path: Path, summary: dict[str, Any]
) -> str:
    from PIL import Image

    assets = []
    names = {}
    candidate_id = result["task_spec"]["metadata"]["candidate_id"]
    for candidate in result["task_candidates"]["candidates"]:
        if candidate["candidate_id"] == candidate_id:
            for ref in candidate["scene_request"]["references"]:
                uid = result["image_grounding"]["bindings"][ref["reference_id"]][0]
                names.setdefault(uid, ref["reference"])
    with Image.open(image_path) as source:
        for item in result["image_grounding"]["objects"]:
            if item["bbox"] is None:
                assets.append(
                    {
                        "uid": item["id"],
                        "name": names.get(item["id"], item["id"]),
                        "description": item["description"],
                        "role": item["provenance"],
                        "render": None,
                        "cutout": None,
                    }
                )
                continue
            x0, y0, x1, y1 = item["bbox"]
            crop = source.crop(
                (
                    math.floor(x0 * source.width),
                    math.floor(y0 * source.height),
                    math.ceil(x1 * source.width),
                    math.ceil(y1 * source.height),
                )
            ).convert("RGB")
            crop.thumbnail((256, 256))
            buffer = BytesIO()
            crop.save(buffer, format="PNG")
            assets.append(
                {
                    "uid": item["id"],
                    "name": names.get(item["id"], item["id"]),
                    "description": item["description"],
                    "category": "image reference",
                    "role": "visual",
                    "render": None,
                    "cutout": "data:image/png;base64,"
                    + base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "cutout_source": f"source image crop: {item['bbox']}",
                }
            )
    return render_semantic_task_graph_html(
        result["semantic_task_graph"],
        task_spec=result["task_spec"],
        validation=result["semantic_graph_validation"],
        presentation={
            "robot": "未绑定（图片语义预览）",
            "assets": assets,
            "created_at": summary["created_at"],
            "wall_seconds": summary["elapsed_seconds"],
            "wall_source": summary["elapsed_scope"],
        },
    )


if __name__ == "__main__":
    main()
