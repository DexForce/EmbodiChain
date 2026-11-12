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

"""Build synthetic display fixtures through the installed viewer contract.

These records exercise presentation only and do not qualify physical execution.
An optional frozen HTML reference supplies the previous frontend for comparison.
"""

from __future__ import annotations

import argparse
import base64
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from PIL import Image

from embodichain.gen_sim.task_engine.semantic_graph_html import (
    render_semantic_task_graph_html,
)
from embodichain.gen_sim.task_engine.contracts import canonical_hash


def _image(color: str) -> str:
    image = Image.new("RGB", (48, 48), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def build_fixtures(output: Path, reference: Path | None = None) -> None:
    """Write deterministic HTML and expected display facts for browser checks.

    Args:
        output: Directory for generated pages, media and the expectation manifest.
        reference: Optional frozen HTML containing the previous frontend.
    """
    output.mkdir(parents=True, exist_ok=True)
    fixture = json.loads(
        Path(__file__)
        .with_name("data")
        .joinpath("semantic_graph_contract.json")
        .read_text()
    )
    spec, original_graph = fixture["task_spec"], fixture["graph"]
    literal = '</script><img src="https://example.invalid/probe" onerror="window.__injection=true"> & 中文'
    assets = [
        dict(
            uid="a",
            name="Original can",
            description="Original description",
            render=_image("#497ea3"),
            cutout=_image("#c07843"),
            render_source="assets/a/render.png",
            cutout_source="assets/a/cutout.png",
            glb_source="assets/a/model.glb",
            glb_sha256="a" * 64,
        ),
        dict(
            uid="table",
            name="Cutout only",
            description="Only a source cutout is available",
            render=None,
            cutout=_image("#68a07d"),
            cutout_source="assets/table/cutout.png",
            glb_source="assets/table/model.glb",
            glb_sha256="b" * 64,
        ),
        dict(uid="missing", name="Missing preview", render=None, cutout=None),
        dict(
            uid="unsafe",
            name="Unsafe image",
            render="https://example.invalid/image.png",
            cutout="data:image/svg+xml,<svg onload='window.__injection=true'/>",
        ),
    ]
    presentation = dict(
        assets=assets,
        acceptance_mode="display_fixture",
        video_href="videos/env_7/front end.mp4",
        localizations=dict(
            title=dict(zh="前端展示验证", en="Frontend display validation"),
            instruction=dict(
                zh="仅用于展示验证，不代表物理运行。",
                en="Synthetic display fixture; not a physical run.",
            ),
            assets={
                "a": dict(
                    name=dict(zh="展示易拉罐", en="Display can"),
                    description=dict(zh="展示描述", en="Display description"),
                )
            },
        ),
    )
    video = output / "videos/env_7/front end.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "color=c=steelblue:s=96x64:r=10",
                "-t",
                "1",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
    reference_html = None if reference is None else reference.read_text()
    names = [
        "static-absent",
        "static-spec",
        "passed",
        "failed",
        "mixed",
        "unavailable",
        "aborted",
        "rejected",
        "literal",
    ]
    manifest: dict[str, Any] = {
        "scope": "Synthetic frontend display checks; no physical execution/certificate",
        "video": video.is_file(),
        "cases": {},
    }
    for name in names:
        graph = deepcopy(original_graph)
        if name == "literal":
            graph["instruction"] = literal
        report = None
        expected: dict[str, Any] = {}
        static = name.startswith("static") or name == "literal"
        if not static:
            rows = []
            for env_id in [7, 2, 19]:
                semantic = (
                    {}
                    if name == "mixed" and env_id == 19
                    else {
                        "step_01": name != "failed" and (name != "mixed" or env_id != 2)
                    }
                )
                rows.append(
                    dict(
                        env_id=env_id,
                        success=name != "failed" and (env_id == 7 or name != "mixed"),
                        terminal_reason="synthetic_display_fixture",
                        semantic_success=semantic,
                    )
                )
            segments = [
                dict(
                    name=node["id"],
                    start_step=index * 10,
                    end_step=(index + 1) * 10,
                    active=[True, True, True],
                    successes=[name != "failed"] * 3,
                    metadata=dict(
                        runtime=dict(
                            env_ids=[7, 2, 19],
                            events=[
                                dict(kind="skill_planned", timestamp=1.0 + index),
                                dict(kind="skill_completed", timestamp=1.5 + index),
                            ],
                        )
                    ),
                )
                for index, node in enumerate(original_graph["nodes"])
            ]
            if name == "mixed":
                segments[0]["successes"][1] = False
                segments[1]["active"][1] = False
                segments[2]["metadata"]["runtime"]["env_ids"] = [7, 19]
                segments[2]["active"] = segments[2]["successes"] = [True, True]
                segments[3]["metadata"]["runtime"]["env_ids"] = [7, 7, 19]
                segments[4]["active"][1] = 1
            evaluation_rows = []
            for env_id in [7, 2, 19]:
                status = (
                    "unavailable"
                    if name == "unavailable" or name in {"aborted", "rejected"}
                    else (
                        {7: "satisfied", 2: "failed", 19: "unavailable"}[env_id]
                        if name == "mixed"
                        else "failed" if name == "failed" else "satisfied"
                    )
                )
                clauses = dict(
                    init=[],
                    goal=[
                        dict(
                            predicate_id="goal_001",
                            status=status,
                            reason="synthetic_display_fixture",
                        )
                    ],
                    invariants=[],
                    temporal=[],
                    milestones=[dict(predicate_id="milestone:step_01", status=status)],
                )
                evaluation_rows.append(
                    dict(
                        env_id=env_id,
                        status=status,
                        certificate_ready=status == "satisfied",
                        **clauses,
                    )
                )
            evaluation = dict(
                schema_version="task_spec_evaluation/v1",
                task_spec_digest=canonical_hash(spec),
                graph_digest=canonical_hash(graph),
                environments=evaluation_rows,
            )
            report = dict(
                schema_version="task_program_execution_report/v1",
                status=(
                    name if name in {"aborted", "rejected", "failed"} else "succeeded"
                ),
                task_id=graph["task_id"],
                semantic_call_count=len(graph["nodes"]),
                integration_fingerprint=graph["integration_fingerprint"],
                record_dir="synthetic_display_fixture",
                environments=rows,
                runtime_result=dict(
                    length=60,
                    lengths=[60, 60, 60],
                    success=(
                        [True, False, False]
                        if name == "mixed"
                        else [name != "failed"] * 3
                    ),
                    segments=segments,
                    task_spec_evaluation=evaluation,
                ),
                failure=None,
            )
        for env in ["all"] if static else ["all", "7", "2", "19"]:
            calls = ["unknown"] * 6 if static else ["succeeded"] * 6
            group = "unknown" if static else "succeeded"
            goal = (
                "unknown"
                if static
                else (
                    "unavailable"
                    if name in {"unavailable", "aborted", "rejected"}
                    else "satisfied"
                )
            )
            strict = "satisfied" if goal == "satisfied" else "unavailable"
            if name == "failed":
                calls = ["failed"] * 6
                group = goal = strict = "failed"
            if name == "mixed":
                calls = {
                    "all": [
                        "failed",
                        "unknown",
                        "unknown",
                        "unknown",
                        "unknown",
                        "succeeded",
                    ],
                    "7": [
                        "succeeded",
                        "succeeded",
                        "succeeded",
                        "unknown",
                        "unknown",
                        "succeeded",
                    ],
                    "2": [
                        "failed",
                        "unknown",
                        "unknown",
                        "unknown",
                        "unknown",
                        "succeeded",
                    ],
                    "19": [
                        "succeeded",
                        "succeeded",
                        "succeeded",
                        "unknown",
                        "unknown",
                        "succeeded",
                    ],
                }[env]
                group = {
                    "all": "failed",
                    "7": "succeeded",
                    "2": "failed",
                    "19": "unknown",
                }[env]
                goal = {
                    "all": "failed",
                    "7": "satisfied",
                    "2": "failed",
                    "19": "unavailable",
                }[env]
                strict = (
                    "satisfied"
                    if goal == "satisfied"
                    else "failed" if goal == "failed" else "unavailable"
                )
            expected[env] = dict(
                calls=dict(
                    zip((node["id"] for node in original_graph["nodes"]), calls)
                ),
                groups={"step_01": group},
                goals={} if name == "static-absent" else {"goal_001": goal},
                strict=strict,
            )
        display = (
            deepcopy(presentation) if not static else dict(assets=[], video_href=None)
        )
        if name == "unavailable":
            display["assets"][0]["cutout"] = None
        if name == "aborted":
            display["assets"][0]["render"] = None
        if name == "literal":
            display["assets"] = [
                dict(
                    uid="a",
                    name=literal,
                    render="javascript:window.__injection=true",
                    cutout=None,
                )
            ]
            display["video_href"] = "https://example.invalid/video.mp4"
        if name == "rejected":
            display["video_href"] = "videos/env_7/unpublished.partial.mp4"
        html = render_semantic_task_graph_html(
            graph,
            report,
            task_spec=None if name == "static-absent" else spec,
            validation=(
                None
                if name == "static-absent"
                else dict(
                    schema_version="semantic_graph_coverage/v1",
                    coverage=fixture["coverage"],
                    source=dict(
                        candidate=fixture["candidate"],
                        role_bindings=fixture["role_bindings"],
                    ),
                )
            ),
            presentation=display,
        )
        match = re.search(
            r'(<script id="graph-data" type="application/json">)(.*?)(</script>)',
            html,
            re.S,
        )
        assert match is not None
        data = json.loads(match[2])
        if name == "failed":
            from embodichain.gen_sim.task_spec import TaskSpec
            from embodichain.gen_sim.task_spec.publication import freeze_task_spec
            from embodichain.gen_sim.task_engine.semantic_graph_html import (
                _viewer_document,
            )

            review_spec = freeze_task_spec(
                TaskSpec.from_dict(spec),
                image_sha256="a" * 64,
                image_ref="source.png",
                role_descriptions={"role_001": "can"},
                interpretation=[],
                generalization=[],
            )
            call_id = graph["nodes"][1]["id"]
            review = dict(
                spec=review_spec,
                pairing={
                    "role_entities": {"role_001": ["a"]},
                    "goal_groups": spec["metadata"]["goal_sources"],
                    "milestone_groups": spec["metadata"]["milestone_sources"],
                },
                pending=0,
                attempt={
                    "status": "failed",
                    "physical": "unavailable",
                    "verdict": "not_evaluated",
                    "calls": {
                        call_id: {
                            "success": False,
                            "start_step": 2,
                            "end_step": 7,
                            "issues": [
                                {
                                    "kind": "session_failed",
                                    "message": "fixture measured failure",
                                    "timestamp": 1.0,
                                }
                            ],
                            "env_ids": [0],
                            "successes": [False],
                            "active": [True],
                        }
                    },
                },
            )
            (output / "unified.html").write_text(
                _viewer_document({**data, "review": review})
            )
        (output / f"{name}.html").write_text(html)
        (output / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )
        if reference_html:
            replaced, count = re.subn(
                r'(<script id="graph-data" type="application/json">)(.*?)(</script>)',
                lambda m: m[1] + match[2] + m[3],
                reference_html,
                flags=re.S,
            )
            assert count == 1
            (output / f"reference-{name}.html").write_text(replaced)
        manifest["cases"][name] = dict(
            expected=expected,
            reference=reference_html is not None,
            instruction=graph["instruction"],
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    build_fixtures(args.output, args.reference)
