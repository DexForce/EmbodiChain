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
"""Publish a portable unified viewer for the recorded task1186 pilot."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .stack_pilot import digest
from embodichain.gen_sim.workflow import load_task_plan
from embodichain.gen_sim.task_engine.semantic_graph_html import (
    _graph_data,
    _viewer_document,
)


def render(root: Path, output: Path, replay: bool) -> None:
    def read(path):
        return json.loads((root / path).read_text())

    r = read("review.json")
    plan = load_task_plan(root / "plan")
    graph = read("plan/semantic_task_graph.json")
    for key, actual in {
        "spec_digest": plan["spec_digest"],
        "graph_digest": plan["semantic_graph_sha256"],
        "deployment_digest": digest(root / "prepared/deployment_manifest.json"),
        "program_digest": digest(root / "prepared/deployment/program.json"),
        "scene_digest": digest(root / "scene/generated_scene.json"),
    }.items():
        if r[key] != actual:
            raise ValueError("Mismatched " + key)
    data = _graph_data(graph)
    segments = read("execution.json")["segments"]
    if [n["id"] for n in graph["nodes"]] != [s["name"] for s in segments] or [
        s["node"] for s in r["segments"]
    ] != [s["name"] for s in segments]:
        raise ValueError("Execution nodes mismatch")
    calls = {}
    for n, s in zip(r["segments"], segments, strict=True):
        if (n["start"], n["end"], n["success"]) != (
            s["start_step"],
            s["end_step"],
            s["success"],
        ):
            raise ValueError("Segment boundary mismatch")
        calls[s["name"]] = {
            **{
                k: s[k]
                for k in (
                    "start_step",
                    "end_step",
                    "success",
                    "failure_reason",
                    "active",
                    "successes",
                    "start_steps",
                    "end_steps",
                )
            },
            "env_ids": [0],
            "holders": n["holders"],
            "issues": [
                e
                for e in s["metadata"]["runtime"]["events"]
                if e["kind"]
                in {
                    "skill_completed",
                    "phase_effect_gate_satisfied",
                    "skill_planning_failed",
                    "phase_effect_gate_failed",
                }
            ],
        }
    terminal = graph["nodes"][-1]["id"]
    calls[terminal].update(terminal=r["terminal"], tray_lift_m=r["tray_lift_m"])
    spec = plan["task_document"]
    data["review"] = dict(
        spec=spec,
        pending=0,
        displayNames={
            k: dict(zh=v, en=k.replace("_", " "))
            for k, v in dict(
                apple="苹果", orange="橙子", tray="果盘", service_pad="服务垫"
            ).items()
        },
        replayVideo=replay,
        pairing=dict(
            role_entities=read("plan/plan_coverage.json")["role_entities"],
            milestone_groups={
                m["id"]: "plan" for m in spec["objectives"]["milestones"]
            },
            goal_groups={g["id"]: ["plan"] for g in spec["objectives"]["goal"]},
        ),
        attempt=dict(
            status=r["execution"],
            verdict="geometry",
            physical=(
                "satisfied"
                if all(v["satisfied"] for v in r["terminal"].values())
                else "failed"
            ),
            step=segments[-1]["end_step"],
            missing=[],
            identity={k: v for k, v in r.items() if k.endswith("_digest")},
            calls=calls,
        ),
    )
    output.write_text(_viewer_document(data), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--replay",
        action="store_true",
        help="Include sibling execution.mp4 (separate deployment replay)",
    )
    args = p.parse_args()
    render(args.run, args.output, args.replay)
