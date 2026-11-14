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
"""Run paired live task1186 authoring trials with one shared per-route budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib

from embodichain.gen_sim.services import GenSimServices, GenSimServicesCfg
from embodichain.gen_sim.task_engine.goal_planning import publish
from embodichain.gen_sim.task_engine.hybrid import publish_hybrid, verify_witness
from embodichain.gen_sim.task_spec import TaskSpec
from embodichain.gen_sim.task_engine.image_planning import (
    generate_semantic_task_graph_from_image,
)


def inspect_1186(graph):
    """Check the declared fruit roles, transfer sequence and named destination."""
    calls = [n["call"] for n in graph["nodes"]]
    kinds = [c.get("call_id", c["kind"]) for c in calls]
    sequence = ["pick", "hand_over", "place"] * 2 + ["simulation.coordinated_transport"]
    fruit_ok = len(calls) == 7 and all(
        fruit in str(c.get("object", "")).lower()
        for fruit, block in (("apple", calls[:3]), ("orange", calls[3:6]))
        for c in block
    )
    assignments = [
        {"primary": "right"},
        {"source": "right", "destination": "left"},
        {"primary": "left"},
    ] * 2 + [{"left": "left", "right": "right"}]
    arm_ok = [c["resources"] for c in calls] == assignments
    order_ok = (
        kinds == sequence
        and fruit_ok
        and all(
            n["depends_on"] == ([graph["nodes"][i - 1]["id"]] if i else [])
            for i, n in enumerate(graph["nodes"])
        )
    )
    landing = calls[-1].get("arguments", {})
    plate = calls[2].get("on", calls[2].get("inside")) if len(calls) == 7 else None
    support_ok = bool(
        plate
        and plate == calls[5].get("on", calls[5].get("inside"))
        and landing.get("object") == plate
        and ("plate" in plate or "tray" in plate)
        and landing.get("target", "").lower().replace(" ", "_")
        in {
            "mat",
            "service_pad",
            "service_mat",
            "service_mat_01",
            "serv_mat",
            "serving_mat",
        }
        and kinds[-1] == sequence[-1]
    )
    return dict(
        arms=arm_ok,
        ordered_transfers=order_ok,
        shared_plate_and_final_support=support_ok,
    )


def rescore(root: Path) -> None:
    """Regrade saved graphs without resampling, preserving original trial records."""
    rows = json.loads((root / "comparison.json").read_text())
    reference = root / "hybrid" / "task_spec.json"
    frozen = (
        TaskSpec.from_dict(json.loads(reference.read_text()))
        if reference.exists()
        else None
    )
    for row in rows:
        route = row["route"]
        if route == "template":
            path = root / "template.json"
            if not path.exists():
                continue
            graph = json.loads(path.read_text())["semantic_task_graph"]
        else:
            path = root / route / "semantic_task_graph.json"
            if not path.exists():
                continue
            graph = json.loads(path.read_text())
        row["common_checks"] = inspect_1186(graph)
        if frozen:
            # This pilot fixture has exactly one plate and one service mat.
            # Normalize only reviewed aliases; never infer arbitrary targets.
            aliases = {
                a: role.name
                for role in frozen.roles
                for names in (
                    ("plate", "tray", "plate_01"),
                    (
                        "mat",
                        "service_pad",
                        "service mat",
                        "service_mat",
                        "serv_mat",
                        "service_mat_01",
                    ),
                    ("apple", "apple_01"),
                    ("orange", "orange_01"),
                )
                if role.name in names
                for a in names
            }
            witness = deepcopy(graph)
            for n in witness["nodes"]:
                call = n["call"]
                params = call.get("arguments", call)
                for key in ("object", "on", "inside", "target", "reference"):
                    if isinstance(params.get(key), str):
                        params[key] = aliases.get(params[key], params[key])
            try:
                row["common_frozen_coverage"] = verify_witness(
                    frozen,
                    witness,
                    support_targets=json.loads(
                        (root / "hybrid/pending_support_targets.json").read_text()
                    ),
                )
            except ValueError as exc:
                row["common_frozen_coverage"] = {
                    "status": "rejected",
                    "reason": str(exc),
                }
    (root / "graded.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))


def trial(route, case, image, instruction, output):
    trace = []
    lock = Lock()
    summary = {}
    with GenSimServices() as base:
        roles = (
            "task_interpretation",
            "task_semantic_planning",
            "task_image_grounding",
        )
        models = {k: base.model(k) for k in roles}
        if len(set(models.values())) != 1:
            raise ValueError("Pilot requires identical models across roles")

        class Meter:
            def __init__(self, role):
                self.role = role

            def complete(self, request):
                with lock:
                    if len(trace) >= 5:
                        raise ValueError("Shared budget exhausted")
                    trace.append(self.role)
                return base.get(self.role).complete(request)

        with GenSimServices(
            GenSimServicesCfg(), injected={k: Meter(k) for k in roles}
        ) as services:
            try:
                if route == "template":
                    result = generate_semantic_task_graph_from_image(
                        image, instruction, task_id=case, services=services
                    )
                    (output / "template.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2)
                    )
                elif route == "goal":
                    result = publish(
                        instruction,
                        case,
                        output / route,
                        None,
                        services,
                        image=image,
                    )
                else:
                    result = publish_hybrid(
                        instruction,
                        case,
                        image,
                        output / route,
                        services,
                    )
                summary.update(
                    status=result.get(
                        "coverage",
                        result.get("semantic_graph_validation", {}).get("coverage"),
                    )["status"]
                )
                graph = result["semantic_task_graph"]
                if case == "task1186":
                    summary["common_checks"] = inspect_1186(graph)
                if route == "hybrid" and case == "task1186":
                    spec = TaskSpec.from_dict(result["task_spec"])
                    summary["negative_controls"] = {}
                    for name in (
                        "wrong_arm",
                        "wrong_plate",
                        "wrong_support",
                        "wrong_order",
                    ):
                        bad = deepcopy(graph)
                        nodes = bad["nodes"]
                        positions = next(
                            i
                            for i, n in enumerate(nodes)
                            if n["call"]["kind"] == "place"
                        )
                        place = nodes[positions]["call"]
                        destination_key = "inside" if "inside" in place else "on"
                        if name == "wrong_arm":
                            swap = nodes[positions - 1]["call"]["resources"]
                            swap["source"], swap["destination"] = (
                                swap["destination"],
                                swap["source"],
                            )
                        elif name == "wrong_plate":
                            place[destination_key] = nodes[-1]["call"]["arguments"][
                                "target"
                            ]
                        elif name == "wrong_support":
                            nodes[-1]["call"]["arguments"]["target"] = place[
                                destination_key
                            ]
                        else:
                            original = [n["call"] for n in nodes]
                            for n, c in zip(
                                nodes, original[3:6] + original[:3] + original[6:]
                            ):
                                n["call"] = c
                        try:
                            verdict = verify_witness(
                                spec,
                                bad,
                                support_targets=result["pending_support_targets"],
                            )
                        except ValueError as e:
                            verdict = {"status": "rejected", "reason": str(e)}
                        summary["negative_controls"][name] = verdict
            except Exception as exc:
                summary.update(status="rejected", reason=str(exc)[:1200])
    summary.update(
        route=route,
        case=case,
        model=models,
        limit=5,
        calls=trace,
        instruction=instruction,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--routes",
        nargs="+",
        choices=("template", "goal", "hybrid"),
        default=("template", "goal", "hybrid"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    instructions = json.loads((args.dataset / "task_descriptions.json").read_text())

    def execute(pair):
        case, route = pair
        folder = args.output / case if args.all else args.output
        folder.mkdir(exist_ok=True)
        row = trial(
            route,
            case,
            args.dataset / (case + ".png"),
            instructions[case]["zh"],
            folder,
        )
        (folder / (route + "-trial.json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2)
        )
        print(case, route, row["status"], len(row["calls"]), flush=True)
        return row

    with ThreadPoolExecutor(5) as pool:
        rows = list(
            pool.map(
                execute,
                (
                    (c, r)
                    for c in (sorted(instructions) if args.all else ["task1186"])
                    for r in args.routes
                ),
            )
        )
    (args.output / "comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2)
    )
    if not args.all:
        rescore(args.output)


if __name__ == "__main__":
    main()
