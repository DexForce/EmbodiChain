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
"""Compare independent text authoring with pinned image-template response replay."""

from __future__ import annotations
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from embodichain.gen_sim.task_engine.goal_planning import publish
from embodichain.gen_sim.task_engine.image_planning import (
    generate_semantic_task_graph_from_image,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live-template",
        action="store_true",
        help="Also generate a fresh template plan with the same service configuration",
    )
    args = parser.parse_args()
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "tests/gen_sim/task_engine")
    )
    from image_case_acceptance import check_image_case

    instructions = json.loads((args.dataset / "task_descriptions.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)

    def compare(item):
        case, langs = item
        row = {"case": case, "instruction": langs["zh"]}
        folder = args.output / case
        folder.mkdir()
        if args.live_template:
            try:
                a = generate_semantic_task_graph_from_image(
                    args.dataset / (case + ".png"), langs["zh"], task_id=case
                )
                row["live_template_oracle"] = check_image_case(case, a)
                (folder / "live-template.json").write_text(
                    json.dumps(a, ensure_ascii=False, indent=2)
                )
            except Exception as exc:
                row["live_template_error"] = str(exc)[:700]
        try:
            archive = next(
                p / case
                for p in args.archives
                if (p / case / "semantic_task_graph.json").is_file()
            )
            row["template_archive"] = str(archive)
            attempts = json.loads((archive / "model_attempts.json").read_text())

            def replay(stage):
                queue = deque(a["response"] for a in attempts if a["stage"] == stage)
                return lambda **kwargs: queue.popleft()

            old = generate_semantic_task_graph_from_image(
                args.dataset / (case + ".png"),
                langs["zh"],
                task_id=case,
                instruction_caller=replay("interpretation"),
                image_caller=replay("grounding"),
                planning_caller=replay("planning"),
            )
            row["template_oracle"] = check_image_case(case, old)
            row["template_calls"] = [
                n["call"] for n in old["semantic_task_graph"]["nodes"]
            ]
            if case == "task1186":
                assert [c["resources"] for c in row["template_calls"]] == [
                    {"primary": "right"},
                    {"source": "right", "destination": "left"},
                    {"primary": "left"},
                ] * 2 + [{"left": "left", "right": "right"}]
            (folder / "template.json").write_text(
                json.dumps(old, ensure_ascii=False, indent=2)
            )
        except Exception as exc:
            row["template_error"] = str(exc)[:700]
        try:
            new = publish(
                langs["zh"],
                case,
                folder / "goal",
                None,
                None,
                image=args.dataset / (case + ".png"),
            )
            row["goal_coverage"] = new["coverage"]
            row["goal_preflight"] = new["deployment_preflight"]["status"]
        except Exception as exc:
            row["goal_error"] = str(exc)[:700]
        (folder / "comparison.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2)
        )
        print(
            case,
            row.get("goal_coverage", {}).get("status", row.get("goal_error", "")[:80]),
            flush=True,
        )
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(compare, sorted(instructions.items())))
    (args.output / "comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
