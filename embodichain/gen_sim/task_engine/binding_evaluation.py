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

"""Frozen scene-reference controls without simulation or instruction generation.

Run: python -m embodichain.gen_sim.task_engine.binding_evaluation --output-dir PATH
Add --run-model to enable provider calls; the default only records frozen inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import psutil

from .orchestration.grounding import (
    _UnresolvedSceneReference,
    ground_scene_references,
)
from .orchestration.scene_inventory import SceneInventory

__all__: list[str] = []


def _cases() -> list[dict[str, Any]]:
    """Return independent, labeled controls fixed before any model response."""
    tray = {
        "uid": "tray_001",
        "role": "rigid_object",
        "category": "tray",
        "name": "green serving tray",
        "description": "A green rectangular plastic serving tray with two handles.",
        "attributes": {"color": "green", "shape": "rectangular", "material": "plastic"},
    }
    plate = {
        "uid": "plate_001",
        "role": "rigid_object",
        "category": "plate",
        "name": "white dinner plate",
        "description": "A white round ceramic dinner plate.",
        "attributes": {"color": "white", "shape": "round", "material": "ceramic"},
    }
    cases = []
    for case_id, reference, scene, status, uids, group, steps in (
        ("broad_plate", "盘子", [tray], "resolved", ["tray_001"], "tray_variants", 1),
        ("tray", "托盘", [tray], "resolved", ["tray_001"], "tray_variants", 1),
        (
            "green_plate",
            "绿色盘子",
            [tray],
            "resolved",
            ["tray_001"],
            "tray_variants",
            1,
        ),
        (
            "english_tray",
            "serving tray",
            [tray],
            "resolved",
            ["tray_001"],
            "tray_variants",
            1,
        ),
        ("distinguish_plate", "餐盘", [tray, plate], "resolved", ["plate_001"], "", 1),
        ("distinguish_tray", "托盘", [tray, plate], "resolved", ["tray_001"], "", 1),
        ("attribute_conflict", "白色圆形陶瓷餐盘", [tray], "not_found", [], "", 1),
        (
            "two_trays",
            "托盘",
            [tray, {**tray, "uid": "tray_002"}],
            "ambiguous",
            [],
            "",
            1,
        ),
        ("missing_coaster", "杯垫", [tray], "not_found", [], "", 1),
        ("exact_uid", "tray_001", [tray, plate], "resolved", ["tray_001"], "", 1),
        ("decomposed", "盘子", [tray], "resolved", ["tray_001"], "tray_variants", 2),
    ):
        cases.append(
            {
                "id": case_id,
                "group": group,
                "instruction": f"用双臂把{reference}端起来再放下。",
                "scene": deepcopy(scene),
                "intent": {
                    "steps": [
                        {
                            "id": f"step_{index + 1:02d}",
                            "task_type": "E5",
                            "relation": "none",
                            "object": {
                                "kind": "scene_ref",
                                "reference": reference,
                                "quantifier": "one",
                                "count": 0,
                            },
                            "target": {"kind": "none"},
                        }
                        for index in range(steps)
                    ]
                },
                "expected": {
                    f"step_{index + 1:02d}.object": {
                        "status": status,
                        "uids": list(uids),
                    }
                    for index in range(steps)
                },
            }
        )
    return cases


class _DryRun(Exception):
    """Stop after the production grounding path has generated its request."""


def _evaluate_case(
    case: Mapping[str, Any],
    *,
    repeat: int,
    model: str | None,
    caller: Callable[..., Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Capture every native grounding call, including its single format repair."""
    calls: list[dict[str, Any]] = []

    def recorded(**kwargs: Any) -> Mapping[str, Any]:
        call = deepcopy(kwargs)
        calls.append(call)
        if caller is None:
            raise _DryRun()
        started = perf_counter()
        try:
            call["response"] = deepcopy(caller(**kwargs))
            return call["response"]
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            call["latency_seconds"] = perf_counter() - started

    record: dict[str, Any] = {
        "case_id": case["id"],
        "repeat": repeat,
        "completed": False,
        "observed": {},
        "calls": calls,
    }
    inventory = SceneInventory(case["scene"], robot_profile="franka")
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = perf_counter()
    try:
        ground_scene_references(
            instruction=case["instruction"],
            intent=case["intent"],
            inventory=inventory,
            scene_objects=case["scene"],
            model=model,
            caller=recorded,
        )
        record["completed"] = True
    except _UnresolvedSceneReference as error:
        # A claimed resolution rejected for contradictory evidence is not a
        # completed semantic refusal and must not score as a correct binding.
        record.update(completed=error.semantic_rejection, rejection=str(error))
        if not error.semantic_rejection:
            record["error"] = str(error)
    except _DryRun:
        record["mode"] = "dry_run"
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["latency_seconds"] = perf_counter() - started
    record["cpu_delta_mb"] = (process.memory_info().rss - rss_before) / 1024**2
    if record["completed"]:
        rows = calls[-1]["response"]["bindings"]
        observed = {
            row["reference_id"]: {"status": row["status"], "uids": row["uids"]}
            for row in rows
        }
        if len(rows) != len(observed) or set(observed) != set(case["expected"]):
            record.update(
                completed=False,
                error="Response does not cover every reference exactly once.",
            )
        else:
            record["observed"] = observed
    return record


def _ratio(correct: int, total: int) -> dict[str, Any]:
    """Retain numerator and denominator, including undefined empty cohorts."""
    return {
        "correct": correct,
        "total": total,
        "rate": correct / total if total else None,
    }


def _metrics(
    cases: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
) -> dict[str, Any]:
    """Use all requested references as primary denominators, including failures."""
    by_key = {(row["case_id"], row["repeat"]): row for row in records}
    correct = wrong = rejected = completed = positive = negative = total = 0
    for case in cases:
        for repeat in range(repeats):
            record = by_key.get((case["id"], repeat), {})
            for reference_id, expected in case["expected"].items():
                total += 1
                is_positive = expected["status"] == "resolved"
                positive += is_positive
                negative += not is_positive
                if not record.get("completed"):
                    continue
                actual = record["observed"][reference_id]
                completed += 1
                equal = actual["status"] == expected["status"] and (
                    not is_positive or set(actual["uids"]) == set(expected["uids"])
                )
                correct += is_positive and equal
                rejected += not is_positive and equal
                wrong += actual["status"] == "resolved" and not equal
    groups = {case["group"] for case in cases if case["group"]}
    consistent = eligible = correct_consistent = 0
    for group in groups:
        members = [case for case in cases if case["group"] == group]
        for repeat in range(repeats):
            rows = [by_key.get((case["id"], repeat), {}) for case in members]
            if not all(row.get("completed") for row in rows):
                continue
            eligible += 1
            signatures = [
                {
                    (value["status"], tuple(sorted(value["uids"])))
                    for value in row["observed"].values()
                }
                for row in rows
            ]
            same = all(signature == signatures[0] for signature in signatures)
            consistent += same
            correct_consistent += same and all(
                row["observed"] == case["expected"] for row, case in zip(rows, members)
            )
    return {
        "requested_references": total,
        "completed_references": completed,
        "correct_binding": _ratio(correct, positive),
        "wrong_substitution": _ratio(wrong, total),
        "reject_accuracy": _ratio(rejected, negative),
        "overall_accuracy": _ratio(correct + rejected, total),
        "paraphrase_consistency": _ratio(consistent, eligible),
        "correct_paraphrase_consistency": _ratio(correct_consistent, eligible),
        "paraphrase_groups_requested": len(groups) * repeats,
    }


def _write_json(path: Path, value: Any) -> None:
    """Persist auditable UTF-8 JSON without replacing existing run artifacts."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _run(
    output_dir: Path,
    *,
    repeats: int,
    model: str | None,
    caller: Callable[..., Mapping[str, Any]] | None = None,
    model_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze controls before execution and write one three-table summary."""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    cases = _cases()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "dry_run" if caller is None else "model",
        "repeats": repeats,
        "max_provider_calls": 2 * len(cases) * repeats,
        "cases": cases,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "model_config": dict(model_config or {"requested_model": model}),
        "cases_sha256": hashlib.sha256(
            json.dumps(cases, sort_keys=True).encode()
        ).hexdigest(),
        "scope": "Synthetic text grounding only; no interpreter, visual fallback, simulation or physical success.",
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "calls").mkdir()
    records = []
    for repeat in range(repeats):
        for case in cases:
            record = _evaluate_case(case, repeat=repeat, model=model, caller=caller)
            _write_json(
                output_dir / "calls" / f"{case['id']}_{repeat:02d}.json", record
            )
            records.append(record)
    metrics = _metrics(cases, records, repeats=repeats)
    _write_json(output_dir / "metrics.json", metrics)
    lines = [
        "# 绑定对照评估",
        "",
        f"模式：{manifest['mode']}。仅评估文本绑定，不代表物理成功。",
        "主指标分母包含请求失败；改写一致率仅统计全部完成的对照组。",
        "",
        "## Time & Memory",
        "",
        "| cost_time_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |",
        "| --- | --- | --- | --- |",
        f"| {sum(row['latency_seconds'] for row in records) * 1000:.2f} | {sum(row['cpu_delta_mb'] for row in records):.2f} | N/A | N/A |",
        "",
        "## Success & Other Metrics",
        "",
        "| metric | count | denominator | rate |",
        "| --- | --- | --- | --- |",
    ]
    for name, value in metrics.items():
        if isinstance(value, dict):
            lines.append(
                f"| {name} | {value['correct']} | {value['total']} | {value['rate']} |"
            )
    lines.extend(
        [
            "",
            "## Leaderboard",
            "",
            "| rank | algorithm | success_rate | completed_references | requested_references |",
            "| --- | --- | --- | --- | --- |",
            f"| 1 | native_text_grounding | {metrics['overall_accuracy']['rate']} | {metrics['completed_references']} | {metrics['requested_references']} |",
            "",
            f"已请求改写对照组：{metrics['paraphrase_groups_requested']}。GPU 内存不适用：此评估不启动本地推理或仿真。",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics


def _main(argv: Sequence[str] | None = None) -> int:
    """Require explicit opt-in before any provider configuration or network use."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args(argv)
    caller = None
    model = args.model
    config: dict[str, Any] = {"requested_model": model}
    if args.run_model:
        from .interpretation import _default_instruction_caller, _load_llm_settings

        settings = _load_llm_settings(model=model)
        endpoint = urlsplit(settings["base_url"] or "https://api.openai.com/v1")
        model = settings["model"]
        config = {
            "model": model,
            "endpoint_host": endpoint.hostname,
            "temperature": 0,
            "default_query_keys": sorted(settings["default_query"]),
        }
        caller = _default_instruction_caller
    metrics = _run(
        args.output_dir,
        repeats=args.repeats,
        model=model,
        caller=caller,
        model_config=config,
    )
    print(f"Report: {args.output_dir / 'report.md'}")
    return 0 if caller is None or metrics["overall_accuracy"]["rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(_main())
