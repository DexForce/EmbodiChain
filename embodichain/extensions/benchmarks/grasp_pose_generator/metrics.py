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

"""Fixed-budget physical grasp statistics by gripper and object category."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

__all__ = ["summarize"]

_METRICS = ("Success@1", "Hit@5", "Hit@10", "Precision@10")


def _case_metrics(case):
    rows = case["trials"]
    if (
        case.get("error")
        or case.get("finished") is False
        or any(r["status"] != "ok" for r in rows)
    ):
        return None
    outcomes = {r["rank"]: r["success"] for r in rows}
    expected = set(range(1, case["count"] + 1))
    if outcomes.keys() != expected or len(outcomes) != len(rows):
        return None
    ranked = [outcomes.get(i, False) for i in range(1, 11)]
    return np.array(
        [ranked[0], any(ranked[:5]), any(ranked), sum(ranked) / 10], dtype=float
    )


def _diagnostics(rows):
    info = {}
    info["hold_retention"] = {
        str(t): (
            np.mean([r["retention"][str(t)] for r in rows]).item() if rows else None
        )
        for t in (0.5, 1.0, 3.0)
    }
    for phase in ("hold", "disturbance"):
        windows = [
            r[phase] for r in rows if r.get(phase) is not None and r[phase]["entered"]
        ]
        passed = sum(w["success"] for w in windows)
        info[phase] = {
            "entered": len(windows),
            "passed": passed,
            "conditional_retention": passed / len(windows) if windows else None,
            "drops": sum(w["drop_s"] is not None for w in windows),
        }
        for metric in ("slip_mm", "rotation_deg"):
            info[phase][metric + "_p50_p95"] = (
                np.percentile([w[metric] for w in windows], [50, 95]).tolist()
                if windows
                else None
            )
    return info


def _ci95(draws):
    bounds = np.percentile(draws, [2.5, 97.5], axis=0).T.tolist()
    return dict(zip(_METRICS, bounds))


def _group_summary(scenes, methods, repeats):
    pairs = sorted({pair for pair, _ in scenes})
    objects = sorted({obj for obj, _ in pairs})
    group = {
        "objects": len(objects),
        "object_gripper_pairs": len(pairs),
        "algorithms": {},
        "paired_difference_ci95": {},
    }
    complete_scores = {}
    for method in methods:
        scores = defaultdict(list)
        trials = []
        missing = 0
        for pair, observations in scenes:
            result = observations.get(method)
            if result is None:
                missing += 1
            else:
                case_scores, case_trials = result
                scores[pair].append(case_scores)
                trials.extend(case_trials)

        stats = {
            "completed_cases": len(scenes) - missing,
            "missing_cases": missing,
            "trials": len(trials),
            "metrics": None,
            "ci95": None,
            "diagnostics": _diagnostics(trials),
        }
        if not missing:
            values = np.array([np.mean(scores[pair], axis=0) for pair in pairs])
            complete_scores[method] = values
            stats["metrics"] = dict(zip(_METRICS, values.mean(axis=0).tolist()))
        group["algorithms"][method] = stats

    if len(objects) >= 2 and complete_scores:
        # Keep every hand and method paired when resampling an object instance.
        object_indices = {obj: i for i, obj in enumerate(objects)}
        indices = [object_indices[obj] for obj, _ in pairs]
        probabilities = np.full(len(objects), 1 / len(objects))
        values = np.stack(list(complete_scores.values()))
        rng = np.random.default_rng(0)
        batches = []
        for start in range(0, repeats, 32):
            counts = rng.multinomial(
                len(objects), probabilities, size=min(32, repeats - start)
            )
            weights = counts[:, indices]
            batches.append((weights / weights.sum(1, keepdims=True)) @ values)
        draws = dict(zip(complete_scores, np.concatenate(batches, axis=1)))
        for method, sample in draws.items():
            group["algorithms"][method]["ci95"] = _ci95(sample)
        for first, second in combinations(draws, 2):
            group["paired_difference_ci95"][f"{first} minus {second}"] = _ci95(
                draws[first] - draws[second]
            )
    return group


def summarize(cases: list[dict], *, repeats: int = 2000) -> dict:
    """Score algorithms overall, by gripper, by object category, and by both.

    Each object-instance/gripper pair has equal weight within its group after
    averaging its placements and seeds. Missing observations suppress the
    affected method's scores in that group. Empty candidate sets score zero.
    """
    if not cases or repeats < 1:
        raise ValueError("Cases and positive bootstrap repetitions are required")
    methods = sorted({case["algorithm"] for case in cases})
    population = defaultdict(dict)
    for case in cases:
        obj = (case["category"], case["object"])
        pair = (obj, case["gripper"])
        scene = (pair, case.get("placement", "default"), case.get("seed", 0))
        method = case["algorithm"]
        if method in population[scene]:
            raise ValueError(f"Duplicate benchmark case {scene} for {method}")
        scores = _case_metrics(case)
        population[scene][method] = (
            (scores, case["trials"]) if scores is not None else None
        )

    groups = defaultdict(list)
    for (pair, _, _), observations in population.items():
        (category, _), hand = pair
        for key in ((None, None), (hand, None), (None, category), (hand, category)):
            groups[key].append((pair, observations))

    overall = _group_summary(groups.pop((None, None)), methods, repeats)
    missing = sum(stats["missing_cases"] for stats in overall["algorithms"].values())
    report = {
        "complete": missing == 0,
        "missing_cases": missing,
        "overall": overall,
        "by_gripper": {},
        "by_category": {},
        "by_gripper_category": {},
    }
    for (hand, category), scenes in groups.items():
        result = _group_summary(scenes, methods, repeats)
        if category is None:
            report["by_gripper"][hand] = result
        elif hand is None:
            report["by_category"][category] = result
        else:
            report["by_gripper_category"].setdefault(hand, {})[category] = result
    return report
