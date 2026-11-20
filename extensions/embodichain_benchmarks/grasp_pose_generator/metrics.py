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

"""Fixed-budget physical grasp statistics and paired family bootstrap."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

__all__ = ["summarize"]


def _case_metrics(case):
    rows = case["trials"]
    if case.get("error") or any(r["status"] != "ok" for r in rows):
        return None
    outcomes = {(r["rank"], r["condition"]): r["success"] for r in rows}
    expected = {
        (rank, d) for rank in range(1, case["count"] + 1) for d in ("base", "x", "y")
    }
    if outcomes.keys() != expected or len(outcomes) != len(rows):
        return None
    base = [outcomes.get((i, "base"), False) for i in range(1, 11)]
    return np.array(
        [
            base[0],
            any(base[:5]),
            any(base),
            sum(base) / 10,
            all(outcomes.get((1, d), False) for d in ("base", "x", "y")),
        ],
        dtype=float,
    )


def summarize(
    cases: list[dict], held_grippers: list[str], *, repeats: int = 2000
) -> dict:
    """Aggregate paired scenes, with equal object/hand weights within each split."""
    metrics = ["Success@1", "Hit@5", "Hit@10", "Precision@10", "Robust@1"]
    methods = sorted({c["algorithm"] for c in cases})
    pairs = defaultdict(list)
    families = {}
    diagnostics = defaultdict(list)
    if not cases or repeats < 1:
        raise ValueError("Cases and positive bootstrap repetitions are required")
    missing = 0
    population = defaultdict(set)
    for c in cases:
        key = (
            c["object"],
            c["gripper"],
            c.get("placement", "default"),
            c.get("seed", 0),
        )
        if key in population[c["algorithm"]]:
            raise ValueError(f"Duplicate benchmark case {key}")
        population[c["algorithm"]].add(key)
        value = _case_metrics(c)
        if value is None:
            missing += 1
            continue
        group = (c["object"], c["gripper"], c["object_holdout"])
        families[group[0]] = c["family"]
        pairs[c["algorithm"], group].append(value)
        diagnostics[c["algorithm"]].extend(c["trials"])
    intended = set.union(*population.values())
    missing += sum(len(intended - scenes) for scenes in population.values())
    report = {
        "complete": not missing,
        "missing_cases": missing,
        "folds": {},
        "diagnostics": {},
    }
    for method, rows in diagnostics.items():
        info = {}
        nominal = [r for r in rows if r["condition"] == "base"]
        info["base_retention"] = {
            str(t): (
                np.mean([r["retention"][str(t)] for r in nominal]).item()
                if nominal
                else None
            )
            for t in (0.5, 1.0, 3.0)
        }
        for d in ("x", "y"):
            trials = [r for r in rows if r["condition"] == d and r["hold"]["success"]]
            info[d] = {
                "started": len(trials),
                "passed": sum(r["success"] for r in trials),
                "conditional_retention": (
                    sum(r["success"] for r in trials) / len(trials) if trials else None
                ),
            }
        for phase in ("hold", "disturbance"):
            windows = [
                r[phase]
                for r in rows
                if r.get(phase) is not None
                and r[phase].get("entered", True)
                and (phase != "hold" or r["condition"] == "base")
            ]
            info[phase] = {"drops": sum(w["drop_s"] is not None for w in windows)}
            for metric in ("slip_mm", "rotation_deg"):
                info[phase][metric + "_p50_p95"] = (
                    np.percentile([w[metric] for w in windows], [50, 95]).tolist()
                    if windows
                    else None
                )
        report["diagnostics"][method] = info
    if missing:
        return report
    for held in held_grippers:
        fold = {}
        for obj_new, hand_new, label in (
            (False, False, "seen"),
            (True, False, "new_object"),
            (False, True, "new_gripper"),
            (True, True, "both_new"),
        ):
            groups = sorted(
                {g for _, g in pairs if g[2] == obj_new and (g[1] == held) == hand_new}
            )
            if not groups:
                fold[label] = None
                continue
            values = np.array(
                [[np.mean(pairs[m, g], axis=0) for g in groups] for m in methods]
            )
            family_ids = np.array([families[g[0]] for g in groups])
            family_names = np.unique(family_ids)
            rng = np.random.default_rng(0)
            scores = values.mean(1)
            bootstrap = []
            # Bound temporary storage independently of the number of draws.
            indices = np.searchsorted(family_names, family_ids)
            for start in range(0, repeats, 32):
                counts = rng.multinomial(
                    len(family_names),
                    np.full(len(family_names), 1 / len(family_names)),
                    size=min(32, repeats - start),
                )
                weights = counts[:, indices]
                bootstrap.append((weights / weights.sum(1, keepdims=True)) @ values)
            draws = np.concatenate(bootstrap, axis=1)
            stats = {}
            for i, m in enumerate(methods):
                stats[m] = {
                    "metrics": dict(zip(metrics, scores[i].tolist())),
                    "ci95": dict(
                        zip(
                            metrics,
                            np.percentile(draws[i], [2.5, 97.5], axis=0).T.tolist(),
                        )
                    ),
                }
            comparisons = {}
            for i, j in combinations(range(len(methods)), 2):
                comparisons[f"{methods[i]} minus {methods[j]}"] = dict(
                    zip(
                        metrics,
                        np.percentile(
                            draws[i] - draws[j], [2.5, 97.5], axis=0
                        ).T.tolist(),
                    )
                )
            fold[label] = {
                "object_gripper_pairs": len(groups),
                "families": len(family_names),
                "algorithms": stats,
                "paired_difference_ci95": comparisons,
            }
        if fold["seen"] and fold["both_new"]:
            fold["drop_percentage_points"] = {
                m: 100
                * (
                    fold["seen"]["algorithms"][m]["metrics"]["Success@1"]
                    - fold["both_new"]["algorithms"][m]["metrics"]["Success@1"]
                )
                for m in methods
            }
        report["folds"][held] = fold
    return report
