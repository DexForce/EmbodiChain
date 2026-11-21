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

"""Recount grouped ACT checkpoint decisions, reserved outcomes and release receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import math


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _panel(path: Path, base: int, count: int, fingerprint: str) -> dict:
    panel = _read(path)
    rows = panel["records"]
    if [row["seed"] for row in rows] != list(range(base, base + count)) or panel[
        "policy_sha256"
    ] != fingerprint:
        raise ValueError(f"{path}: different scenes or policy")
    for row in rows:
        xyz = row["final_cube_xyz"]
        distance = row["goal_distance"]
        if (
            not math.isclose(math.dist(xyz[:2], row["goal_xy"]), distance, abs_tol=2e-5)
            or row["success"] != (distance < 0.1 and 0 <= xyz[2] <= 0.08)
            or not 1 <= row["steps"] <= 300
        ):
            raise ValueError(f"{path}: inconsistent physical outcome")
    if panel["successes"] != sum(row["success"] for row in rows):
        raise ValueError(f"{path}: inconsistent success count")
    return panel


def _audit(path: Path, spec: dict, seed: int) -> dict:
    from embodichain.extensions.models.policy.act import policy_fingerprint

    recipe = _read(path / "recipe.json")
    supply = _read(path / "supply.json")
    if (
        recipe["protocol"] != spec
        or recipe["seed"] != seed
        or len(recipe["gpu"].split(",")) != 1
        or not recipe["gpu"]
    ):
        raise ValueError("Run configuration or GPU assignment differs")
    events = _read(path / "development.json")
    if [e["step"] for e in events] != spec["checkpoints"][: len(events)]:
        raise ValueError("Development checkpoint schedule differs")
    for e in events:
        d = _panel(
            path / f"dev_{e['step']}.json",
            spec["dev_base"],
            spec["dev_scenes"],
            e["sha256"],
        )
        if (
            d["successes"] != e["successes"]
            or e["save_seconds"] > e["result_seconds"]
            or d["mean_goal_distance"] != e["goal_distance"]
        ):
            raise ValueError("Development report disagrees with event")
    cutoff = spec["deadline_seconds"] - spec.get("submission_margin_seconds", 0)
    candidates = [
        b
        for a, b in zip(events, events[1:])
        if b["result_seconds"] <= cutoff
        and min(a["successes"], b["successes"]) >= spec["dev_min_successes"]
    ]
    chosen = max(
        candidates,
        key=lambda e: (e["successes"], -e["goal_distance"], -e["step"]),
        default=None,
    )
    receipt = _read(path / "deadline.json")
    if (
        receipt["budget_seconds"] != spec["deadline_seconds"]
        or receipt["chosen"] != chosen
    ):
        raise ValueError("Locked policy disagrees with in-budget events")
    if "submission_margin_seconds" in spec:
        if (
            receipt["selection_cutoff_seconds"] != cutoff
            or not 0 <= receipt["locked_seconds"] <= spec["deadline_seconds"]
        ):
            raise ValueError("Submission receipt missed its physical deadline")
        if chosen and receipt["locked_seconds"] < chosen["result_seconds"]:
            raise ValueError("Submission receipt precedes the development result")
    score = None
    if chosen:
        if policy_fingerprint(path / chosen["checkpoint"]) != chosen["sha256"]:
            raise ValueError("Native checkpoint fingerprint differs")
        score = _panel(
            path / "test.json", spec["test_base"], spec["test_scenes"], chosen["sha256"]
        )["successes"]
    m = _read(path / "metrics.json")
    if (
        m["selected"] != chosen
        or m["reserved_successes"] != score
        or (not chosen and (path / "test.json").exists())
    ):
        raise ValueError("Reported result differs from verified checkpoint")
    width, slots = spec["producer_envs"], spec["shared_slots"]
    counts = []
    for generation, receipt in enumerate(supply["releases"], 2):
        first = (generation - 1) * width - slots
        ids = list(range(first, first + width)) if first >= 0 else []
        credits = receipt["committed_draws"]
        if (
            receipt["requested_generation"] != generation
            or receipt["episode_ids"] != ids
            or len(credits) != len(ids)
            or any(type(v) is not int or v < spec["draw_floor"] for v in credits)
        ):
            raise ValueError("Group authorization violated the committed-draw floor")
        counts.extend(credits)
    qualified = score is not None and score >= spec["required_reserved_successes"]
    if m["seed"] != seed or m["pass_gate"] != qualified:
        raise ValueError("Inconsistent run gate")
    return dict(
        seed=seed,
        qualified=qualified,
        reserved_successes=score,
        locked_seconds=receipt.get("locked_seconds"),
        chosen=chosen,
        updates=supply["step"],
        first_update=supply["first_update"],
        sampled_chunks=supply["engine"]["sampled_chunks"],
        successful_episodes=supply["engine"]["successful_episodes"],
        authorized_episode_replacements=len(counts),
        minimum_at_release=min(counts, default=None),
        releases=supply["releases"],
        development=events,
        source_hashes=recipe["source_hashes"],
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.glob("*.json"))
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("grouped_protocol.json"),
    )
    args = parser.parse_args()
    spec = _read(args.protocol)
    rows = []
    errors = []
    for seed in spec["seeds"]:
        try:
            rows.append(_audit(args.root / str(seed), spec, seed))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"{seed}: {exc}")
    n = sum(r["qualified"] for r in rows)
    result = dict(
        passed=not errors and n == len(spec["seeds"]),
        qualified=n,
        intended=len(spec["seeds"]),
        issues=errors,
        protocol=spec,
        protocol_sha256=hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        runs=rows,
    )
    (args.root / "acceptance.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Qualified {n}/{len(spec['seeds'])}; audit issues: {errors}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
