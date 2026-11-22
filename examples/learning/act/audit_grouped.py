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

__all__ = ["main"]


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


def _audit_supply(recipe: dict, supply: dict, spec: dict) -> list[int]:
    """Check delivered draws and each producer's complete authorization chain."""
    layouts = {"per-rank": 2, "shared": 1}
    world_size = recipe["world_size"]
    layout = recipe["producer_layout"]
    producer_count = layouts[layout]
    if world_size not in (1, 2) or producer_count > world_size:
        raise ValueError("Unexpected rank or producer count")
    ranks = recipe["ranks"]
    if (
        [rank["rank"] for rank in ranks] != list(range(world_size))
        or len({rank["pid"] for rank in ranks}) != world_size
        or len({rank["gpu_uuid"] for rank in ranks}) != world_size
    ):
        raise ValueError("Training ranks have inconsistent GPU assignments")
    steps = supply["step"]
    draw_counts = supply["ranks"]
    if [row["rank"] for row in draw_counts] != list(range(world_size)) or any(
        row["committed_draws"] * world_size != steps * spec["batch_size"]
        for row in draw_counts
    ):
        raise ValueError("Rank draw counts disagree with completed optimizer batches")
    producers = supply["producers"]
    if (
        [producer["rank"] for producer in producers] != list(range(producer_count))
        or len({producer["pid"] for producer in producers}) != producer_count
        or len({producer["engine"]["run_id"] for producer in producers})
        != producer_count
    ):
        raise ValueError("Producer identities differ from the requested topology")

    counts: list[int] = []
    for producer in producers:
        width, slots, cap = (producer[key] for key in ("width", "slots", "cap"))
        rank = producer["rank"]
        stats = producer["engine"]
        completed, requested = (
            stats[key] for key in ("completed_generation", "requested_generation")
        )
        releases = producer["releases"]
        if (
            producer["gpu_uuid"] != ranks[rank]["gpu_uuid"]
            or producer["pid"] in {record["pid"] for record in ranks}
            or width * producer_count != spec["producer_envs"]
            or slots * producer_count != spec["shared_slots"]
            or cap * producer_count != spec["episodes"]
            or stats["sampled_chunks"] * producer_count != steps * spec["batch_size"]
        ):
            raise ValueError(f"Producer {rank} differs from the supply contract")
        if (
            completed < 1
            or requested not in (completed, completed + 1)
            or len(releases) != requested - 1
            or requested * width > cap
            or stats["successful_episodes"] != completed * width
        ):
            raise ValueError(f"Producer {rank} has an incomplete release chain")
        counts.extend(_audit_releases(releases, width, slots, spec["draw_floor"]))
    return counts


def _audit_releases(
    releases: list[dict], width: int, slots: int, floor: int
) -> list[int]:
    counts = []
    for generation, release in enumerate(releases, 2):
        first = (generation - 1) * width - slots
        ids = list(range(first, first + width)) if first >= 0 else []
        credits = release["committed_draws"]
        if (
            release["requested_generation"] != generation
            or release["episode_ids"] != ids
            or len(credits) != len(ids)
            or any(type(draw) is not int or draw < floor for draw in credits)
        ):
            raise ValueError("Group authorization violated the committed-draw floor")
        counts.extend(credits)
    return counts


def _audit(path: Path, spec: dict, seed: int) -> dict:
    from embodichain.extensions.models.policy.act import policy_fingerprint

    recipe = _read(path / "recipe.json")
    supply = _read(path / "supply.json")
    current = "world_size" in recipe
    if (
        recipe["protocol"] != spec
        or recipe["seed"] != seed
        or recipe.get("smoke", False)
        or (not current and len(recipe["gpu"].split(",")) != 1)
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
            or not 0 <= receipt["locked_seconds"]
            or (not current and receipt["locked_seconds"] > spec["deadline_seconds"])
        ):
            raise ValueError("Submission receipt missed its physical deadline")
        if chosen and receipt["locked_seconds"] < chosen["result_seconds"]:
            raise ValueError("Submission receipt precedes the development result")
    published = None
    if spec.get("publish_receipt"):
        evidence = _read(path / "publication.json")
        published = evidence["published_seconds"]
        if (
            not receipt["locked_seconds"] <= published
            or (not current and published > spec["deadline_seconds"])
            or evidence["decision_sha256"]
            != hashlib.sha256((path / "deadline.json").read_bytes()).hexdigest()
        ):
            raise ValueError("Committed deadline receipt differs or was late")
    score = None
    manifest_hash = None
    if chosen:
        if policy_fingerprint(path / chosen["checkpoint"]) != chosen["sha256"]:
            raise ValueError("Native checkpoint fingerprint differs")
        checkpoint = path / chosen["checkpoint"]
        manifest_file = checkpoint / "embodichain_act.json"
        manifest = _read(manifest_file)
        image = _read(checkpoint / "config.json")["input_features"][
            manifest["image_key"]
        ]
        side = spec.get("backbone_image_side", 224)
        if (
            manifest["task_identity"] != recipe["task_identity"]
            or manifest["update"] != chosen["step"]
            or manifest["policy_sha256"] != chosen["sha256"]
            or manifest.get("query_period", 1) != spec.get("query_period", 1)
            or image["shape"] != [3, side, side]
            or manifest["selection"]["seeds"]
            != list(range(spec["dev_base"], spec["dev_base"] + spec["dev_scenes"]))
            or (manifest.get("elapsed_control_scale") is not None)
            != spec.get("elapsed_control_input", False)
        ):
            raise ValueError("Deployed ACT manifest differs from the frozen recipe")
        manifest_hash = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
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
    if current:
        counts = _audit_supply(recipe, supply, spec)
        engine_stats = [producer["engine"] for producer in supply["producers"]]
        releases = supply["producers"]
    else:
        engine_stats = [supply["engine"]]
        if supply["engine"]["sampled_chunks"] != supply["step"] * spec["batch_size"]:
            raise ValueError("Sample count disagrees with completed optimizer batches")
        releases = supply["releases"]
        counts = _audit_releases(
            releases, spec["producer_envs"], spec["shared_slots"], spec["draw_floor"]
        )
    submission = published if published is not None else receipt.get("locked_seconds")
    qualified = (
        score is not None
        and score >= spec["required_reserved_successes"]
        and (submission is None or submission <= spec["deadline_seconds"])
    )
    if m["seed"] != seed or m["pass_gate"] != qualified:
        raise ValueError("Inconsistent run gate")
    return dict(
        seed=seed,
        qualified=qualified,
        reserved_successes=score,
        locked_seconds=receipt.get("locked_seconds"),
        published_seconds=published,
        chosen=chosen,
        chosen_manifest_sha256=manifest_hash,
        updates=supply["step"],
        first_update=supply["first_update"],
        sampled_chunks=sum(engine["sampled_chunks"] for engine in engine_stats),
        successful_episodes=sum(
            engine["successful_episodes"] for engine in engine_stats
        ),
        authorized_episode_replacements=len(counts),
        minimum_at_release=min(counts, default=None),
        releases=releases,
        development=events,
        source_hashes=recipe["source_hashes"],
        artifact_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.glob("*.json"))
        },
    )


def main() -> None:
    """Write the audited evidence and print the gate for each training seed."""
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
    print("seed    published_s  reserved  min_draws_at_release")
    for row in rows:
        published = row["published_seconds"]
        score = row["reserved_successes"]
        draws = row["minimum_at_release"]
        seconds = f"{published:.3f}" if published is not None and row["chosen"] else "-"
        reserved = f"{score}/{spec['test_scenes']}" if score is not None else "-"
        print(
            f"{row['seed']:<8}{seconds:>11}  {reserved:>8}  {draws if draws is not None else '-':>20}"
        )
    print(f"Qualified {n}/{len(spec['seeds'])}; audit issues: {errors}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
