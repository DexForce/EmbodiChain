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

"""CPU-only, post-terminal diagnostic. Default inspect never loads checkpoints.

Only --execute enables model loading and existing framehash/episode metadata reads.
Never opens videos, decodes frames, runs simulation/training, or alters run files.
Checkpoints must be trusted outputs of this experiment. YUV agreement never proves
agreement of original RGB tensors consumed by either learner.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterator
from fractions import Fraction
import itertools
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, TypeAlias

__all__ = ["main"]

SourceMapping: TypeAlias = tuple[
    float, list[tuple[int, int, int, list[int]]], dict[int, set[int]]
]

DEFAULT_ROOT = Path("/tmp/ods-visual-training-formal-20261021-v3")


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def contracts(left: Path, right: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sides = []
    for run in (left, right):
        summary, trace = read(run / "summary.json"), read(run / "training-trace.json")
        initial = read(run / "initial-contract.json")
        metric = summary["training_metrics"]
        budget = [
            metric[k]
            for k in ("completed_updates", "trained_chunks", "valid_target_occurrences")
        ]
        recipe = trace["recipe"]
        valid = budget == [
            recipe["updates"],
            recipe["updates"] * recipe["batch_size"],
            recipe["updates"] * recipe["batch_size"] * recipe["prediction_horizon"],
        ]
        valid = valid and len(trace["updates"]) == recipe["updates"]
        valid = valid and all(
            len(row["pairs"]) == recipe["batch_size"] for row in trace["updates"]
        )
        sides.append(
            {
                "summary": summary,
                "trace": trace,
                "initial": initial,
                "metric": metric,
                "budget": budget,
                "budget_valid": valid,
            }
        )
    a, b = sides
    numeric = lambda side: [
        [row[k] for k in ("ordinal", "generation", "length", "numeric_sha256")]
        for row in side["trace"]["episodes"]
    ]
    checks = {
        "initial_model_equal": a["metric"]["initial_model_sha256"]
        == b["metric"]["initial_model_sha256"],
        "normalization_equal": a["metric"]["normalization_sha256"]
        == b["metric"]["normalization_sha256"],
        "initial_contracts_self_consistent": all(
            side["initial"]["model_sha256"] == side["metric"]["initial_model_sha256"]
            and side["initial"]["normalization_sha256"]
            == side["metric"]["normalization_sha256"]
            and side["trace"]["initial_model_sha256"]
            == side["metric"]["initial_model_sha256"]
            and side["trace"]["normalization_sha256"]
            == side["metric"]["normalization_sha256"]
            for side in sides
        ),
        "recipe_equal": a["trace"]["recipe"] == b["trace"]["recipe"],
        "numeric_episode_manifests_equal": numeric(a) == numeric(b),
        "ordered_updates_and_anchors_equal": a["trace"]["updates"]
        == b["trace"]["updates"],
        "completed_budget_matches": a["budget_valid"]
        and b["budget_valid"]
        and a["budget"] == b["budget"],
        "budget_overlap_serial": [a["budget"], b["budget"]],
    }
    return checks, sides


def model_difference(left: Path, right: Path) -> dict[str, Any]:
    import torch

    # Map trusted PyTorch ZIP files privately; touch model tensors, not optimizer buffers.
    a = torch.load(
        left / "final.pt", map_location="cpu", weights_only=False, mmap=True
    )["model"]
    b = torch.load(
        right / "final.pt", map_location="cpu", weights_only=False, mmap=True
    )["model"]
    modules = defaultdict(
        lambda: dict(
            tensors=0,
            elements=0,
            changed_tensors=0,
            nonfinite_elements=0,
            max_abs=0.0,
            squared_error=0.0,
            reference_squared=0.0,
        )
    )
    for name in sorted(a.keys() & b.keys()):
        if not isinstance(a[name], torch.Tensor) or not isinstance(
            b[name], torch.Tensor
        ):
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f"Tensor shape/dtype mismatch: {name}")
        module = modules[
            name.rsplit(".", 1)[0] if "." in name else "normalization_buffers"
        ]
        module["tensors"] += 1
        module["elements"] += x.numel()
        identical = torch.equal(
            x.contiguous().reshape(-1).view(torch.uint8),
            y.contiguous().reshape(-1).view(torch.uint8),
        )
        module["changed_tensors"] += int(not identical)
        x, y = x.double(), y.double()
        bad = ~(torch.isfinite(x) & torch.isfinite(y))
        module["nonfinite_elements"] += int(bad.sum())
        if bad.any():
            continue
        delta = y - x
        module["max_abs"] = max(module["max_abs"], float(delta.abs().max()))
        module["squared_error"] += float(delta.square().sum())
        module["reference_squared"] += float(x.square().sum())
    for module in modules.values():
        error, reference = module.pop("squared_error"), module.pop("reference_squared")
        module["rms"] = math.sqrt(error / module["elements"])
        module["relative_l2"] = (
            math.sqrt(error / reference) if reference else (0.0 if not error else None)
        )
        module["bitwise_equal"] = module["changed_tensors"] == 0
        if module["nonfinite_elements"]:
            module.update(max_abs=None, rms=None, relative_l2=None)
    extras = lambda state: {
        k: v for k, v in state.items() if not isinstance(v, torch.Tensor)
    }
    return {
        "tensor_keys_equal": sorted(a) == sorted(b),
        "keys_only_overlap": sorted(a.keys() - b.keys()),
        "keys_only_serial": sorted(b.keys() - a.keys()),
        "non_tensor_state_equal": extras(a) == extras(b),
        "modules": dict(modules),
        "relative_l2_definition": "||serial-overlap||2 / ||overlap||2; null for nonfinite or zero reference with nonzero delta",
        "parity_is_not_a_training_budget_requirement": True,
    }


def rows(run: Path, record: dict[str, Any]) -> Iterator[tuple[str, float | None]]:
    path = Path(record["decoded_frame_hashes"])
    if not path.is_file():
        path = run / path.parent.name / path.name
    if path.stat().st_size > 64 * 1024**2:
        raise ValueError("Framehash text exceeds 64 MiB diagnostic limit")
    base = None
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line.startswith("#tb 0:"):
                base = Fraction(line.split(":", 1)[1].strip())
            if line.startswith("#") or not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if (
                len(parts) != 6
                or parts[0] != "0"
                or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[-1])
            ):
                raise ValueError(f"Malformed framehash row in {path}")
            yield parts[-1].lower(), (
                float(int(parts[2]) * base) if base is not None else None
            )


def mapping(run: Path, record: dict[str, Any], side: dict[str, Any]) -> SourceMapping:
    """Return mapping only when official spans, audited source IDs and PTS agree."""
    import pyarrow.parquet as pq

    audit = side["summary"]["post_audit"]
    if not audit.get("passed") or not audit["recording"].get(
        "all_numeric_sha256_verified"
    ):
        raise ValueError("Source-to-episode numeric audit is unavailable")
    corpus, camera = run / "lerobot_dataset", record["camera"]
    if side["summary"]["config"].get("history_length") != 2:
        raise ValueError("H2 source attribution requires declared H=2")
    fingerprints = audit["full_video_integrity"].get("metadata_sha256", {})
    if not fingerprints:
        raise ValueError("Official metadata fingerprints are unavailable")
    for relative, wanted in fingerprints.items():
        path = corpus / relative
        if (
            path.stat().st_size > 64 * 1024**2
            or hashlib.sha256(path.read_bytes()).hexdigest() != wanted
        ):
            raise ValueError(
                "Official metadata changed or exceeds diagnostic size limit"
            )
    info = read(corpus / "meta/info.json")
    fps = float(info["fps"])
    columns = ["episode_index", "length"] + [
        f"videos/{camera}/{key}"
        for key in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    files = sorted((corpus / "meta/episodes").rglob("*.parquet"))
    metadata = pq.read_table(files, columns=columns, use_threads=False).to_pylist()
    trace = {row["ordinal"]: row for row in side["trace"]["episodes"]}
    spans, used = [], defaultdict(set)
    for update in side["trace"]["updates"]:
        for ordinal, anchor in update["pairs"]:
            used[ordinal].update((anchor - 1, anchor))
    for row in metadata:
        prefix = f"videos/{camera}/"
        relative = info["video_path"].format(
            video_key=camera,
            chunk_index=row[prefix + "chunk_index"],
            file_index=row[prefix + "file_index"],
        )
        if relative != record["file"]:
            continue
        start, stop = row[prefix + "from_timestamp"], row[prefix + "to_timestamp"]
        first, last = round(start * fps), round(stop * fps)
        ordinal = row["episode_index"]
        if abs(start - first / fps) > 1e-4 or abs(stop - last / fps) > 1e-4:
            raise ValueError("Official video span is not aligned to frame grid")
        if last - first != row["length"] or trace[ordinal]["length"] != row["length"]:
            raise ValueError("Official/source span lengths differ")
        spans.append((first, last, ordinal, trace[ordinal]["source_id"]))
    spans.sort()
    end = 0
    for first, last, _, _ in spans:
        if first != end:
            raise ValueError("Video source spans overlap or contain gaps")
        end = last
    if (
        end != record["decoded_frames"]
        or len({row[2] for row in spans}) != len(spans)
        or {row[2] for row in spans} != set(record["episodes"])
    ):
        raise ValueError("Video source inventory/count mapping differs")
    return fps, spans, used


def source_at(
    index: int, mapped: SourceMapping | None, valid: bool, reason: str | None
) -> dict[str, Any]:
    if not valid or mapped is None:
        return {
            "status": "unknown",
            "reason": reason or "PTS/count grid is not verified",
        }
    for first, last, ordinal, identity in mapped[1]:
        if first <= index < last:
            frame = index - first
            return {
                "status": "known",
                "source_id": identity,
                "ordinal": ordinal,
                "frame_index": frame,
                "used_as_H2_training_image": frame in mapped[2][ordinal],
            }
    return {"status": "unknown", "reason": "Frame outside verified source spans"}


def compare_yuv(
    left: Path, right: Path, sides: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    inventories = [
        {
            (r["camera"], r["file"]): r
            for r in side["summary"]["post_audit"]["full_video_integrity"]["files"]
        }
        for side in sides
    ]
    output = []
    for key in sorted(inventories[0].keys() | inventories[1].keys()):
        result = {
            "camera": key[0],
            "file": key[1],
            "hash_semantics": "decoded YUV checksum column only",
            "alignment": "same camera/file and decoded row index; source attribution is conditional",
        }
        try:
            records = [inventory[key] for inventory in inventories]
            maps, reasons = [], []
            for run, record, side in zip((left, right), records, sides):
                try:
                    maps.append(mapping(run, record, side))
                    reasons.append(None)
                except Exception as error:
                    maps.append(None)
                    reasons.append(str(error))
            same = different = 0
            count, grid = [0, 0], [True, True]
            first_difference = None
            for index, pair in enumerate(
                itertools.zip_longest(rows(left, records[0]), rows(right, records[1]))
            ):
                for arm, value in enumerate(pair):
                    if value is not None:
                        count[arm] += 1
                        grid[arm] &= (
                            maps[arm] is not None
                            and value[1] is not None
                            and abs(value[1] - index / maps[arm][0]) <= 1e-4
                        )
                if (
                    pair[0] is not None
                    and pair[1] is not None
                    and pair[0][0] == pair[1][0]
                ):
                    same += 1
                else:
                    different += 1
                    if first_difference is None:
                        first_difference = {
                            "file_frame_index": index,
                            "hashes_overlap_serial": [
                                value[0] if value else None for value in pair
                            ],
                        }
            if first_difference is not None:
                for arm, label in enumerate(("overlap_source", "serial_source")):
                    valid = grid[arm] and count[arm] == records[arm]["decoded_frames"]
                    first_difference[label] = source_at(
                        first_difference["file_frame_index"],
                        maps[arm],
                        valid,
                        reasons[arm],
                    )
            result.update(
                same_hashes=same,
                different_or_missing_hashes=different,
                frame_counts_overlap_serial=count,
                first_difference=first_difference,
                mapping_unknown_reasons=reasons,
                pts_grid_verified=grid,
            )
        except Exception as error:
            result.update(status="unknown", reason=str(error))
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    pairs = [
        (
            vision,
            seed,
            root / "ods_overlap" / vision / f"seed{seed}",
            root / "ods_serial" / vision / f"seed{seed}",
        )
        for vision in ("single", "dual")
        for seed in (19001, 19002, 19003)
    ]
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "inspect_only",
                    "root": str(root),
                    "pairs": [
                        {
                            "vision": v,
                            "seed": s,
                            "checkpoint_files_present": [
                                (a / "final.pt").is_file(),
                                (b / "final.pt").is_file(),
                            ],
                        }
                        for v, s, a, b in pairs
                    ],
                    "next": "Only after all 30 runs are terminal, add --execute --output /tmp/<new-diagnostic>.json",
                },
                indent=2,
            )
        )
        return
    if args.output is None:
        parser.error("--execute requires a new --output JSON path")
    output = args.output.resolve()
    if root == output or root in output.parents or output.exists():
        parser.error("Output must be new and outside the immutable experiment root")
    if not (root / "suite-summary.json").is_file():
        parser.error(
            "Terminal suite-summary.json is not present; no checkpoint/framehash read performed"
        )
    terminal = read(root / "suite-summary.json")
    if (
        terminal.get("status") not in ("complete", "failed", "unsupported")
        or len(terminal.get("results", [])) != 30
        or any(
            r.get("status") not in ("complete", "failed", "unsupported")
            for r in terminal["results"]
        )
    ):
        parser.error(
            "All 30 runs must be terminal before any checkpoint/framehash reads"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.update(
        OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    import torch

    torch.set_num_threads(1)
    report = {
        "diagnostic_only": True,
        "source_suite_status": terminal["status"],
        "source_run_status_counts": {
            status: sum(row["status"] == status for row in terminal["results"])
            for status in ("complete", "failed", "unsupported")
        },
        "raw_RGB_equivalence": "unknown; encoded or decoded YUV is not the original model RGB input",
        "pairs": [],
    }
    for vision, seed, left, right in pairs:
        result = {"vision": vision, "seed": seed, "source_run_statuses": {}}
        for label, run in (("overlap", left), ("serial", right)):
            try:
                summary = read(run / "summary.json")
                result["source_run_statuses"][label] = {
                    key: summary.get(key)
                    for key in (
                        "status",
                        "qualification_passed",
                        "post_audit_status",
                        "evaluation_status",
                        "quality_error",
                    )
                }
            except Exception as error:
                result["source_run_statuses"][label] = {
                    "status": "unavailable",
                    "error": repr(error),
                }
        try:
            checks, sides = contracts(left, right)
            result["contracts"] = checks
            result["declared_final_model_sha_equal_diagnostic_only"] = (
                sides[0]["metric"]["final_model_sha256"]
                == sides[1]["metric"]["final_model_sha256"]
            )
            result["models"] = model_difference(left, right)
            result["decoded_yuv"] = compare_yuv(left, right, sides)
        except Exception as error:
            result.update(status="unavailable", error=repr(error))
        report["pairs"].append(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"diagnostic": str(output), "pairs": len(report["pairs"])}))


if __name__ == "__main__":
    main()
