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

"""Separate measured motion, physical-condition, visual and source coverage reports."""

from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from embodichain.data_pipeline.recording.media import _atomic_json

__all__ = ["build_expansion_report"]


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _scalars(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _scalars(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _scalars(item, f"{prefix}[{index}]")
    elif (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
    ):
        yield prefix, float(value)


def build_expansion_report(
    directory: str | Path, *, receipts=None, variants=None, write: bool = False
) -> dict[str, object]:
    """Report orthogonal coverage of confirmed artifacts without changing admission.

    Geometry buckets quantize the existing joint-limit/arc-length descriptor at
    1e-3 resolution. They are descriptive counts, not a statistical coverage
    certificate or a new rejection threshold. Physical and visual variants
    retain separate identities even when their joint paths occupy one bucket.
    """
    from .persistence import ArtifactEpisodeSink
    from .visual import inspect_observation_variants

    root = Path(directory)
    receipts = ArtifactEpisodeSink(root).inspect() if receipts is None else receipts
    variants = inspect_observation_variants(root) if variants is None else variants
    geometry = set()
    timing = set()
    physical = set()
    requested_physical = set()
    sources = Counter()
    profiles = Counter()
    actual = {}
    sampled = {}
    missing = 0
    for receipt in receipts:
        with np.load(receipt.storage_id, allow_pickle=False) as arrays:
            metadata = json.loads(arrays["manifest"].tobytes())["metadata"]
        descriptor = metadata.get("coverage_descriptor")
        if descriptor:
            samples = np.asarray(descriptor["geometry"], dtype=np.float64)
            geometry.add(
                _digest(
                    (
                        descriptor["phase_ids"],
                        np.rint(samples / 1e-3).astype(np.int64).tolist(),
                    )
                )
            )
            timing.add(
                _digest(
                    (
                        descriptor["phase_ids"],
                        np.rint(np.asarray(descriptor["timing"]) / 1e-3)
                        .astype(np.int64)
                        .tolist(),
                    )
                )
            )
        else:
            missing += 1
        recipe = metadata.get("expansion_recipe", metadata.get("recipe", {}))
        if not isinstance(recipe, dict):
            recipe = {
                "physics": next(
                    (
                        json.loads(operation["payload_json"])
                        for operation in metadata.get("recipe_spec", {}).get(
                            "operations", ()
                        )
                        if operation["name"] == "physical_parameters"
                    ),
                    {},
                )
            }
        values = metadata.get("physics_parameters", {})
        if values:
            physical.add(_digest(values))
        requested_physical.add(_digest(recipe.get("physics", {})))
        for target, data in ((sampled, recipe.get("physics", {})), (actual, values)):
            for key, value in _scalars(data):
                row = target.setdefault(
                    key, {"count": 0, "minimum": value, "maximum": value}
                )
                row["count"] += 1
                row["minimum"] = min(row["minimum"], value)
                row["maximum"] = max(row["maximum"], value)
        ancestors = metadata.get("source_episode_uuids") or (
            [metadata["source_episode_uuid"]]
            if metadata.get("source_episode_uuid")
            else []
        )
        sources.update(set(ancestors))
    for variant in variants:
        profiles[f"{variant['profile_id']}@{variant['profile_revision']}"] += 1
    result = {
        "schema_version": 1,
        "confirmed_physical_episodes": len(receipts),
        "geometry": {
            "buckets": len(geometry),
            "timing_buckets": len(timing),
            "descriptor": "joint_limits_arc_length",
            "quantization": 0.001,
            "missing_descriptors": missing,
        },
        "physics": {
            "condition_fingerprints": len(physical),
            "requested_conditions": len(requested_physical),
            "sampled_ranges": sampled,
            "measured_ranges": actual,
        },
        "visual": {
            "variants": len(variants),
            "profiles": dict(profiles),
            "physical_success_credit": 0,
        },
        "sources": {"ancestor_groups": dict(sources)},
        "commit_ids": sorted(receipt.commit_id for receipt in receipts),
    }
    if write:
        _atomic_json(root / "coverage.json", result)
    return result
