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

"""Validated source-independent v4 scene layout packages."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .contracts import ScenePoseChange, SceneVariant, _variant_id
from .layout import (
    SceneLayoutInput,
    PreparedSceneRandomization,
    SceneRandomizationBatch,
    _VariantPool,
    _digest,
    _policy_data,
    prepare_scene_randomization,
)
from ._layout_numeric import _numeric_geometry

__all__ = ["LoadedSceneVariants", "save_scene_variants", "load_scene_variants"]


def _write_atomic(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("Scene package destination must not be a symlink.")
    # The C encoder avoids Python iteration over millions of source-geometry
    # scalars. Bound the additional text allocation to one package file at a
    # time; canonical bytes and atomic fsync/replace semantics stay identical.
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict | list:
    if path.is_symlink():
        raise ValueError("Scene package files must not be symlinks.")

    def unique(pairs: list) -> dict:
        data = {}
        for key, value in pairs:
            if key in data:
                raise ValueError(f"Duplicate scene package field: {key!r}")
            data[key] = value
        return data

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON: {value}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError("Invalid scene package JSON.") from exc


def _pool_records(pool: Sequence[SceneVariant]) -> list[dict]:
    if isinstance(pool, _VariantPool):
        records = []
        for index, ordinal in enumerate(pool.ordinals):
            metadata = tuple(
                sorted((*pool.metadata, ("proposal_ordinal", str(int(ordinal)))))
            )
            changes = tuple(
                (uid, tuple(tuple(float(x) for x in row) for row in pose))
                for uid, pose in zip(pool.uids, pool.poses[index])
            )
            records.append(
                {
                    "parent_scene_id": pool.parent,
                    "seed": pool.seed,
                    "ordinal": int(ordinal),
                    "poses": dict(changes),
                    "metadata": metadata,
                    "variant_id": _variant_id(
                        pool.parent, pool.seed, int(ordinal), changes, metadata
                    ),
                }
            )
        return records
    return [
        {
            "parent_scene_id": v.parent_scene_id,
            "seed": v.seed,
            "ordinal": v.ordinal,
            "poses": {c.entity_uid: c.arena_pose for c in v.pose_changes},
            "metadata": v.metadata,
            "variant_id": v.variant_id,
        }
        for v in pool
    ]


def _validate_records(
    prepared: PreparedSceneRandomization, records: list[dict]
) -> None:
    geometry, policy = prepared._geometry, {
        **prepared.policy,
        "_table": prepared._table,
    }
    nodes = geometry["nodes"]
    roots = [i for i, node in enumerate(nodes) if node["parent"] < 0]
    targets = [
        i
        for i, node in enumerate(nodes)
        if node["uid"] in policy["intra_group"]["targets"]
    ]
    uids = [node["uid"] for node in nodes]
    proxies = geometry.get("fixed_orientation_group_proxies")
    if proxies is not None:
        for root in roots:
            descendants = {root}
            for index in range(root + 1, len(nodes)):
                if nodes[index]["parent"] in descendants:
                    descendants.add(index)
            clouds = []
            for index in descendants:
                uid = uids[index]
                pose = np.asarray(prepared.source.poses[uid])
                clouds.append(
                    prepared.source.vertices[uid] @ pose[:3, :3].T + pose[:3, 3]
                )
            cloud = np.concatenate(clouds)[:, :2]
            box = np.asarray(proxies[uids[root]], dtype=float)
            if (
                box.shape != (2, 2)
                or not np.isfinite(box).all()
                or np.any(box[0] > cloud.min(0) + 1e-7)
                or np.any(box[1] < cloud.max(0) - 1e-7)
            ):
                raise ValueError(
                    "Saved group proxy does not conservatively cover its source geometry."
                )
    previous = -1
    for record in records:
        if set(record) != {
            "parent_scene_id",
            "seed",
            "ordinal",
            "poses",
            "metadata",
            "variant_id",
        }:
            raise ValueError("Unsupported scene variant fields.")
        if (
            type(record["ordinal"]) is not int
            or record["ordinal"] <= previous
            or type(record["seed"]) is not int
            or record["seed"] < 0
            or record["parent_scene_id"] != prepared.source.scene_id
            or set(record["poses"]) != set(prepared.movable_uids)
        ):
            raise ValueError("Invalid scene variant identity, ordering or permissions.")
        previous = record["ordinal"]
        if any(isinstance(item, str) for item in record["metadata"]):
            raise ValueError("Provenance requires explicit key/value pairs.")
        metadata = tuple(tuple(item) for item in record["metadata"])
        if (
            any(
                len(item) != 2 or not all(isinstance(x, str) for x in item)
                for item in metadata
            )
            or len({key for key, _ in metadata}) != len(metadata)
            or tuple(sorted(metadata)) != metadata
        ):
            raise ValueError("Invalid scene variant provenance.")
        if any(not key or key != key.strip() for key, _ in metadata):
            raise ValueError("Provenance keys must be nonempty identifiers.")
        changes = tuple(
            (
                uid,
                tuple(
                    tuple(float(value) for value in row) for row in record["poses"][uid]
                ),
            )
            for uid in sorted(record["poses"])
        )
        if (
            _variant_id(
                record["parent_scene_id"],
                record["seed"],
                record["ordinal"],
                changes,
                metadata,
            )
            != record["variant_id"]
        ):
            raise ValueError("Scene variant identity does not match saved geometry.")
    for start in range(0, len(records), 1024):
        batch = records[start : start + 1024]
        actual = np.broadcast_to(
            np.array([prepared.source.poses[uid] for uid in uids]),
            (len(batch), len(uids), 4, 4),
        ).copy()
        for row, record in enumerate(batch):
            for uid, pose in record["poses"].items():
                matrix = np.asarray(pose, dtype=np.float64)
                if matrix.shape != (4, 4):
                    raise ValueError("Saved poses must be 4x4 matrices.")
                actual[row, uids.index(uid)] = matrix
        if not np.isfinite(actual).all() or not np.allclose(
            actual[:, :, 3], [0, 0, 0, 1], rtol=0, atol=1e-10
        ):
            raise ValueError("Saved poses must be finite homogeneous matrices.")
        delta = np.stack(
            [
                actual[:, i, :2, 3] - np.asarray(prepared.source.poses[uids[i]])[:2, 3]
                for i in roots
            ],
            axis=1,
        )
        for col, root in enumerate(roots):
            if root not in prepared._movable_roots and np.any(
                np.abs(delta[:, col]) > 1e-10
            ):
                raise ValueError("Saved layout moves a fixed group root.")
        relative = (
            np.stack(
                [
                    np.linalg.inv(actual[:, nodes[i]["parent"]]) @ actual[:, i]
                    for i in targets
                ],
                axis=1,
            )
            if targets
            else np.empty((len(batch), 0, 4, 4))
        )
        proxies = geometry.get("fixed_orientation_group_proxies")
        if proxies is not None:
            if targets or set(proxies) != {uids[root] for root in roots}:
                raise ValueError(
                    "Fixed-orientation proxy certificates require complete unrotated groups."
                )
            boxes = []
            for col, root in enumerate(roots):
                box = (
                    np.asarray(proxies[uids[root]], dtype=float)[None]
                    + delta[:, col, None]
                )
                if box.shape != (len(batch), 2, 2) or not np.isfinite(box).all():
                    raise ValueError("Invalid saved group proxy bounds.")
                if np.any(box[:, 0] < np.asarray(prepared._table)[0] - 1e-7) or np.any(
                    box[:, 1] > np.asarray(prepared._table)[1] + 1e-7
                ):
                    raise ValueError("Saved group proxy lies outside the support.")
                for other in boxes:
                    if np.any(
                        np.maximum(
                            box[:, 0] - other[:, 1], other[:, 0] - box[:, 1]
                        ).max(-1)
                        < policy["collision_margin_m"] - 1e-7
                    ):
                        raise ValueError("Saved group proxy violates clearance.")
                boxes.append(box)
        expected_uids, expected, _ = _numeric_geometry(
            geometry, policy, delta, relative, validate=proxies is None
        )
        for col, uid in enumerate(expected_uids):
            if not np.allclose(
                expected[:, col], actual[:, uids.index(uid)], rtol=0, atol=1e-8
            ):
                raise ValueError("Saved poses do not preserve the declared hierarchy.")


def save_scene_variants(
    result: SceneRandomizationBatch,
    output_dir: str | Path,
    *,
    storage: str = "packed",
    source_context: dict | None = None,
) -> Path:
    """Publish a v4 layout package, writing the completion manifest last.

    Args:
        result: Bounded generation result with its owned source and geometry.
        output_dir: New run directory; existing paths are not overwritten.
        storage: Individual ``json`` files or one ``packed`` candidate file.
        source_context: Optional inert host references; the shared reader never executes them.

    Returns:
        Absolute path to the completed run.
    """
    if storage not in ("json", "packed"):
        raise ValueError("storage must be json or packed.")
    prepared = result.prepared
    if _digest(prepared._geometry) != prepared.geometry_sha256:
        raise ValueError("Prepared geometry changed after generation.")
    records = _pool_records(result.variants)
    _validate_records(prepared, records)
    requested = (
        len(records) if result.requested_count is None else result.requested_count
    )
    budget = (
        max(result.attempts, requested)
        if result.max_attempts is None
        else result.max_attempts
    )
    if (
        result.status not in ("completed", "budget_exhausted")
        or type(requested) is not int
        or requested < 1
        or type(budget) is not int
        or budget < requested
        or not len(records) <= requested
        or not len(records) <= result.attempts <= budget
        or (result.status == "completed") != (len(records) == requested)
        or result.next_ordinal != result.start_ordinal + result.attempts
    ):
        raise ValueError(
            "Generation accounting differs from its requested count or budget."
        )
    root = Path(output_dir).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=False)
    source = prepared.source.to_dict()
    _write_atomic(root / "scene_input.json", source)
    _write_atomic(root / "policy.json", prepared.policy)
    prepared_data = {
        "geometry": prepared._geometry,
        "table": prepared._table,
        "movable_uids": prepared.movable_uids,
        "movable_roots": prepared._movable_roots,
    }
    _write_atomic(root / "prepared_geometry.json", prepared_data)
    listing = []
    chunks = (
        [("candidates.json", records)]
        if storage == "packed"
        else [(f"candidate_{i:06d}.json", [record]) for i, record in enumerate(records)]
    )
    for name, values in chunks:
        _write_atomic(root / "layouts" / name, values)
        listing.append({"path": name, "sha256": _digest(values)})
    _write_atomic(
        root / "run_result.json",
        {
            "schema_version": 4,
            "layout_storage": storage,
            "source_sha256": prepared.source.fingerprint,
            "policy_sha256": _digest(prepared.policy),
            "geometry_sha256": prepared.geometry_sha256,
            "prepared_sha256": _digest(prepared_data),
            "generated_count": len(records),
            "requested_count": requested,
            "max_attempts": budget,
            "attempts": result.attempts,
            "status": result.status,
            "seed": result.seed,
            "start_ordinal": result.start_ordinal,
            "next_ordinal": result.next_ordinal,
            "layouts": listing,
            "source_context": deepcopy(source_context or {}),
        },
    )
    return root


class _RecordVariants(Sequence[SceneVariant]):
    def __init__(self, records: list[dict]) -> None:
        self._records = deepcopy(records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(
        self, index: int | slice
    ) -> SceneVariant | tuple[SceneVariant, ...]:
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        record = self._records[index]
        return SceneVariant(
            record["parent_scene_id"],
            record["seed"],
            record["ordinal"],
            tuple(ScenePoseChange(uid, pose) for uid, pose in record["poses"].items()),
            tuple(tuple(item) for item in record["metadata"]),
        )


@dataclass(frozen=True)
class LoadedSceneVariants:
    """Fully validated source, prepared geometry and lazy saved proposals."""

    prepared: PreparedSceneRandomization
    variants: Sequence[SceneVariant]
    manifest: dict


def load_scene_variants(
    path: str | Path,
    *,
    scene_input: SceneLayoutInput | None = None,
) -> LoadedSceneVariants:
    """Validate every file, source, geometry and identity before exposing layouts.

    Args:
        path: Completed v4 run directory or its layouts subdirectory.
        scene_input: Optional independently supplied expected authored source.

    Returns:
        Validated layout sequence; reading never resamples candidates.
    """
    root = Path(path).expanduser().absolute()
    if root.name == "layouts":
        root = root.parent
    try:
        manifest = _read_json(root / "run_result.json")
        if manifest["schema_version"] != 4:
            raise ValueError(
                "Source-independent scene loading requires layout schema v4."
            )
        source_data = _read_json(root / "scene_input.json")
        source = SceneLayoutInput(**source_data)
        if source.fingerprint != manifest["source_sha256"] or (
            scene_input is not None and scene_input.fingerprint != source.fingerprint
        ):
            raise ValueError("Saved source differs from its expected snapshot.")
        policy = _read_json(root / "policy.json")
        if _digest(policy) != manifest["policy_sha256"]:
            raise ValueError("Saved policy changed.")
        if "prepared_sha256" in manifest:
            saved = _read_json(root / "prepared_geometry.json")
            if _digest(saved) != manifest["prepared_sha256"]:
                raise ValueError("Prepared geometry snapshot changed.")
            geometry = saved["geometry"]
            uids = [node["uid"] for node in geometry["nodes"]]
            if set(uids) != set(source.parent_by_uid) or len(set(uids)) != len(uids):
                raise ValueError("Cached geometry does not match its source entities.")
            for index, node in enumerate(geometry["nodes"]):
                uid = node["uid"]
                parent = source.parent_by_uid[uid]
                expected_index = (
                    -1 if parent == source.support_uid else uids.index(parent)
                )
                relative = (
                    np.asarray(source.poses[uid])
                    if expected_index < 0
                    else np.linalg.inv(source.poses[parent]) @ source.poses[uid]
                )
                points = source.vertices[uid]
                import itertools

                corners = np.array(
                    list(itertools.product(*zip(points.min(0), points.max(0))))
                )
                if (
                    node["parent"] != expected_index
                    or expected_index >= index
                    or not np.allclose(
                        node["relative_pose"], relative, rtol=0, atol=1e-10
                    )
                    or not np.allclose(node["corners"], corners, rtol=0, atol=1e-10)
                ):
                    raise ValueError(
                        "Cached hierarchy or geometry changed from its source."
                    )
            moving = tuple(saved["movable_uids"])
            if set(moving) - set(source.movable_entity_ids):
                raise ValueError(
                    "Cached layout permissions exceed the authored source."
                )
            bounds = np.asarray(saved["table"], dtype=float)
            if (
                bounds.shape != (2, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[0] >= bounds[1])
            ):
                raise ValueError("Invalid saved support bounds.")
            prepared = PreparedSceneRandomization(
                source,
                _policy_data(policy),
                manifest["geometry_sha256"],
                moving,
                geometry,
                tuple(tuple(row) for row in bounds),
                tuple(saved["movable_roots"]),
            )
        else:
            prepared = prepare_scene_randomization(source, policy)
        if _digest(prepared._geometry) != manifest["geometry_sha256"]:
            raise ValueError("Saved geometry certificate differs from its source.")
        records, listed = [], set()
        for entry in manifest["layouts"]:
            name = entry["path"]
            if not isinstance(name, str) or Path(name).name != name or name in listed:
                raise ValueError("Invalid or duplicate candidate path.")
            listed.add(name)
            values = _read_json(root / "layouts" / name)
            if not isinstance(values, list) or _digest(values) != entry["sha256"]:
                raise ValueError("Saved candidate container changed.")
            records.extend(values)
        actual = (
            {p.name for p in (root / "layouts").iterdir()}
            if (root / "layouts").exists()
            else set()
        )
        if actual != listed or len(records) != manifest["generated_count"]:
            raise ValueError(
                "Completion manifest must list every candidate exactly once."
            )
        if (
            any(
                type(manifest[key]) is not int or manifest[key] < 0
                for key in (
                    "attempts",
                    "seed",
                    "start_ordinal",
                    "next_ordinal",
                    "generated_count",
                )
            )
            or manifest["next_ordinal"]
            != manifest["start_ordinal"] + manifest["attempts"]
            or len(records) > manifest["attempts"]
            or manifest["status"] not in ("completed", "budget_exhausted")
        ):
            raise ValueError("Invalid generation accounting.")
        if "requested_count" in manifest:
            requested, budget = manifest["requested_count"], manifest["max_attempts"]
            if (
                type(requested) is not int
                or requested < 1
                or type(budget) is not int
                or budget < requested
                or len(records) > requested
                or manifest["attempts"] > budget
                or (manifest["status"] == "completed") != (len(records) == requested)
            ):
                raise ValueError(
                    "Saved completion or budget accounting is inconsistent."
                )
        if any(
            record["seed"] != manifest["seed"]
            or not manifest["start_ordinal"]
            <= record["ordinal"]
            < manifest["next_ordinal"]
            for record in records
        ):
            raise ValueError("Candidate ordinal or seed is outside its generation run.")
        _validate_records(prepared, records)
    except (KeyError, TypeError, AttributeError, OverflowError, IndexError) as exc:
        raise ValueError("Malformed saved scene layout package.") from exc
    return LoadedSceneVariants(prepared, _RecordVariants(records), deepcopy(manifest))
