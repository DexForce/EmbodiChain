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

"""Load saved group centers and cycle them through the existing scene preview."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import yaml

from embodichain.lab.sim.scene.expansion import ScenePoseChange, SceneVariant
from embodichain.lab.sim.scene.operations import SceneChangeSet, apply_changes
from embodichain.lab.sim.scene.query import SceneView

from .baseline import (
    _cached_base,
    _entry_pose,
    _read_json,
    _read_pose,
    _rigid_entries,
    _source_digest,
    _support_groups,
)
from .cfg import (
    _BATCH_GENERATOR,
    _INTRA_GENERATORS,
    _LEGACY_GENERATOR,
    _UniqueKeyLoader,
    _validate_policy,
)
from .persistence import _check_geometry

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager
    from embodichain.lab.visualization import VisualizationCfg

__all__ = ["LoadedLayouts", "load_layouts", "preview_layouts", "main"]


@dataclass(frozen=True)
class LoadedLayouts:
    """Validated source workspace, original poses and ordered saved variants.

    Args:
        output_root: Source Scene Engine workspace.
        baseline_poses: Original arena-local poses keyed by entity UID.
        variants: Reconstructed layouts in proposal order. Batch-generated
            layouts use an owned lazy sequence after full file/identity checks.
        names: Candidate names corresponding to ``variants``.
    """

    output_root: Path
    baseline_poses: dict[str, np.ndarray]
    variants: Sequence[SceneVariant]
    names: tuple[str, ...]


def load_layouts(layout_dir: str | Path) -> LoadedLayouts:
    """Read every saved candidate once and reconstruct all group member poses.

    Args:
        layout_dir: A completed run's ``layouts/`` directory containing JSON files.

    Returns:
        Source workspace, original entity poses and variants in proposal order.

    Raises:
        ValueError: Files are incomplete, changed or inconsistent with their
            source, baseline, policy, geometry or saved variant identities.
        OSError: Referenced files cannot be read.
    """
    directory = Path(layout_dir).expanduser().resolve()
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError("The layout directory contains no candidate JSON files.")
    try:
        return _load(directory, paths)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("Invalid saved scene layout data.") from exc


def _load(directory: Path, paths: list[Path]) -> LoadedLayouts:
    from embodichain.gen_sim.scene_engine.pipeline.utils.scene_importer import (
        SceneExportImporter,
    )

    run = directory.parent
    _, manifest = _read_json(run / "run_result.json")
    listed, records = _layout_records(directory, paths, manifest)
    for _, record in records:
        if (
            type(record["proposal_ordinal"]) is not int
            or record["proposal_ordinal"] < 0
        ):
            raise ValueError(
                "Candidate proposal ordinals must be nonnegative integers."
            )
    records.sort(key=lambda item: item[1]["proposal_ordinal"])
    first = records[0][1]
    base_path = _relative(directory, first["base_data"]["path"])
    export_root = _relative(directory, first["source_scene"]["export_dir"])
    if export_root.name != "scene_export":
        raise ValueError("Scene preview requires a scene_export directory.")
    config_bytes, config = _read_json(export_root / "scene_config.json")
    graph_bytes, graph_data = _read_json(export_root / "scene_graph.json")
    entries = _rigid_entries(config)
    graph = SceneExportImporter._scene_graph_from_data(graph_data)
    graph.validate()
    if set(graph.node_by_id()) != {entry["uid"] for entry in entries}:
        raise ValueError("Source graph and entity UIDs differ.")
    source = {
        "scene_id": config["scene_id"],
        "content_sha256": _source_digest(
            export_root,
            config_bytes,
            graph_bytes,
            [export_root / entry["shape"]["fpath"] for entry in entries],
        ),
        "export_dir": Path(os.path.relpath(export_root, base_path.parent)).as_posix(),
    }
    baseline = _cached_base(base_path, source, entries, _support_groups(graph))
    if baseline is None:
        raise ValueError(
            "Saved baseline or source no longer matches; regenerate layouts."
        )
    policy_path = _relative(run, manifest["policy_file"])
    try:
        policy = yaml.load(
            policy_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader
        )
    except yaml.YAMLError as exc:
        raise ValueError("Invalid resolved policy YAML.") from exc
    _validate_policy(policy)
    policy_hash = hashlib.sha256(
        json.dumps(
            policy, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        manifest["policy_sha256"] != policy_hash
        or manifest["generated_count"] != len(records)
        or manifest["seed"] != policy["seed"]
        or manifest["requested_count"] != policy["candidate_count"]
        or manifest["max_attempts"] != policy["max_attempts"]
        or type(manifest["attempts"]) is not int
        or not len(records) <= manifest["attempts"] <= policy["max_attempts"]
        or manifest["algorithm_version"]
        not in (
            _INTRA_GENERATORS
            if policy["schema_version"] == 3
            else (
                (_BATCH_GENERATOR,)
                if policy["schema_version"] == 2
                else (_LEGACY_GENERATOR,)
            )
        )
        or manifest["status"] not in {"completed", "budget_exhausted", "unsupported"}
    ):
        raise ValueError("Manifest and resolved policy are inconsistent.")
    _check_references(manifest, run, source, export_root, base_path, baseline)
    variants, names, ordinals, centers = [], [], set(), []
    hierarchical = manifest["algorithm_version"] in _INTRA_GENERATORS
    geometry = None
    if hierarchical:
        counts = manifest["solver_status_counts"]
        if (
            manifest["schema_version"] != 3
            or manifest.get("layout_storage") not in ("json", "packed")
            or set(counts) != {"solved", "no_solution"}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or counts["solved"] != len(records)
            or sum(counts.values()) != manifest["attempts"]
            or (
                manifest["status"] == "completed"
                and (
                    len(records) != policy["candidate_count"]
                    or records[-1][1]["proposal_ordinal"] + 1 != manifest["attempts"]
                )
            )
            or (
                manifest["status"] == "budget_exhausted"
                and (
                    len(records) >= policy["candidate_count"]
                    or manifest["attempts"] != policy["max_attempts"]
                )
            )
            or manifest["status"] == "unsupported"
        ):
            raise ValueError("Hierarchical run accounting is inconsistent.")
        from ._intra_geometry import _cache_path, _load_geometry
        from .baseline import PreparedSceneBaseline
        from .cfg import _decode_intra

        intra_cfg = _decode_intra(policy["intra_group"])
        geometry_path = _relative(run, manifest["support_geometry"]["path"])
        geometry = _load_geometry(geometry_path, baseline, intra_cfg)
        expected_path = _cache_path(
            PreparedSceneBaseline(base_path, True, baseline),
            intra_cfg,
            extractor=geometry["inputs"]["extractor"],
        )
        if geometry_path != expected_path:
            raise ValueError("Unexpected support cache reference.")
        if manifest["support_geometry"]["data_sha256"] != geometry["data_sha256"]:
            raise ValueError("Support cache digest differs from saved run.")
    batched = manifest["algorithm_version"] == _BATCH_GENERATOR
    for index, (path, record) in enumerate(records):
        _check_references(record, directory, source, export_root, base_path, baseline)
        ordinal = record["proposal_ordinal"]
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != (3 if hierarchical else 1)
            or record["coordinate_frame"] != "local_arena_z_up"
            or record["geometry_model"]
            != (
                "hierarchical_support_grid_xy_yaw"
                if hierarchical
                else "fixed_orientation_aabb"
            )
            or record["algorithm_version"] != manifest["algorithm_version"]
            or type(record["seed"]) is not int
            or record["seed"] != policy["seed"]
            or record["policy_sha256"] != policy_hash
            or _relative(directory, record["policy_file"]) != policy_path
            or ordinal in ordinals
            or ordinal >= manifest["attempts"]
            or record["candidate_id"] != f"candidate_{ordinal:06d}"
            or path.stem != record["candidate_id"]
            or type(record["variant_ordinal"]) is not int
            or record["variant_ordinal"] != index
            or any(
                record[key] != listed[path][key]
                for key in ("candidate_id", "proposal_ordinal", "variant_id")
            )
        ):
            raise ValueError(f"Candidate metadata does not match its run: {path.name}.")
        if hierarchical:
            if (
                _relative(directory, record["support_geometry"]["path"])
                != geometry_path
                or record["support_geometry"]["data_sha256"] != geometry["data_sha256"]
            ):
                raise ValueError(
                    "Candidate support cache reference differs from its run."
                )
        elif batched:
            group_centers = _record_centers(record, baseline)
            _check_variant_metadata(record, baseline)
            centers.append(
                [group_centers[group["root_id"]] for group in baseline["groups"]]
            )
        else:
            variants.append(_variant(record, baseline, policy["collision_margin_m"]))
        names.append(record["candidate_id"])
        ordinals.add(ordinal)
    if hierarchical:
        from ._intra import _LoadedIntraVariants

        variants = _LoadedIntraVariants(
            baseline, geometry, [record for _, record in records], policy
        )
    elif batched:
        from ._batch import _check_centers, _LoadedVariants

        numeric_centers = np.asarray(centers, dtype=np.float64)
        _check_centers(baseline, numeric_centers, policy["collision_margin_m"])
        variants = _LoadedVariants(
            baseline, numeric_centers, [record for _, record in records], policy["seed"]
        )
    else:
        variants = tuple(variants)
    return LoadedLayouts(
        export_root.parent,
        {entry["uid"]: _entry_pose(entry) for entry in entries},
        variants,
        tuple(names),
    )


def _layout_records(
    directory: Path, paths: list[Path], manifest: dict
) -> tuple[dict, list[tuple[Path, dict]]]:
    """Read legacy files or a single indexed atomic candidate container."""
    run = directory.parent
    listing = manifest["layouts"]
    version = manifest["schema_version"]
    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unsupported layout manifest schema.")
    if version == 1 or (version == 3 and manifest.get("layout_storage") == "json"):
        listed = {_relative(run, item["path"]): item for item in listing}
        if set(listed) != set(paths) or len(listed) != len(listing):
            raise ValueError("Manifest must list every candidate file exactly once.")
        return listed, [(path, _read_json(path)[1]) for path in paths]
    packed_path = directory / "candidates.json"
    if (
        manifest.get("layout_storage") != "packed"
        or _relative(run, manifest["candidate_file"]) != packed_path
        or set(paths) != {packed_path}
    ):
        raise ValueError(
            "Packed manifest must reference exactly one candidate container."
        )
    _, packed = _read_json(packed_path)
    if type(packed["schema_version"]) is not int or packed["schema_version"] != 1:
        raise ValueError("Unsupported candidate container schema.")
    candidates = packed["candidates"]
    if (
        not isinstance(candidates, list)
        or not candidates
        or len(candidates) != len(listing)
    ):
        raise ValueError("Packed candidates and manifest listing differ.")
    listed, records = {}, []
    for index, (item, candidate) in enumerate(zip(listing, candidates, strict=True)):
        if (
            type(item["index"]) is not int
            or item["index"] != index
            or _relative(run, item["path"]) != packed_path
        ):
            raise ValueError("Packed candidate indices must be complete and ordered.")
        path = directory / f"{candidate['candidate_id']}.json"
        if path in listed:
            raise ValueError("Packed candidate identities must be unique.")
        listed[path] = item
        records.append((path, candidate))
    return listed, records


def _relative(parent: Path, value: object) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("Saved file references must be nonempty relative paths.")
    return (parent / value).resolve()


def _check_references(
    record: dict,
    parent: Path,
    source: dict,
    export_root: Path,
    base_path: Path,
    baseline: dict,
) -> None:
    if (
        record["source_scene"]["scene_id"] != source["scene_id"]
        or record["source_scene"]["content_sha256"] != source["content_sha256"]
        or _relative(parent, record["source_scene"]["export_dir"]) != export_root
        or _relative(parent, record["base_data"]["path"]) != base_path
        or record["base_data"]["data_sha256"] != baseline["data_sha256"]
    ):
        raise ValueError(
            "Layouts must reference the same unchanged source and baseline."
        )


def _record_centers(record: dict, baseline: dict) -> dict[str, np.ndarray]:
    """Decode a complete group mapping before geometry validation."""
    groups = record["groups"]
    centers = {}
    for group in groups:
        center = group["center_xy"]
        if (
            set(group) != {"root_id", "center_xy", "yaw_delta_deg"}
            or group["root_id"] in centers
            or type(group["yaw_delta_deg"]) not in (int, float)
            or group["yaw_delta_deg"] != 0
            or not isinstance(center, list)
            or len(center) != 2
            or any(type(value) not in (int, float) for value in center)
            or not np.isfinite(center).all()
        ):
            raise ValueError(
                "Saved groups require unique IDs, finite XY centers and zero yaw."
            )
        centers[group["root_id"]] = np.asarray(center)
    if set(centers) != {group["root_id"] for group in baseline["groups"]}:
        raise ValueError(
            "Every baseline group must appear exactly once in each layout."
        )
    return centers


def _check_variant_metadata(record: dict, baseline: dict) -> None:
    """Validate source and proposal provenance before materialization."""
    metadata = record["variant_metadata"]
    if (
        metadata.get("generator") != record["algorithm_version"]
        or metadata.get("baseline_sha256") != baseline["data_sha256"]
        or metadata.get("proposal_ordinal") != str(record["proposal_ordinal"])
    ):
        raise ValueError("Saved variant provenance is inconsistent.")


def _variant(record: dict, baseline: dict, margin: float) -> SceneVariant:
    centers = _record_centers(record, baseline)
    _check_geometry(baseline, centers, margin)
    changes = []
    for group in baseline["groups"]:
        uid = group["root_id"]
        pose = _read_pose(group["root_pose"])
        pose[:2, 3] += centers[uid] - group["aabb_2d"]["center_xy"]
        changes.append(ScenePoseChange(uid, pose.tolist()))
        changes.extend(
            ScenePoseChange(
                member["entity_id"],
                (pose @ _read_pose(member["pose_in_root"])).tolist(),
            )
            for member in group["members"]
        )
    _check_variant_metadata(record, baseline)
    variant = SceneVariant(
        baseline["source_scene"]["content_sha256"],
        record["seed"],
        record["variant_ordinal"],
        tuple(changes),
        tuple(record["variant_metadata"].items()),
    )
    if variant.variant_id != record["variant_id"]:
        raise ValueError(
            "Reconstructed variant identity does not match the saved layout."
        )
    return variant


class _Playback:
    def __init__(self, loaded: LoadedLayouts, interval: float) -> None:
        self.loaded = loaded
        self.interval = interval
        self.view: SceneView | None = None
        self.index = -1
        self.deadline = 0.0

    def update(self, sim: SimulationManager) -> bool:
        now = time.monotonic()
        if self.view is None:
            self.view = SceneView(sim)
        elif now < self.deadline:
            return False
        else:
            self.index = (self.index + 1) % (len(self.loaded.variants) + 1)
        if self.index == -1 or self.index == len(self.loaded.variants):
            poses, name = self.loaded.baseline_poses, "original"
        else:
            poses = {
                change.entity_uid: change.arena_pose
                for change in self.loaded.variants[self.index].pose_changes
            }
            name = self.loaded.names[self.index]
        apply_changes(self.view, SceneChangeSet(poses, tuple(poses)))
        self.deadline = now + self.interval
        print(f"Layout preview: {name}")
        return True


def preview_layouts(
    layout_dir: str | Path,
    *,
    interval: float = 3.0,
    device: str = "cpu",
    headless: bool = False,
    visualization: VisualizationCfg | None = None,
) -> None:
    """Cycle original and saved layouts in the existing scene preview window.

    Args:
        layout_dir: Completed run's directory of candidate JSON files.
        interval: Positive finite seconds to display each layout.
        device: Existing preview simulation device, normally ``cpu``.
        headless: Load without a native window; Viser follows its existing mode.
        visualization: Existing preview visualization configuration.
    """
    if (
        type(interval) not in (int, float)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError("interval must be finite and greater than zero.")
    loaded = load_layouts(layout_dir)
    from embodichain.gen_sim.scene_engine.cli.preview import preview_scene_export

    preview_scene_export(
        output_root=loaded.output_root,
        device=device,
        headless=headless,
        visualization=visualization,
        joint_control=False,
        on_frame=_Playback(loaded, float(interval)).update,
        advance_physics=False,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Parse standalone saved-layout preview arguments.

    Args:
        argv: Explicit arguments, or process arguments when omitted.
    """
    from embodichain.lab.visualization import (
        add_viser_args_to_parser,
        visualization_cfg_from_args,
    )

    parser = argparse.ArgumentParser(
        description="Preview every JSON layout in a saved layouts directory."
    )
    parser.add_argument(
        "--layout-dir",
        type=Path,
        required=True,
        help="Saved run's layouts/ directory containing candidate JSON files.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds to display the original scene and each layout (default: 3).",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--headless", action="store_true")
    add_viser_args_to_parser(parser)
    args = parser.parse_args(argv)
    try:
        preview_layouts(
            args.layout_dir,
            interval=args.interval,
            device=args.device,
            headless=args.headless,
            visualization=visualization_cfg_from_args(args),
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
