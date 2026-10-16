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
from .cfg import _GENERATION_FIELDS, _UniqueKeyLoader, _validate_generation_parameters
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
        variants: Reconstructed layouts in proposal order.
        names: Candidate names corresponding to ``variants``.
    """

    output_root: Path
    baseline_poses: dict[str, np.ndarray]
    variants: tuple[SceneVariant, ...]
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
    listing = manifest["layouts"]
    listed = {_relative(run, item["path"]): item for item in listing}
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or set(listed) != set(paths)
        or len(listed) != len(listing)
    ):
        raise ValueError("Manifest must list every candidate file exactly once.")
    records = [(path, _read_json(path)[1]) for path in paths]
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
    if (
        set(policy) != {"schema_version", *_GENERATION_FIELDS}
        or type(policy["schema_version"]) is not int
        or policy["schema_version"] != 1
    ):
        raise ValueError("Unsupported resolved policy schema.")
    _validate_generation_parameters(**{key: policy[key] for key in _GENERATION_FIELDS})
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
        or manifest["algorithm_version"] != "scene_engine.group_randomization/v1"
        or manifest["status"] not in {"completed", "budget_exhausted", "unsupported"}
    ):
        raise ValueError("Manifest and resolved policy are inconsistent.")
    _check_references(manifest, run, source, export_root, base_path, baseline)
    variants, names, ordinals = [], [], set()
    for index, (path, record) in enumerate(records):
        _check_references(record, directory, source, export_root, base_path, baseline)
        ordinal = record["proposal_ordinal"]
        if (
            type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["coordinate_frame"] != "local_arena_z_up"
            or record["geometry_model"] != "fixed_orientation_aabb"
            or record["algorithm_version"] != manifest["algorithm_version"]
            or record["seed"] != policy["seed"]
            or record["policy_sha256"] != policy_hash
            or _relative(directory, record["policy_file"]) != policy_path
            or ordinal in ordinals
            or ordinal >= manifest["attempts"]
            or record["candidate_id"] != f"candidate_{ordinal:06d}"
            or path.stem != record["candidate_id"]
            or record["variant_ordinal"] != index
            or any(
                record[key] != listed[path][key]
                for key in ("candidate_id", "proposal_ordinal", "variant_id")
            )
        ):
            raise ValueError(f"Candidate metadata does not match its run: {path.name}.")
        variant = _variant(record, baseline, policy["collision_margin_m"])
        variants.append(variant)
        names.append(record["candidate_id"])
        ordinals.add(ordinal)
    return LoadedLayouts(
        export_root.parent,
        {entry["uid"]: _entry_pose(entry) for entry in entries},
        tuple(variants),
        tuple(names),
    )


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


def _variant(record: dict, baseline: dict, margin: float) -> SceneVariant:
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
    metadata = record["variant_metadata"]
    if (
        metadata.get("generator") != record["algorithm_version"]
        or metadata.get("baseline_sha256") != baseline["data_sha256"]
        or metadata.get("proposal_ordinal") != str(record["proposal_ordinal"])
    ):
        raise ValueError("Saved variant provenance is inconsistent.")
    variant = SceneVariant(
        baseline["source_scene"]["content_sha256"],
        record["seed"],
        record["variant_ordinal"],
        tuple(changes),
        tuple(metadata.items()),
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
        apply_changes(
            self.view, SceneChangeSet(poses, tuple(self.loaded.baseline_poses))
        )
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
