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

"""Read-only inputs and controls for the native expansion-result browser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation

from .comparison import MeasuredTrajectory
from .persistence import ArtifactEpisodeSink, load_expansion_episode

__all__: list[str] = []


def pose_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


@dataclass
class PreviewClip:
    measured: MeasuredTrajectory
    space: str
    affordance: str
    variant: str
    targets: np.ndarray
    phase_names: tuple[str, ...]
    anchors: np.ndarray
    candidate_ids: tuple[str, ...]
    label: str = ""
    links: np.ndarray | None = None
    tips: np.ndarray | None = None


def annotations(metadata: Any, measured: MeasuredTrajectory) -> dict[str, Any]:
    records = metadata.get("expansion", ())
    env_ids = {r["env_id"] for r in records if r.get("env_id") is not None}
    if metadata.get("source_env_id") is not None:
        source_row = metadata["source_env_id"]
        if type(source_row) is not int or source_row < 0:
            raise ValueError("source_env_id must be a nonnegative integer")
        env_ids.add(source_row)
    if len(env_ids) > 1:
        raise ValueError("Episode expansion annotations refer to multiple rows")
    env_id = next(iter(env_ids), None)
    branches, candidates, targets, phases = [], [], [], []
    phase_safe = True
    for segment in metadata.get("program_segments", ()):
        runtime = segment.get("runtime", segment.get("metadata", {}).get("runtime", {}))
        for call in runtime.get("calls", ()):
            for event in call.get("events", ()):
                if event.get("kind") == "trajectory_segment_entered":
                    mask = event.get("env_mask")
                    if mask:
                        if env_id is not None:
                            index = 0 if len(mask) == 1 else env_id
                            if index >= len(mask):
                                phase_safe = False
                                continue
                            if not mask[index]:
                                continue
                        elif len(set(mask)) > 1:
                            phase_safe = False
                            continue
                        elif not any(mask):
                            continue
                    phases.append(
                        (
                            str(call["call"]["call_kind"])
                            + "/"
                            + str(event["segment_name"]),
                            float(event["timestamp"]),
                        )
                    )
            # Only the successful final planning attempt describes the executed target.
            attempts = call.get("plan_attempts", ())
            if not attempts:
                continue
            sample = (
                attempts[-1]
                .get("planner_diagnostics", {})
                .get("metadata", {})
                .get("affordance_sample", {})
                .get("grasp")
            )
            if not sample:
                continue
            rows = list(sample["env_ids"])
            row = (
                rows.index(env_id)
                if env_id in rows
                else 0 if env_id is None and len(rows) == 1 else None
            )
            if row is None:
                # Legacy numeric evidence remains usable; never guess another row's grasp.
                continue
            if not sample["success"][row]:
                continue
            sampling = sample.get("sampling") or {}
            overrides = sampling.get("branch_overrides")
            branch = overrides[row] if overrides else sampling.get("branch_override")
            if branch is None and sampling.get("count") == 1:
                branch = 0
            branches.append("unrecorded" if branch is None else str(branch))
            candidates.append(str(sample["candidate_ids"][row]))
            uid = (
                call.get("call", {})
                .get("arguments", {})
                .get("object", {})
                .get("entity_id")
            )
            if uid not in measured.object_poses:
                continue
            reference = np.asarray(sample["reference_poses"][row], dtype=float)
            selected = np.asarray(sample["selected_poses"][row], dtype=float)
            if (
                reference.shape != (4, 4)
                or selected.shape != (4, 4)
                or not np.isfinite([reference, selected]).all()
            ):
                raise ValueError("Invalid recorded grasp transform")
            targets.append(
                pose_matrix(measured.object_poses[uid][0])
                @ np.linalg.inv(reference)
                @ selected
            )
    phases = [v for i, v in enumerate(phases) if i == 0 or v != phases[i - 1]]
    anchors = np.array([p[1] for p in phases] + [measured.timestamps[-1]])
    if (
        not phase_safe
        or not phases
        or anchors[0] < 0
        or not np.isfinite(anchors).all()
        or not np.all(np.diff(anchors) > 0)
    ):
        phases = []
        anchors = np.empty(0)
    variants = [
        str(r["trajectory_selected"])
        for r in records
        if r.get("trajectory_requested") is not None
    ]
    if not variants:
        variants = [
            str(r["operator"])
            for r in records
            if r.get("operator")
            in {"joint_residual", "via_points", "nullspace_residual"}
        ]
        variants = list(dict.fromkeys(variants))
    return dict(
        affordance="/".join(branches) or "unrecorded",
        variant="/".join(variants) or "unrecorded",
        targets=np.asarray(targets, dtype=float).reshape(-1, 4, 4),
        phase_names=tuple(p[0] for p in phases),
        anchors=anchors,
        candidate_ids=tuple(candidates),
    )


def _object_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objects = {}
    for obj in [*(cfg.get("rigid_object") or []), *(cfg.get("background") or [])]:
        uid = obj.get("uid")
        if not isinstance(uid, str) or not uid or uid in objects:
            raise ValueError("Preview requires unique recorded rigid object IDs")
        if obj.get("visuals") is not None or obj.get("visual_shape"):
            raise ValueError(
                "Preview does not yet support explicit rigid visual replacements"
            )
        shape, source = obj.get("shape") or {}, obj.get("source")
        if source:
            if not isinstance(source, dict) or not source.get("file_path") or shape:
                raise ValueError("Preview requires an unambiguous source object")
            filename = source["file_path"]
        elif shape.get("shape_type") == "Mesh":
            filename = shape.get("fpath")
        elif shape.get("shape_type") in {
            "Cube",
            "Sphere",
            "Cylinder",
            "Capsule",
            "Cone",
        }:
            filename = None
        else:
            raise ValueError("Unsupported recorded rigid visual geometry")
        if (source or shape.get("shape_type") == "Mesh") and (
            not filename or not Path(filename).expanduser().is_file()
        ):
            raise ValueError(f"Recorded object asset is unavailable: {filename}")
        scale = obj.get("body_scale")
        if scale is not None and (
            np.asarray(scale).shape != (3,)
            or not np.isfinite(scale).all()
            or np.any(np.asarray(scale) <= 0)
        ):
            raise ValueError("Invalid recorded object scale")
        objects[uid] = {
            key: obj[key]
            for key in ("shape", "source", "body_scale", "body_type")
            if key in obj
        }
    return objects


def _scene(job: dict[str, Any]) -> tuple[dict, dict[str, tuple[float, ...] | None]]:
    cfg = job.get("resolved_deployment")
    if not isinstance(cfg, dict) or not isinstance(cfg.get("robot"), dict):
        raise ValueError(
            "Preview requires job.json with a resolved single-robot deployment"
        )
    unsupported = (
        "articulation",
        "rigid_object_group",
        "cable_object",
        "fluid_object",
        "soft_object",
        "cloth_object",
        "deformable_object",
    )
    if any(cfg.get(k) for k in unsupported) or job.get("config", {}).get("scene_atlas"):
        raise ValueError(
            "Preview requires a fixed layout, one URDF robot and recorded rigid objects"
        )
    robot = cfg["robot"]
    if robot.get("visuals") or robot.get("link_visuals"):
        raise ValueError("Preview requires the URDF-owned robot visual geometry")
    if not np.allclose(robot.get("body_scale", [1, 1, 1]), [1, 1, 1]):
        raise ValueError("Scaled robot assets are not supported by this preview")
    boxes = {}
    for uid, obj in _object_specs(cfg).items():
        shape = obj.get("shape") or {}
        if shape.get("shape_type") != "Cube":
            boxes[uid] = None
            continue
        size = np.asarray(shape["size"], float) * np.asarray(
            obj.get("body_scale") or [1, 1, 1], float
        )
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("Invalid recorded Cube dimensions")
        boxes[uid] = tuple(size)
    return robot, boxes


@dataclass
class PreviewData:
    clips: list[PreviewClip]
    reference: PreviewClip | None
    asset: Path
    eef_link: str
    boxes: dict[str, tuple[float, ...] | None]
    ground_visible: bool = True
    coverage: dict[str, Any] | None = None
    object_specs: dict[str, dict[str, Any]] | None = None

    @property
    def all_clips(self) -> list[PreviewClip]:
        return self.clips + ([self.reference] if self.reference is not None else [])

    @property
    def phase_alignment(self) -> bool:
        signature = self.clips[0].phase_names
        return bool(signature) and all(
            c.phase_names == signature for c in self.all_clips
        )


def load_preview(
    outputs: list[Path],
    reference: Path | None = None,
    *,
    robot_asset: Path | None = None,
    eef_link: str | None = None,
    max_episodes: int = 32,
) -> PreviewData:
    if not outputs or not 1 <= max_episodes <= 64:
        raise ValueError("Provide result directories and max_episodes in [1,64]")
    coverage = None
    if len(outputs) == 1 and (Path(outputs[0]) / "preview-plan.json").is_file():
        root = Path(outputs[0]).resolve()
        coverage = json.loads((root / "preview-plan.json").read_text())
        outputs = []
        for record in coverage["records"]:
            path = (root / record["result"]).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Preview result path escapes the manifest directory")
            if record["confirmed"] > 0:
                outputs.append(path)
        if not outputs:
            raise ValueError(
                "Preview plan has no qualified trajectories; inspect its rejection/unrun records"
            )
    paths = [Path(p).expanduser().resolve() for p in outputs]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate output directory")
    if reference is not None:
        paths.append(Path(reference).expanduser().resolve())
    clips = []
    reference_clip = None
    asset = None
    boxes = None
    robot_spec = None
    object_specs = None
    end = None
    ground = True
    seen = set()
    frames = 0
    for group, path in enumerate(paths):
        job_path = path / "job.json"
        if not job_path.is_file():
            raise ValueError(f"Not an expansion job output: {path} (missing job.json)")
        job = json.loads(job_path.read_text())
        robot, shapes = _scene(job)
        recorded_specs = _object_specs(job["resolved_deployment"])
        if object_specs is not None and recorded_specs != object_specs:
            raise ValueError(
                "Result directories disagree about recorded object geometry"
            )
        object_specs = recorded_specs
        spec = {k: robot.get(k) for k in ("uid", "control_parts", "solver_cfg")}
        if robot_spec is not None and (spec != robot_spec or shapes != boxes):
            raise ValueError(
                "Result directories must share robot/control topology and object geometry"
            )
        robot_spec = spec
        boxes = shapes
        path_value = robot.get("fpath") or (robot.get("urdf_cfg") or {}).get("fpath")
        resolved_asset = Path(robot_asset or path_value or "").expanduser().resolve()
        if not resolved_asset.is_file() or resolved_asset.suffix.lower() != ".urdf":
            raise ValueError(
                "Recorded URDF is unavailable; supply --robot-asset with matching visuals"
            )
        # URDF assembly comments may differ; compare the actual XML model.
        import xml.etree.ElementTree as ET

        if asset is not None and ET.tostring(ET.parse(asset).getroot()) != ET.tostring(
            ET.parse(resolved_asset).getroot()
        ):
            raise ValueError("Output directories reference different robot models")
        asset = resolved_asset
        ends = {
            s.get("end_link_name")
            for s in robot.get("solver_cfg", {}).values()
            if s.get("end_link_name")
        }
        inferred = eef_link or (next(iter(ends)) if len(ends) == 1 else None)
        if not inferred:
            raise ValueError(
                "Specify --eef-link when the recorded solver endpoint is ambiguous"
            )
        end = inferred
        if group == 0:
            ground = job["resolved_deployment"].get("ground_plane_visible") is not False
        elif ground != (
            job["resolved_deployment"].get("ground_plane_visible") is not False
        ):
            raise ValueError("Outputs disagree about ground visibility")
        receipts = [r for r in ArtifactEpisodeSink(path).inspect() if r.confirmed]
        is_reference = reference is not None and group == len(outputs)
        if not receipts or (is_reference and len(receipts) != 1):
            raise ValueError(
                "Each output needs confirmed episodes; reference must contain exactly one"
            )
        if len(clips) + len(receipts) > (max_episodes + int(is_reference)):
            raise ValueError(
                "Preview episode budget exceeded; select smaller output directories"
            )
        for receipt in receipts:
            episode = load_expansion_episode(receipt.storage_id)
            if not episode.validation.accepted:
                raise ValueError("Unqualified episode cannot be previewed")
            if episode.episode_id in seen:
                raise ValueError("Duplicate episode across inputs/reference")
            seen.add(episode.episode_id)
            measured = MeasuredTrajectory.load_episode(
                receipt.storage_id, robot_uid=robot["uid"]
            )
            if len(measured.timestamps) != len(episode.actions) + 1 or set(
                measured.object_poses
            ) != set(boxes):
                raise ValueError(
                    "Incomplete T+1 scene state or unsupported scene entities"
                )
            if (
                any(
                    x["kind"] not in ("robot", "rigid_object")
                    for x in episode.metadata["initial_state"]
                )
                or sum(x["kind"] == "robot" for x in episode.metadata["initial_state"])
                != 1
            ):
                raise ValueError(
                    "Preview supports one robot and recorded rigid objects only"
                )
            frames += len(measured.timestamps)
            if frames > 100_000:
                raise ValueError("Preview exceeds the 100000 measured-state budget")
            clip = PreviewClip(
                measured=measured,
                space=f"S{group+1} · {path.name}",
                **annotations(episode.metadata, measured),
            )
            if is_reference:
                reference_clip = clip
                clip.label = "Reference"
            else:
                clips.append(clip)
                clip.label = f"C-{len(clips):03d}"
    return PreviewData(
        clips, reference_clip, asset, end, boxes, ground, coverage, object_specs
    )


class PreviewState:
    def __init__(self, data: PreviewData) -> None:
        self.data = data
        self.values = dict(
            space="all",
            aff="all",
            variant="all",
            selected=0,
            progress=0.0,
            playing=False,
            reference=data.reference is not None,
            ghosts=True,
            paths=True,
            targets=True,
            positions=True,
            align="phase" if data.phase_alignment else "time",
            yaw=-2.15,
            elevation=0.55,
            distance=1.65,
        )
        self.options = {
            "space": sorted({c.space for c in data.clips}),
            "aff": sorted({c.affordance for c in data.clips}),
            "variant": sorted({c.variant for c in data.clips}),
            "align": ["time"] + (["phase"] if data.phase_alignment else []),
        }

    def validate(self, update: Any) -> dict[str, Any]:
        if (
            not isinstance(update, dict)
            or not update
            or set(update) - self.values.keys()
        ):
            raise ValueError("Invalid preview controls")
        bounds = {
            "selected": (0, len(self.data.clips) - 1),
            "progress": (0, 1),
            "yaw": (-20, 20),
            "elevation": (0.1, 1.5),
            "distance": (0.2, 20),
        }
        for key, value in update.items():
            if key in self.options:
                if value not in self.options[key] + ([] if key == "align" else ["all"]):
                    raise ValueError(key)
            elif key in bounds:
                if (
                    type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not bounds[key][0] <= value <= bounds[key][1]
                    or (key == "selected" and type(value) is not int)
                ):
                    raise ValueError(key)
            elif type(value) is not bool:
                raise ValueError(key)
        return dict(update)

    def apply(self, update: dict[str, Any]) -> None:
        self.values.update(self.validate(update))
        visible = self.visible()
        if visible and self.values["selected"] not in visible:
            self.values["selected"] = visible[0]
        if not visible:
            self.values["playing"] = False

    def visible(self) -> list[int]:
        v = self.values
        return [
            i
            for i, c in enumerate(self.data.clips)
            if all(
                v[k] == "all" or v[k] == actual
                for k, actual in [
                    ("space", c.space),
                    ("aff", c.affordance),
                    ("variant", c.variant),
                ]
            )
        ]

    def frame_at(self, clip: PreviewClip) -> int:
        p = self.values["progress"]
        if self.values["align"] == "phase":
            n = len(clip.anchors) - 1
            segment = min(n - 1, int(p * n))
            t = np.interp(p * n - segment, [0, 1], clip.anchors[segment : segment + 2])
        else:
            t = p * self.duration
        return clip.measured.frame_at(t)

    @property
    def duration(self) -> float:
        return max(float(c.measured.timestamps[-1]) for c in self.data.all_clips)

    def snapshot(self) -> dict[str, Any]:
        visible = self.visible()
        c = self.data.clips[self.values["selected"]]
        f = self.frame_at(c)
        return {
            **self.values,
            "visible": visible,
            "count": len(visible),
            "total": len(self.data.clips),
            "options": self.options,
            "has_reference": self.data.reference is not None,
            "coverage": self.data.coverage,
            "label": c.label if visible else "No matches",
            "source_frame": f if visible else None,
            "source_frames": len(c.measured.timestamps) if visible else 0,
            "time": float(c.measured.timestamps[f]) if visible else None,
            "episode": c.measured.name if visible else "",
            "candidate": ", ".join(c.candidate_ids) or "unrecorded",
            "phase": (
                next(
                    (
                        name
                        for name, t in reversed(list(zip(c.phase_names, c.anchors)))
                        if t <= c.measured.timestamps[f]
                    ),
                    "unrecorded",
                )
                if visible
                else ""
            ),
            "clips": [
                dict(
                    index=i,
                    label=c.label,
                    space=c.space,
                    aff=c.affordance,
                    variant=c.variant,
                )
                for i, c in enumerate(self.data.clips)
            ],
        }
