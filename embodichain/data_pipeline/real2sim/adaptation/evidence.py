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
"""Prepare bounded visual and numerical evidence without model inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw

from embodichain.data_pipeline.real2sim.adaptation.inputs import RobotModel
from embodichain.data_pipeline.real2sim.adaptation.trajectory_tools import (
    TrajectoryTools,
)
from embodichain.data_pipeline.real2sim.adaptation.video_fusion import (
    decode_video_frames,
)

__all__ = ["prepare_trajectory_evidence"]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _sheet(
    video: Path,
    frames: list[int],
    samples: list[int],
    times: np.ndarray,
    label: str,
    destination: Path,
) -> None:
    images = decode_video_frames(video, frames)
    sheet = Image.new("RGB", (1280, ((len(samples) + 3) // 4) * 264), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (sample, frame) in enumerate(zip(samples, frames)):
        x, y = index % 4 * 320, index // 4 * 264
        picture = Image.fromarray(images[frame])
        picture.thumbnail((320, 240))
        sheet.paste(picture, (x, y + 24))
        draw.text(
            (x + 3, y + 3),
            f"{label} sample {sample} / {times[sample]:.2f}s",
            fill="black",
        )
    sheet.save(destination, quality=90)


def prepare_trajectory_evidence(
    stage: str,
    context: dict,
    output: Path,
    *,
    annotation: dict | None = None,
    phases: list[dict] | None = None,
) -> tuple[dict, list[Path]]:
    """Read source/replay observations and prepare images for semantic review.

    Args:
        stage: Annotate source observations or review a completed replay.
        context: Explicit source, robot, scene and replay artifact paths.
        output: Directory for evidence JSON and contact sheets.
        annotation: Host-validated Task Program annotation for review.
        phases: Inclusive source intervals projected by the host from annotation.

    Returns:
        Numerical observations and attached image paths; no task decisions.
    """
    if stage not in ("annotate", "review"):
        raise ValueError(f"Unknown evidence stage: {stage}")
    if stage == "review" and (annotation is None or phases is None):
        raise ValueError("Review evidence requires annotation and bound phases")
    output.mkdir(parents=True, exist_ok=True)
    phases = [] if phases is None else phases
    model_data = json.loads(Path(context["robot_model_path"]).read_text())
    model_data["urdf"] = Path(model_data["urdf"])
    for key in ("base_pose_world", "flange_to_tcp"):
        model_data[key] = np.asarray(model_data[key])
    model = RobotModel(**model_data)
    tools = TrajectoryTools(model)
    with np.load(context["source_trajectory"], allow_pickle=False) as data:
        original = data["qpos"].copy()
        times = data["timestamps"] - data["timestamps"][0]
        if context["time_unit"] == "ms":
            times = times / 1000
        camera = context["source_videos"][context["primary_camera"]]
        source_frames = data[camera["frame_indices_key"]].copy()
    samples = set(np.linspace(0, len(times) - 1, 24).round().astype(int).tolist())
    if annotation:
        for phase in phases:
            samples.update(
                [
                    phase["start_frame"],
                    (phase["start_frame"] + phase["end_frame"]) // 2,
                    phase["end_frame"],
                ]
            )
    samples = sorted(samples)
    # Cap inference evidence independently of video duration and scene complexity.
    if len(samples) > 60:
        samples = [
            samples[i] for i in np.linspace(0, len(samples) - 1, 60).round().astype(int)
        ]
    images = [output / "source.jpg"]
    _sheet(
        Path(camera["video"]),
        [int(source_frames[i]) for i in samples],
        samples,
        times,
        "REAL",
        images[0],
    )
    scene = json.loads(Path(context["scene"]).read_text())
    evidence = dict(
        instruction=context["instruction"],
        sample_count=len(times),
        arms=list(model.side_names),
        correction_limits=context["correction_limits"],
        correction_failure=context.get("correction_failure"),
        objects=[
            {k: o[k] for k in ("id", "name", "category", "bounds") if k in o}
            for o in scene["objects"]
        ],
        annotation=annotation,
        phases=phases,
        samples=samples,
        times_s=np.round(times[samples], 4),
        coordinate_convention="World coordinates, metres; rotation vector in world axes, radians; opening is full aperture.",
    )
    current = original
    if stage == "review":
        with np.load(context["current_trajectory"], allow_pickle=False) as data:
            current = data["qpos"].copy()
        rollout = Path(context["rollout"])
        diagnostics = rollout / "diagnostics"
        physics = diagnostics / "physics_config.json"
        if physics.is_file():
            evidence["native_physics_config"] = json.loads(physics.read_text())
        initialization = diagnostics / "initialization.json"
        if initialization.is_file():
            evidence["native_initialization"] = json.loads(initialization.read_text())
            evidence["replay_clock"] = (
                "Source time zero follows native physical initialization. Use the "
                "observed object poses for correction; scene bounds describe the "
                "registered layout before initialization."
            )
        with np.load(diagnostics / "physics_rollout.npz", allow_pickle=False) as data:
            indices = np.abs(data["timestamps"][:, None] - times[samples]).argmin(
                axis=0
            )
            observed = data["qpos_observed"][indices]
            object_names = data["object_names"].tolist()
            object_poses = data["object_poses"][indices]
            evidence["object_poses_at_samples"] = {
                str(name): np.round(data["object_poses"][indices, i], 5)
                for i, name in enumerate(data["object_names"])
            }
        contacts = json.loads((diagnostics / "contacts.json").read_text())
        evidence["contact_actors"] = contacts["actors"]
        evidence["phase_contacts"] = []
        for phase in phases:
            pairs = {}
            dropped = 0
            for row in contacts.get("interval_samples", contacts["samples"]):
                if (
                    times[phase["start_frame"]]
                    <= row["timestamp"]
                    <= times[phase["end_frame"]]
                ):
                    dropped += row.get("dropped_contact_rows", 0)
                    for index, pair in enumerate(row["user_ids"]):
                        key = str(pair)
                        item = pairs.setdefault(
                            key,
                            dict(
                                first_s=row["timestamp"],
                                last_s=row["timestamp"],
                                count=0,
                            ),
                        )
                        item["last_s"] = row["timestamp"]
                        item["count"] += 1
                        forces = row.get("peak_force_on_first_actor_n")
                        if forces is not None:
                            force = forces[index]
                            magnitude = float(np.linalg.norm(force))
                            if magnitude > item.get("peak_force_n", -1):
                                item.update(
                                    peak_force_n=magnitude,
                                    peak_force_on_first_actor_n=force,
                                    peak_force_time_s=row["timestamp"],
                                )
            evidence["phase_contacts"].append(
                dict(phase_id=phase["id"], pairs=pairs, dropped_contact_rows=dropped)
            )
        evidence["contact_sampling"] = contacts.get("sampling")
        evidence.update(
            revisions_remaining=context["revisions_remaining"],
            previous_reviews=[
                json.loads(Path(path).read_text())
                for path in [
                    *context.get("previous_run_reviews", []),
                    *(h["review"] for h in context["history"]),
                ]
            ],
        )
        for name in ("simulation", "overview"):
            video = rollout / "videos" / f"{name}.mp4"
            with av.open(str(video)) as container:
                stream = container.streams.video[0]
                fps, count = float(stream.average_rate), stream.frames
            frames = [
                min(count - 1, max(0, int(round(times[i] * fps)))) for i in samples
            ]
            path = output / f"{name}.jpg"
            _sheet(video, frames, samples, times, name.upper(), path)
            images.append(path)
    evidence["kinematics"] = {}
    evidence["source_joint_limits"] = tools.joint_limit_report(original)
    evidence["commanded_joint_limits"] = tools.joint_limit_report(current)
    for side, name in enumerate(model.side_names):
        records = dict(source=original[samples], commanded=current[samples])
        if stage == "review":
            records["observed"] = observed
        evidence["kinematics"][name] = {
            key: dict(
                tcp_world=np.round(tools.fk(q, side), 5),
                aperture_m=np.round(2 * tools.robot.half_open(q, side), 5),
            )
            for key, q in records.items()
        }
    if stage == "review":
        # TCP origins and body origins are not contact surfaces. Give the reviewer
        # measured geometry at the existing semantic samples without prescribing
        # a grasp pose or modifying the scene.
        initial = evidence["native_initialization"]
        initial_poses = dict(
            zip(initial["object_names"], initial["initial_object_poses"])
        )
        with np.load(context["scene_vertices"], allow_pickle=False) as data:
            local_vertices = {}
            for uid in object_names:
                pose = np.asarray(initial_poses[uid])
                local_vertices[uid] = (data[uid] - pose[:3, 3]) @ pose[:3, :3]
        evidence["contact_geometry"] = []
        for phase in phases:
            side = model.side_names.index(phase["acting_arm"])
            for sample in (
                phase["start_frame"],
                (phase["start_frame"] + phase["end_frame"]) // 2,
                phase["end_frame"],
            ):
                if sample not in samples:
                    continue
                i = samples.index(sample)
                tcp = tools.fk(observed[i], side)

                def bounds(points: np.ndarray) -> dict:
                    local = (points - tcp[:3, 3]) @ tcp[:3, :3]
                    return dict(
                        world_m=np.round([points.min(0), points.max(0)], 5),
                        tcp_m=np.round([local.min(0), local.max(0)], 5),
                    )

                objects = {}
                for uid in object_names:
                    pose = object_poses[i, object_names.index(uid)]
                    points = local_vertices[uid] @ pose[:3, :3].T + pose[:3, 3]
                    objects[uid] = bounds(points)
                evidence["contact_geometry"].append(
                    dict(
                        phase_id=phase["id"],
                        sample=sample,
                        side=phase["acting_arm"],
                        hand_collision_bounds={
                            link: bounds(points)
                            for link, points in tools.hand_surfaces(
                                observed[i], side
                            ).items()
                        },
                        arm_collision_bounds={
                            kind: {
                                link: bounds(points)
                                for link, points in tools.arm_surfaces(q, side).items()
                            }
                            for kind, q in (
                                ("commanded", current[sample]),
                                ("observed", observed[i]),
                            )
                        },
                        object_mesh_bounds=objects,
                    )
                )
        evidence["contact_geometry_convention"] = (
            "Bounds are [min,max] of authored URDF collision surface samples or reconstructed "
            "mesh vertices at observed poses, in world and observed TCP axes. They are not "
            "native convex hulls or guaranteed contact patches. TCP/body origins are not "
            "finger-pad/object centers; infer their offsets from the supplied geometry."
        )
    _write_json(output / "evidence.json", evidence)
    return evidence, images
