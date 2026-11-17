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

"""Minimal portable fitting inputs, with optional video for web annotation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import yaml

from embodichain.data_pipeline.skill_fitting import file_digest, load_joint_recording
from ._pipeline import FitInputs
from ._robot import RobotProfile

__all__: list[str] = []


def _contained(root: Path, value: str) -> Path:
    path = (root / value).resolve(strict=True)
    if Path(value).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Bundle resources must use contained relative paths.")
    return path


def pack_inputs(
    directory: Path,
    inputs: FitInputs,
    document: dict[str, Any],
    clock: np.ndarray,
    durations: np.ndarray,
    *,
    video: Path | None = None,
    video_stream: int | None = None,
) -> Path:
    """Publish self-contained robot, scene, joint and annotation inputs."""
    from ._collision import prepare_scene_collision, resolve_scene_path
    from ._export import _copy_urdf
    from ._scene import load_scene

    directory = directory.resolve()
    if directory.exists():
        raise FileExistsError(directory)
    profile = RobotProfile.load(inputs.robot_profile)
    models = {Path(s["urdf_path"]) for s in profile.solvers.values()}
    if len(models) != 1:
        raise ValueError("A portable fitting bundle requires one complete robot URDF.")
    scene = resolve_scene_path(inputs.scene, inputs.joints)
    if scene is None:
        raise ValueError("A reconstructed scene is required for an input bundle.")
    objects, scene_files = load_scene(scene, include_ungraspable=True)
    collisions = prepare_scene_collision(scene)
    scene_files.update(collisions["source_files_sha256"])
    source_yaml = scene / "scene_assets.yaml" if scene.is_dir() else scene
    raw = yaml.safe_load(source_yaml.read_text())
    recording = load_joint_recording(
        inputs.joints,
        joint_names=profile.joint_names,
        semantics=inputs.qpos_semantics,
        qpos_key=inputs.qpos_key,
        timestamps_key=inputs.timestamps_key,
        sample_dt=inputs.joint_dt,
        env_index=inputs.env_index,
    )
    original_files = {
        str(Path(inputs.robot_profile).resolve()): file_digest(inputs.robot_profile),
        str(Path(inputs.joints).resolve()): file_digest(inputs.joints),
        **scene_files,
    }
    if video_stream is not None and (
        video is None or type(video_stream) is not int or video_stream < 0
    ):
        raise ValueError("video_stream requires a video and a non-negative integer.")
    if video is not None:
        video = video.expanduser().resolve(strict=True)
        original_files[str(video)] = file_digest(video)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fitting-inputs-", dir=directory.parent
    ) as tmp:
        root = Path(tmp) / "bundle"
        root.mkdir()
        _copy_urdf(models.pop(), root)
        portable = deepcopy(profile.metadata())
        for solver in portable["solvers"].values():
            solver["urdf_path"] = "robot/robot.urdf"
        (root / "robot_profile.json").write_text(json.dumps(portable, indent=2))
        (root / "scene").mkdir()
        for index, obj in enumerate(objects):
            declaration = raw["objects"][obj.identifier]
            declaration["pose"] = obj.pose.tolist()
            target = root / "scene" / (f"{index:03d}" + obj.mesh_path.suffix)
            shutil.copyfile(obj.mesh_path, target)
            declaration["shape"]["fpath"] = target.name
            if obj.identifier in collisions["objects"]:
                cached = collisions["objects"][obj.identifier]
                if file_digest(target) != cached["sha256"]:
                    target = target.with_name(f"{index:03d}.convex.obj")
                    shutil.copyfile(cached["fpath"], target)
                declaration["shape"]["collision"]["decomposition_fpath"] = target.name
        # The contract deliberately excludes exporter reports and Blender metadata.
        scene_data = {"objects": raw["objects"]}
        if "skill_fitting" in raw:
            scene_data["skill_fitting"] = raw["skill_fitting"]
        (root / "scene" / "scene_assets.yaml").write_text(
            yaml.safe_dump(scene_data, sort_keys=False)
        )
        np.savez_compressed(
            root / "joints.npz",
            qpos=recording.qpos,
            timestamps=recording.timestamps,
            joint_names=np.asarray(profile.joint_names),
        )
        np.savez_compressed(
            root / "frame_clock.npz", timestamps=clock, durations=durations
        )
        (root / "annotations.json").write_text(
            json.dumps(document, ensure_ascii=False, indent=2)
        )
        frame_source = None
        if video is not None:
            from embodichain.toolkits.trajectory_annotator._cfg import AnnotationCfg
            from embodichain.toolkits.trajectory_annotator._storage import (
                AnnotationStore,
            )
            from embodichain.toolkits.trajectory_annotator.video import VideoFrameSource
            import threading

            # Validate content, selected stream and image timing against the
            # annotation seed before publishing a portable web project.
            source = VideoFrameSource(video, stream_index=video_stream)
            store = AnnotationStore(
                root / "annotations.json", AnnotationCfg().document_bytes
            )
            try:
                info = source.prepare(threading.Event(), lambda **_: None)
                store.initialize(info)
                source.check_unchanged()
            finally:
                source.close()
            (root / "video").mkdir()
            target = root / "video" / video.name
            shutil.copyfile(video, target)
            # Refresh descriptive provenance; no original source directory is
            # needed to reopen the seed. The content identity is unchanged.
            document = store.snapshot()["document"]
            document["source"]["path"] = target.relative_to(root).as_posix()
            (root / "annotations.json").write_text(
                json.dumps(document, ensure_ascii=False, indent=2)
            )
            frame_source = dict(
                kind="video",
                path=target.relative_to(root).as_posix(),
                stream_index=video_stream,
            )
        settings = asdict(inputs)
        settings.update(
            joints="joints.npz",
            robot_profile="robot_profile.json",
            scene="scene/scene_assets.yaml",
            qpos_key="qpos",
            timestamps_key="timestamps",
            env_index=0,
            joint_dt=None,
            device="cpu",
        )
        manifest = dict(
            schema="scene-skill-fitting-inputs/v1",
            inputs=settings,
            annotations="annotations.json",
            frame_clock="frame_clock.npz",
            files={
                p.relative_to(root).as_posix(): file_digest(p)
                for p in sorted(root.rglob("*"))
                if p.is_file()
            },
        )
        if frame_source is not None:
            manifest["frame_source"] = frame_source
        (root / "fitting.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False)
        )
        (root / "README.md").write_text(
            "# Portable scene fitting inputs\n\n"
            "All lengths are metres, angles radians; poses are world-from-object 4x4 matrices. "
            "Mesh vertices are object-local. Frame bounds are inclusive; joint clocks are seconds.\n\n"
            "```bash\nconda activate py312_bk\n"
            "embodichain fit-trajectory --bundle . --output fit.npz --run-task-output run-task\n```\n\n"
            "`fitting.json` pins input resources by SHA-256. `frame_clock.npz` supports offline fitting. "
            "`joints.npz` is the original full-joint recording; `annotations.json` defines skill intervals. "
            "`robot_profile.json` owns control parts, solver/TCP calibration and explicit simulation settings. "
            "`scene/scene_assets.yaml` owns object meshes, poses, dynamics and fitting defaults; "
            "convex-part OBJ files are prepared colliders, so run-task does not decompose again. "
            "The installed EmbodiChain/py312_bk runtime is required. No original source directory, "
            "Blender project, export report, USDA or prior fit is required.\n"
            + (
                "\nThe original encoded video is included with its recorded frame timing. Start the web workflow with:\n\n"
                "```bash\nembodichain annotate-trajectory --bundle . --port 8898 --no-browser\n```\n\n"
                "The web editor saves `annotations.edited.json`, independently of the sealed annotation seed. "
                "Fit the full sequence and download its run-task ZIP in the webpage. The runtime bundle includes "
                "`expansion.yaml`; execute it with `--skill-sequence skills.yaml --expansion-profile expansion.yaml`. "
                "No original recording directory is required.\n"
                if frame_source is not None
                else "Video pixels are unnecessary for this offline-only bundle.\n"
            )
        )
        for path, digest in original_files.items():
            if file_digest(path) != digest:
                raise ValueError("Source resources changed during packaging.")
        load_inputs(root)
        root.rename(directory)
    return directory


def load_inputs(
    directory: Path,
) -> tuple[FitInputs, dict[str, Any], np.ndarray, np.ndarray]:
    """Validate file identity and resolve a moved folder without source lookups."""
    root = directory.resolve(strict=True)
    manifest = json.loads((root / "fitting.json").read_text())
    required = {"schema", "inputs", "annotations", "frame_clock", "files"}
    if (
        not required <= manifest.keys()
        or manifest.keys() - (required | {"frame_source"})
        or manifest["schema"] != "scene-skill-fitting-inputs/v1"
    ):
        raise ValueError("Unsupported fitting input bundle.")
    for name, digest in manifest["files"].items():
        if file_digest(_contained(root, name)) != digest:
            raise ValueError(f"Bundle resource changed: {name}")
    settings = deepcopy(manifest["inputs"])
    if settings.keys() - FitInputs.__dataclass_fields__.keys():
        raise ValueError("Unknown fitting bundle input field.")
    for key in ("joints", "robot_profile", "scene"):
        settings[key] = str(_contained(root, settings[key]))
    inputs = FitInputs(**settings)
    inputs.check()
    # Robot/scene dependencies are also confined to the sealed file inventory.
    profile = RobotProfile.load(inputs.robot_profile)
    from ._scene import load_scene

    _, sources = load_scene(Path(inputs.scene), include_ungraspable=True)
    dependencies = set(sources) | {s["urdf_path"] for s in profile.solvers.values()}
    import xml.etree.ElementTree as ET
    import yourdfpy

    for solver in profile.solvers.values():
        model_path = Path(solver["urdf_path"])
        tree = ET.parse(model_path)
        for element in tree.findall(".//mesh") + tree.findall(".//texture"):
            uri = element.get("filename")
            if uri:
                dependencies.add(
                    str(
                        Path(
                            yourdfpy.filename_handler_magic(
                                uri, dir=str(model_path.parent)
                            )
                        ).resolve(strict=True)
                    )
                )
    scene_data = yaml.safe_load(Path(inputs.scene).read_text())
    for obj in scene_data["objects"].values():
        decomposition = (
            obj.get("shape", {}).get("collision", {}).get("decomposition_fpath")
        )
        if decomposition:
            dependencies.add(
                str((Path(inputs.scene).parent / decomposition).resolve(strict=True))
            )
    dependencies.update(
        {
            inputs.joints,
            inputs.robot_profile,
            str(_contained(root, manifest["annotations"])),
            str(_contained(root, manifest["frame_clock"])),
        }
    )
    visual = _video_source(root, manifest)
    if visual is not None:
        dependencies.add(str(visual[0]))
    inventory = {str(_contained(root, p)) for p in manifest["files"]}
    if dependencies - inventory:
        raise ValueError("Bundle has unlisted or external robot/scene dependencies.")
    document = json.loads(_contained(root, manifest["annotations"]).read_text())
    with np.load(_contained(root, manifest["frame_clock"]), allow_pickle=False) as data:
        clock, durations = data["timestamps"].copy(), data["durations"].copy()
    if (
        clock.ndim != 1
        or len(clock) != document["source"]["frame_count"]
        or durations.shape != clock.shape
        or not np.isfinite(clock).all()
        or not np.isfinite(durations).all()
        or np.any(np.diff(clock) <= 0)
        or np.any(durations <= 0)
    ):
        raise ValueError("Invalid frame clock in fitting bundle.")
    return inputs, document, clock, durations


def _video_source(
    root: Path, manifest: dict[str, Any]
) -> tuple[Path, int | None] | None:
    """Decode the optional closed, contained web frame-source declaration."""
    if "frame_source" not in manifest:
        return None
    source = manifest["frame_source"]
    if (
        not isinstance(source, dict)
        or source.keys() != {"kind", "path", "stream_index"}
        or source["kind"] != "video"
        or not isinstance(source["path"], str)
        or (
            source["stream_index"] is not None
            and (type(source["stream_index"]) is not int or source["stream_index"] < 0)
        )
    ):
        raise ValueError("Bundle frame_source requires a video path and stream_index.")
    return _contained(root, source["path"]), source["stream_index"]


def load_annotation_inputs(
    directory: Path, annotations: Path | None = None
) -> tuple[Path, int | None, Path, dict[str, Any]]:
    """Resolve a web project and isolate mutable edits from sealed input files."""
    root = directory.expanduser().resolve(strict=True)
    inputs, document, _, _ = load_inputs(root)
    manifest = json.loads((root / "fitting.json").read_text())
    visual = _video_source(root, manifest)
    if visual is None:
        raise ValueError(
            "This offline-only bundle has no video. Repackage with --package-video <video>."
        )
    destination = (
        (annotations or root / "annotations.edited.json").expanduser().resolve()
    )
    protected = [_contained(root, name) for name in manifest["files"]]
    protected += [root / "fitting.json", root / "README.md"]
    if destination in protected or (
        destination.exists()
        and any(p.exists() and destination.samefile(p) for p in protected)
    ):
        raise ValueError("Annotation edits must not overwrite sealed bundle inputs.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
    return *visual, destination, asdict(inputs)
