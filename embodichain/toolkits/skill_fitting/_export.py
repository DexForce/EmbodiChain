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

"""Portable native run-task bundles for a fitted mixed-skill recording."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from ._pipeline import FitInputs

__all__ = []


def _copy_urdf(source: Path, root: Path) -> tuple[Any, Path]:
    """Copy referenced model files and rewrite only their URI spelling."""
    import yourdfpy

    model = yourdfpy.URDF.load(source, load_meshes=False, load_collision_meshes=False)
    tree = ET.parse(source)
    target = root / "robot" / "robot.urdf"
    target.parent.mkdir(parents=True)
    copied = {}
    for element in tree.findall(".//mesh") + tree.findall(".//texture"):
        uri = element.get("filename")
        if not uri:
            continue
        path = Path(
            yourdfpy.filename_handler_magic(uri, dir=str(source.parent))
        ).resolve(strict=True)
        if path not in copied:
            folder = target.parent / "assets" / f"{len(copied):04d}"
            folder.mkdir(parents=True)
            destination = folder / path.name
            shutil.copy2(path, destination)
            # Preserve OBJ sidecars in the same relative layout. URDF/GLB assets
            # usually have no sidecars; copying their directory is unnecessary.
            if path.suffix.lower() == ".obj":
                for line in path.read_text(errors="replace").splitlines():
                    if line.startswith("mtllib "):
                        material = (path.parent / line[7:].strip()).resolve(strict=True)
                        if material.parent != path.parent:
                            raise ValueError(
                                "OBJ materials must be beside their mesh for portable export."
                            )
                        shutil.copy2(material, folder / material.name)
                        for entry in material.read_text(errors="replace").splitlines():
                            if entry.lstrip().startswith(
                                ("map_", "bump ", "disp ", "decal ")
                            ):
                                texture = entry.split()[-1]
                                texture_path = Path(texture)
                                if (
                                    texture_path.is_absolute()
                                    or ".." in texture_path.parts
                                ):
                                    raise ValueError(
                                        "OBJ textures must use contained relative paths."
                                    )
                                (folder / texture).parent.mkdir(
                                    parents=True, exist_ok=True
                                )
                                shutil.copy2(
                                    material.parent / texture, folder / texture
                                )
            copied[path] = destination.relative_to(target.parent).as_posix()
        element.set("filename", copied[path])
    # pytorch_kinematics consumes URDF as a Unicode string; an XML encoding
    # declaration would make that otherwise portable copied model unreadable.
    tree.write(target, encoding="utf-8", xml_declaration=False)
    return model, target


def _placement(profile: dict[str, Any], model: Any) -> np.ndarray:
    bases = []
    for part, solver in profile["solvers"].items():
        root_link = solver.get("root_link_name") or model.base_link
        base = np.asarray(profile["base_poses"].get(part, np.eye(4)), dtype=float)
        bases.append(base @ np.linalg.inv(model.get_transform(root_link)))
    if not bases or any(
        not np.allclose(b, bases[0], atol=1e-5, rtol=0) for b in bases[1:]
    ):
        raise ValueError("运行配置需要一个基座标定一致的完整机器人 URDF。")
    return bases[0]


def export_run_task(
    directory: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    inputs: FitInputs,
) -> Path:
    """Publish a complete run-task bundle without leaving partial deployments."""
    directory = directory.resolve()
    archive = Path(str(directory) + ".zip")
    if directory.exists() or archive.exists():
        raise FileExistsError(f"运行包输出目录或 ZIP 已存在：{directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fitted-run-task-", dir=directory.parent
    ) as temporary:
        staged = Path(temporary) / "bundle"
        staged_zip = _write_run_task(staged, arrays, report, inputs)
        if directory.exists():
            raise FileExistsError(directory)
        # Hard-link publication is exclusive and stays on this filesystem.
        os.link(staged_zip, archive)
        try:
            staged.rename(directory)
        except BaseException:
            archive.unlink()
            raise
    return archive


def _write_run_task(
    directory: Path,
    arrays: dict[str, np.ndarray],
    report: dict[str, Any],
    inputs: FitInputs,
) -> Path:
    """Write native invocation parameters, shared assets and demo-only tensors."""
    import torch
    from scipy.spatial.transform import Rotation
    import yaml

    from embodichain.data_pipeline.skill_fitting import file_digest
    from ._scene import load_scene

    if report.get("scope") != "episode" or not report["accepted"]:
        raise ValueError("仅在全轨迹所有原子技能均通过拟合校验后导出运行包。")
    archive = Path(str(directory) + ".zip")
    if archive.exists():
        raise FileExistsError(archive)
    directory.mkdir(parents=True, exist_ok=False)
    profile = report["robot_profile"]
    profile_path = Path(inputs.robot_profile).expanduser().resolve()
    if file_digest(profile_path) != report["robot_profile_sha256"]:
        raise ValueError("拟合后机器人配置发生变化，请重新拟合。")
    models = {Path(s["urdf_path"]).resolve() for s in profile["solvers"].values()}
    if len(models) != 1:
        raise ValueError(
            "运行配置需要一个完整机器人 URDF；分离模型请先提供合并后的机器人配置。"
        )
    source = models.pop()
    if any(
        file_digest(source) != digest
        for digest in report["robot_model_sha256"].values()
    ):
        raise ValueError("拟合后机器人 URDF 发生变化。")
    model, target = _copy_urdf(source, directory)
    names = report["joint_names"]
    moving = {joint.name for joint in model.robot.joints if joint.type != "fixed"}
    if moving != set(names):
        raise ValueError(
            "完整机器人 URDF 的活动/被动关节必须与录制配置的全部 DOF 一致。"
        )
    base = _placement(profile, model)
    active_names = []
    for joints in profile["control_parts"].values():
        for name in joints:
            if model.joint_map[name].mimic is None and name not in active_names:
                active_names.append(name)
    if set(active_names) != {n for n in names if model.joint_map[n].mimic is None}:
        raise ValueError("配置的控制部件必须覆盖完整轨迹的全部非 mimic 关节。")
    times = arrays["timestamps"]
    _, dof = arrays["original_qpos"].shape
    dt = float((times[-1] - times[0]) / (len(times) - 1))
    timeline = deepcopy(report["timeline"])
    (directory / "skills").mkdir()
    for i, span in enumerate(timeline):
        if span["kind"] == "atomic_skill":
            continue
        start, stop = span["start"], span["stop"]
        filename = f"skills/{i:03d}.execute_demo_skill.pt"
        torch.save(
            dict(
                schema="fitted-skill-qpos/v1",
                layout="DOF x n_frame",
                skill_id="execute_demo_skill",
                joint_names=names,
                timestamps=torch.from_numpy(np.array(times[start:stop], copy=True)),
                qpos=torch.from_numpy(
                    np.array(arrays["original_qpos"][start:stop], copy=True)
                ).T.contiguous(),
                source_sample_start=start,
                source_sample_stop=stop,
                qpos_semantics=report["qpos_semantics"],
            ),
            directory / filename,
        )
        span.update(
            execution="recorded_qpos_replay",
            tensor=filename,
            tensor_sha256=file_digest(directory / filename),
        )

    scene_path = (
        Path(inputs.scene).expanduser().resolve()
        if inputs.scene
        else Path(inputs.joints).resolve().parent / "blender_scene" / "physical_export"
    )
    physical_objects = []
    scene_files = {}
    collision_assets = {}
    if inputs.scene or scene_path.exists():
        from ._collision import prepare_scene_collision

        prepared = prepare_scene_collision(scene_path)
        objects, scene_files = load_scene(scene_path, include_ungraspable=True)
        scene_files.update(prepared["source_files_sha256"])
        scene_yaml = (
            scene_path / "scene_assets.yaml" if scene_path.is_dir() else scene_path
        )
        declarations = yaml.safe_load(scene_yaml.read_text())["objects"]
        boxes = {}
        for obj in objects:
            world = obj.vertices @ obj.pose[:3, :3].T + obj.pose[:3, 3]
            boxes[obj.identifier] = (world.min(axis=0), world.max(axis=0))
        for obj in objects:
            declaration = declarations[obj.identifier]
            lower, upper = boxes[obj.identifier]
            supports = any(
                other != obj.identifier
                and abs(other_lower[2] - upper[2]) < 0.002
                and np.all((other_lower[:2] + other_upper[:2]) / 2 >= lower[:2])
                and np.all((other_lower[:2] + other_upper[:2]) / 2 <= upper[:2])
                for other, (other_lower, other_upper) in boxes.items()
            )
            path = (
                directory
                / "scene"
                / f"{len(physical_objects):03d}{obj.mesh_path.suffix}"
            )
            path.parent.mkdir(exist_ok=True)
            shutil.copy2(obj.mesh_path, path)
            shape = deepcopy(declaration["shape"])
            shape["fpath"] = "./" + path.relative_to(directory).as_posix()
            if obj.identifier in prepared["objects"]:
                cached = prepared["objects"][obj.identifier]
                collision_path = path
                if file_digest(path) != cached["sha256"]:
                    collision_path = path.with_name(f"{path.stem}.convex.obj")
                    shutil.copyfile(cached["fpath"], collision_path)
                if file_digest(collision_path) != cached["sha256"]:
                    raise ValueError("缓存碰撞网格在打包期间发生变化，请重新加载场景。")
                relative = "./" + collision_path.relative_to(directory).as_posix()
                shape["collision"]["decomposition_fpath"] = relative
                collision_assets[obj.identifier] = dict(
                    fpath=relative,
                    sha256=cached["sha256"],
                    hull_count=cached["hull_count"],
                    recipe=cached["recipe"],
                )
            body_scale = np.broadcast_to(shape.pop("scale", 1), (3,)).tolist()
            physical_objects.append(
                dict(
                    uid=obj.identifier,
                    shape=shape,
                    init_pos=obj.pose[:3, 3].tolist(),
                    init_rot=Rotation.from_matrix(obj.pose[:3, :3])
                    .as_euler("xyz", degrees=True)
                    .tolist(),
                    body_type=declaration.get(
                        "body_type", "static" if supports else "dynamic"
                    ),
                    attrs=declaration.get("attrs", {}),
                    body_scale=body_scale,
                )
            )
    config = dict(
        id="EmbodiedEnv-v1",
        physics="default",
        physics_config=dict(physics_dt=dt / 4),
        num_envs=1,
        headless=True,
        enable_sensor=False,
        max_episode_steps=200_000,
        robot={
            "uid": "fitted_recording",
            "fpath": "./" + target.relative_to(directory).as_posix(),
            "control_parts": profile["control_parts"],
            "init_qpos": [0.0] * dof,
            "init_pos": base[:3, 3].tolist(),
            "init_rot": Rotation.from_matrix(base[:3, :3])
            .as_euler("xyz", degrees=True)
            .tolist(),
            "root_props": dict(fixed_base=True),
            **deepcopy(profile.get("simulation", {})),
        },
        rigid_object=[obj for obj in physical_objects if obj["body_type"] == "dynamic"],
        background=[obj for obj in physical_objects if obj["body_type"] != "dynamic"],
        env=dict(
            control_parts=list(profile["control_parts"]),
            sim_steps_per_control=4,
            task_instruction="Execute native atomic skills and original demonstration transitions.",
            expert_trajectory=dict(joint_command_mode="position"),
        ),
    )
    portable_profile = deepcopy(profile)
    for solver in portable_profile["solvers"].values():
        solver["urdf_path"] = "./robot/robot.urdf"
    (directory / "robot_profile.json").write_text(
        json.dumps(portable_profile, indent=2, ensure_ascii=False)
    )
    for path, digest in scene_files.items():
        if file_digest(path) != digest:
            raise ValueError("导出过程中重建场景发生变化。")
    for fit in report["fits"]:
        for path, digest in fit.get("scene_files_sha256", {}).items():
            if file_digest(path) != digest:
                raise ValueError("拟合后重建场景发生变化，请重新拟合。")
    if file_digest(profile_path) != report["robot_profile_sha256"]:
        raise ValueError("导出过程中机器人配置发生变化，请重新拟合。")
    from ._native_export import write_native_files

    write_native_files(
        directory, arrays, report, config, timeline, collision_assets, dt
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                stream.write(path, path.relative_to(directory))
    return archive
