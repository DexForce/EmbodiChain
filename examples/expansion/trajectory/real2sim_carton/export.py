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

"""Export a local offline viewer from verified recording attachments.

The browser performs only URDF forward kinematics and displays precomputed
observations. No physics, data collection or receipt creation occurs here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import urllib.request
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
import yaml

from embodichain.data_pipeline.real2sim import load_prototype, retarget_motion_prototype
from embodichain.data_pipeline.real2sim._geometry import (
    _canonical_robot_geometry_sha256,
)
from embodichain.data_pipeline.real2sim.replay import export_real2sim_replay_manifest
from embodichain.lab.sim.motion.expansion.real2sim_collision import (
    SampledRobotCollisionChecker,
    PlaneCollisionBody,
)
from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
from embodichain.lab.scripts.run_task import _create_parser
from examples.expansion.trajectory.real2sim_carton.prepare import (
    ROOT,
    digest,
    write_json,
)

__all__: list[str] = []


def _origin(node) -> np.ndarray:
    result = np.eye(4)
    if node is not None:
        result[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def _robot(urdf: Path, output: Path) -> dict:
    output.mkdir()
    root = ET.parse(urdf).getroot()
    links = []
    for link in root.findall("link"):
        scene = trimesh.Scene()
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                box = visual.find("geometry/box")
                if box is not None:
                    geometry = trimesh.creation.box(
                        np.fromstring(box.get("size"), sep=" ")
                    )
                    geometry.apply_transform(_origin(visual.find("origin")))
                    scene.add_geometry(geometry)
                continue
            path = Path(mesh.get("filename"))
            if not path.is_absolute():
                path = urdf.parent / path
            loaded = trimesh.load(path, force="scene", process=False)
            scale = np.eye(4)
            scale[:3, :3] = np.diag(np.fromstring(mesh.get("scale", "1 1 1"), sep=" "))
            for name in loaded.graph.nodes_geometry:
                transform, geometry_name = loaded.graph[name]
                geometry = loaded.geometry[geometry_name].copy()
                geometry.apply_transform(
                    _origin(visual.find("origin")) @ scale @ transform
                )
                scene.add_geometry(geometry)
        filename = None
        if len(scene.geometry):
            filename = link.get("name") + ".glb"
            scene.export(output / filename)
        links.append(
            dict(
                name=link.get("name"),
                mesh=filename,
                mesh_sha256=digest(output / filename) if filename else None,
            )
        )
    joints = []
    children = set()
    for joint in root.findall("joint"):
        origin = _origin(joint.find("origin"))
        child = joint.find("child").get("link")
        children.add(child)
        axis = joint.find("axis")
        joints.append(
            dict(
                name=joint.get("name"),
                type=joint.get("type"),
                parent=joint.find("parent").get("link"),
                child=child,
                position=origin[:3, 3].tolist(),
                quaternion=Rotation.from_matrix(origin[:3, :3]).as_quat().tolist(),
                axis=(
                    np.fromstring(axis.get("xyz", "0 0 1"), sep=" ").tolist()
                    if axis is not None
                    else [0, 0, 1]
                ),
            )
        )
    return dict(
        links=links,
        joints=joints,
        urdf_sha256=digest(urdf),
        roots=[link["name"] for link in links if link["name"] not in children],
    )


def _vendor(output: Path) -> list[dict]:
    output.mkdir()
    items = json.loads((ROOT / "viewer.vendor.lock.json").read_text())["files"]
    result = []
    for item in items:
        data = urllib.request.urlopen(item["url"], timeout=60).read()
        import hashlib

        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("pinned viewer dependency differs")
        if item["path"] == "GLTFLoader.js":
            data = data.replace(
                b"../utils/BufferGeometryUtils.js", b"./BufferGeometryUtils.js"
            )
        path = output / item["path"]
        path.write_bytes(data)
        result.append({**item, "export_sha256": digest(path)})
    return result


def _offset(pose: np.ndarray, values) -> np.ndarray:
    out = pose.copy()
    out[:2, 3] += values[:2]
    out[:3, :3] = Rotation.from_euler("z", values[2]).as_matrix() @ out[:3, :3]
    return out


def _comparison_plot(output: Path, index: list[dict]) -> None:
    """Plot the measured baseline and their explicitly unexecuted targets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(14, 4))
    handles = []
    for row_number, origin in enumerate(("measured",)):
        entry = next(
            item
            for item in index
            if item["origin"] == origin
            and item["layout"] == "baseline"
            and item["recipe"] == "nominal"
        )
        case = json.loads((output / entry["url"]).read_text())
        if case["case"]["status"] != "passed":
            raise ValueError(
                "comparison plots require independently verified baselines"
            )
        times = np.array(case["frames"]["t"])
        for column, curve in enumerate(case["curves"]):
            axes = figure.add_subplot(
                1, 3, row_number * 3 + column + 1, projection="3d"
            )
            for key, color, label in (
                ("input", "#3c74d6", "Input target (unexecuted)"),
                ("rigid", "#9a8db9", "Rigid target (unexecuted)"),
                ("corrected", "#e5832e", "Refined target"),
            ):
                points = np.array(curve[key])
                (line,) = axes.plot(*points.T, color=color, linewidth=1.3, label=label)
                if row_number == 0 and column == 0:
                    handles.append(line)
            mask = np.zeros(len(times), dtype=bool)
            for phase in case["phases"]:
                if phase["call_index"] == curve["call_index"]:
                    mask |= (times >= phase["start"]) & (times <= phase["stop"])
            (line,) = axes.plot(
                *np.array(case["frames"]["tcp"])[mask].T,
                color="#07816b",
                linewidth=1.2,
                label="Observed simulator TCP",
            )
            if row_number == 0 and column == 0:
                handles.append(line)
            axes.set_title(
                "DROID" + " / " + curve["skill_id"],
                fontsize=10,
            )
            axes.set_xlabel("X / m", fontsize=8)
            axes.set_ylabel("Y / m", fontsize=8)
            axes.set_zlabel("Z / m", fontsize=8)
            axes.tick_params(labelsize=7)
    figure.suptitle(
        "Frozen inputs, geometric comparison targets and verified observed motion",
        fontsize=14,
    )
    figure.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    figure.subplots_adjust(
        left=0.03, right=0.97, bottom=0.10, top=0.91, wspace=0.08, hspace=0.24
    )
    figure.savefig(output / "trajectory-comparison.png", dpi=160)
    figure.savefig(output / "trajectory-comparison.svg")
    plt.close(figure)


def export(prepared: Path, matrix: Path, source_directory: Path, output: Path) -> Path:
    report = json.loads(matrix.read_text())
    scene = json.loads((prepared / "scene.lock.json").read_text())
    qualification = json.loads((prepared / "qualification.json").read_text())
    source_lock = json.loads((ROOT / "source.lock.json").read_text())
    if scene["source_lock_sha256"] != digest(ROOT / "source.lock.json"):
        raise ValueError("the prepared scene does not match its pinned source lock")
    for source_file in source_lock["files"]:
        path = source_directory / source_file["path"]
        if (
            not path.is_file()
            or path.stat().st_size != source_file["size"]
            or digest(path) != source_file["sha256"]
        ):
            raise ValueError(
                "an original source video or recording differs from its pinned input"
            )
    output.mkdir(parents=True, exist_ok=False)
    roots = [Path(row["dataset"]) for row in report["cases"] if row.get("dataset")]
    raw_manifest = export_real2sim_replay_manifest(
        roots,
        output / "recordings.json",
        source_assets={
            "qualification": dict(
                path=str(prepared / "qualification.json"),
                sha256=digest(prepared / "qualification.json"),
            ),
            **{
                f"camera_{serial}": dict(
                    path=str(source_directory / f"recordings/MP4/{serial}.mp4"),
                    sha256=digest(source_directory / f"recordings/MP4/{serial}.mp4"),
                )
                for serial in ("20521388", "24259877", "13062452")
            },
        },
    )
    archive = json.loads(raw_manifest.read_text())
    rows = {item["dataset"]: item for item in archive["episodes"]}
    arguments = _create_parser().parse_args(
        ["--gym-config", str(prepared / "task.measured.yaml")]
    )
    environment_cfg, _, _ = build_env_cfg_from_args(arguments)
    cfg = environment_cfg.robot
    urdf = Path(cfg.urdf_cfg.assemble_urdf())
    geometry = SampledRobotCollisionChecker(
        urdf,
        joint_names=scene["geometry"]["joint_names"],
        bodies=(PlaneCollisionBody("ground", np.eye(4)),),
        world_from_base=np.eye(4),
    )
    if geometry.geometry_sha256 != _canonical_robot_geometry_sha256(
        scene["geometry"]["robot_geometry_sha256"]
    ):
        raise ValueError("replay robot geometry differs from the qualified recording")
    robot = _robot(urdf, output / "robot")
    vendor = _vendor(output / "vendor")
    media = output / "media"
    media.mkdir()
    for serial in ("20521388", "24259877", "13062452"):
        shutil.copyfile(
            source_directory / f"recordings/MP4/{serial}.mp4", media / f"{serial}.mp4"
        )
    shutil.copyfile(ROOT / "viewer.html", output / "index.html")
    for source, name in (
        (matrix, "acceptance-report.json"),
        (prepared / "qualification.json", "qualification.json"),
        (prepared / "scene.lock.json", "scene.lock.json"),
    ):
        shutil.copyfile(source, output / name)
    for name in ("source.lock.json", "assets.lock.json"):
        shutil.copyfile(ROOT / name, output / name)
    case_directory = output / "cases"
    case_directory.mkdir()
    clock = qualification["frame_clock_map"]
    annotation = json.loads((ROOT / "annotations/staples.json").read_text())
    arm_indices = [
        scene["geometry"]["joint_names"].index(name)
        for name in cfg.control_parts["arm"]
    ]
    chain = cfg.build_pk_serial_chain()["arm"]
    tcp = np.array(cfg.solver_cfg["arm"].tcp)
    index = []
    for number, case in enumerate(report["cases"]):
        profile = yaml.safe_load(Path(case["profile"]).read_text())
        layout = profile["layouts"][0]
        obj = _offset(np.array(scene["object_pose"]), layout["object_xyyaw"])
        target = _offset(np.array(scene["target_pose"]), layout["target_xyyaw"])
        item = dict(
            case=case,
            object_pose=obj.tolist(),
            target_pose=target.tolist(),
            frames=None,
            curves=[],
            phases=[],
            failed_attempts=[
                entry
                for entry in archive["failed_attempts"]
                if entry["dataset"] == case.get("dataset")
            ],
        )
        stored = rows.get(case.get("dataset"))
        if case["status"] == "passed" and stored is None:
            raise ValueError(
                "a passed matrix case is missing its verified recording episode"
            )
        if stored is not None:
            row = stored["metadata"]
            if (row["origin"], row["layout"], row["recipe"], row["candidate_id"]) != (
                case["origin"],
                case["layout"],
                case["recipe"],
                case["audit"]["candidate_id"],
            ):
                raise ValueError(
                    "the verified episode differs from its matrix candidate"
                )
            root = Path(stored["dataset"])
            evidence = row["real2sim_episode_evidence"]
            with np.load(root / evidence["path"], allow_pickle=False) as arrays:
                q = arrays[f"observation_{evidence['observation_names'].index('qpos')}"]
                times = arrays["timestamps"]
            with np.load(
                root / row["real2sim_observed_geometry"]["path"], allow_pickle=False
            ) as arrays:
                objects = arrays["object_pose"]
                item["target_pose"] = arrays["target_pose"].tolist()
            actual_tcp = (
                chain.forward_kinematics(
                    torch.as_tensor(q[:, arm_indices].copy(), dtype=torch.float32)
                )
                .get_matrix()
                .numpy()
                @ tcp
            )
            with np.load(
                root / row["real2sim_lineage"]["trace"]["path"], allow_pickle=False
            ) as arrays:
                trace = {name: arrays[name] for name in arrays.files}
            calls = {
                call["call_index"]: call for call in row["real2sim_lineage"]["calls"]
            }
            item["refinement_reports"] = [
                dict(call_index=i, **call["refinement"]) for i, call in calls.items()
            ]
            prototypes = {
                i: load_prototype(
                    root
                    / row["real2sim_artifacts"][call["source_bundle"]][
                        "prototype_manifest"
                    ]["path"]
                )
                for i, call in calls.items()
            }
            (
                source_time,
                source_frame,
                phase_names,
                input_xyz,
                refined_xyz,
                rigid_xyz,
            ) = ([], [], [], [], [], [])
            phase_ranges = {}
            for i, call in calls.items():
                original = trace[f"call_{i}_refinement_original_tcp"]
                corrected = trace[f"call_{i}_refinement_target_tcp"]
                anchor = call["refinement"]["anchor_index"]
                rigid = corrected[anchor] @ np.linalg.inv(original[anchor]) @ original
                item["curves"].append(
                    dict(
                        call_index=i,
                        skill_id=prototypes[i].unit.source.skill_id,
                        input=original[:, :3, 3].tolist(),
                        corrected=corrected[:, :3, 3].tolist(),
                        rigid=rigid[:, :3, 3].tolist(),
                    )
                )
            for step in range(len(times)):
                call_index = int(trace["step_call_index"][min(step, len(times) - 2)])
                plan_index = int(trace["step_plan_index"][min(step, len(times) - 2)])
                mapped = None
                phase_name = "等待 / 验收"
                pts = (None, None, None)
                if call_index >= 0 and plan_index >= 0:
                    call, proto = calls[call_index], prototypes[call_index]
                    prefix = call["source_sample_offset"]
                    left, right = [
                        int(trace[f"call_{call_index}_{key}"][plan_index]) - prefix
                        for key in ("left", "right")
                    ]
                    weight = float(trace[f"call_{call_index}_weight"][plan_index])
                    if left >= 0:
                        tm = proto.unit.report.time_map

                        def original_time(j):
                            return (1 - tm.weight[j]) * proto.unit.source.timestamps[
                                tm.left[j]
                            ] + tm.weight[j] * proto.unit.source.timestamps[tm.right[j]]

                        mapped = float(
                            (1 - weight) * original_time(left)
                            + weight * original_time(right)
                        )
                        phase_name = next(
                            p.name
                            for p in proto.unit.phases
                            if p.start <= left < p.stop
                        )
                        phase_ranges.setdefault((call_index, phase_name), [step, step])[
                            1
                        ] = step
                        curve = item["curves"][call_index]
                        pts = tuple(
                            (
                                (1 - weight) * np.array(curve[key][left])
                                + weight * np.array(curve[key][right])
                            ).tolist()
                            for key in ("input", "corrected", "rigid")
                        )
                        if case["origin"] == "measured":
                            start = annotation["phases"][call_index]["start"]
                            mapped += clock["source_seconds"][start]
                    else:
                        phase_name = "入口连接 · 无源帧"
                source_time.append(mapped)
                source_frame.append(
                    float(
                        np.interp(
                            mapped, clock["source_seconds"], clock["encoding_seconds"]
                        )
                    )
                    if mapped is not None and case["origin"] == "measured"
                    else None
                )
                phase_names.append(phase_name)
                input_xyz.append(pts[0])
                refined_xyz.append(pts[1])
                rigid_xyz.append(pts[2])
            item.update(
                frames=dict(
                    t=np.round(times, 8).tolist(),
                    q=np.round(q, 8).tolist(),
                    object=np.round(objects, 8).tolist(),
                    tcp=np.round(actual_tcp[:, :3, 3], 8).tolist(),
                    source_seconds=source_time,
                    video_seconds=source_frame,
                    phase=phase_names,
                    input=input_xyz,
                    corrected=refined_xyz,
                    rigid=rigid_xyz,
                ),
                phases=[
                    dict(
                        call_index=i,
                        name=name,
                        start=times[bounds[0]],
                        stop=times[min(bounds[1] + 1, len(times) - 1)],
                    )
                    for (i, name), bounds in phase_ranges.items()
                ],
                receipt=stored["receipt"],
                source_lineage=stored["sources"],
                metadata_url=os.path.relpath(
                    root / "meta/embodichain_episodes.jsonl", output
                ),
            )
        else:
            base = prepared / "measured"
            prototypes = [
                load_prototype(base / f"prototypes/{i}/prototype.json")
                for i in range(3)
            ]
            relation = prototypes[0].object_to_tcp
            for i, proto in enumerate(prototypes):
                poses = retarget_motion_prototype(
                    proto,
                    object_pose=obj if i == 0 else None,
                    target_pose=target if i else None,
                    held_object_to_tcp=relation if i else None,
                    object_supported_regrasp=i == 0,
                )
                item["curves"].append(
                    dict(
                        call_index=i,
                        skill_id=proto.unit.source.skill_id,
                        input=poses[:, :3, 3].tolist(),
                        corrected=None,
                        rigid=None,
                    )
                )
            item["notice"] = (
                "未取得可验收的完整执行；仅展示输入目标路径。失败原因保留在报告中。"
            )
            partials = [
                failure
                for failure in item["failed_attempts"]
                if "partial_rollout" in failure["detail"]
            ]
            if partials:
                failure = partials[-1]
                with np.load(
                    Path(failure["dataset"])
                    / failure["detail"]["partial_rollout"]["path"],
                    allow_pickle=False,
                ) as arrays:
                    q, times, objects = (
                        arrays["qpos"],
                        arrays["timestamps"],
                        arrays["object_pose"],
                    )
                observed_tcp = (
                    chain.forward_kinematics(
                        torch.as_tensor(q[:, arm_indices].copy(), dtype=torch.float32)
                    )
                    .get_matrix()
                    .numpy()
                    @ tcp
                )
                item.update(
                    partial_execution=True,
                    frames=dict(
                        t=times.tolist(),
                        q=q.tolist(),
                        object=objects.tolist(),
                        tcp=observed_tcp[:, :3, 3].tolist(),
                        source_seconds=[None] * len(times),
                        video_seconds=[None] * len(times),
                        phase=["失败候选的部分实测 · 无成功回执"] * len(times),
                    ),
                    notice="展示安全中止前保存的部分实测，不是完整成功 episode；来源视频没有逐步映射。",
                )
        path = case_directory / f"{number:03d}.json"
        write_json(path, item)
        index.append(
            {
                **{
                    key: case[key]
                    for key in ("origin", "layout", "recipe", "status", "error")
                },
                "url": f"cases/{number:03d}.json",
            }
        )
    initial = cfg.init_qpos
    if isinstance(initial, dict):
        initial = [initial[name] for name in scene["geometry"]["joint_names"]]
    write_json(
        output / "manifest.json",
        dict(
            schema="real2sim-carton-view/v1",
            read_only=True,
            cases=index,
            sources=report["sources"],
            robot=robot,
            initial_qpos=list(initial),
            source_interval=[clock["source_seconds"][25], clock["source_seconds"][93]],
            joint_names=scene["geometry"]["joint_names"],
            object_size=scene["object_size_m"],
            support_size=[0.65, 0.65, scene["support_height_m"]],
            qualification=qualification,
            scope=report["scope"],
            vendor=vendor,
        ),
    )
    _comparison_plot(output, index)
    return output / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "matrix", "source-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        export(
            args.prepared.resolve(),
            args.matrix.resolve(),
            args.source_dir.resolve(),
            args.output.resolve(),
        )
    )


if __name__ == "__main__":
    main()
