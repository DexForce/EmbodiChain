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

"""Convert a public RoboCasa365 demonstration into an offline EmbodiChain fixture."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
import pyarrow.parquet as parquet

__all__ = ["main"]

_SCALAR_JOINT_TYPES = (int(mj.mjtJoint.mjJNT_SLIDE), int(mj.mjtJoint.mjJNT_HINGE))
_ROBOT_PARTS = {
    "arm": [f"robot0_joint{i}" for i in range(1, 8)],
    "base": [f"mobilebase0_joint_mobile_{axis}" for axis in ("forward", "side", "yaw")],
    "torso": ["mobilebase0_joint_torso_height"],
    "eef": [f"gripper0_right_finger_joint{i}" for i in (1, 2)],
}


def _read_episode(
    read: Callable[[str], bytes], episode: int, task: str
) -> tuple[ET.Element, dict[str, Any], np.ndarray, np.ndarray]:
    if json.loads(read("extras/dataset_meta.json"))["env"] != task:
        raise ValueError(f"The demonstration source does not contain {task}.")
    directory = f"extras/episode_{episode:06d}"
    xml = ET.fromstring(gzip.decompress(read(f"{directory}/model.xml.gz")))
    metadata = json.loads(read(f"{directory}/ep_meta.json"))
    with np.load(io.BytesIO(read(f"{directory}/states.npz"))) as recording:
        states = recording["states"]
    path = f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
    table = parquet.read_table(io.BytesIO(read(path)), columns=["action"])
    actions = np.asarray(table["action"].to_pylist())
    return xml, metadata, states, actions


def _source_asset(filename: str) -> tuple[str, str]:
    prefix, relative = filename.split("/models/assets/", 1)
    repository = "robosuite" if "robosuite" in prefix.split("/") else "robocasa"
    path = Path(os.path.normpath(relative))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid source asset path {filename!r}.")
    return repository, path.as_posix()


def _resolve_source_files(xml: ET.Element, casa: Path, suite: Path) -> None:
    for item in xml.iter():
        filename = item.get("file")
        if filename is None:
            continue
        repository, relative = _source_asset(filename)
        root = suite if repository == "robosuite" else casa
        local = (root / relative).resolve()
        if not local.is_relative_to(root) or not local.is_file():
            raise FileNotFoundError(f"Missing source asset {local}.")
        item.set("file", str(local))


def _serialise_multi_joint_bodies(xml: ET.Element) -> None:
    """Keep joint axes and pivots when expanding MJCF's co-located joints."""
    for parent in list(xml.iter()):
        for body in parent.findall("body"):
            joints = body.findall("joint")
            if len(joints) < 2:
                continue
            name = body.attrib["name"]
            root_attrs = body.attrib.copy()
            body.attrib = {"name": name}
            index = list(parent).index(body)
            parent.remove(body)
            for number, joint in enumerate(joints[:-1]):
                label = f"{name}_serial_{number}"
                if number:
                    link = ET.SubElement(link, "body", {"name": label})
                else:
                    link = ET.Element("body", {**root_attrs, "name": label})
                    parent.insert(index, link)
                ET.SubElement(
                    link,
                    "inertial",
                    {"pos": "0 0 0", "mass": "1e-5", "diaginertia": "1e-7 1e-7 1e-7"},
                )
                body.remove(joint)
                link.append(joint)
            link.append(body)


def _normalise_usd(path: Path, model: mj.MjModel) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdPhysics.FixedJoint):
            continue
        joint = UsdPhysics.FixedJoint(prim)
        parents = joint.GetBody0Rel().GetTargets()
        children = joint.GetBody1Rel().GetTargets()
        if len(children) != 1 or (parents and parents != [root.GetPath()]):
            continue
        body = stage.GetPrimAtPath(children[0])
        if not body.HasAPI(UsdPhysics.ArticulationRootAPI):
            body.RemoveAPI(UsdPhysics.RigidBodyAPI)
            stage.RemovePrim(prim.GetPath())

    body_ids = {model.body(i).name: i for i in range(1, model.nbody)}
    geom_ids = {model.geom(i).name: i for i in range(model.ngeom)}
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            uv = UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("st")
            if not uv or not uv.HasValue():
                # Several collision meshes share a textured material with their
                # render mesh but carry no texture coordinates.
                UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
        name = prim.GetDisplayName() or prim.GetName()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and name in body_ids:
            i = body_ids[name]
            if model.body_mass[i] > 0:
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(float(model.body_mass[i]))
                mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, model.body_ipos[i])))
                mass.CreateDiagonalInertiaAttr(
                    Gf.Vec3f(*map(float, model.body_inertia[i]))
                )
                w, *xyz = map(float, model.body_iquat[i])
                mass.CreatePrincipalAxesAttr(Gf.Quatf(w, Gf.Vec3f(*xyz)))
        if prim.HasAPI(UsdPhysics.CollisionAPI) and name in geom_ids:
            i = geom_ids[name]
            for field, value in (
                ("mjc:gap", model.geom_gap[i]),
                ("mjc:margin", model.geom_margin[i]),
                ("newton:torsionalFriction", model.geom_friction[i, 1]),
                ("newton:rollingFriction", model.geom_friction[i, 2]),
            ):
                prim.CreateAttribute(field, Sdf.ValueTypeNames.Double).Set(float(value))
        if prim.IsA(UsdPhysics.PrismaticJoint) or prim.IsA(UsdPhysics.RevoluteJoint):
            degrees = 180.0 / math.pi if prim.IsA(UsdPhysics.RevoluteJoint) else 1.0
            for field, upstream, scale in (
                ("armature", "armature", 1.0),
                ("damping", "damping", degrees),
                ("frictionloss", "friction", 1.0),
            ):
                value = prim.GetAttribute(f"newton:{upstream}").Get()
                if value is not None:
                    prim.CreateAttribute(f"mjc:{field}", Sdf.ValueTypeNames.Double).Set(
                        float(value) * scale
                    )
    stage.GetRootLayer().Save()


def _prepare_textures(xml: ET.Element, working: Path) -> None:
    textures: dict[str, set[Path]] = defaultdict(set)
    for texture in xml.iter("texture"):
        if filename := texture.get("file"):
            path = Path(filename)
            textures[path.name].add(path)
            if texture.get("type") == "cube":
                texture.set("type", "2d")
                texture.attrib.pop("gridsize", None)
                texture.attrib.pop("gridlayout", None)

    # The converter flattens its texture directory. Give different source
    # images with the same basename distinct names before conversion.
    renamed: dict[str, str] = {}
    for paths in textures.values():
        if len(paths) == 1:
            continue
        target_dir = working / "textures"
        target_dir.mkdir(exist_ok=True)
        for path in paths:
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()[:16]
            target = target_dir / f"{path.stem}-{digest}{path.suffix}"
            shutil.copyfile(path, target)
            renamed[str(path)] = str(target)
    for texture in xml.iter("texture"):
        filename = texture.get("file")
        if filename is not None and filename in renamed:
            texture.set("file", renamed[filename])


def _save_reference(
    path: Path,
    model: mj.MjModel,
    states: np.ndarray,
    actions: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    if (
        states.shape != (len(actions), 1 + model.nq + model.nv)
        or actions.shape[1] != 12
    ):
        raise ValueError("Unexpected PandaOmron recording schema.")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("The recording contains non-finite values.")
    time = states[:, 0] - states[0, 0]
    intervals = np.diff(time)
    # The demonstration recorder can leave one extra interval after the reset
    # snapshot before writing its first post-action state.
    if (
        len(intervals) < 1
        or not np.isclose(intervals[0], (0.05, 0.10)).any()
        or not np.allclose(intervals[1:], 0.05)
    ):
        raise ValueError("Expected a 20 Hz control recording.")
    qpos = states[:, 1 : 1 + model.nq]
    qvel = states[:, 1 + model.nq :]
    joints = [joint for part in _ROBOT_PARTS.values() for joint in part]
    pos_ids = [int(model.joint(joint).qposadr[0]) for joint in joints]
    vel_ids = [int(model.joint(joint).dofadr[0]) for joint in joints]
    target = qpos[:, pos_ids]
    excursion = float(np.linalg.norm(target[:, 7:9] - target[0, 7:9], axis=1).max())
    if excursion < 0.25:
        raise ValueError(f"The recorded base excursion is only {excursion:.3f} m.")

    data = mj.MjData(model)
    bias = np.empty_like(target)
    for i, (q, v) in enumerate(zip(qpos, qvel, strict=True)):
        data.qpos[:] = q
        data.qvel[:] = v
        mj.mj_forward(model, data)
        bias[i] = data.qfrc_bias[vel_ids]

    data.qpos[:] = qpos[0]
    data.qvel[:] = qvel[0]
    mj.mj_forward(model, data)
    movable: dict[str, dict[str, list[float]]] = {}
    for i, kind in enumerate(model.jnt_type):
        if int(kind) == int(mj.mjtJoint.mjJNT_FREE):
            body = int(model.jnt_bodyid[i])
            w, *xyz = map(float, data.xquat[body])
            velocity = np.empty(6)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body, velocity, 0)
            movable[model.body(body).name] = {
                "pose": [*data.xpos[body], *xyz, w],
                "velocity": np.concatenate([velocity[3:], velocity[:3]]).tolist(),
            }

    _save_goal_metadata(metadata, model, data, qpos)
    metadata.update(
        parts=_ROBOT_PARTS,
        initial_scalar_joints={
            model.joint(i).name: [
                float(qpos[0, model.jnt_qposadr[i]]),
                float(qvel[0, model.jnt_dofadr[i]]),
            ]
            for i, kind in enumerate(model.jnt_type)
            if int(kind) in _SCALAR_JOINT_TYPES
        },
        initial_free_bodies=movable,
        reference_base_excursion_m=excursion,
    )
    data.qpos[:] = qpos[-1]
    data.qvel[:] = qvel[-1]
    mj.mj_forward(model, data)
    metadata["reference_final_positions"] = {
        name: data.xpos[model.body(name).id].tolist() for name in movable
    }
    np.savez_compressed(
        path / "reference.npz",
        time=time,
        position=target,
        velocity=qvel[:, vel_ids],
        bias=bias,
        gripper=actions[:, -1],
    )


def _save_goal_metadata(
    metadata: dict[str, Any], model: mj.MjModel, initial: mj.MjData, qpos: np.ndarray
) -> None:
    refs = metadata["fixture_refs"]
    task = metadata["task"]
    if task == "OpenMicrowave":
        door = refs["fxtr"]
        metadata["target_objects"] = []
        _save_door_limits(metadata, model, door, ("microjoint", "_door_joint"))
        return
    if task == "SetBowlsForSoup":
        metadata["target_objects"] = [
            "bowl1_main",
            "bowl2_main",
            "plate1_main",
            "plate2_main",
        ]
        metadata["receptacle_thresholds"] = _receptacle_thresholds(
            model, ("plate1", "plate2")
        )
        return
    if task == "LoadPreparedFood":
        _save_refrigerator_metadata(metadata, model, initial)
        return

    goal = {
        "CollectWashingSupplies": "counter",
        "RefillCondimentStation": "dining_table",
        "TongBuffetSetup": "dining_counter",
    }[task]
    objects = {
        "CollectWashingSupplies": ("supply1_main", "supply2_main"),
        "RefillCondimentStation": ("condiment_main",),
        "TongBuffetSetup": ("tongs_main", "food_container_main"),
    }[task]
    metadata.update(support=refs[goal], target_objects=list(objects))
    if task == "CollectWashingSupplies":
        signs = np.asarray(list(product((-1, 1), repeat=3)))
        sink = model.geom(f"{refs['sink']}_reg_main").id
        sink_points = (signs * model.geom_size[sink]) @ initial.geom_xmat[sink].reshape(
            3, 3
        ).T + initial.geom_xpos[sink]
        metadata["sink_bounds"] = [
            sink_points.min(axis=0).tolist(),
            sink_points.max(axis=0).tolist(),
        ]
        metadata["object_bbox_points"] = {}
        for name in objects:
            geom = model.geom(f"{name.removesuffix('_main')}_reg_bbox").id
            metadata["object_bbox_points"][name] = (
                signs * model.geom_size[geom] + model.geom_pos[geom]
            ).tolist()
    else:
        fixture = refs["cab" if task == "RefillCondimentStation" else "drawer"]
        candidates = []
        for i in range(model.njnt):
            joint = model.joint(i)
            if (
                joint.name.startswith(fixture)
                and int(model.jnt_type[i]) in _SCALAR_JOINT_TYPES
            ):
                positions = qpos[:, model.jnt_qposadr[i]]
                delta = positions - positions[0]
                excursion = float(delta[np.abs(delta).argmax()])
                candidates.append(
                    (abs(excursion), joint.name, float(positions[0]), excursion)
                )
        _, joint, start, excursion = max(candidates)
        metadata["opening"] = {
            "fixture": fixture,
            "joint": joint,
            "start": start,
            "reference_excursion": excursion,
        }


def _receptacle_thresholds(
    model: mj.MjModel, names: tuple[str, ...]
) -> dict[str, float]:
    return {
        f"{name}_main": float(
            0.7 * np.linalg.norm(model.geom(f"{name}_reg_bbox").size[:2])
        )
        for name in names
    }


def _save_door_limits(
    metadata: dict[str, Any], model: mj.MjModel, fixture: str, suffixes: tuple[str, ...]
) -> None:
    limits = {
        model.joint(i).name: model.jnt_range[i].tolist()
        for i in range(model.njnt)
        if model.joint(i).name.startswith(fixture)
        and model.joint(i).name.endswith(suffixes)
    }
    if not limits:
        raise ValueError(f"Missing door joints on {fixture}.")
    metadata["door"] = {"fixture": fixture, "limits": limits}


def _save_refrigerator_metadata(
    metadata: dict[str, Any], model: mj.MjModel, initial: mj.MjData
) -> None:
    fixture = metadata["fixture_refs"]["fridge"]
    existing = [
        f"existing_item{i}_{rack}" for rack in ("top", "second") for i in (1, 2)
    ]
    existing.append("target_rack_third_item")
    items = ["tupperware", "meat_in_tupperware", "veg_in_tupperware", *existing]
    metadata["target_objects"] = [f"{name}_main" for name in items]
    metadata["receptacle_thresholds"] = _receptacle_thresholds(model, ("tupperware",))
    signs = np.asarray(list(product((-1, 1), repeat=3)))
    metadata["object_bbox_points"] = {}
    for name in ("tupperware", *existing):
        geom = model.geom(f"{name}_reg_bbox")
        metadata["object_bbox_points"][f"{name}_main"] = (
            signs * geom.size + geom.pos
        ).tolist()
    regions = [
        i
        for i in range(model.ngeom)
        if model.geom(i).name.startswith(f"{fixture}_reg_fridge_shelf")
        and model.geom_size[i, 2] >= 0.1
    ]
    regions.sort(key=lambda i: initial.geom_xpos[i, 2] - model.geom_size[i, 2])
    metadata["fridge_racks"] = {
        model.geom(i).name: {
            "center": initial.geom_xpos[i].tolist(),
            "axes": initial.geom_xmat[i].reshape(3, 3).tolist(),
            "half_size": model.geom_size[i].tolist(),
        }
        for i in regions
    }
    index = metadata["task_refs"]["target_rack_index"]
    metadata["target_rack"] = model.geom(regions[index]).name
    _save_door_limits(metadata, model, fixture, ("_fridge_door_joint",))


def _prepare_bundle(
    recording: tuple[ET.Element, dict[str, Any], np.ndarray, np.ndarray],
    *,
    task: str,
    episode: int,
    archive_sha256: str,
    robocasa_assets: Path,
    robosuite_assets: Path,
    converter_python: Path,
    output: Path,
) -> None:
    xml, source, states, actions = recording
    _resolve_source_files(xml, robocasa_assets, robosuite_assets)
    xml.set("model", "KitchenEpisode")
    model = mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    metadata = {
        "task": task,
        "episode": episode,
        "archive_sha256": archive_sha256,
        "layout": source["layout_id"],
        "style": source["style_id"],
        "instruction": source["lang"],
        "fixture_refs": source["fixture_refs"],
        "task_refs": source.get("refs", {}),
    }

    with tempfile.TemporaryDirectory(dir=output.parent) as working_dir:
        working = Path(working_dir)
        bundle = working / "bundle"
        _serialise_multi_joint_bodies(xml)
        for material in xml.iter("material"):
            for attr in ("metallic", "shininess"):
                value = material.get(attr)
                if value is not None and float(value) != -1:
                    material.set(attr, str(np.clip(float(value), 0.0, 1.0)))
        _prepare_textures(xml, working)
        source_xml = working / "serial.xml"
        ET.ElementTree(xml).write(source_xml, encoding="unicode")
        subprocess.run(
            [
                str(converter_python.absolute()),
                "-I",
                "-m",
                "mujoco_usd_converter",
                str(source_xml),
                str(bundle),
                "--no-layer-structure",
                "--no-openpbr",
            ],
            check=True,
        )
        converted_model = mj.MjModel.from_xml_path(str(source_xml))
        _normalise_usd(bundle / "KitchenEpisode.usdc", converted_model)
        _save_reference(bundle, model, states, actions, metadata)
        (bundle / "episode.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        credits = Path(__file__).parent
        shutil.copyfile(credits / "ATTRIBUTION.md", bundle / "ATTRIBUTION.md")
        shutil.copytree(credits / "LICENSES", bundle / "LICENSES")
        os.rename(bundle, output)


def main() -> None:
    """Prepare one selected demonstration and its source receipt."""
    from ._sources import _TASKS

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--task", choices=_TASKS, required=True)
    cli.add_argument("--archive", type=Path, required=True)
    cli.add_argument("--robocasa-assets", type=Path, required=True)
    cli.add_argument("--robosuite-assets", type=Path, required=True)
    cli.add_argument("--converter-python", type=Path, required=True)
    cli.add_argument("--episode", type=int)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    episode = _TASKS[args.task]["episode"] if args.episode is None else args.episode
    output = args.output.resolve()
    if output.exists():
        cli.error(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.archive) as archive:

        def read(path: str) -> bytes:
            stream = archive.extractfile(f"lerobot/{path}")
            if stream is None:
                raise ValueError(f"Invalid demonstration member {path!r}.")
            with stream:
                return stream.read()

        recording = _read_episode(read, episode, args.task)
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    _prepare_bundle(
        recording,
        task=args.task,
        episode=episode,
        archive_sha256=digest,
        robocasa_assets=args.robocasa_assets.resolve(),
        robosuite_assets=args.robosuite_assets.resolve(),
        converter_python=args.converter_python,
        output=output,
    )
    print(output)


if __name__ == "__main__":
    main()
