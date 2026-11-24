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

"""Mount the official RP Hands in a fixed carrying pose."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from embodichain.data import get_data_path

__all__: list[str] = []

RP_HAND_ASSET = "RoboPartyRP1/mjcf/robot_hand.xml"


def _numbers(values: np.ndarray) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def add_rp_hands(scene: ET.Element) -> None:
    """Rigidly merge each official hand into the corresponding wrist."""
    source_path = Path(get_data_path(RP_HAND_ASSET))
    source = ET.parse(source_path).getroot()
    model = mujoco.MjModel.from_xml_path(str(source_path))
    data = mujoco.MjData(model)
    _set_carrying_pose(source, model, data)
    mujoco.mj_kinematics(model, data)

    defaults = ET.SubElement(scene.find("default"), "default", {"class": "hand"})
    defaults.append(deepcopy(source.find("default/default[@class='hand']/geom")))
    mesh_root = source_path.parent / source.find("compiler").get("meshdir", "")
    meshes = {
        node.get("name", Path(node.attrib["file"]).stem): node
        for node in source.findall("asset/mesh")
    }
    asset = scene.find("asset")
    for side, prefix in (("left", "lh_"), ("right", "rh_")):
        wrist_name = f"{side}_wrist_pitch_link"
        wrist = scene.find(f".//body[@name='{wrist_name}']")
        hand = source.find(f".//body[@name='{prefix}base_link']")
        wrist_index = model.body(wrist_name).id
        origin = data.xpos[wrist_index]
        to_wrist = Rotation.from_quat(data.xquat[wrist_index, [1, 2, 3, 0]]).inv()
        masses, centers, tensors = _wrist_inertia(wrist)
        for link in hand.iter("body"):
            index = model.body(link.attrib["name"]).id
            position = to_wrist.apply(data.xpos[index] - origin)
            rotation = to_wrist * Rotation.from_quat(data.xquat[index, [1, 2, 3, 0]])
            for shape in link.findall("geom"):
                geom = deepcopy(shape)
                geom.attrib.setdefault("class", "hand")
                if "fromto" in geom.attrib:
                    endpoints = np.fromstring(geom.attrib["fromto"], sep=" ").reshape(
                        2, 3
                    )
                    geom.set(
                        "fromto",
                        _numbers((rotation.apply(endpoints) + position).ravel()),
                    )
                else:
                    pos = np.fromstring(geom.get("pos", "0 0 0"), sep=" ")
                    quat = np.fromstring(geom.get("quat", "1 0 0 0"), sep=" ")
                    orientation = rotation * Rotation.from_quat(quat[[1, 2, 3, 0]])
                    geom.set("pos", _numbers(rotation.apply(pos) + position))
                    geom.set("quat", _numbers(orientation.as_quat()[[3, 0, 1, 2]]))
                if (mesh_name := geom.get("mesh")) is not None:
                    mesh = deepcopy(meshes[mesh_name])
                    mesh.set("file", str((mesh_root / mesh.attrib["file"]).resolve()))
                    asset.append(mesh)
                wrist.append(geom)

            masses.append(model.body_mass[index])
            centers.append(to_wrist.apply(data.xipos[index] - origin))
            axes = to_wrist.as_matrix() @ data.ximat[index].reshape(3, 3)
            tensors.append(axes @ np.diag(model.body_inertia[index]) @ axes.T)
        _set_combined_inertia(wrist.find("inertial"), masses, centers, tensors)


def _set_carrying_pose(
    source: ET.Element, model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    # Set the six driven joints per hand and evaluate the official mimic curves.
    for side in ("lh", "rh"):
        data.joint(f"{side}_finger11").qpos[:] = 0.18
        data.joint(f"{side}_finger12").qpos[:] = 0.22
        for finger in range(2, 6):
            data.joint(f"{side}_finger{finger}1").qpos[:] = 0.53
    for equality in source.findall("equality/joint"):
        follower = equality.attrib["joint1"]
        leader = equality.attrib["joint2"]
        coeffs = np.fromstring(equality.attrib["polycoef"], sep=" ")
        angle = np.polynomial.polynomial.polyval(data.joint(leader).qpos, coeffs)
        data.joint(follower).qpos[:] = np.clip(angle, *model.joint(follower).range)


def _wrist_inertia(
    wrist: ET.Element,
) -> tuple[list[float], list[np.ndarray], list[np.ndarray]]:
    inertial = wrist.find("inertial")
    xx, yy, zz, xy, xz, yz = np.fromstring(inertial.attrib["fullinertia"], sep=" ")
    tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return (
        [float(inertial.attrib["mass"])],
        [np.fromstring(inertial.attrib["pos"], sep=" ")],
        [tensor],
    )


def _set_combined_inertia(
    inertial: ET.Element,
    masses: list[float],
    centers: list[np.ndarray],
    tensors: list[np.ndarray],
) -> None:
    mass = sum(masses)
    center = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for component_mass, component_center, tensor in zip(
        masses, centers, tensors, strict=True
    ):
        offset = component_center - center
        inertia += tensor + component_mass * (
            offset @ offset * np.eye(3) - np.outer(offset, offset)
        )
    inertial.set("mass", str(mass))
    inertial.set("pos", _numbers(center))
    inertial.set(
        "fullinertia", _numbers(inertia[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]])
    )
