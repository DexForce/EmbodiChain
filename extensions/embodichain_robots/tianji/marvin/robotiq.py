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

"""Complete Marvin body with explicitly fixed Robotiq 2F-140 wrist assemblies."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import torch

from embodichain.data import get_data_path
from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    JointDrivePropertiesCfg,
    RobotCfg,
)
from embodichain.lab.sim.motion.solvers import PytorchSolverCfg
from embodichain.lab.sim.utility.cfg_utils import merge_robot_cfg
from embodichain.utils import configclass

if TYPE_CHECKING:
    import pytorch_kinematics as pk

__all__ = ["TianjiMarvinRobotiq140Cfg"]

# Increment when the model-owned assembly recipe changes.
_ASSEMBLY_REVISION = "1"
_TCP = [
    [0.0, -1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.2],
    [0.0, 0.0, 0.0, 1.0],
]
_HAND_JOINTS = {
    "left_eef": [
        "left_finger_joint",
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "left_right_outer_knuckle_joint",
        "left_right_inner_knuckle_joint",
        "left_right_inner_finger_joint",
    ],
    "right_eef": [
        "right_finger_joint",
        "right_left_inner_knuckle_joint",
        "right_left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    ],
}


def _write_assembly(source: Path, hand_path: str, output_dir: Path) -> Path:
    """Keep the entire bare robot and attach the same two processed Robotiq hands."""
    from embodichain.toolkits.urdf_assembly import URDFAssemblyManager

    bare = ET.parse(source).getroot()
    source_links = {link.attrib["name"] for link in bare.findall("link")}
    if not {"left_ee", "right_ee", "left_arm_base", "right_arm_base"} <= source_links:
        raise ValueError("TianjiMarvin bare URDF lacks the declared arm/tool frames.")
    movable = [j for j in bare.findall("joint") if j.get("type") != "fixed"]
    if len(movable) != 14 or any(j.find("mimic") is not None for j in movable):
        raise ValueError(
            "TianjiMarvin requires the complete bare fourteen-joint model."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manager = URDFAssemblyManager()
    manager.base_link_name = "gen_sim_base"
    manager.name_case = {"joint": "original", "link": "original"}
    if not manager.add_component("chassis", str(source)):
        raise ValueError("Unable to assemble the TianjiMarvin body.")
    for side in ("left", "right"):
        if not manager.add_component(f"{side}_hand", hand_path):
            raise ValueError(f"Unable to assemble the {side} Robotiq hand.")
    path = output_dir / "tianji_marvin_robotiq_arg2f_140.urdf"
    root = manager.merge_urdfs(str(path), use_signature_check=False)
    # Whole-body URDFs are not split into the assembler's left/right arm slots.
    # Add explicit fixed wrist mounts without changing body or hand kinematics.
    for side in ("left", "right"):
        hand_root = f"{side}_robotiq_arg2f_base_link"
        connectors = [
            joint
            for joint in root.findall("joint")
            if joint.find("child").get("link") == hand_root
        ]
        if not connectors:
            joint = ET.SubElement(
                root, "joint", name=f"{side}_robotiq_mount", type="fixed"
            )
            ET.SubElement(joint, "parent", link=f"{side}_ee")
            ET.SubElement(joint, "child", link=hand_root)
            ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
        elif len(connectors) != 1 or connectors[0].get("type") != "fixed":
            raise ValueError(f"Ambiguous {side} Robotiq mounting connector.")
        else:
            connectors[0].find("parent").set("link", f"{side}_ee")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def _assembled_model(directory: str | None) -> str:
    """Publish the deterministic model atomically; catalog imports perform no IO."""
    source = Path(get_data_path("TianjiMarvin/robot_acd.urdf"))
    hand = Path(get_data_path("Robotiq/robotiq_arg2f_140/robotiq_arg2f_140.urdf"))
    parent = (
        Path(directory)
        if directory is not None
        else Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
        / "assembled"
        / "tianji_marvin_robotiq_arg2f_140"
    )
    if any(
        parent.resolve().is_relative_to(model.parent.resolve())
        for model in (source, hand)
    ):
        raise ValueError(
            "Assembly output must be outside the source asset directories."
        )
    digest = hashlib.sha256(_ASSEMBLY_REVISION.encode())
    # Include model resources, not just URDF text: in-place mesh/texture repairs
    # must not select a stale generated model. Stream files to bound memory.
    for model in (source, hand):
        digest.update(str(model.resolve()).encode() + b"\0")
        for resource in sorted(model.parent.rglob("*")):
            if not resource.is_file():
                continue
            digest.update(str(resource.relative_to(model.parent)).encode() + b"\0")
            content = hashlib.sha256()
            with resource.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    content.update(block)
            digest.update(content.digest())
    output = parent / digest.hexdigest()
    parent.mkdir(parents=True, exist_ok=True)
    target = output / "tianji_marvin_robotiq_arg2f_140.urdf"
    if not target.is_file():
        if output.exists():
            raise ValueError(f"Incomplete generated model cache: {output}")
        with TemporaryDirectory(prefix=".assembly-", dir=parent) as temporary:
            _write_assembly(source, str(hand), Path(temporary))
            # The URDF references generated Visual/Collision files beside it.
            # Publish the entire model so readers never see a partial resource set.
            try:
                os.rename(temporary, output)
            except OSError:
                if not target.is_file():
                    raise
    return str(target)


@configclass
class TianjiMarvinRobotiq140Cfg(RobotCfg):
    """Assemble the full fourteen-joint bare body with two six-joint Robotiq hands.

    This preset preserves the source fixed wrist mounts and TCP calibration. It
    does not inherit Marvin's native-gripper TCP, SSIK solver, cameras or material
    overlays. The reusable embodiment owns the explicit zero home, sampling
    budget, solver tolerance and simulation armature calibration; task-dependent
    gain/friction overrides remain deployment declarations. Catalog listing is
    lazy; model assembly occurs only when building an actual configuration.
    """

    assembly_directory: str | None = None
    """Optional generated-model cache root; the standard data assembly cache is used otherwise."""

    @classmethod
    def from_dict(cls, init_dict: dict[str, Any]) -> TianjiMarvinRobotiq140Cfg:
        """Build assembly defaults and apply independently owned overrides.

        Args:
            init_dict: Robot fields; an explicit fpath bypasses model assembly.

        Returns:
            A complete model configuration with both arm chains and full hands.
        """
        cfg = cls()
        cfg._build_defaults(init_dict)
        return merge_robot_cfg(cfg, init_dict)

    def _build_defaults(self, init_dict: dict[str, Any] | None = None) -> None:
        values = init_dict or {}
        self.assembly_directory = values.get(
            "assembly_directory", self.assembly_directory
        )
        if self.assembly_directory is not None and (
            type(self.assembly_directory) is not str or not self.assembly_directory
        ):
            raise ValueError("assembly_directory must be a nonempty path or None.")
        self.uid = "TianjiMarvin"
        self.fpath = values.get("fpath") or _assembled_model(self.assembly_directory)
        self.control_parts = {
            f"{side}_arm": [
                f"{joint}_{label}_J{index}"
                for index, joint in enumerate(
                    (
                        "SHOULDER_PITCH",
                        "SHOULDER_ROLL",
                        "ELBOW_PITCH",
                        "ELBOW_YAW",
                        "WRIST_PITCH",
                        "WRIST_YAW",
                        "WRIST_ROLL",
                    ),
                    1,
                )
            ]
            for side, label in (("left", "L"), ("right", "R"))
        }
        self.control_parts.update(
            {part: list(names) for part, names in _HAND_JOINTS.items()}
        )
        self.control_parts["dual_arm"] = [
            *self.control_parts["left_arm"],
            *self.control_parts["right_arm"],
        ]
        self.solver_cfg = {
            f"{side}_arm": PytorchSolverCfg(
                root_link_name=f"{side}_arm_base",
                end_link_name=f"{side}_ee",
                tcp=[list(row) for row in _TCP],
            )
            for side in ("left", "right")
        }
        self.joint_drive_props = JointDrivePropertiesCfg(
            drive_type="force",
            friction=0.0,
            armature=0.0,
            stiffness={
                "left_arm": 7e4,
                "right_arm": 7e4,
                "left_eef": 50.0,
                "right_eef": 50.0,
            },
            damping={
                "left_arm": 1e3,
                "right_arm": 1e3,
                "left_eef": 5.0,
                "right_eef": 5.0,
            },
            max_effort={
                "left_arm": 3e6,
                "right_arm": 3e6,
                "left_eef": 500.0,
                "right_eef": 500.0,
            },
        )
        self.root_props = ArticulationRootPropertiesCfg(
            fixed_base=True, min_position_iters=8, min_velocity_iters=2
        )
        self.sensors = []

    @property
    def _pk_urdf_path(self) -> str:
        if self.fpath is None:
            raise ValueError(
                "Build the robot configuration before requesting its arm chains."
            )
        return self.fpath

    def build_pk_serial_chain(
        self, device: torch.device = torch.device("cpu"), **kwargs: Any
    ) -> dict[str, pk.SerialChain]:
        """Build raw seven-joint arm chains from the exact simulation assembly.

        Args:
            device: Device for the kinematics tensors.
            **kwargs: Reserved RobotCfg protocol arguments.

        Returns:
            Left/right chains ending at the authored wrists. Apply the declared
            solver TCP separately; no hand-joint serial chain is inferred.
        """
        from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain

        return {
            f"{side}_arm": create_pk_serial_chain(
                urdf_path=self._pk_urdf_path,
                device=device,
                root_link_name=f"{side}_arm_base",
                end_link_name=f"{side}_ee",
            )
            for side in ("left", "right")
        }
