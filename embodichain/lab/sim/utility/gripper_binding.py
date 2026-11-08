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

"""Validate explicit hand bindings and derive solver TCPs after URDF assembly."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np
from embodichain.toolkits.graspkit.pg_grasp._urdf_geometry import _origin_transform

if TYPE_CHECKING:
    from embodichain.lab.sim.cfg import RobotCfg

__all__ = ["prepare_robot_grippers"]


def _fixed_transform(document: ET.Element, source: str, target: str) -> np.ndarray:
    if source == target:
        return np.eye(4)
    graph = {}
    for joint in document.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        a, b = joint.find("parent").get("link"), joint.find("child").get("link")
        transform = _origin_transform(joint.find("origin"))
        graph.setdefault(a, []).append((b, transform))
        graph.setdefault(b, []).append((a, np.linalg.inv(transform)))
    pending = deque([(source, np.eye(4))])
    seen = {source}
    while pending:
        node, transform = pending.popleft()
        for child, edge in graph.get(node, ()):
            if child == target:
                return transform @ edge
            if child not in seen:
                seen.add(child)
                pending.append((child, transform @ edge))
    raise ValueError(
        f"Solver end {source!r} and hand root {target!r} must share a fixed chain."
    )


def prepare_robot_grippers(cfg: RobotCfg) -> None:
    """Validate final hand topology and set effective TCPs before robot creation.

    Args:
        cfg: Complete selected RobotCfg with resolved gripper bindings. This
            updates its assembled path and effective solver TCPs idempotently.

    Raises:
        ValueError: A binding disagrees with actual hand joints or solver frames.
    """
    if not cfg.gripper_bindings:
        return
    if cfg.fpath is None:
        if cfg.urdf_cfg is None:
            raise ValueError("Gripper bindings require a concrete robot source.")
        cfg.fpath = cfg.urdf_cfg.assemble_urdf()
    document = ET.parse(Path(cfg.fpath)).getroot()
    joints = {j.get("name"): j for j in document.findall("joint")}
    links = {j.get("name") for j in document.findall("link")}
    movable = [
        j.get("name") for j in document.findall("joint") if j.get("type") != "fixed"
    ]
    override_limits = cfg.qpos_limits
    resolved_limits = {}
    if isinstance(override_limits, dict):
        from embodichain.utils.string import resolve_matching_names_values

        _, names, values = resolve_matching_names_values(override_limits, movable)
        resolved_limits = dict(zip(names, values))
    elif override_limits is not None:
        if hasattr(override_limits, "detach"):
            override_limits = override_limits.detach().cpu().numpy()
        values = np.asarray(override_limits)
        if values.shape != (len(movable), 2):
            raise ValueError(
                "qpos_limits must match source-order joints with shape (num_dofs, 2)."
            )
        resolved_limits = dict(zip(movable, values))
    for instance, binding in cfg.gripper_bindings.items():
        declaration = binding["declaration"]
        defaults = binding["defaults"]
        root = binding["root_link"]
        expected_control = [
            binding["joint_names"][native] for native in binding["control_joint_names"]
        ]
        if (cfg.control_parts or {}).get(
            declaration["control_part"]
        ) != expected_control:
            raise ValueError(
                "Actual hand control-part order disagrees with its binding."
            )
        if root not in links:
            raise ValueError(
                f"Hand {instance!r} root {root!r} absent from the actual robot."
            )
        for native, actual in binding["joint_names"].items():
            joint = joints.get(actual)
            if joint is None:
                raise ValueError(f"Bound hand joint {actual!r} is absent.")
            limit = joint.find("limit")
            if limit is None:
                raise ValueError("Bound hand joints require explicit source limits.")
            effective = resolved_limits.get(
                actual, [float(limit.get("lower")), float(limit.get("upper"))]
            )
            for command in defaults["joint_commands"].values():
                if not effective[0] - 1e-9 <= command[native] <= effective[1] + 1e-9:
                    raise ValueError(
                        f"Bound command exceeds actual joint {actual!r} limits."
                    )
            mimic = joint.find("mimic")
            if mimic is not None:
                parent = next(
                    (
                        n
                        for n, a in binding["joint_names"].items()
                        if a == mimic.get("joint")
                    ),
                    None,
                )
                if parent is None:
                    raise ValueError("A hand mimic cannot borrow another hand's joint.")
                for command in defaults["joint_commands"].values():
                    expected = command[parent] * float(
                        mimic.get("multiplier", "1")
                    ) + float(mimic.get("offset", "0"))
                    if not np.isclose(command[native], expected, rtol=0, atol=1e-8):
                        raise ValueError(
                            f"Command contradicts actual mimic {actual!r}."
                        )
        from embodichain.data import get_data_path
        import hashlib

        source_path = Path(get_data_path(defaults["fpath"]))
        if (
            defaults.get("source_sha256") is not None
            and hashlib.sha256(source_path.read_bytes()).hexdigest()
            != defaults["source_sha256"]
        ):
            raise ValueError(
                "Gripper source asset differs from its pinned description."
            )
        expected = ET.parse(source_path).getroot()
        source_joints = {j.get("name"): j for j in expected.findall("joint")}
        from embodichain.toolkits.urdf_assembly.name_normalizer import NameNormalizer

        normalizer = NameNormalizer(
            cfg.urdf_cfg.name_case
            if cfg.urdf_cfg is not None
            else {"joint": "original", "link": "original"}
        )
        prefixes = (
            dict(cfg.urdf_cfg.component_prefix) if cfg.urdf_cfg is not None else {}
        )
        prefix = prefixes.get(declaration.get("component_type"))
        subtree = {defaults["grasp_cfg"]["root_link"]}
        while True:
            children = {
                j.find("child").get("link")
                for j in source_joints.values()
                if j.find("parent").get("link") in subtree
            }
            if children <= subtree:
                break
            subtree |= children
        for native, a in source_joints.items():
            if a.find("parent").get("link") not in subtree:
                continue
            actual_name = normalizer.component_name("joint", native, prefix)
            b = joints.get(actual_name)
            if (
                b is None
                or a.get("type") != b.get("type")
                or not np.allclose(
                    _origin_transform(a.find("origin")),
                    _origin_transform(b.find("origin")),
                    rtol=0,
                    atol=1e-6,
                )
            ):
                raise ValueError(
                    "Actual gripper joint geometry disagrees with its model."
                )
            for tag in ("parent", "child"):
                mapped_link = normalizer.component_name(
                    "link", a.find(tag).get("link"), prefix
                )
                if b.find(tag).get("link") != mapped_link:
                    raise ValueError(
                        "Actual gripper joint topology disagrees with its model."
                    )
            axis_a = (
                a.find("axis").get("xyz", "1 0 0")
                if a.find("axis") is not None
                else "1 0 0"
            )
            axis_b = (
                b.find("axis").get("xyz", "1 0 0")
                if b.find("axis") is not None
                else "1 0 0"
            )
            if not np.allclose(
                np.fromstring(axis_a, sep=" "),
                np.fromstring(axis_b, sep=" "),
                rtol=0,
                atol=1e-7,
            ):
                raise ValueError("Actual gripper joint axis disagrees with its model.")
        source_path = Path(get_data_path(defaults["fpath"]))
        for native, actual in binding["link_names"].items():
            a = expected.find(f"./link[@name='{native}']")
            b = document.find(f"./link[@name='{actual}']")
            if b is None or len(a.findall("collision")) != len(b.findall("collision")):
                raise ValueError(
                    "Actual gripper collision links disagree with the model."
                )
            for ca, cb in zip(a.findall("collision"), b.findall("collision")):
                if not np.allclose(
                    _origin_transform(ca.find("origin")),
                    _origin_transform(cb.find("origin")),
                    rtol=0,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "Actual gripper collision origin disagrees with the model."
                    )
                ma, mb = ca.find("geometry/mesh"), cb.find("geometry/mesh")
                if ma is not None:
                    if mb is None:
                        raise ValueError("Actual gripper mesh is missing.")
                    if not np.allclose(
                        np.fromstring(ma.get("scale", "1 1 1"), sep=" "),
                        np.fromstring(mb.get("scale", "1 1 1"), sep=" "),
                        rtol=0,
                        atol=1e-9,
                    ):
                        raise ValueError(
                            "Actual gripper mesh scale disagrees with its model."
                        )
                    filename = ma.get("filename")
                    original = (
                        get_data_path(defaults["mesh_paths"][filename])
                        if filename in defaults.get("mesh_paths", {})
                        else source_path.parent / filename
                    )
                    actual_path = Path(cfg.fpath).parent / mb.get("filename")
                    if (
                        hashlib.sha256(Path(original).read_bytes()).digest()
                        != hashlib.sha256(actual_path.read_bytes()).digest()
                    ):
                        raise ValueError(
                            "Actual gripper mesh differs from the selected asset."
                        )
                else:
                    ga, gb = ca.find("geometry"), cb.find("geometry")
                    if len(ga) != len(gb):
                        raise ValueError(
                            "Actual gripper primitive geometry disagrees with its model."
                        )
                    for pa, pb in zip(ga, gb):
                        if (
                            pa.tag != pb.tag
                            or pa.attrib.keys() != pb.attrib.keys()
                            or any(
                                not np.allclose(
                                    np.fromstring(pa.get(k), sep=" "),
                                    np.fromstring(pb.get(k), sep=" "),
                                    rtol=0,
                                    atol=1e-9,
                                )
                                for k in pa.attrib
                            )
                        ):
                            raise ValueError(
                                "Actual gripper primitive geometry disagrees with its model."
                            )
        if declaration["mode"] == "mounted":
            connector = next(
                (
                    j
                    for j in document.findall("joint")
                    if j.find("child").get("link") == root
                ),
                None,
            )
            if connector is None or connector.get("type") != "fixed":
                raise ValueError("A mounted gripper requires a fixed connector.")
            declared_mount = np.asarray(declaration.get("transform", np.eye(4)))
            if not np.allclose(
                _origin_transform(connector.find("origin")),
                declared_mount,
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError(
                    "Actual gripper installation disagrees with its declared mount."
                )
        solver = (cfg.solver_cfg or {}).get(declaration["motion_part"])
        if solver is None:
            continue
        solver_end = solver.end_link_name
        if solver_end not in links:
            solver_end = normalizer.component_name("link", solver_end, prefix)
        end_from_root = _fixed_transform(document, solver_end, root)
        if declaration["mode"] == "mounted":
            # Retain authored matrix precision rather than the assembler's RPY formatting.
            end_from_parent = _fixed_transform(
                document, solver_end, connector.find("parent").get("link")
            )
            end_from_root = end_from_parent @ np.asarray(
                declaration.get("transform", np.eye(4))
            )
        tcp = end_from_root @ np.asarray(defaults["default_root_from_tcp"])
        explicit = binding.get("explicit_solver_tcp")
        if explicit is not None and not np.allclose(explicit, tcp, rtol=0, atol=1e-6):
            raise ValueError(
                "Explicit solver TCP disagrees with the selected gripper binding."
            )
        solver.tcp = tcp.tolist()
    movable = [
        j.get("name") for j in document.findall("joint") if j.get("type") != "fixed"
    ]
    current = cfg.init_qpos
    if current is None or len(current) != len(movable):
        opened = {
            actual: binding["defaults"]["joint_commands"]["open"][native]
            for binding in cfg.gripper_bindings.values()
            for native, actual in binding["joint_names"].items()
        }
        others = [n for n in movable if n not in opened]
        if current is not None and len(current) != len(others):
            raise ValueError(
                "Robot initial positions must cover the body or complete assembled source order."
            )
        values = dict(
            zip(others, current if current is not None else [0.0] * len(others))
        )
        values.update(opened)
        cfg.init_qpos = [values[n] for n in movable]


def _apply_gripper_tcp_defaults(cfg: RobotCfg, solver_overrides: object = None) -> None:
    """Project declared local references without resolving source assets."""
    for binding in cfg.gripper_bindings.values():
        declaration, defaults = binding["declaration"], binding["defaults"]
        solver = (cfg.solver_cfg or {}).get(declaration["motion_part"])
        if solver is None or not hasattr(solver, "end_link_name"):
            continue
        if declaration["mode"] == "mounted":
            end_from_root = np.asarray(declaration.get("transform", np.eye(4)))
        elif solver.end_link_name == binding.get("reference_link"):
            end_from_root = np.linalg.inv(np.asarray(defaults["root_from_reference"]))
        else:
            # Custom end links require the actual fixed-chain check at assembly.
            continue
        tcp = end_from_root @ np.asarray(defaults["default_root_from_tcp"])
        explicit = binding.get("explicit_solver_tcp")
        if isinstance(solver_overrides, dict):
            provided = solver_overrides.get(declaration["motion_part"])
            if isinstance(provided, dict) and provided.get("tcp") is not None:
                explicit = provided["tcp"]
            elif provided is not None and getattr(provided, "tcp", None) is not None:
                explicit = provided.tcp
        if explicit is not None and not np.allclose(explicit, tcp, rtol=0, atol=1e-6):
            raise ValueError("Explicit solver TCP disagrees with the selected gripper.")
        solver.tcp = tcp.tolist()
