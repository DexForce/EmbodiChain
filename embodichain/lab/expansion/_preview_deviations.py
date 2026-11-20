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

"""Time-aligned deviations from the fixed reference in the expansion viewer."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from embodichain.robot_registry import _validate_gripper_transform
from embodichain.utils.string import resolve_matching_names

if TYPE_CHECKING:
    from ._preview_data import PreviewClip, PreviewData

__all__: list[str] = []


def recorded_coordinates(
    robot: dict, endpoint: str, model: ET.Element
) -> tuple[dict[str, list[str]], tuple[str, np.ndarray] | None]:
    """Resolve the selected motion group and its own recorded gripper calibration."""
    solvers = robot.get("solver_cfg") or {}
    bindings = list((robot.get("gripper_bindings") or {}).values())
    parts = [
        part for part, cfg in solvers.items() if cfg.get("end_link_name") == endpoint
    ]

    def motion_part(binding: dict) -> str | None:
        declaration = binding.get("declaration") or {}
        defaults = binding.get("defaults", {}).get("binding") or {}
        return declaration.get("motion_part") or defaults.get("motion_part")

    if len(parts) == 1:
        part = parts[0]
        matched = [b for b in bindings if motion_part(b) == part]
        if len(bindings) == len(solvers) == 1 and motion_part(bindings[0]) is None:
            matched = bindings
    elif not parts:
        matched = [b for b in bindings if b.get("root_link") == endpoint]
        if len(matched) != 1:
            return {}, None
        part = motion_part(matched[0])
    else:
        return {}, None
    solver = solvers.get(part, {})
    names = solver.get("joint_names") or (robot.get("control_parts") or {}).get(part)
    joints = {j.get("name"): j for j in model.findall("joint")}
    units = {"continuous": "deg", "revolute": "deg", "prismatic": "mm"}
    if names and any(name not in joints for name in names):
        try:
            _, names = resolve_matching_names(
                names, [n for n, j in joints.items() if j.get("type") in units]
            )
        except (ValueError, re.error):
            names = []
    elif not names:
        # Older jobs can declare just the solver's root and end links.
        parents = {j.find("child").get("link"): j for j in joints.values()}
        link = solver.get("end_link_name") or endpoint
        root = solver.get("root_link_name")
        path = []
        while link != root and link in parents:
            joint = parents.pop(link)
            if joint.get("type") != "fixed":
                path.append(joint.get("name"))
            link = joint.find("parent").get("link")
        names = list(reversed(path)) if root is None or link == root else []

    groups: dict[str, list[str]] = {}
    for name in names:
        if name not in joints or joints[name].get("type") not in units:
            groups = {}
            break
        groups.setdefault(units[joints[name].get("type")], []).append(name)

    if len(matched) == 1:
        binding = matched[0]
        offset = binding.get("defaults", {}).get("default_root_from_tcp")
        if binding.get("root_link") and offset is not None:
            try:
                return groups, (
                    binding["root_link"],
                    np.asarray(_validate_gripper_transform(offset)),
                )
            except ValueError:
                pass  # The joint measurements still support a qpos comparison.
    return groups, None


def _tcp(clip: PreviewClip, links: dict[str, int]) -> np.ndarray:
    root, offset = clip.tcp_frame
    return clip.links[:, links[root]] @ offset


def deviation_curves(data: PreviewData, link_names: list[str]) -> dict[str, Any]:
    """Report the range and mean of each deviation on the reference clock."""
    ref = data.solid_clip
    times = ref.measured.timestamps
    candidates = [data.clips[i] for i in data.ghost_indices]
    counts = np.zeros(len(times), dtype=np.int64)
    sums = {}
    links = {name: i for i, name in enumerate(link_names)}
    tcp_available = all(
        c.tcp_frame is not None and c.tcp_frame[0] in links for c in (ref, *candidates)
    )
    if tcp_available:
        original = _tcp(ref, links)
        original_inverse = Rotation.from_matrix(original[:, :3, :3]).inv()
        sums["translation_mm"] = np.zeros(len(times))
        sums["rotation_deg"] = np.zeros(len(times))
    for unit in data.arm_joint_groups:
        sums[f"qpos_{unit}"] = np.zeros(len(times))
    ranges = {key: (np.full(len(times), np.inf), np.zeros(len(times))) for key in sums}

    def accumulate(key: str, values: np.ndarray) -> None:
        end = len(values)
        sums[key][:end] += values
        low, high = ranges[key]
        np.minimum(low[:end], values, out=low[:end])
        np.maximum(high[:end], values, out=high[:end])

    for c in candidates:
        sample = c.measured
        end = times.searchsorted(sample.timestamps[-1], side="right")
        grid = times[:end]
        counts[:end] += 1
        if tcp_available:
            tcp = _tcp(c, links)
            xyz = np.column_stack(
                [
                    np.interp(grid, sample.timestamps, tcp[:, axis, 3])
                    for axis in range(3)
                ]
            )
            rotations = Slerp(sample.timestamps, Rotation.from_matrix(tcp[:, :3, :3]))(
                grid
            )
            accumulate(
                "translation_mm",
                1000 * np.linalg.norm(xyz - original[:end, :3, 3], axis=1),
            )
            accumulate(
                "rotation_deg",
                np.rad2deg((original_inverse[:end] * rotations).magnitude()),
            )
        for unit, names in data.arm_joint_groups.items():
            worst = np.zeros(end)
            for name in names:
                q = sample.qpos[:, sample.joint_names.index(name)]
                q_ref = ref.measured.qpos[:end, ref.measured.joint_names.index(name)]
                delta = np.interp(grid, sample.timestamps, q) - q_ref
                np.maximum(worst, np.abs(delta), out=worst)
            accumulate(
                f"qpos_{unit}", np.rad2deg(worst) if unit == "deg" else 1000 * worst
            )

    # Every recording starts at zero, so uncovered reference samples form a tail.
    covered = np.count_nonzero(counts)
    empty_tail = [None] * (len(times) - covered)
    return {
        "time_s": times.tolist(),
        "count": counts.tolist(),
        "episodes": len(candidates),
        "explicit_reference": data.reference is not None,
        "arm_joint_groups": data.arm_joint_groups,
        "mean": {
            key: (values[:covered] / counts[:covered]).tolist() + empty_tail
            for key, values in sums.items()
        },
        "range": {
            key: {
                "min": low[:covered].tolist() + empty_tail,
                "max": high[:covered].tolist() + empty_tail,
            }
            for key, (low, high) in ranges.items()
        },
    }
