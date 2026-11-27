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

"""Resolve configured motion/hand pairs without interpreting their names."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING
import xml.etree.ElementTree as ET

import numpy as np

if TYPE_CHECKING:
    from ._robot import RobotProfile

__all__ = []


def _candidate_pairs(profile: RobotProfile) -> list[dict[str, str]]:
    pairs = {(p["motion"], p["grasp"]) for p in profile.participants.values()}
    documents = {}
    for arm, solver in profile.solvers.items():
        path = Path(solver["urdf_path"])
        if path not in documents:
            root = ET.parse(path).getroot()
            documents[path] = (
                {j.attrib["name"]: j for j in root.findall("joint")},
                {j.find("child").attrib["link"]: j for j in root.findall("joint")},
            )
        joints, by_child = documents[path]
        arm_joints = set(profile.control_parts[arm])
        for hand in profile.hand_commands:
            if hand not in profile.control_parts:
                continue
            upstream = []
            for name in profile.control_parts[hand]:
                joint = joints.get(name)
                ancestors = set()
                while joint is not None:
                    ancestors.add(joint.attrib["name"])
                    joint = by_child.get(joint.find("parent").attrib["link"])
                upstream.append(ancestors)
            if upstream and all(arm_joints <= chain for chain in upstream):
                pairs.add((arm, hand))
    return [dict(motion=arm, grasp=hand) for arm, hand in sorted(pairs)]


def select_participants(
    profile: RobotProfile,
    qpos: np.ndarray,
    *,
    skill_id: str,
    selection: str,
    phase_hints: list[dict[str, Any]] | None,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    """Choose a single observed closing hand, retaining explicit handover roles.

    Configured joint lists and URDF ancestry pair endpoints. Selection never
    depends on words such as ``left``, ``arm`` or ``eef`` in their names.
    Ambiguous or unobservable closures require an explicit endpoint choice.
    """
    from embodichain.lab.sim.atomic_skills.fitting import _transitions

    if selection == "configured" or skill_id == "hand_over":
        return dict(profile.participants), {"mode": "configured"}
    pairs = _candidate_pairs(profile)
    if selection != "auto":
        matches = [p for p in pairs if selection in (p["motion"], p["grasp"])]
        if len(matches) != 1:
            raise ValueError(
                f"Participant {selection!r} must identify one configured motion/hand pair."
            )
        return {"primary": matches[0]}, {"mode": "explicit"}
    evidence = []
    eligible = []
    for pair in pairs:
        hand = pair["grasp"]
        ids = [profile.joint_names.index(n) for n in profile.control_parts[hand]]
        commands = profile.hand_commands[hand]
        closes, opens = _transitions(
            qpos[:, ids], np.asarray(commands["open"]), np.asarray(commands["grasp"])
        )
        evidence.append({**pair, "closures": closes, "openings": opens})
        if len(closes) == 1:
            eligible.append(pair)
    # A single explicitly configured manipulator can use supplied phase hints
    # when measured hand positions do not reveal the command event.
    if not eligible and len(pairs) == 1 and phase_hints:
        eligible = pairs
    if len(eligible) != 1:
        names = ", ".join(p["motion"] for p in pairs)
        raise ValueError(
            "Cannot automatically identify one participating hand from closure evidence; "
            f"select --participant from [{names}] or use configured bindings."
        )
    selected = eligible[0]
    if sum(p["grasp"] == selected["grasp"] for p in pairs) != 1:
        raise ValueError(
            "The closing hand belongs to multiple motion groups; select --participant explicitly."
        )
    return {"primary": selected}, {"mode": "auto", "evidence": evidence}
