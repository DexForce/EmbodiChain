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

"""Explicit independent and URDF-mimic gripper contact projection.

Projection predicts where a closing command first meets the object. It is
separate from measured post-execution contact/effect validation and never
changes the submitted controller targets.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from xml.etree import ElementTree

import numpy as np

from embodichain.utils import configclass
from .real2sim_collision import SampledRobotCollisionChecker

__all__ = ["GripperContactProjectionCfg", "GripperContactProjector"]


@configclass
class GripperContactProjectionCfg:
    """Select contact prediction from explicitly declared gripper mechanics.

    Args:
        kind: Independent prismatic fingers or a single URDF mimic drive.
        driver_joint: Required independent drive for coupled mimic mechanics.
        mechanism_contacts: Explicit internal linkage contacts and depth bounds
            in metres, separate from object/environment contact permissions.
        measured_coupling_tolerance: Maximum observed follower error in joint
            units. Command paths always retain the strict 1e-6 coupling check.
    """

    kind: str = "independent_prismatic"
    driver_joint: str | None = None
    mechanism_contacts: tuple[tuple[str, str, float], ...] = ()
    measured_coupling_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if (
            type(self.measured_coupling_tolerance) not in (int, float)
            or not np.isfinite(self.measured_coupling_tolerance)
            or not 0 <= self.measured_coupling_tolerance <= 0.01
        ):
            raise ValueError("measured coupling tolerance must lie in [0, 0.01]")
        if self.kind not in {"independent_prismatic", "coupled_mimic"}:
            raise ValueError("unknown gripper contact projection kind")
        if self.kind == "coupled_mimic":
            if not isinstance(self.driver_joint, str) or not self.driver_joint.strip():
                raise ValueError("coupled gripper requires an explicit driver_joint")
        elif self.driver_joint is not None:
            raise ValueError("independent gripper cannot declare a shared driver")
        contacts = []
        pairs = set()
        for item in self.mechanism_contacts:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError("mechanism contact needs two links and a depth bound")
            first, second, depth = item
            if (
                not isinstance(first, str)
                or not first
                or not isinstance(second, str)
                or not second
                or first == second
                or type(depth) not in (int, float)
                or not np.isfinite(depth)
                or not 0 <= depth <= 0.05
            ):
                raise ValueError("invalid bounded mechanism contact")
            pair = tuple(sorted((first, second)))
            if pair in pairs:
                raise ValueError("duplicate mechanism contact")
            pairs.add(pair)
            contacts.append((*pair, float(depth)))
        if contacts and self.kind != "coupled_mimic":
            raise ValueError("mechanism contacts require a coupled mimic declaration")
        self.mechanism_contacts = tuple(contacts)


class GripperContactProjector:
    """Predict a contact-limited joint path without mutating a nominal plan.

    The coupled route obtains mimic multipliers and offsets from the exact
    collision URDF, validates its complete descendant layout, and advances every
    linked joint with one drive coordinate. Only explicitly selected deployments
    are qualified; this representation alone does not establish physical contact.

    Args:
        cfg: Explicit contact model selection.
        checker: Collision checker with this robot's full joint layout.
        urdf_path: Same assembled URDF used by the checker.
        joint_names: Complete source-order joint layout.
        finger_links: Collision links that may establish initial grasp contact.
        finger_joints: Complete gripper drive/follower joint declaration.
        closing_directions: Closing sign of each declared joint.
    """

    def __init__(
        self,
        cfg: GripperContactProjectionCfg,
        checker: SampledRobotCollisionChecker,
        *,
        urdf_path: str | Path | None,
        joint_names: Sequence[str],
        finger_links: Sequence[str],
        finger_joints: Sequence[str],
        closing_directions: Sequence[int],
    ) -> None:
        if not isinstance(cfg, GripperContactProjectionCfg):
            raise TypeError("gripper contact requires a typed projection config")
        self.cfg = cfg.copy()
        self.cfg.__post_init__()
        self.cfg.validate()
        cfg = self.cfg
        self._checker = checker
        self._names = tuple(joint_names)
        self._links = tuple(finger_links)
        fingers = tuple(finger_joints)
        self._directions = tuple(closing_directions)
        if (
            not self._links
            or len(set(self._links)) != len(self._links)
            or not fingers
            or len(set(fingers)) != len(fingers)
            or any(name not in self._names for name in fingers)
            or len(fingers) != len(self._directions)
            or any(
                type(sign) is not int or sign not in (-1, 1)
                for sign in self._directions
            )
        ):
            raise ValueError("invalid explicit gripper joint/link declarations")
        self.joint_indices = tuple(self._names.index(name) for name in fingers)
        self._driver = None
        self._multipliers = self._offsets = None
        if cfg.kind == "independent_prismatic":
            if len(fingers) != len(self._links):
                raise ValueError("every independent finger needs one prismatic joint")
            for link, joint in zip(self._links, fingers):
                checker.validate_prismatic_link(link, joint)
        else:
            if urdf_path is None:
                raise ValueError("coupled gripper requires its exact collision URDF")
            tree = ElementTree.parse(urdf_path).getroot()
            joints = {node.attrib["name"]: node for node in tree.findall("joint")}
            if (
                cfg.driver_joint not in joints
                or joints[cfg.driver_joint].find("mimic") is not None
            ):
                raise ValueError("coupled drive must be an independent URDF joint")
            coefficients = {cfg.driver_joint: (1.0, 0.0)}
            while True:
                before = len(coefficients)
                for name, node in joints.items():
                    mimic = node.find("mimic")
                    if (
                        name in coefficients
                        or mimic is None
                        or mimic.attrib["joint"] not in coefficients
                    ):
                        continue
                    a, b = coefficients[mimic.attrib["joint"]]
                    m = float(mimic.get("multiplier", "1"))
                    o = float(mimic.get("offset", "0"))
                    coefficients[name] = (m * a, m * b + o)
                if len(coefficients) == before:
                    break
            if set(fingers) != set(coefficients) or len(coefficients) < 2:
                raise ValueError(
                    "coupled declaration must cover the complete URDF mimic drive"
                )
            if any(
                joints[name].get("type") not in {"revolute", "prismatic"}
                for name in fingers
            ):
                raise ValueError("coupled projection requires bounded scalar joints")
            self._driver = self._names.index(cfg.driver_joint)
            self._driver_sign = self._directions[fingers.index(cfg.driver_joint)]
            self._multipliers, self._offsets = np.array(
                [coefficients[name] for name in fingers]
            ).T
            if (
                not np.isfinite(self._multipliers).all()
                or not np.isfinite(self._offsets).all()
                or np.any(self._multipliers == 0)
                or tuple(np.sign(self._multipliers * self._driver_sign).astype(int))
                != self._directions
            ):
                raise ValueError(
                    "mimic mapping and declared closing directions disagree"
                )
            # Every fingertip collision must be driven by this coupled subtree.
            child_parent = {
                node.find("child").attrib["link"]: node
                for node in tree.findall("joint")
            }
            declared_links = set(self._links)
            declared_links.update(
                link
                for first, second, _ in cfg.mechanism_contacts
                for link in (first, second)
            )
            for link in declared_links:
                ancestors = []
                while link in child_parent:
                    node = child_parent[link]
                    ancestors.append(node.attrib["name"])
                    link = node.find("parent").attrib["link"]
                if cfg.driver_joint not in ancestors and not set(ancestors) & set(
                    coefficients
                ):
                    raise ValueError(
                        "finger collision link is outside the coupled drive"
                    )

    @property
    def mechanism_contacts(self) -> MappingProxyType:
        """Return explicitly bounded internal linkage contact permissions."""
        return MappingProxyType(
            {
                ("robot:" + first, "robot:" + second): depth
                for first, second, depth in self.cfg.mechanism_contacts
            }
        )

    def _positions(
        self, positions: np.ndarray, *, measured: bool = False
    ) -> np.ndarray:
        values = np.array(positions, dtype=np.float64, copy=True)
        if (
            values.ndim != 2
            or values.shape[1] != len(self._names)
            or not np.isfinite(values).all()
        ):
            raise ValueError("gripper projection requires finite full-joint samples")
        if self._driver is not None:
            expected = values[:, self._driver, None] * self._multipliers + self._offsets
            tolerance = self.cfg.measured_coupling_tolerance if measured else 1e-6
            if not np.allclose(
                values[:, self.joint_indices], expected, atol=tolerance, rtol=0
            ):
                raise ValueError(
                    "planned or measured gripper state violates URDF coupling"
                )
        return values

    def project_close(
        self,
        positions: np.ndarray,
        *,
        start: int,
        stop: int,
        object_uid: str,
        object_pose: np.ndarray,
    ) -> np.ndarray:
        """Predict the first closing contact and preserve all joint couplings.

        Args:
            positions: Full nominal command path; returned data owns its storage.
            start: First close-phase sample.
            stop: First sample after the stationary close phase.
            object_uid: Physical object's collision ID.
            object_pose: Frozen world object pose at closure.

        Returns:
            A separately owned collision-only joint path, including later holds.
        """
        values = self._positions(positions)
        if not 0 <= start < stop <= len(values):
            raise ValueError("invalid gripper close phase")
        first, last = values[start].copy(), values[stop - 1].copy()
        others = [i for i in range(values.shape[1]) if i not in self.joint_indices]
        if not np.allclose(
            values[start:stop, others], first[others], atol=1e-5, rtol=0
        ):
            raise ValueError(
                "gripper contact projection requires a stationary close phase"
            )
        if self._driver is None:
            for link, index, sign in zip(
                self._links, self.joint_indices, self._directions
            ):
                if sign * (last[index] - first[index]) <= 0:
                    raise ValueError(
                        "gripper command does not follow its declared closing direction"
                    )
                end = first.copy()
                end[index] = last[index]
                fraction = self._checker.first_link_body_contact(
                    first, end, link=link, body_uid=object_uid, body_pose=object_pose
                )
                contact = first[index] + fraction * (last[index] - first[index])
                values[start:, index] = (np.minimum if sign == 1 else np.maximum)(
                    values[start:, index], contact
                )
        else:
            sign = self._driver_sign
            if sign * (last[self._driver] - first[self._driver]) <= 0:
                raise ValueError(
                    "gripper command does not follow its declared closing direction"
                )
            fraction = self._checker.first_link_body_contact(
                first,
                last,
                link=self._links,
                body_uid=object_uid,
                body_pose=object_pose,
            )
            contact = first[self._driver] + fraction * (
                last[self._driver] - first[self._driver]
            )
            drive = (np.minimum if sign == 1 else np.maximum)(
                values[start:, self._driver], contact
            )
            values[start:, self.joint_indices] = (
                drive[:, None] * self._multipliers + self._offsets
            )
        return values

    def project_held(
        self, positions: np.ndarray, measured_qpos: np.ndarray
    ) -> np.ndarray:
        """Limit closing predictions to the currently observed held configuration.

        Args:
            positions: Nominal full-joint commands, including opening/retract.
            measured_qpos: Current measured full-joint state from the host.

        Returns:
            A separate collision prediction; opening commands remain free to open.
        """
        values = self._positions(positions)
        current = self._positions(np.asarray(measured_qpos)[None], measured=True)[0]
        if self._driver is None:
            for index, sign in zip(self.joint_indices, self._directions):
                values[:, index] = (np.minimum if sign == 1 else np.maximum)(
                    values[:, index], current[index]
                )
        else:
            drive = (np.minimum if self._driver_sign == 1 else np.maximum)(
                values[:, self._driver], current[self._driver]
            )
            values[:, self.joint_indices] = (
                drive[:, None] * self._multipliers + self._offsets
            )
        return values
