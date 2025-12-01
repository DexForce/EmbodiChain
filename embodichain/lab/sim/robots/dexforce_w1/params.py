# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass, field
from embodichain.lab.sim.robots.dexforce_w1.types import (
    DexforceW1HandBrand,
    DexforceW1ArmSide,
    DexforceW1ArmKind,
    DexforceW1Version,
)


@dataclass
class W1ArmKineParams:
    """Kinematics parameters for W1 arm variants.

    - arm_kind and W1Version enum types expected to be defined elsewhere.
    - dh_params stored as numpy array of shape (7,4).
    - qpos_limits stored as numpy array of shape (7,2) in radians.
    """

    arm_side: "DexforceW1ArmSide"
    arm_kind: "DexforceW1ArmKind"
    version: "DexforceW1Version" = field(default_factory=lambda: DexforceW1Version.V021)

    # (initialized in __post_init__)
    # physical constants
    d_list: list[float] = field(init=False, default_factory=list)
    link_lengths: list[float] = field(init=False, default_factory=list)
    rotation_directions: list[float] = field(init=False, default_factory=list)

    # transforms
    T_b_ob: np.ndarray = field(init=False)
    T_e_oe: np.ndarray = field(init=False)

    # kinematic parameters
    dh_params: np.ndarray = field(init=False)
    qpos_limits: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.version == DexforceW1Version.V021:
            self.d_list = np.array([0.0, 0.0, 0.260, 0.0, 0.166, 0.098, 0.0])
            self.link_lengths = np.array(
                [
                    self.d_list[0] + self.d_list[1],
                    self.d_list[2] + self.d_list[3],
                    self.d_list[4] + self.d_list[5],
                    self.d_list[6],
                ]
            )
        else:
            raise ValueError(f"W1Version {self.version} are not supported.")

        # helpers: create DH rows and clamp limits
        def dh_row(d, alpha, a, theta):
            return [d, alpha, a, theta]

        def deg2rad_list(list_of_pairs):
            return np.deg2rad(np.array(list_of_pairs, dtype=float))

        T_b_ob = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1025],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Build parameters per arm_kind and side, minimizing duplication
        if self.arm_kind == DexforceW1ArmKind.INDUSTRIAL:
            # default tcp for industrial
            T_e_oe = np.array(
                [
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.066],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

            # fmt: off
            dh = [
                dh_row(self.link_lengths[0],  -np.pi / 2,  0.0,  0.0),
                dh_row(0.0,                    np.pi / 2,  0.0,  0.0),
                dh_row(self.link_lengths[1],   np.pi / 2,  0.0,  np.pi / 2),
                dh_row(0.0,                   -np.pi / 2,  0.0,  0.0),
                dh_row(self.link_lengths[2],  -np.pi / 2,  0.0,  0.0),
                dh_row(0.0,                    np.pi / 2,  0.0,  0.0),
                dh_row(self.link_lengths[3],   0.0,        0.0,  0.0),
            ]

            # fmt: on
            if self.arm_side == DexforceW1ArmSide.LEFT:
                limits = [
                    [-170.0, 170.0],
                    [-120.0, 90.0],
                    [-170.0, 170.0],
                    [-135.0, 90.0],
                    [-170.0, 170.0],
                    [-90.0, 90.0],
                    [-170.0, 170.0],
                ]
                rotation_directions = np.array([1, 1, 1, 1, 1, -1, 1])
            else:
                limits = [
                    [-170.0, 170.0],
                    [-90.0, 120.0],
                    [-170.0, 170.0],
                    [-90.0, 135.0],
                    [-170.0, 170.0],
                    [-90.0, 90.0],
                    [-170.0, 170.0],
                ]
                rotation_directions = np.array([1, 1, 1, -1, 1, 1, 1])

            self.T_e_oe = T_e_oe

        elif self.arm_kind == DexforceW1ArmKind.ANTHROPOMORPHIC:
            T_e_oe = np.array(
                [
                    [0.0, 0.0, -1.0, -0.066],
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            # fmt: off
            dh = [
                dh_row(self.link_lengths[0],  -np.pi / 2,  0.0,  0.0),
                dh_row(0.0,                    np.pi / 2,  0.0,  0.0),
                dh_row(self.link_lengths[1],   np.pi / 2,  0.0,  np.pi / 2),
                dh_row(0.0,                   -np.pi / 2,  0.0,  0.0),
                dh_row(self.link_lengths[2],  -np.pi / 2,  0.0,  0.0),
                dh_row(0.0,                    np.pi / 2,  0.0,  np.pi / 2),
                dh_row(self.link_lengths[3],   0.0,        0.0,  0.0),
            ]
            # fmt: on

            if self.arm_side == DexforceW1ArmSide.LEFT:
                limits = [
                    [-170.0, 170.0],
                    [-120.0, 90.0],
                    [-170.0, 170.0],
                    [-135.0, 90.0],
                    [-170.0, 170.0],
                    [-45.0, 45.0],
                    [-90.0, 60.0],
                ]
                rotation_directions = np.array([1, 1, 1, 1, 1, -1, 1])
            else:
                limits = [
                    [-170.0, 170.0],
                    [-90.0, 120.0],
                    [-170.0, 170.0],
                    [-90.0, 135.0],
                    [-170.0, 170.0],
                    [-45.0, 45.0],
                    [-60.0, 90.0],
                ]
                rotation_directions = np.array([1, 1, 1, -1, 1, 1, 1])
        else:
            raise ValueError(f"Unsupported arm_kind: {self.arm_kind}")

        self.T_b_ob = T_b_ob
        self.T_e_oe = T_e_oe

        # finalize arrays
        self.dh_params = np.array(dh, dtype=float)
        self.qpos_limits = deg2rad_list(limits)
        self.rotation_directions = rotation_directions

        # sanity checks
        assert self.dh_params.shape == (7, 4), "dh_params must be shape (7,4)"
        assert self.qpos_limits.shape == (7, 2), "qpos_limits must be shape (7,2)"

    def as_dict(self) -> dict:
        return {
            "arm_side": self.arm_side.name,
            "arm_kind": self.arm_kind.name,
            "version": self.version.name,
            "link_lengths": self.link_lengths.tolist(),
            "T_b_ob": self.T_b_ob.tolist(),
            "T_e_oe": self.T_e_oe.tolist(),
            "dh_params": self.dh_params.tolist(),
            "qpos_limits": self.qpos_limits.tolist(),
            "rotation_directions": self.rotation_directions.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "W1ArmKineParams":
        arm_side = (
            DexforceW1ArmSide[data["arm_side"]]
            if isinstance(data.get("arm_side"), str)
            else data.get("arm_side")
        )

        arm_kind = (
            DexforceW1ArmKind[data["arm_kind"]]
            if isinstance(data.get("arm_kind"), str)
            else data.get("arm_kind")
        )
        version = (
            DexforceW1Version[data["version"]]
            if isinstance(data.get("version"), str)
            else data.get("version", DexforceW1Version.V021)
        )
        inst = cls(arm_side=arm_side, arm_kind=arm_kind, version=version)

        # allow overriding computed arrays if provided
        if "dh_params" in data:
            object.__setattr__(
                inst, "dh_params", np.array(data["dh_params"], dtype=float)
            )
        if "qpos_limits" in data:
            (
                object.__setattr__(
                    inst,
                    "qpos_limits",
                    np.deg2rad(np.array(data["qpos_limits"], dtype=float)),
                )
                if np.max(np.abs(np.array(data["qpos_limits"]))) > 2 * np.pi
                else object.__setattr__(
                    inst, "qpos_limits", np.array(data["qpos_limits"], dtype=float)
                )
            )
        if "link_lengths" in data:
            object.__setattr__(
                inst, "link_lengths", np.array(data["link_lengths"], dtype=float)
            )
        if "T_b_ob" in data:
            object.__setattr__(inst, "T_b_ob", np.array(data["T_b_ob"], dtype=float))
        if "T_e_oe" in data:
            object.__setattr__(inst, "T_e_oe", np.array(data["T_e_oe"], dtype=float))
        if "rotation_directions" in data:
            object.__setattr__(
                inst,
                "rotation_directions",
                np.array(data["rotation_directions"], dtype=float),
            )
        inst.validate()
        return inst

    def to_torch(
        self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32
    ) -> dict:
        dev = torch.device("cpu") if device is None else device
        return {
            "dh_params": torch.tensor(self.dh_params, dtype=dtype, device=dev),
            "qpos_limits": torch.tensor(self.qpos_limits, dtype=dtype, device=dev),
            "T_b_ob": torch.tensor(self.T_b_ob, dtype=dtype, device=dev),
            "T_e_oe": torch.tensor(self.T_e_oe, dtype=dtype, device=dev),
            "rotation_directions": torch.tensor(
                self.rotation_directions, dtype=dtype, device=dev
            ),
        }
