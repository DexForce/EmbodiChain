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

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["SplitRobotProxy"]


@dataclass(frozen=True)
class _PartRoute:
    side: str
    local_part: str


class _SplitRobotBodyData:
    """Concatenate body data from multiple independent robots."""

    def __init__(self, proxy: SplitRobotProxy) -> None:
        self._proxy = proxy

    @property
    def qpos(self) -> torch.Tensor:
        return self._proxy.get_qpos()

    @property
    def target_qpos(self) -> torch.Tensor:
        return self._proxy.get_qpos(target=True)

    @property
    def qvel(self) -> torch.Tensor:
        return self._proxy.get_qvel()

    @property
    def target_qvel(self) -> torch.Tensor:
        return self._proxy.get_qvel(target=True)

    @property
    def qf(self) -> torch.Tensor:
        return self._proxy.get_qf()

    @property
    def qpos_limits(self) -> torch.Tensor:
        return torch.cat(
            [
                self._proxy.robots[side].body_data.qpos_limits
                for side in self._proxy.side_order
            ],
            dim=1,
        )

    @property
    def qvel_limits(self) -> torch.Tensor:
        return torch.cat(
            [
                self._proxy.robots[side].body_data.qvel_limits
                for side in self._proxy.side_order
            ],
            dim=1,
        )

    @property
    def qf_limits(self) -> torch.Tensor:
        return torch.cat(
            [
                self._proxy.robots[side].body_data.qf_limits
                for side in self._proxy.side_order
            ],
            dim=1,
        )


class SplitRobotProxy:
    """Robot-like facade over independent left/right robot articulations.

    The action-agent stack expects a single ``env.robot`` with left/right control
    parts. This proxy keeps that interface while routing actual simulation calls
    to two separately loaded UR10 robots.
    """

    def __init__(
        self,
        uid: str,
        robots: dict[str, Robot],
        part_routes: dict[str, tuple[str, str]],
    ) -> None:
        self.uid = uid
        self.robots = robots
        self.side_order = [side for side in ("left", "right") if side in robots]
        self._routes = {
            part: _PartRoute(side=route[0], local_part=route[1])
            for part, route in part_routes.items()
        }
        self._offsets: dict[str, int] = {}
        offset = 0
        for side in self.side_order:
            self._offsets[side] = offset
            offset += self.robots[side].dof
        self.dof = offset
        self.active_joint_ids = list(range(self.dof))
        self.control_parts = {
            part: self.robots[route.side].control_parts[route.local_part]
            for part, route in self._routes.items()
        }
        self.device = self.robots[self.side_order[0]].device
        self.body_data = _SplitRobotBodyData(self)

    @property
    def num_instances(self) -> int:
        return self.robots[self.side_order[0]].num_instances

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for side in self.side_order:
            names.extend(self.robots[side].joint_names)
        return names

    @property
    def active_joint_names(self) -> list[str]:
        return [self.joint_names[idx] for idx in self.active_joint_ids]

    @property
    def mimic_ids(self) -> list[int | None]:
        ids: list[int | None] = []
        for side in self.side_order:
            offset = self._offsets[side]
            ids.extend(
                None if mimic_id is None else mimic_id + offset
                for mimic_id in self.robots[side].mimic_ids
            )
        return ids

    def _route(self, name: str) -> _PartRoute:
        if name not in self._routes:
            raise ValueError(
                f"Unknown split robot control part '{name}'. "
                f"Available parts: {list(self._routes)}."
            )
        return self._routes[name]

    def _globalize_ids(self, side: str, local_ids: Sequence[int]) -> list[int]:
        offset = self._offsets[side]
        return [offset + int(joint_id) for joint_id in local_ids]

    def get_robot_uid_for_part(self, name: str) -> str:
        route = self._route(name)
        return self.robots[route.side].uid

    def get_local_control_part(self, name: str) -> str:
        return self._route(name).local_part

    def get_local_joint_ids(
        self, name: str | None = None, remove_mimic: bool = False
    ) -> list[int]:
        if name is None:
            return list(range(self.robots[self.side_order[0]].dof))
        route = self._route(name)
        return self.robots[route.side].get_joint_ids(
            route.local_part, remove_mimic=remove_mimic
        )

    def get_joint_ids(
        self, name: str | None = None, remove_mimic: bool = False
    ) -> list[int]:
        if name is None:
            return (
                list(range(self.dof))
                if not remove_mimic
                else [
                    joint_id
                    for side in self.side_order
                    for joint_id in self._globalize_ids(
                        side,
                        self.robots[side].active_joint_ids,
                    )
                ]
            )

        route = self._route(name)
        local_ids = self.robots[route.side].get_joint_ids(
            route.local_part, remove_mimic=remove_mimic
        )
        return self._globalize_ids(route.side, local_ids)

    def get_qpos(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        if name is not None:
            route = self._route(name)
            return self.robots[route.side].get_qpos(route.local_part, target=target)
        return torch.cat(
            [self.robots[side].get_qpos(target=target) for side in self.side_order],
            dim=1,
        )

    def get_qvel(self, name: str | None = None, target: bool = False) -> torch.Tensor:
        if name is not None:
            route = self._route(name)
            return self.robots[route.side].get_qvel(route.local_part, target=target)
        return torch.cat(
            [self.robots[side].get_qvel(target=target) for side in self.side_order],
            dim=1,
        )

    def get_qf(self, name: str | None = None) -> torch.Tensor:
        if name is not None:
            route = self._route(name)
            robot = self.robots[route.side]
            joint_ids = robot.get_joint_ids(route.local_part)
            return robot.body_data.qf[:, joint_ids]
        return torch.cat(
            [self.robots[side].body_data.qf for side in self.side_order], dim=1
        )

    def _prepare_values(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        else:
            values = values.to(device=self.device, dtype=torch.float32)
        if values.dim() == 1:
            values = values.unsqueeze(0)
        return values

    def _as_full_values(
        self,
        values: torch.Tensor,
        joint_ids: Sequence[int] | None,
        current_values: torch.Tensor,
    ) -> torch.Tensor:
        values = self._prepare_values(values)

        if joint_ids is None:
            return values

        full = current_values.clone()
        full[:, list(joint_ids)] = values
        return full

    def set_qpos(
        self,
        qpos: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        target: bool = True,
        name: str | None = None,
    ) -> None:
        if name is not None:
            route = self._route(name)
            self.robots[route.side].set_qpos(
                qpos=qpos,
                env_ids=env_ids,
                target=target,
                name=route.local_part,
            )
            return

        full = self._as_full_values(
            qpos,
            joint_ids,
            current_values=self.get_qpos(target=target),
        )
        for side in self.side_order:
            offset = self._offsets[side]
            robot = self.robots[side]
            robot.set_qpos(
                full[:, offset : offset + robot.dof],
                env_ids=env_ids,
                target=target,
            )

    def set_qvel(
        self,
        qvel: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        target: bool = True,
        name: str | None = None,
    ) -> None:
        if name is not None:
            route = self._route(name)
            self.robots[route.side].set_qvel(
                qvel=qvel,
                env_ids=env_ids,
                target=target,
                name=route.local_part,
            )
            return

        full = self._as_full_values(
            qvel,
            joint_ids,
            current_values=self.get_qvel(target=target),
        )
        for side in self.side_order:
            offset = self._offsets[side]
            robot = self.robots[side]
            robot.set_qvel(
                full[:, offset : offset + robot.dof],
                env_ids=env_ids,
                target=target,
            )

    def set_qf(
        self,
        qf: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
        name: str | None = None,
    ) -> None:
        if name is not None:
            route = self._route(name)
            self.robots[route.side].set_qf(
                qf=qf,
                env_ids=env_ids,
                name=route.local_part,
            )
            return

        full = self._as_full_values(
            qf,
            joint_ids,
            current_values=self.get_qf(),
        )
        for side in self.side_order:
            offset = self._offsets[side]
            robot = self.robots[side]
            robot.set_qf(full[:, offset : offset + robot.dof], env_ids=env_ids)

    def get_qpos_limits(
        self, name: str | None = None, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        if name is not None:
            route = self._route(name)
            return self.robots[route.side].get_qpos_limits(route.local_part, env_ids)
        if env_ids is None:
            return self.body_data.qpos_limits
        return self.body_data.qpos_limits[env_ids]

    def compute_fk(self, qpos, name: str | None = None, **kwargs) -> torch.Tensor:
        route = self._route(name)
        return self.robots[route.side].compute_fk(
            qpos=qpos, name=route.local_part, **kwargs
        )

    def compute_ik(
        self,
        pose,
        joint_seed=None,
        name: str | None = None,
        **kwargs,
    ):
        route = self._route(name)
        return self.robots[route.side].compute_ik(
            pose=pose,
            joint_seed=joint_seed,
            name=route.local_part,
            **kwargs,
        )

    def compute_batch_fk(self, qpos, name: str, **kwargs):
        route = self._route(name)
        return self.robots[route.side].compute_batch_fk(
            qpos=qpos, name=route.local_part, **kwargs
        )

    def compute_batch_ik(self, pose, joint_seed, name: str, **kwargs):
        route = self._route(name)
        return self.robots[route.side].compute_batch_ik(
            pose=pose,
            joint_seed=joint_seed,
            name=route.local_part,
            **kwargs,
        )

    def get_control_part_base_pose(self, name: str, **kwargs) -> torch.Tensor:
        route = self._route(name)
        return self.robots[route.side].get_control_part_base_pose(
            route.local_part, **kwargs
        )

    def get_proprioception(self) -> TensorDict:
        return TensorDict(
            {
                "qpos": self.get_qpos(),
                "qvel": self.get_qvel(),
                "qf": self.get_qf(),
            },
            batch_size=[self.num_instances],
            device=self.device,
        )

    def build_pk_serial_chain(self) -> None:
        for robot in self.robots.values():
            robot.build_pk_serial_chain()
