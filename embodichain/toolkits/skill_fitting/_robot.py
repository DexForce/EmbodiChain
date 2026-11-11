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

"""Simulator-free robot port using the existing robot-owned FK/IK solvers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from embodichain.lab.sim.motion.solvers import SolverCfg
from embodichain.lab.sim.utility.solver_utils import create_pk_serial_chain
from embodichain.utils import configclass

__all__ = []


@configclass
class RobotProfile:
    """Explicit robot calibration and command layout, independent of objects."""

    joint_names: tuple[str, ...] = ()
    control_parts: dict[str, list[str]] = {}
    participants: dict[str, dict[str, str]] = {}
    solvers: dict[str, dict[str, Any]] = {}
    base_poses: dict[str, list[list[float]]] = {}
    hand_commands: dict[str, dict[str, list[float]]] = {}
    lower: list[float] = []
    upper: list[float] = []
    velocity_limits: list[float] = []
    acceleration_limits: list[float] = []
    calibration_revision: str = ""

    @classmethod
    def load(cls, path: str | Path) -> RobotProfile:
        """Load a closed profile; resolve solver URDF paths against the file."""
        source = Path(path).resolve()
        raw = json.loads(source.read_text())
        allowed = set(cls.__dataclass_fields__)
        if set(raw) - allowed:
            raise ValueError(
                f"Unknown robot profile fields: {sorted(set(raw) - allowed)}"
            )
        raw["joint_names"] = tuple(raw.get("joint_names", ()))
        result = cls(**raw)
        result.validate()
        result.check()
        for solver in result.solvers.values():
            model = Path(solver["urdf_path"]).expanduser()
            solver["urdf_path"] = str(
                (source.parent / model).resolve() if not model.is_absolute() else model
            )
        return result

    def check(self) -> None:
        """Validate named layout, finite limits and explicit endpoint commands."""
        names = self.joint_names
        if not names or len(set(names)) != len(names) or not self.calibration_revision:
            raise ValueError(
                "Profile needs unique joint names and a calibration revision."
            )
        for key in ("lower", "upper", "velocity_limits", "acceleration_limits"):
            value = np.asarray(getattr(self, key), dtype=float)
            if value.shape != (len(names),) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be one finite value per joint.")
        if (
            np.any(np.asarray(self.lower) >= self.upper)
            or np.any(np.asarray(self.velocity_limits) <= 0)
            or np.any(np.asarray(self.acceleration_limits) <= 0)
        ):
            raise ValueError("Profile has invalid position or derivative limits.")
        for part, joints in self.control_parts.items():
            if (
                not joints
                or len(set(joints)) != len(joints)
                or not set(joints) <= set(names)
            ):
                raise ValueError(f"Invalid explicit joint list for {part}.")
        for role, endpoints in self.participants.items():
            if set(endpoints) != {"motion", "grasp"}:
                raise ValueError(f"{role} must bind motion and grasp.")
            arm, hand = endpoints["motion"], endpoints["grasp"]
            if arm not in self.solvers or hand not in self.hand_commands:
                raise ValueError(f"{role} lacks a solver or command profile.")
            if arm not in self.control_parts or hand not in self.control_parts:
                raise ValueError("Endpoint refers to an undeclared control part.")
        for part, commands in self.hand_commands.items():
            if set(commands) != {"open", "grasp"} or part not in self.control_parts:
                raise ValueError("Hand profiles require exactly open/grasp commands.")
            for values in commands.values():
                ids = [names.index(n) for n in self.control_parts[part]]
                if (
                    len(values) != len(ids)
                    or not np.isfinite(values).all()
                    or np.any(np.asarray(values) < np.asarray(self.lower)[ids])
                    or np.any(np.asarray(values) > np.asarray(self.upper)[ids])
                ):
                    raise ValueError("Hand command is invalid or outside joint limits.")
            if np.linalg.norm(np.subtract(commands["open"], commands["grasp"])) < 1e-8:
                raise ValueError("Open and grasp commands must be distinguishable.")

    def metadata(self) -> dict[str, Any]:
        """Return a JSON-compatible copy for fit provenance."""
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


class RobotPort:
    """Borrowed kinematics-only Robot interface; never acquires a simulator."""

    def __init__(
        self, profile: RobotProfile, initial_qpos: np.ndarray, device: str
    ) -> None:
        import warp as wp

        wp.init()
        profile.check()
        self.profile = profile
        self.device = torch.device(device)
        self.joint_names = list(profile.joint_names)
        self.dof = len(self.joint_names)
        self.num_instances = 1
        self.control_parts = profile.control_parts
        self._qpos = torch.tensor(
            initial_qpos, device=self.device, dtype=torch.float32
        )[None]
        self.mimic_ids: list[int] = []
        self.mimic_parents: list[int] = []
        self.mimic_multipliers: list[float] = []
        self.solvers = {}
        self.bases = {}
        for part, declaration in profile.solvers.items():
            cfg = SolverCfg.from_dict(declaration)
            joints = profile.control_parts[part]
            cfg.joint_names = list(joints)
            chain = create_pk_serial_chain(
                urdf_path=cfg.urdf_path,
                end_link_name=cfg.end_link_name,
                root_link_name=cfg.root_link_name,
                device=self.device,
            )
            if chain.get_joint_parameter_names() != list(joints):
                raise ValueError(f"URDF chain joint order differs for {part}.")
            ids = self.get_joint_ids(part)
            cfg.user_qpos_limits = [
                [profile.lower[i] for i in ids],
                [profile.upper[i] for i in ids],
            ]
            self.solvers[part] = cfg.init_solver(
                device=self.device, pk_serial_chain=chain
            )
            base = torch.tensor(
                profile.base_poses.get(part, np.eye(4).tolist()),
                device=self.device,
                dtype=torch.float32,
            )
            from embodichain.lab.sim.atomic_skills.goals import validate_pose_goal

            validate_pose_goal(base, f"{part} base", allow_waypoints=False)
            if base.shape != (4, 4):
                raise ValueError("Each arm base must be one calibrated transform.")
            self.bases[part] = base

    def get_joint_ids(self, name: str | None = None) -> list[int]:
        return (
            list(range(self.dof))
            if name is None
            else [self.joint_names.index(n) for n in self.control_parts[name]]
        )

    def get_qpos(self, name: str | None = None) -> torch.Tensor:
        return self._qpos[:, self.get_joint_ids(name)].clone()

    def get_qvel(self) -> torch.Tensor:
        return torch.zeros_like(self._qpos)

    def get_qvel_limits(
        self, name: str | None = None, joint_ids: list[int] | None = None
    ) -> torch.Tensor:
        ids = self.get_joint_ids(name) if joint_ids is None else joint_ids
        return torch.tensor(self.profile.velocity_limits, device=self.device)[ids][None]

    def get_qpos_limits(
        self, name: str | None = None, joint_ids: list[int] | None = None
    ) -> torch.Tensor:
        ids = self.get_joint_ids(name) if joint_ids is None else joint_ids
        return torch.tensor(
            np.column_stack((self.profile.lower, self.profile.upper)),
            device=self.device,
            dtype=torch.float32,
        )[None, ids]

    def get_solver(self, name: str) -> Any:
        return self.solvers[name]

    def compute_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool = True, **kwargs: Any
    ) -> torch.Tensor:
        return self.bases[name] @ self.solvers[name].get_fk(qpos)

    def compute_batch_fk(
        self, qpos: torch.Tensor, name: str, to_matrix: bool = True, **kwargs: Any
    ) -> torch.Tensor:
        return self.compute_fk(qpos.reshape(-1, qpos.shape[-1]), name).reshape(
            *qpos.shape[:-1], 4, 4
        )

    def compute_ik(
        self, pose: torch.Tensor, name: str, joint_seed: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = torch.linalg.inv(self.bases[name]) @ pose
        return self.solvers[name].get_ik(target_xpos=target, qpos_seed=joint_seed)

    def compute_batch_ik(
        self, pose: torch.Tensor, name: str, joint_seed: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shape = pose.shape[:-2]
        seed = joint_seed
        if seed.ndim == 2:
            seed = seed[:, None].expand(*shape, seed.shape[-1])
        valid, q = self.compute_ik(
            pose.reshape(-1, 4, 4), name, seed.reshape(-1, seed.shape[-1])
        )
        return valid.reshape(shape), q.reshape(*shape, q.shape[-1])

    def geometry(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Batch FK into compact arm-link skeletons and TCPs for standalone viewing."""
        skeletons, tcp = [], []
        for part, solver in self.solvers.items():
            joints = torch.as_tensor(
                qpos[:, self.get_joint_ids(part)],
                device=self.device,
                dtype=torch.float32,
            )
            base = self.bases[part]
            with torch.no_grad():
                transforms = solver.pk_serial_chain.forward_kinematics_tensor(joints)
                nodes = (base @ transforms)[:, :, :3, 3].permute(1, 0, 2)
                tool = self.compute_fk(joints, part)[:, :3, 3]
            skeletons.append(torch.cat((nodes, tool[:, None]), dim=1).cpu().numpy())
            tcp.append(tool.cpu().numpy())
        # Separate arms are separate polylines; no invented link between bases.
        max_nodes = max(v.shape[1] for v in skeletons)
        padded = [
            np.pad(v, ((0, 0), (0, max_nodes - v.shape[1]), (0, 0)), mode="edge")
            for v in skeletons
        ]
        return np.stack(padded, axis=1), np.stack(tcp, axis=1)
