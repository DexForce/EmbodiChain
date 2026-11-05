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

"""Analytic redundant translation model for the production CPU IK adapter."""

from __future__ import annotations

from pathlib import Path
import hashlib
from types import SimpleNamespace, MethodType
import torch
from embodichain.lab.sim.motion.solvers.pytorch_solver import PytorchSolver


def branch_fixture(directory: Path, *, failed: bool = False):
    path = directory / "redundant.urdf"
    path.write_text(
        '<robot name="redundant"><link name="base"/><link name="mid"/><link name="tip"/><joint name="q0" type="prismatic"><parent link="base"/><child link="mid"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="10"/></joint><joint name="q1" type="prismatic"><parent link="mid"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="10" velocity="10"/></joint></robot>'
    )
    solver = object.__new__(PytorchSolver)
    solver.device = torch.device("cpu")
    solver._seed_sampler = None
    solver._pos_eps = solver._rot_eps = 1e-4
    solver.lower_qpos_limits = torch.tensor([-2.0, -2.0])
    solver.upper_qpos_limits = torch.tensor([2.0, 2.0])
    solver.urdf_path = str(path)
    solver.root_link_name, solver.end_link_name = "base", "tip"
    solver.tcp_xpos = torch.eye(4)
    solver.tcp_xpos[2, 3] = 0.2
    calls = []

    def inverse(self, pose, seeds):
        calls.append(
            {
                "pose_shape": list(pose.shape),
                "pose_dtype": str(pose.dtype),
                "pose_sha256": hashlib.sha256(
                    pose.contiguous().numpy().tobytes()
                ).hexdigest(),
                "seed_shape": list(seeds.shape),
                "seed_dtype": str(seeds.dtype),
                "seed_sha256": hashlib.sha256(
                    seeds.contiguous().numpy().tobytes()
                ).hexdigest(),
                "rng": torch.get_rng_state().clone(),
            }
        )
        q = seeds.clone()
        q[:, 0] = pose[:, 0, 3] - q[:, 1]
        valid = torch.full((len(q),), not failed, dtype=torch.bool)
        return valid, q

    solver._compute_inverse_kinematics = MethodType(inverse, solver)

    def get_ik(self, pose, qpos_seed):
        ok, q = inverse(
            self, pose @ torch.linalg.inv(self.tcp_xpos.to(pose)), qpos_seed
        )
        bounded, q = self._qpos_map_to_limits(q)
        return ok & bounded, q

    solver.get_ik = MethodType(get_ik, solver)
    root = torch.eye(4)[None]
    root[0, :3, 3] = torch.tensor([1.0, -2.0, 0.3])
    pose_reads = []

    def get_pose(name, *, to_matrix):
        assert name == "base" and to_matrix
        pose_reads.append(name)
        return root.clone()

    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(path)),
        num_instances=1,
        device=torch.device("cpu"),
        joint_names=["q0", "q1"],
        control_parts={"arm": ["q0", "q1"]},
        get_solver=lambda part: solver,
        get_link_pose=get_pose,
        get_joint_ids=lambda name: [0, 1],
        get_qvel_limits=lambda name: torch.full((1, 2), 10.0),
    )
    return robot, solver, root, calls, pose_reads
