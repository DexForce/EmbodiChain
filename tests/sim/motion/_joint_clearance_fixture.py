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

"""Analytic two-link clearance model with explicit sampled-query facts."""

from __future__ import annotations
from types import SimpleNamespace
import hashlib, json, random
import numpy as np
import torch
from embodichain.lab.sim.motion.mesh_collision import MotionCollisionReport


def clearance_fixture(tmp_path, mode):
    path = tmp_path / "offset.urdf"
    limit = "0" if mode == "joint_limit" else "1"
    path.write_text(
        f"""<robot name="clearance"><link name="base"/><link name="arm"/><link name="tcp"/><link name="protected"/>
    <joint name="offset_joint" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-{limit}" upper="{limit}" effort="10" velocity="1"/></joint>
    <joint name="tcp_fixed" type="fixed"><parent link="arm"/><child link="tcp"/><origin xyz="1 0 0"/></joint>
    <joint name="protected_fixed" type="fixed"><parent link="base"/><child link="protected"/><origin xyz="1 1 0"/></joint></robot>"""
    )
    robot = SimpleNamespace(
        cfg=SimpleNamespace(fpath=str(path)),
        joint_names=["offset_joint"],
        get_solver=lambda name: SimpleNamespace(
            end_link_name="tcp" if name == "active" else "protected"
        ),
    )
    calls = []

    class Query:
        def inspect(self, positions, *, control_part):
            assert control_part == "active"
            calls.append(
                dict(
                    positions=hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
                    shape=list(positions.shape),
                    last=positions[-1].tolist(),
                    draws=[
                        random.random(),
                        float(np.random.rand()),
                        float(torch.rand(())),
                    ],
                )
            )
            accepted = mode == "first" or mode == "second" and len(calls) == 2
            return (
                MotionCollisionReport(True, frames=len(positions))
                if accepted
                else MotionCollisionReport(
                    False, frame=0, kind="scene", pairs=(("tcp", "obstacle"),)
                )
            )

    return robot, torch.zeros(1, 1), Query(), calls


def record_clearance(run, calls):
    random.seed(921)
    np.random.seed(921)
    torch.manual_seed(921)
    try:
        result = {"offset": run()}
    except ValueError as error:
        result = {"error": str(error)}
    result.update(
        calls=calls,
        after_rng=[random.random(), float(np.random.rand()), float(torch.rand(()))],
    )
    return result
