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

from pathlib import Path

import pytest
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, physics_cfg_for_backend
from embodichain.utils.math import matrix_from_quat

pytestmark = [pytest.mark.gpu, pytest.mark.requires_sim]


def _tensor(moments: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    rotation = matrix_from_quat(pose[..., 3:])
    return rotation @ torch.diag_embed(moments) @ rotation.transpose(-1, -2)


@pytest.mark.parametrize("backend", ["default", "newton"])
def test_live_inertia_frames_and_partial_reset(tmp_path: Path, backend: str) -> None:
    """Creation, live batch reads and reset agree for a rotated source tensor."""
    source = tmp_path / "inertial.urdf"
    link = """<link name="{name}">
      <inertial><origin xyz=".01 .02 .03" rpy=".3 .5 .7"/><mass value="2"/>
      <inertia ixx=".02" ixy="0" ixz="0" iyy=".03" iyz="0" izz=".04"/></inertial>
      <visual><geometry><box size=".1 .1 .1"/></geometry></visual>
      <collision><geometry><box size=".1 .1 .1"/></geometry></collision>
    </link>"""
    source.write_text(
        '<robot name="inertial">'
        + link.format(name="base")
        + link.format(name="tip")
        + '<joint name="hinge" type="revolute"><parent link="base"/><child link="tip"/>'
        '<origin xyz="0 0 .3"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="5"/></joint></robot>'
    )
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cuda:0",
            num_envs=3,
            physics_cfg=physics_cfg_for_backend(backend),
        )
    )
    art = None
    try:
        art = sim.add_articulation(
            ArticulationCfg(uid="inertial", fpath=str(source), init_pos=(0, 0, 1))
        )
        sim.prepare()
        data = art.body_data
        defaults = tuple(value.clone() for value in data.read_physical_properties())
        link_id = art.link_names.index("tip")
        selected = [2, 0]
        names = ["tip"]
        mass = defaults[0][selected, link_id : link_id + 1] * 1.2
        art.set_mass(mass, link_names=names, env_ids=selected)
        moments = art.get_inertia(names, selected).clone() * 1.25
        old_pose = art.get_com_pose(names, selected).clone()
        art.set_inertia(moments, link_names=names, env_ids=selected)
        torch.testing.assert_close(
            _tensor(
                art.get_inertia(names, selected), art.get_com_pose(names, selected)
            ),
            _tensor(moments, old_pose),
            atol=1e-5,
            rtol=1e-4,
        )
        # Read the paired principal values before changing their orientation.
        moments = art.get_inertia(names, selected).clone()
        pose = old_pose.clone()
        pose[..., :3] += 0.02
        pose[..., 3:] = torch.tensor([1.0, 2.0, 3.0, 4.0], device=art.device) / 30**0.5
        art.set_com_pose(pose, link_names=names, env_ids=selected)
        observed = tuple(value.clone() for value in data.read_physical_properties())
        expected = _tensor(moments, pose)
        torch.testing.assert_close(
            _tensor(
                observed[1][selected, link_id : link_id + 1],
                observed[2][selected, link_id : link_id + 1],
            ),
            expected,
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(
            observed[2][selected, link_id : link_id + 1, :3], pose[..., :3]
        )
        if backend == "newton":
            descriptors = art.get_newton_link_properties(names, selected)
            for row, descriptor in enumerate(descriptors):
                torch.testing.assert_close(
                    torch.as_tensor(descriptor.inertia, device=art.device),
                    expected[row, 0],
                    atol=1e-5,
                    rtol=1e-4,
                )
        art.reset(env_ids=[2])
        restored = tuple(value.clone() for value in data.read_physical_properties())
        for actual, baseline in zip(restored, defaults):
            torch.testing.assert_close(actual[1], baseline[1])
        torch.testing.assert_close(restored[0][2], defaults[0][2])
        torch.testing.assert_close(
            _tensor(restored[1][2], restored[2][2]),
            _tensor(defaults[1][2], defaults[2][2]),
            atol=1e-5,
            rtol=1e-4,
        )
        torch.testing.assert_close(restored[2][2, :, :3], defaults[2][2, :, :3])
        torch.testing.assert_close(
            _tensor(restored[1][0], restored[2][0]),
            _tensor(observed[1][0], observed[2][0]),
            atol=1e-5,
            rtol=1e-4,
        )
        sim.update()
        assert torch.isfinite(art.root_state).all()
    finally:
        art = None
        sim.destroy()
        SimulationManager.flush_cleanup_queue()
