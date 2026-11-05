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

"""Standard prismatic provider owns cached nominal and observed late hand capture."""

from __future__ import annotations
from types import SimpleNamespace
from dataclasses import replace
import pytest
import torch
from preshape_hand_support import preshape_fixture
from embodichain.lab.sim.atomic_actions.bindings import JointPositionTarget
from embodichain.lab.sim.evidence import PhysicalObservationStamp
from embodichain.lab.task_program.integrations.simulation.slide_preshape import (
    SlidePreshapeBinding,
    SimulationSlideHandClearanceProvider,
)


def test_provider_keeps_nominal_cached_and_fractional_captures_observation_bound(
    tmp_path,
):
    robot, cfg = preshape_fixture(tmp_path)
    stamp = PhysicalObservationStamp("source", "default.contacts", 7, 1.0)
    stamps = [stamp]
    table = SimpleNamespace(
        get_local_pose=lambda **kwargs: torch.eye(4)[None],
        get_vertices=lambda **kwargs: torch.tensor(
            [[[-1.0, -1.0, -0.1], [1.0, 1.0, 0.0]]]
        ),
    )
    simulation = SimpleNamespace(
        num_envs=1,
        physics_backend="default",
        get_robot=lambda uid: robot,
        get_rigid_object=lambda uid: table,
    )
    binding = SlidePreshapeBinding("p/segment:0", "custom_hand", "surface_native")
    provider = SimulationSlideHandClearanceProvider(
        robot,
        simulation,
        bindings=(binding,),
        hand_geometry_cfgs={"custom_hand": cfg},
        read_source_stamp=lambda: stamps[0],
    )
    assert len(robot.fk_calls) == 1
    targets = {
        "motion": JointPositionTarget("custom_arm", (0,)),
        "grasp": JointPositionTarget("custom_hand", (2, 1)),
    }
    request = SimpleNamespace(
        skill_id="slide",
        invocation_id=binding.invocation_id,
        binding=SimpleNamespace(
            endpoint=lambda role, name: SimpleNamespace(
                require_target=lambda cls: targets[name]
            )
        ),
    )
    context = SimpleNamespace(
        batch_size=1,
        env_ids=torch.tensor([0]),
        world_source_stamp=stamp,
        observed_robot=None,
        robot=SimpleNamespace(qpos=robot.qpos.clone()),
    )
    nominal = provider.nominal(request, context)
    original = nominal.points.clone()
    nominal.points.fill_(9.0)
    assert torch.equal(provider.nominal(request, context).points, original)
    assert len(robot.fk_calls) == 1
    measured = provider.measure(request, context, 0.78)
    assert measured.aperture > provider.nominal(request, context).aperture
    assert len(robot.fk_calls) == 2
    stamps[0] = replace(stamp, source_id="foreign")
    with pytest.raises(ValueError, match="original complete"):
        provider.measure(request, context, 0.77)
    assert len(robot.fk_calls) == 2
    stamps[0] = stamp
    robot.qpos += 1
    with pytest.raises(ValueError, match="actual B1"):
        provider.measure(request, context, 0.77)
