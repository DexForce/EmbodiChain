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
"""Default backend contact-point multiplicity regression contract."""

from __future__ import annotations

import pytest
import warp as wp
import torch

from embodichain.lab.sim.cfg import (
    DefaultPhysicsCfg,
    MassPropertiesCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.sensors import ContactSensorCfg, ContactPermissionChecker
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

pytestmark = pytest.mark.requires_sim


@pytest.mark.parametrize(
    ("device", "expected_contacts_per_env", "direct_gpu_api"),
    [
        pytest.param("cpu", 4, True, id="cpu"),
        pytest.param("cuda:0", 4, True, marks=pytest.mark.gpu, id="cuda"),
        pytest.param(
            "cuda:0", 4, False, marks=pytest.mark.gpu, id="cuda-native-reports"
        ),
    ],
)
def test_default_contact_sensor_preserves_contact_multiplicity(
    device: str,
    expected_contacts_per_env: int,
    direct_gpu_api: bool,
) -> None:
    wp.init()
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is required for the GPU contact E2E contract.")

    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            num_envs=2,
            arena_space=2.0,
            physics_cfg=DefaultPhysicsCfg(device=device, direct_gpu_api=direct_gpu_api),
        )
    )
    try:
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="ground",
                shape=CubeCfg(size=[1.0, 1.0, 0.1]),
                attrs=RigidBodyPhysicsCfg(),
                body_type="static",
            )
        )
        sim.add_rigid_object(
            RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[0.2, 0.2, 0.2]),
                attrs=RigidBodyPhysicsCfg(mass_props=MassPropertiesCfg(mass=1.0)),
                body_type="dynamic",
                init_pos=(0.0, 0.0, 0.4),
            )
        )
        sensor = sim.add_sensor(
            ContactSensorCfg(
                uid="contacts",
                rigid_uid_list=["cube"],
                filter_need_both_actor=False,
                max_contacts_per_env=16,
            )
        )

        counts = [0, 0]
        for _ in range(80):
            sim.update(step=1)
            sensor.update()
            counts = sensor._num_contacts_per_env.cpu().tolist()
            if counts == [expected_contacts_per_env] * 2:
                break

        assert not sensor.contact_capabilities.friction
        assert counts == [expected_contacts_per_env] * 2
        data = sensor.get_data()
        valid = data["is_valid"]
        assert (data["impulse"][valid] > 1.0e-7).all()
        positions = data["position"][valid]
        assert positions[:, 0].abs().max().item() < 0.2
        assert (positions[:, 2] - 0.05).abs().max().item() < 0.02
        actor_ids = data["user_ids"][valid]
        assert (actor_ids >= 0).any(dim=1).all()
        cube_actor_ids = set(sensor.item_user_ids.cpu().tolist())
        assert all(
            any(actor_id in cube_actor_ids for actor_id in pair)
            for pair in actor_ids.cpu().tolist()
        )
        assert all(
            sensor.get_actor_info(actor_id).path.endswith("/cube")
            for actor_id in cube_actor_ids
        )
        assert sim.is_use_gpu_physics == device.startswith("cuda")
        assert sim.get_world().get_physics_scene().is_direct_gpu() == (
            device.startswith("cuda") and direct_gpu_api
        )
        if device == "cpu" or not direct_gpu_api:
            maps = [{} for _ in range(2)]
            for uid in ("cube", "ground", "default_plane"):
                values = sensor.get_actor_ids(uid)[:, 0].cpu().tolist()
                for row, actor in enumerate(values):
                    maps[row][actor] = uid
            checker = ContactPermissionChecker(
                sensor,
                maps,
                allowed_contacts=(("cube", "ground"),),
                moving_frames=("cube",),
            )
            accepted = checker.check()
            assert accepted[:, 0].cpu().tolist() == [0, 0]
            refused = ContactPermissionChecker(
                sensor, maps, allowed_contacts=(), moving_frames=("cube",)
            )
            assert refused.check()[:, 0].cpu().tolist() == [3, 3]
            assert accepted[:, 0].cpu().tolist() == [0, 0]
            if device.startswith("cuda"):
                other = torch.cuda.Stream()
                with torch.cuda.stream(other):
                    repeated = checker.check()
                other.synchronize()
                assert repeated[:, 0].cpu().tolist() == [0, 0]
            bounded = sim.add_sensor(
                ContactSensorCfg(
                    uid="bounded",
                    rigid_uid_list=["cube"],
                    filter_need_both_actor=False,
                    max_contacts_per_env=1,
                )
            )
            bounded.update()
            overflow = ContactPermissionChecker(
                bounded,
                maps,
                allowed_contacts=(("cube", "ground"),),
                moving_frames=("cube",),
            )
            assert overflow.check()[:, 0].cpu().tolist() == [4, 4]
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()
