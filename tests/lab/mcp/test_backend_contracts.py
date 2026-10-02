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

"""Provider-free regression tests for MCP lifecycle and protocol contracts."""

from __future__ import annotations

import asyncio

import pytest

from embodichain.mcp import (
    EmbodiChainMCPService,
    InMemorySimulationBackend,
    SimulationManagerBackend,
    create_server,
)
from embodichain.mcp.server import serve


def test_in_memory_restore_keeps_scene_revision_and_rejects_stale_snapshot() -> None:
    backend = InMemorySimulationBackend()
    state = backend.create_world(backend="in-memory", seed=3)
    world_id = state["world_id"]
    backend.load_scene(world_id, task_id="demo", scene=None)
    snapshot_id = backend.snapshot_world(world_id)["snapshot_id"]
    backend.step_simulation(world_id, steps=3)

    restored = backend.restore_world(world_id, snapshot_id)

    assert restored["scene_revision"] == 1
    assert restored["time_s"] == pytest.approx(0.0)

    backend.destroy_world(world_id)
    with pytest.raises(ValueError, match="Unknown world_id"):
        backend.restore_world(world_id, snapshot_id)


def test_service_limits_step_and_trajectory_expansion() -> None:
    service = EmbodiChainMCPService()
    try:
        world_id = service.create_world()["world_id"]
        with pytest.raises(ValueError, match="between 1 and"):
            service.step_simulation(world_id, steps=100_001)
        with pytest.raises(ValueError, match="expanded trajectory"):
            service.generate_robot_trajectory(
                world_id,
                "demo_planar_arm",
                [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]],
                samples_per_segment=50_001,
            )
    finally:
        service.close()


def test_service_destroy_purges_trajectory_handles() -> None:
    service = EmbodiChainMCPService()
    try:
        world_id = service.create_world()["world_id"]
        result = service.generate_robot_trajectory(
            world_id,
            "demo_planar_arm",
            [[0.0, 0.0], [0.1, 0.1]],
        )
        trajectory_id = result["result"]["trajectory_id"]

        service.destroy_world(world_id)

        with pytest.raises(ValueError, match="Unknown resource URI"):
            service.read_resource(f"embodichain://trajectories/{trajectory_id}")
    finally:
        service.close()


def test_task_catalog_does_not_expose_internal_resource_handles() -> None:
    internal_resource = object()
    service = EmbodiChainMCPService(
        task_provider=lambda: [
            {
                "task_id": "demo:task",
                "deployments": [
                    {"name": "default", "_config_resource": internal_resource}
                ],
            }
        ]
    )
    try:
        task = service.list_tasks()[0]
        assert task["deployments"] == [{"name": "default"}]
    finally:
        service.close()


def test_manager_backend_restores_current_state_and_closes_worlds() -> None:
    class FakePhysics:
        name = "default"

    class FakeRobot:
        def __init__(self) -> None:
            self.current_qpos = [[0.0, 0.0]]
            self.current_qvel = [[0.0, 0.0]]
            self.targets: list[tuple[str, list[list[float]]]] = []

        def get_qpos(self):
            import torch

            return torch.tensor(self.current_qpos)

        def get_qvel(self):
            import torch

            return torch.tensor(self.current_qvel)

        def set_qpos(self, qpos, *, target=True):
            values = qpos.detach().cpu().tolist()
            self.targets.append(("qpos", values))
            if not target:
                self.current_qpos = values

        def set_qvel(self, qvel, *, target=True):
            values = qvel.detach().cpu().tolist()
            self.targets.append(("qvel", values))
            if not target:
                self.current_qvel = values

        def clear_dynamics(self):
            return None

        def get_qpos_limits(self):
            import torch

            return torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])

    class FakeManager:
        physics = FakePhysics()

        def __init__(self) -> None:
            self.robot = FakeRobot()
            self.destroyed = False

        def get_robot_uid_list(self):
            return ["fake_robot"]

        def get_robot(self, robot_id):
            return self.robot if robot_id == "fake_robot" else None

        def get_sensor_uid_list(self):
            return []

        def reset_objects_state(self):
            return None

        def update(self, *, step, render_final_step):
            return None

        def destroy(self, *, exit_process):
            assert exit_process is False
            self.destroyed = True

    managers: list[FakeManager] = []

    def factory(backend, seed):
        manager = FakeManager()
        managers.append(manager)
        return manager

    backend = SimulationManagerBackend(factory)
    world_id = backend.create_world(backend="default", seed=7)["world_id"]
    snapshot_id = backend.snapshot_world(world_id)["snapshot_id"]
    managers[0].robot.current_qpos = [[0.8, 0.7]]
    restored = backend.restore_world(world_id, snapshot_id)

    assert restored["robots"]["fake_robot"]["qpos"] == [[0.0, 0.0]]
    assert ("qpos", [[0.0, 0.0]]) in managers[0].robot.targets

    backend.close()
    assert managers[0].destroyed is True
    assert backend._managers == {}


def test_service_close_closes_owned_manager_backend_worlds() -> None:
    class FakeManager:
        class Physics:
            name = "default"

        physics = Physics()
        destroyed = False

        def get_robot_uid_list(self):
            return []

        def get_sensor_uid_list(self):
            return []

        def get_robot(self, robot_id):
            return None

        def destroy(self, *, exit_process):
            self.destroyed = True

    managers: list[FakeManager] = []

    def factory(backend, seed):
        manager = FakeManager()
        managers.append(manager)
        return manager

    backend = SimulationManagerBackend(factory)
    service = EmbodiChainMCPService(backend=backend)
    service.create_world()
    service.close()

    assert managers[0].destroyed is True
    assert backend._managers == {}


def test_manager_backend_rejects_unbounded_camera_before_manager_setup() -> None:
    class FakeManager:
        class Physics:
            name = "default"

        physics = Physics()

        def get_robot_uid_list(self):
            return []

        def get_sensor_uid_list(self):
            return []

        def get_robot(self, robot_id):
            return None

        def destroy(self, *, exit_process):
            return None

    backend = SimulationManagerBackend(lambda backend, seed: FakeManager())
    world_id = backend.create_world(backend="default", seed=None)["world_id"]
    with pytest.raises(ValueError, match="Camera width and height"):
        backend.load_scene(
            world_id,
            task_id=None,
            scene={
                "sensor": [
                    {
                        "sensor_type": "Camera",
                        "width": 100_000,
                        "height": 100_000,
                    }
                ]
            },
        )
    backend.close()


def test_server_exposes_stdio_only_and_stateful_annotations() -> None:
    pytest.importorskip("mcp")
    service = EmbodiChainMCPService()
    try:
        server = create_server(service)

        async def inspect():
            return {tool.name: tool for tool in await server.list_tools()}

        tools = asyncio.run(inspect())
        assert service.server_info()["transports"] == ["stdio"]
        assert tools["generate_robot_trajectory"].annotations.read_only_hint is False
        assert tools["plan_motion"].annotations.read_only_hint is True
        assert tools["destroy_world"].annotations.destructive_hint is True
        with pytest.raises(ValueError, match="authenticated MCP gateway"):
            serve("streamable-http", service=service)
    finally:
        service.close()
