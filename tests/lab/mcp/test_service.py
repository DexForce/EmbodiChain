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

"""Contract tests for the provider-neutral MCP service."""

from __future__ import annotations

import asyncio
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from embodichain.mcp import (
    EmbodiChainMCPService,
    SimulationManagerBackend,
    create_server,
)
from embodichain.lab.mcp import EmbodiChainMCPService as LegacyMCPService

DEMO_ROBOT = "demo_planar_arm"
TARGET_POSE = {"position_m": [0.6, 0.2, 0.0]}


@pytest.fixture
def service():
    instance = EmbodiChainMCPService(
        task_provider=lambda: [
            {
                "task_id": "demo:reach",
                "title": "Reach",
                "deployments": [],
            }
        ]
    )
    yield instance
    instance.close()


def test_world_snapshot_restore_preserves_scene_revision(service):
    world = service.create_world(seed=123)
    world_id = world["world_id"]
    service.load_task_or_scene(world_id, task_id="demo:reach")
    snapshot = service.snapshot_world(world_id)

    service.step_simulation(world_id, steps=3)
    restored = service.restore_world(world_id, snapshot["result"]["snapshot_id"])

    assert restored["status"] == "succeeded"
    assert restored["scene_revision"] == 1
    assert restored["result"]["time_s"] == pytest.approx(0.0)


def test_kinematics_and_trajectory_contracts_are_structured(service):
    world_id = service.create_world()["world_id"]
    ik = service.solve_ik(world_id, DEMO_ROBOT, TARGET_POSE)
    fk = service.forward_kinematics(world_id, DEMO_ROBOT, ik["result"]["qpos"])
    plan = service.plan_motion(
        world_id, DEMO_ROBOT, [0.0, 0.0], ik["result"]["qpos"], samples=4
    )
    validation = service.validate_trajectory(
        world_id, DEMO_ROBOT, plan["result"]["positions"]
    )

    assert ik["status"] == "succeeded"
    assert ik["result"]["reachable"] is True
    assert fk["result"]["position_m"][:2] == pytest.approx(
        TARGET_POSE["position_m"][:2]
    )
    assert len(plan["result"]["positions"]) == 4
    assert validation["result"]["valid"] is True


def test_resource_templates_return_json(service):
    world_id = service.create_world(seed=7)["world_id"]

    world_resource = service.read_resource(f"embodichain://worlds/{world_id}/manifest")

    assert f'"world_id": "{world_id}"' in world_resource
    assert '"seed": 7' in world_resource
    assert "demo_planar_arm" in service.read_resource(
        "embodichain://robots/demo_planar_arm/config"
    )


def test_generate_and_execute_trajectory_returns_handle(service):
    world_id = service.create_world()["world_id"]
    result = service.generate_robot_trajectory(
        world_id,
        DEMO_ROBOT,
        [[0.0, 0.0], [0.2, 0.1], [0.0, 0.3]],
        samples_per_segment=4,
    )
    trajectory_id = result["result"]["trajectory_id"]

    assert result["result"]["validation"]["valid"] is True
    resource = service.read_resource(f"embodichain://trajectories/{trajectory_id}")
    assert trajectory_id in resource
    execution = service.execute_trajectory(trajectory_id)

    assert execution["result"]["status"] == "succeeded"
    assert execution["result"]["steps"] == 7


def test_rollout_job_reaches_terminal_state(service):
    world_id = service.create_world()["world_id"]
    started = service.start_rollout(world_id, steps=10)
    run_id = started["run_id"]

    deadline = time.monotonic() + 2.0
    status = service.get_run_status(run_id)
    while status["status"] in {"queued", "running"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
        status = service.get_run_status(run_id)

    assert status["status"] == "succeeded"
    assert service.get_run_metrics(run_id)["final_time_s"] == pytest.approx(0.1)


def test_unknown_resource_and_robot_fail_at_service_boundary(service):
    with pytest.raises(ValueError, match="Unknown robot_id"):
        service.get_robot_info("missing")
    with pytest.raises(ValueError, match="Unknown resource URI"):
        service.read_resource("embodichain://unknown/value")


def test_mcp_server_registers_phase_one_tools_resources_and_prompts(service):
    pytest.importorskip("mcp")
    world_id = service.create_world()["world_id"]
    server = create_server(service)

    async def inspect_server():
        tools = await server.list_tools()
        templates = await server.list_resource_templates()
        prompts = await server.list_prompts()
        result = await server.call_tool(
            "check_reachability",
            {
                "world_id": world_id,
                "robot_id": DEMO_ROBOT,
                "target_pose": TARGET_POSE,
            },
        )
        return tools, templates, prompts, result

    tools, templates, prompts, result = asyncio.run(inspect_server())
    tool_names = {tool.name for tool in tools}
    template_uris = {template.uri_template for template in templates}
    prompt_names = {prompt.name for prompt in prompts}

    assert "solve_ik" in tool_names
    assert "start_rollout" in tool_names
    assert "generate_robot_trajectory" in tool_names
    assert "embodichain://worlds/{world_id}/manifest" in template_uris
    assert prompt_names == {
        "inspect_robot",
        "validate_motion",
        "generate_robot_trajectory",
    }
    assert result.is_error is False
    assert result.structured_content["result"]["reachable"] is True


def test_legacy_lab_import_path_reexports_top_level_service():
    assert LegacyMCPService is EmbodiChainMCPService


def test_top_level_import_does_not_load_simulation_or_optional_sdk():
    """The public MCP entry point must work before a domain runtime is loaded."""
    script = """
import sys
from embodichain.mcp import EmbodiChainMCPService, create_server

assert "embodichain.lab" not in sys.modules
assert "dexsim" not in sys.modules
assert "mcp" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_simulation_manager_backend_adapts_lifecycle_and_kinematics():
    class FakePhysics:
        name = "default"

    class FakeRobot:
        qpos_limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]])

        def get_qpos(self):
            return torch.tensor([[0.1, 0.2]])

        def get_qvel(self):
            return torch.zeros((1, 2))

        def compute_fk(self, qpos):
            return torch.cat((qpos, torch.zeros((qpos.shape[0], 5))), dim=1)

        def compute_ik(self, pose):
            return torch.ones((pose.shape[0],), dtype=torch.bool), torch.tensor(
                [[0.1, 0.2]]
            )

        def set_qpos(self, qpos):
            del qpos

        def set_qvel(self, qvel):
            del qvel

    class FakeManager:
        physics = FakePhysics()

        def __init__(self):
            self.robot = FakeRobot()
            self.updated_steps = 0
            self.destroyed = False

        def get_robot_uid_list(self):
            return ["fake_robot"]

        def get_robot(self, robot_id):
            return self.robot if robot_id == "fake_robot" else None

        def reset_objects_state(self):
            return None

        def update(self, *, step, render_final_step):
            assert render_final_step is False
            self.updated_steps += step

        def destroy(self, *, exit_process):
            assert exit_process is False
            self.destroyed = True

    managers = []

    def factory(backend, seed):
        assert (backend, seed) == ("default", 9)
        manager = FakeManager()
        managers.append(manager)
        return manager

    backend = SimulationManagerBackend(factory)
    state = backend.create_world(backend="default", seed=9)
    world_id = state["world_id"]

    assert backend.list_robots()[0]["robot_id"] == "fake_robot"
    assert backend.forward_kinematics(world_id, "fake_robot", [0.1, 0.2])[
        "pose_xyz_xyzw"
    ]
    assert (
        backend.solve_ik(world_id, "fake_robot", {"position_m": [0.0, 0.0, 0.0]})[
            "reachable"
        ]
        is True
    )
    backend.step_simulation(world_id, steps=3)
    backend.destroy_world(world_id)

    assert managers[0].updated_steps == 3
    assert managers[0].destroyed is True
