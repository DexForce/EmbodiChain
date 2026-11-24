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

"""Run the packaged Franka Pick/Place Task Program with direct Sim stepping."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

from embodichain.lab.gym.utils.composition import resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg
from embodichain.lab.sim import SimulationManager
from embodichain.lab.task_program.integrations import (
    load_configured_task_program_deployment,
)
from embodichain.utils import set_seed
from embodichain.utils.utility import load_config

if TYPE_CHECKING:
    from embodichain.lab.gym.envs.task_program.bridge import TaskProgramDemoBridge
    from embodichain.lab.sim.objects.robot import Robot

__all__ = ["main"]


@dataclass
class _SimulationHost:
    """Live inputs for the configured expert adapter."""

    sim: SimulationManager
    robot: Robot
    step_dt: float


def _run_program(
    host: _SimulationHost, bridge: TaskProgramDemoBridge, sim_steps_per_control: int
) -> bool:
    physics_dt = host.sim.sim_config.physics_dt
    env_ids = bridge.env_ids.tolist()
    for segment in bridge.iter_segments():
        for action in segment.actions:
            host.robot.set_qpos(qpos=action.value, env_ids=env_ids, target=True)
            host.sim.update(physics_dt=physics_dt, step=sim_steps_per_control)
        # Validate the segment before requesting the next one from the bridge.
        accepted = segment.validator()
        print(f"{segment.name}: accepted={accepted.tolist()}")

    completion = bridge.completion_mask
    print(f"{bridge.program_id}: completion_mask={completion.tolist()}")
    return bool(completion.all())


def main() -> int:
    """Run one headless expert program and return its acceptance exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, help="Compatible Task Program YAML.")
    args = parser.parse_args()
    set_seed(7)

    path = Path(
        str(
            files("embodichain.extensions.tasks.configs").joinpath(
                "tasks", "manipulation", "repeated_pick_place", "task.franka.yaml"
            )
        )
    )
    declaration = load_config(path)
    components = resolve_gym_components(declaration, base_dir=path.parent)
    selectors = dict(declaration["task_program"])
    if args.program is not None:
        selectors["program"] = str(args.program.expanduser().resolve())
    deployment = load_configured_task_program_deployment(
        task_program=selectors,
        skill_profile=components.embodiment_skill_profile,
        base_dir=path.parent,
    )
    components.validate_scene_bindings(deployment.scene_binding)
    program = deployment.load_program()

    # Reuse the typed spawn configs from the deployment decoder.
    config = components.config
    config.pop("task_program")
    config.update(headless=True, enable_sensor=False, device="cuda:0")
    cfg = config_to_cfg(config, source_path=path, _register_env=False)
    sim = SimulationManager(cfg.sim_cfg, requires_native_renderer=False)
    try:
        robot = sim.add_robot(cfg.robot)
        for object_cfg in cfg.rigid_object:
            sim.add_rigid_object(object_cfg)
        sim.prepare()
        robot.reset()

        host = _SimulationHost(
            sim, robot, sim.sim_config.physics_dt * cfg.sim_steps_per_control
        )
        adapter = deployment.integration.adapter_factory.create_adapter(host)
        try:
            bridge = adapter.create_bridge(adapter.compile(program))
            return 0 if _run_program(host, bridge, cfg.sim_steps_per_control) else 1
        finally:
            adapter.close()
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    raise SystemExit(main())
