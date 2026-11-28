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

"""Prepare physical grasp labels from the calls selected by a Gym deployment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from embodichain.lab.sim.affordance import AntipodalAffordance
from embodichain.lab.sim.affordance.grasp.cache import ValidatedGraspCache
from embodichain.lab.sim.atomic_skills._grasp_selection import (
    SAMPLED_GRASP_SKILLS,
    prepare_grasp_request,
)

if TYPE_CHECKING:
    from embodichain.lab.task_program import CompiledTaskProgram
    from embodichain.lab.task_program.integrations.environment import (
        TaskProgramRuntimeAssembly,
    )

__all__: list[str] = []

_Pair = tuple[str, str, str]


def prepare_deployment_grasps(
    gym_config_path: str,
    physics: str | None,
    pairs: Sequence[Sequence[str]],
) -> Iterator[tuple[_Pair, Path]]:
    """Open a single no-render deployment and report the labels prepared for each pair."""
    import gymnasium
    from embodichain.lab.gym.envs.managers.record import (
        record_camera_data,
        validation_cameras,
    )
    from embodichain.lab.gym.utils.gym_utils import (
        add_env_launcher_args_to_parser,
        build_env_cfg_from_args,
    )
    from embodichain.lab.gym.utils.registration import (
        discover_task_packages,
        execute_init_hooks,
    )
    from embodichain.lab.sim import SimulationManager

    launcher = argparse.ArgumentParser(add_help=False)
    add_env_launcher_args_to_parser(launcher, require_gym_config=True)
    launcher.set_defaults(
        headless=True,
        num_envs=1,
        renderer="no-render",
        filter_dataset_saving=True,
        disable_sensor=True,
        physics=physics,
    )
    launch_args = launcher.parse_args(["--gym-config", gym_config_path])
    discover_task_packages()
    execute_init_hooks()
    cfg, gym_config, action_config = build_env_cfg_from_args(launch_args)
    # These event functors create their own cameras during world construction,
    # independently of the deployment's sensor acquisition setting.
    for name, event in vars(cfg.events).items():
        func = getattr(event, "func", None)
        if isinstance(func, type) and issubclass(
            func, (record_camera_data, validation_cameras)
        ):
            setattr(cfg.events, name, None)
    env = gymnasium.make(gym_config["id"], cfg=cfg, **action_config)
    try:
        adapter = env.unwrapped.task_program_adapter
        program = adapter.compile(cfg.task_program)
        assembly = adapter.assemble_runtime(program.integration)
        yield from _prepare_pairs(assembly, program, pairs)
    finally:
        env.unwrapped.close(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _prepare_pairs(
    assembly: TaskProgramRuntimeAssembly,
    program: CompiledTaskProgram,
    pairs: Sequence[Sequence[str]],
) -> Iterator[tuple[_Pair, Path]]:
    registry, engine = assembly.scene_registry, assembly.engine
    requested = {}
    for part, obj, affordance_id in pairs:
        semantics = registry.object_semantics(obj, affordance=affordance_id)
        cache = getattr(
            engine.grasp_pose_generators.get(part), "validated_labels", None
        )
        if not isinstance(semantics.affordance, AntipodalAffordance) or not isinstance(
            cache, ValidatedGraspCache
        ):
            raise ValueError(
                f"{part}/{affordance_id} requires a validated antipodal hand and affordance"
            )
        requested[(part, obj, affordance_id)] = semantics

    context = assembly.observation_provider.observe(assembly.runtime.task_state)
    eligible = torch.ones_like(context.env_ids, dtype=torch.bool)
    reported = set()
    for analysis in program.preflight_analyses():
        workflow = assembly.compiler.analyze(
            analysis.calls, workflow_id=analysis.analysis_id, path=analysis.source_path
        )
        for index, call in enumerate(workflow.calls):
            if call.bound.linked.descriptor.skill_id not in SAMPLED_GRASP_SKILLS:
                continue
            endpoints = {
                e.target.target_id
                for e in call.bound.binding.skill_binding.endpoints
                if e.endpoint_id == "grasp"
            }
            linked = {ref.entity_id for ref in call.bound.linked.affordances.values()}
            selected = {
                pair: sem
                for pair, sem in requested.items()
                if pair[0] in endpoints
                and (not linked or registry.resolve(pair[2]).entity_id in linked)
            }
            if not selected:
                continue
            invocation, _ = assembly.compiler.lower_invocation(
                call,
                context,
                eligible,
                path=analysis.source_path,
                invocation_id=f"{workflow.workflow_id}:{index}",
            )
            request = engine.resolve(invocation)
            target = getattr(request.goal, "semantics", None)
            matching_pairs = [
                pair
                for pair, sem in selected.items()
                if target is not None
                and sem.entity_id == target.entity_id
                and isinstance(target.affordance, AntipodalAffordance)
                and (linked or _same_physical_target(sem.affordance, target.affordance))
            ]
            if not matching_pairs:
                continue
            services = {
                pair[0]: engine.grasp_pose_generators[pair[0]]
                for pair in matching_pairs
            }
            paths = prepare_grasp_request(request, services, context)
            for pair in matching_pairs:
                for path in paths.get(pair[0], ()):
                    if (pair, path) not in reported:
                        reported.add((pair, path))
                        yield pair, path

    missing = requested.keys() - {pair for pair, _ in reported}
    if missing:
        raise ValueError(
            f"No sampled grasp invocation in this deployment uses {sorted(missing)}"
        )


def _same_physical_target(
    left: AntipodalAffordance, right: AntipodalAffordance
) -> bool:
    """Match a registered lowerer's target when its descriptor has no affordance reference."""

    def attrs(affordance: AntipodalAffordance) -> str:
        return json.dumps(
            affordance.validation_attrs or {},
            sort_keys=True,
            default=lambda value: value.tolist(),
        )

    return (
        left.mesh_vertices is not None
        and right.mesh_vertices is not None
        and torch.equal(left.mesh_vertices, right.mesh_vertices)
        and torch.equal(left.mesh_triangles, right.mesh_triangles)
        and attrs(left) == attrs(right)
    )
