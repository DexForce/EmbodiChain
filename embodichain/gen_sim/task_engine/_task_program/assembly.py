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

"""Task-owned composition around the unchanged simulation runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from embodichain.lab.task_program.integrations.environment import (
    TaskProgramEnvironmentAdapter,
)
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramFactory,
)
from embodichain.utils.utility import load_config

from ..contracts import canonical_hash
from .stability import StabilityConstraint, TaskStabilityPort
from .configured import compose_deployment
from .grasp_filter import GRASP_FILTER_REVISION, install_grasp_filters

__all__: list[str] = []

ADAPTER_CONTRACT = "gen_sim.task_program/2620929c/v3"


class _TaskFactory(SimulationTaskProgramFactory):
    """Select task-owned observation services, not a different executor."""

    def __init__(
        self,
        *args: Any,
        constraints: dict[str, StabilityConstraint],
        articulation_bindings: tuple = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._task_post_port = TaskStabilityPort(
            self.segment_policy_port,
            self._simulation,
            self._robot,
            self.task_program_registration.scene_binding,
            constraints,
            step_dt=self.step_dt,
        )
        if articulation_bindings:
            from .articulation_slide import (
                ArticulationStabilityPort,
                synchronize_joint_limits,
            )

            for binding in articulation_bindings:
                synchronize_joint_limits(
                    binding, self._simulation.get_articulation(binding.object_id)
                )
            self._task_post_port = ArticulationStabilityPort(
                self._task_post_port,
                self._simulation,
                self._robot,
                articulation_bindings,
                self.step_dt,
            )

    def registration_owned_segment_policy_ports(self) -> tuple[Any, Any]:
        return self._task_post_port, self.segment_policy_port


@dataclass(frozen=True, slots=True)
class TaskAdapterFactory:
    """Immutable identity and lazy service construction for a GenSim bundle."""

    registration: Any
    integration_fingerprint: str
    constraints: tuple[tuple[str, StabilityConstraint], ...]
    grasp_factories: tuple[tuple[str, Any], ...]
    cartesian_approaches: bool = False
    articulation_bindings: tuple = ()

    def create_adapter(self, environment: Any) -> TaskProgramEnvironmentAdapter:
        """Return the exact shared adapter; no Session or Bridge is overridden."""
        self.registration.assert_unchanged()
        motion_factory = None
        if self.cartesian_approaches:
            from embodichain.lab.sim.motion.motion_generator import MotionGenCfg
            from embodichain.lab.sim.motion.planners import ToppraPlannerCfg
            from .motion import ApproachMotionGenerator

            motion_factory = lambda: ApproachMotionGenerator(
                MotionGenCfg(
                    planner_cfg=ToppraPlannerCfg(robot_uid=environment.robot.uid)
                )
            )
        factory = _TaskFactory(
            environment.sim,
            environment.robot,
            self.registration,
            step_dt=environment.step_dt,
            motion_generator_factory=motion_factory,
            grasp_pose_generators=install_grasp_filters(
                self.registration,
                environment.sim,
                {name: create() for name, create in self.grasp_factories},
            ),
            constraints=dict(self.constraints),
            articulation_bindings=self.articulation_bindings,
        )
        return factory.create_adapter()


def load_deployment(
    *, task_program: object, skill_profile: object, base_dir: str | Path
) -> Any:
    """Compose the current integration plus an explicitly versioned local contract."""
    base = compose_deployment(
        task_program=task_program,
        skill_profile=skill_profile,
        base_dir=base_dir,
    )
    path = Path(base_dir) / "task_program" / "constraints.json"
    if not path.is_file():
        raise ValueError(
            "GenSim bundle has no task constraints; regenerate the bundle."
        )
    payload = load_config(path)
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "presets"}
        or payload["schema_version"] != "gen_sim_task_constraints/v1"
    ):
        raise ValueError(
            "Unsupported GenSim task constraint format; regenerate the bundle."
        )
    presets = payload["presets"]
    if type(presets) is not dict or any(
        type(name) is not str or not name.startswith("gen_sim.") or name != name.strip()
        for name in presets
    ):
        raise ValueError("Task stability presets must have exact gen_sim.* names.")
    constraints = {
        name: StabilityConstraint.decode(cfg) for name, cfg in presets.items()
    }
    settle_presets = dict(base.integration.registration.settle_presets)
    if set(settle_presets) & set(constraints):
        raise ValueError("Task stability presets cannot replace core settling presets.")
    for name in constraints:
        settle_presets[name] = settle_presets["rigid_object"].snapshot()
    from .articulation_slide import (
        ArticulationSlideFactory,
        ArticulationWithdrawFactory,
        preset_id,
    )

    slides = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if type(f) is ArticulationSlideFactory
    ]
    withdrawals = [
        f
        for f in base.integration.registration.registered_semantic_lowerer_factories
        if type(f) is ArticulationWithdrawFactory
    ]
    articulation_bindings = ()
    if slides or withdrawals:
        if (
            len(slides) != 1
            or len(withdrawals) != 1
            or slides[0].bindings != withdrawals[0].bindings
        ):
            raise ValueError(
                "E6 Slide and withdrawal require identical declared bindings."
            )
        articulation_bindings = slides[0].bindings
        for binding in articulation_bindings:
            for state in ("open", "closed"):
                name = preset_id(binding, state)
                if name in settle_presets:
                    raise ValueError(
                        "Articulation policies cannot replace existing presets."
                    )
                settle_presets[name] = settle_presets["rigid_object"].snapshot()
    registration = replace(base.integration.registration, settle_presets=settle_presets)
    program = load_config(base.program_path)
    from .align_held import with_held_alignment

    registration = with_held_alignment(
        registration, program=program, constraints=constraints
    )
    from .release_clearance import CLEAR_RELEASED_CALL, with_release_clearance

    registration = with_release_clearance(registration, program=program)
    if any(cfg.kind == "stack" for cfg in constraints.values()):
        from .stack_place import with_stack_placement

        registration = with_stack_placement(
            registration,
            targets=frozenset(
                (cfg.entity, cfg.reference)
                for cfg in constraints.values()
                if cfg.kind == "stack"
            ),
        )
    grasp_factories = base.integration.grasp_factories
    # The pad envelope is needed for thin-object coordinated grasps. Retain
    # the conservative full-finger envelope for inclined single-arm grasps:
    # this three-box model couples palm placement to the finger length.
    # The configured Robotiq grasp command closes its pads to zero gap.
    # Its reference 1 cm sampling cutoff excludes the original tray's
    # measured 5-8 mm contacts. The URDF pad collision mesh spans 65 mm
    # about the configured 0.2 m TCP, not the proxy's symmetric 130 mm.
    # Keep palm/width/thickness checks and all physical commands unchanged.
    coordinated = any(
        item.get("steps", {}).get("call", {}).get("call_id")
        in {"simulation.coordinated_hold", "simulation.coordinated_transport"}
        for item in program["program"]["items"]
    )
    grasp_factories = tuple(
        (
            name,
            (
                replace(
                    factory,
                    min_opening_width=(
                        min(factory.min_opening_width, 0.005)
                        if coordinated
                        else factory.min_opening_width
                    ),
                    finger_length=0.065 if coordinated else factory.finger_length,
                )
                if factory.model_id == "robotiq_arg2f_140"
                else factory
            ),
        )
        for name, factory in grasp_factories
    )
    cartesian_approaches = any(
        item.get("steps", {}).get("call", {}).get("kind") == "hand_over"
        or item.get("steps", {}).get("call", {}).get("call_id")
        in {CLEAR_RELEASED_CALL, "gen_sim.articulation_withdraw"}
        for item in program["program"]["items"]
    )
    fingerprint = canonical_hash(
        {
            "adapter_contract": ADAPTER_CONTRACT,
            "grasp_filter_revision": GRASP_FILTER_REVISION,
            "core_integration": base.integration.integration_fingerprint,
            "registration": registration.fingerprint,
            "task_constraints": payload,
            "program": program,
            "cartesian_approaches": cartesian_approaches,
            "grasp_pose_generators": {
                name: asdict(factory) for name, factory in grasp_factories
            },
        }
    )
    adapter = TaskAdapterFactory(
        registration,
        fingerprint,
        tuple(constraints.items()),
        grasp_factories,
        cartesian_approaches,
        articulation_bindings,
    )
    integration = replace(
        base.integration,
        registration=registration,
        adapter_factory=adapter,
        integration_fingerprint=fingerprint,
    )
    return replace(base, integration=integration)


def register_deployment(
    deployment: Any, *, environment_id: str, max_episode_steps: int
) -> None:
    """Use the common environment and registry with a task-owned adapter factory."""
    from embodichain.lab.gym.envs import EmbodiedEnv
    from embodichain.lab.gym.utils.registration import (
        REGISTERED_ENVS,
        get_env_spec,
        register_env_function,
    )

    if environment_id in REGISTERED_ENVS:
        previous = get_env_spec(environment_id)
        if (
            previous.cls is not EmbodiedEnv
            or previous.task_program_adapter_factory.integration_fingerprint
            != deployment.integration.integration_fingerprint
        ):
            raise ValueError(
                "GenSim environment ID is already bound to another integration."
            )
        return
    register_env_function(
        EmbodiedEnv,
        environment_id,
        max_episode_steps=max_episode_steps,
        task_program_adapter_factory=deployment.integration.adapter_factory,
    )
