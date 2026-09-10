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

"""GenSim declarations and measured post-policies around public Slide execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, ClassVar

import torch

from embodichain.lab.sim.atomic_actions import (
    EndEffectorPoseGoal,
    JointPositionCommand,
    JointPositionTarget,
    MoveEndEffector,
    MoveEndEffectorOptions,
    ObjectSemantics,
    SceneEntityPose,
    Slide,
    SlideAffordance,
    SlideGoal,
    SlideJointTarget,
    SlideOptions,
)
from embodichain.lab.task_program.compiler.lowering import (
    RegisteredSemanticLowerer,
    SemanticLowering,
)
from embodichain.lab.task_program.integrations.extensions import (
    RegisteredSemanticLowererFactory,
)
from embodichain.lab.task_program.semantics import SceneArticulationRef, SceneLinkRef
from .articulation_binding import (
    PrismaticBinding,
    SLIDE_CALL,
    WITHDRAW_CALL,
    handle_mesh,
    inspect_prismatic,
    validate_placement,
)

__all__: list[str] = []


def synchronize_joint_limits(binding: PrismaticBinding, art: Any) -> None:
    """Validate native ownership and refresh only the stale public limit cache."""
    if art is None or art.joint_names != [binding.joint]:
        raise ValueError(
            "Native prismatic joint identity differs from its declaration."
        )
    joint = next(
        j for j in art.get_parent_joint_chain(binding.link) if j.joint_type != "fixed"
    )
    if (
        joint.name != binding.joint
        or joint.joint_type != "prismatic"
        or joint.parent_link_name != binding.parent
    ):
        raise ValueError(
            "The native prismatic joint does not own the declared handle link."
        )
    if joint.joint_limits is None:
        raise ValueError("Native prismatic joint limits are unavailable.")
    cached_limits = art.get_qpos_limits()
    native_limits = cached_limits.new_tensor([[joint.joint_limits]])
    if not torch.allclose(
        native_limits,
        cached_limits.new_tensor([[binding.limits]]),
        atol=1e-6,
        rtol=1e-5,
    ):
        raise ValueError(
            "Native prismatic limits differ from the scaled asset binding."
        )
    if not torch.allclose(cached_limits, native_limits, atol=1e-6, rtol=1e-5):
        # Reapply the already-active native limits to refresh the pre-scale cache.
        art.set_qpos_limits(native_limits)


def _bind(
    bindings: tuple[PrismaticBinding, ...], simulation: Any, registry: Any, engine: Any
) -> dict:
    if simulation.num_envs != 1:
        raise ValueError(
            "GenSim articulation recipes currently require one environment."
        )
    result = {}
    table = simulation.get_rigid_object("table")
    if table is None:
        raise ValueError("GenSim E6 requires a measured runtime table.")
    table_pose = table.get_local_pose(to_matrix=True)[0]
    table_vertices = table.get_vertices(scale=True)[0]
    table_top = float(
        (table_vertices @ table_pose[:3, :3].T + table_pose[:3, 3])[:, 2].max()
    )
    for binding in bindings:
        entry = registry.lookup(binding.link_id, expected_type=SceneLinkRef)
        if (
            entry.parent != SceneArticulationRef(binding.object_id)
            or entry.native_name != binding.link
        ):
            raise ValueError("Prismatic link ownership differs from its declaration.")
        art = simulation.get_articulation(binding.object_id)
        if art is None:
            raise ValueError("Declared prismatic articulation is absent.")
        checked = inspect_prismatic(
            {
                "uid": binding.object_id,
                "fpath": art.cfg.fpath,
                "fix_base": art.cfg.fix_base,
                "body_scale": art.cfg.body_scale,
            }
        )
        if checked != binding or art.joint_names != [binding.joint]:
            raise ValueError(
                "Prismatic asset identity or native joint binding changed."
            )
        validate_placement(
            binding,
            {
                "fpath": art.cfg.fpath,
                "body_scale": art.cfg.body_scale,
                "init_pos": art.cfg.init_pos,
                "init_rot": art.cfg.init_rot,
            },
            table_top,
        )
        synchronize_joint_limits(binding, art)
        vertices, faces = handle_mesh(binding, art.cfg.fpath)
        semantics = ObjectSemantics(
            entity_id=binding.link_id,
            geometry={},
            affordance=SlideAffordance(
                mesh_vertices=torch.tensor(
                    vertices, dtype=torch.float32, device=engine.device
                ),
                mesh_triangles=torch.tensor(
                    faces, dtype=torch.long, device=engine.device
                ),
                translation_axis=torch.tensor(binding.axis, device=engine.device)
                * binding.axis_sign,
                joint_name=binding.joint,
                joint_limits=binding.limits,
            ),
        )
        result[binding.object_id] = (binding, semantics, art)
    return result


class _SlideLowerer(RegisteredSemanticLowerer):
    call_id: ClassVar[str] = SLIDE_CALL
    target_descriptor = Slide.descriptor()
    preserves_symbolic_state: ClassVar[bool] = True

    def __init__(self, bindings: dict, robot: Any) -> None:
        self.bindings, self.robot = bindings, robot

    def _selection(self, call: Any, context: Any, bound: Any) -> tuple:
        args = dict(call.arguments)
        if set(args) != {"object", "state"} or args["object"] not in self.bindings:
            raise ValueError(
                "Articulation calls require an exact declared object and state."
            )
        binding, semantics, art = self.bindings[args["object"]]
        target = binding.target(args["state"])
        endpoint = bound.binding.action_binding.endpoint("primary", "motion")
        held = context.task.get_held_object(endpoint.task_state_key)
        if held is not None and (held.env_mask is None or held.env_mask.any()):
            raise ValueError(
                "Articulation interaction cannot reuse a hand holding another object."
            )
        for resources, held in context.task.coordinated_held_objects.items():
            if endpoint.task_state_key in resources and (
                held.env_mask is None or held.env_mask.any()
            ):
                raise ValueError(
                    "Articulation interaction cannot reuse a coordinated holding hand."
                )
        return binding, semantics, art, target, endpoint

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        if type(option_template) is not SlideOptions:
            raise TypeError("Articulation Slide requires SlideOptions.")
        binding, semantics, _, target, _ = self._selection(call, context, bound)
        return SemanticLowering(
            goal=SlideGoal(
                semantics,
                SceneEntityPose(binding.link_id),
                joint_target=SlideJointTarget(
                    binding.object_id,
                    binding.joint,
                    target,
                    axis_sign=binding.axis_sign,
                ),
            )
        )


class _WithdrawLowerer(_SlideLowerer):
    call_id: ClassVar[str] = WITHDRAW_CALL
    target_descriptor = MoveEndEffector.descriptor()

    def lower(
        self, call: Any, *, context: Any, bound: Any, option_template: Any
    ) -> SemanticLowering:
        if type(option_template) is not MoveEndEffectorOptions:
            raise TypeError("Articulation withdrawal requires MoveEndEffectorOptions.")
        binding, _, art, target, endpoint = self._selection(call, context, bound)
        qpos = art.get_qpos()[
            context.env_ids.tolist(), art.joint_names.index(binding.joint)
        ]
        if (
            not torch.isfinite(qpos).all()
            or (
                (qpos - target).abs() > binding.tolerance(call.arguments["state"])
            ).any()
        ):
            raise ValueError("Articulation target was lost before withdrawal.")
        hand = bound.binding.resources["primary"].endpoints["grasp"]
        command = hand.commands.get("open")
        if type(command) is not JointPositionCommand:
            raise ValueError(
                "Articulation withdrawal requires a declared hand-open command."
            )
        ids = list(hand.runtime_target.joint_ids)
        opened = command.resolve(
            num_envs=context.batch_size,
            control_dof=len(ids),
            device=context.robot.qpos.device,
            dtype=context.robot.qpos.dtype,
        )
        observed = context.robot.qpos[:, ids]
        if (
            not torch.isfinite(observed).all()
            or (observed - opened).abs().amax() > 0.005
        ):
            raise ValueError(
                "The hand has not reached its open posture before withdrawal."
            )
        motion = endpoint.require_target(JointPositionTarget)
        current = self.robot.compute_fk(
            qpos=context.robot.qpos[:, list(motion.joint_ids)],
            name=motion.control_part,
            to_matrix=True,
        )
        link = art.get_link_pose(
            binding.link, env_ids=context.env_ids.tolist(), to_matrix=True
        )
        outward = (
            -(link[:, :3, :3] @ current.new_tensor(binding.axis)) * binding.axis_sign
        )
        withdrawn = current.clone()
        withdrawn[:, :3, 3] += 0.10 * outward
        raised = withdrawn.clone()
        raised[:, 2, 3] += 0.05
        return SemanticLowering(
            goal=EndEffectorPoseGoal(torch.stack((withdrawn, raised), dim=1))
        )


@dataclass(frozen=True, slots=True)
class ArticulationSlideFactory(RegisteredSemanticLowererFactory):
    call_id: ClassVar[str] = SLIDE_CALL
    revision: ClassVar[str] = "2"
    target_descriptor = Slide.descriptor()
    bindings: tuple[PrismaticBinding, ...]

    def __post_init__(self) -> None:
        if (
            type(self.bindings) is not tuple
            or not self.bindings
            or any(type(b) is not PrismaticBinding for b in self.bindings)
        ):
            raise ValueError(
                "Articulation factories require immutable prismatic bindings."
            )
        if len({b.object_id for b in self.bindings}) != len(self.bindings):
            raise ValueError(
                "Articulation bindings must have unique object identities."
            )

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        if engine.robot is not robot:
            raise ValueError("Articulation lowerer must use the factory's robot.")
        return _SlideLowerer(
            _bind(self.bindings, simulation, scene_registry, engine), robot
        )


@dataclass(frozen=True, slots=True)
class ArticulationWithdrawFactory(ArticulationSlideFactory):
    call_id: ClassVar[str] = WITHDRAW_CALL
    target_descriptor = MoveEndEffector.descriptor()

    def create(
        self, *, simulation: Any, robot: Any, scene_registry: Any, engine: Any
    ) -> RegisteredSemanticLowerer:
        if engine.robot is not robot:
            raise ValueError("Articulation lowerer must use the factory's robot.")
        return _WithdrawLowerer(
            _bind(self.bindings, simulation, scene_registry, engine), robot
        )


def preset_id(binding: PrismaticBinding, state: str) -> str:
    binding.target(state)
    return f"gen_sim.articulation.{binding.object_id}.{state}"


class ArticulationStabilityPort:
    """Measured joint retention; not a contact safety gate or a task executor."""

    def __init__(
        self,
        delegate: Any,
        simulation: Any,
        robot: Any,
        bindings: tuple[PrismaticBinding, ...],
        step_dt: float,
    ) -> None:
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError(
                "Articulation post-policy requires a positive control period."
            )
        self.delegate, self.simulation, self.robot, self.dt = (
            delegate,
            simulation,
            robot,
            step_dt,
        )
        self.policies = {
            preset_id(b, state): (b, state)
            for b in bindings
            for state in ("open", "closed")
        }
        self.results, self.metadata = {}, {}

    def validate_policy(self, policy: Any, *, segment: Any) -> None:
        if policy.cfg.preset not in self.policies:
            return self.delegate.validate_policy(policy, segment=segment)
        binding, _ = self.policies[policy.cfg.preset]
        if (
            policy.cfg.kind != "wait_stable"
            or policy.entity.entity_id != binding.object_id
            or not any(p is policy for p in segment.post_policies)
        ):
            raise ValueError(
                "Articulation post-policy differs from its declared segment."
            )

    def actions(self, policy: Any, *, segment: Any, active_mask: torch.Tensor) -> Any:
        self.validate_policy(policy, segment=segment)
        if policy.cfg.preset not in self.policies:
            yield from self.delegate.actions(
                policy, segment=segment, active_mask=active_mask
            )
            return
        binding, state = self.policies[policy.cfg.preset]
        art = self.simulation.get_articulation(binding.object_id)
        index = art.joint_names.index(binding.joint)
        hold = self.robot.get_qpos().clone()
        counts = torch.zeros_like(active_mask, dtype=torch.long)
        anchor = art.get_qpos()[:, index].clone()
        if (
            active_mask.dtype != torch.bool
            or active_mask.shape != anchor.shape
            or active_mask.device != anchor.device
        ):
            raise ValueError(
                "Articulation post-policy rows must match the observed batch."
            )
        low, high = anchor.clone(), anchor.clone()
        accepted = torch.zeros_like(active_mask)
        required = math.ceil(1.0 / self.dt)
        timeout_ticks = math.ceil(5.0 / self.dt)
        for tick in range(timeout_ticks + 1):
            qpos = art.get_qpos()[:, index].clone()
            finite = torch.isfinite(qpos)
            good = finite & (
                (qpos - binding.target(state)).abs() <= binding.tolerance(state)
            )
            low, high = torch.minimum(low, qpos), torch.maximum(high, qpos)
            reset = ~good | ((high - low) > 0.001)
            low, high = torch.where(reset, qpos, low), torch.where(reset, qpos, high)
            counts = torch.where(reset, 0, counts + int(tick > 0))
            accepted = good & (counts >= required) & active_mask
            if (
                (accepted | ~active_mask).all()
                or not finite[active_mask].all()
                or tick == timeout_ticks
            ):
                break
            yield hold.clone()
        self.results[id(policy)] = accepted
        self.metadata[id(policy)] = {
            "joint": binding.joint,
            "target": binding.target(state),
            "observed": [float(v) if math.isfinite(float(v)) else None for v in qpos],
            "tolerance": binding.tolerance(state),
            "stable_seconds": 1.0,
            "accepted": accepted.tolist(),
            "evidence": "joint_position_and_stability",
            "contact_qualification": "not_provided",
        }

    def post_policy_result(self, policy: Any, *, segment: Any) -> Any:
        if policy.cfg.preset not in self.policies:
            return self.delegate.post_policy_result(policy, segment=segment)
        return self.results[id(policy)].clone()

    def post_policy_metadata(self, policy: Any, *, segment: Any) -> Any:
        if policy.cfg.preset not in self.policies:
            return self.delegate.post_policy_metadata(policy, segment=segment)
        return deepcopy(self.metadata[id(policy)])
