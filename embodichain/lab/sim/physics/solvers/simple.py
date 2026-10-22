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
"""Small spring-slider solver demonstrating the DexSim extension contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

import newton
import numpy as np
import torch
import warp as wp
from newton.solvers import SolverBase

try:
    from dexsim.engine.newton_physics import (
        FunctionalDynamics,
        FunctionalSolverState,
        NewtonSolverCapabilities,
        NewtonSolverContext,
        NewtonSolverExtension,
    )
except ImportError as exc:
    raise ImportError(
        "The simple solver requires DexSim's FunctionalDynamics extension API "
        "(MR !1483 service revision); install a compatible DexSim wheel."
    ) from exc

from embodichain.utils.math import quat_apply

__all__ = ["SimpleSolverProvider"]


def _torch_scope(device: wp.context.Device) -> Any:
    return (
        torch.cuda.stream(wp.stream_to_torch(wp.get_stream(device)))
        if device.is_cuda
        else nullcontext()
    )


def _tensor(array: wp.array) -> torch.Tensor:
    # Warp cannot expose a null pointer as a Torch tensor for empty arrays.
    if array.size == 0:
        return torch.as_tensor(array.numpy(), device=str(array.device))
    return wp.to_torch(array)


class _SliderDynamics(FunctionalDynamics):
    def __init__(self, model: newton.Model, options: Mapping[str, Any]) -> None:
        unknown = set(options) - {"stiffness", "damping", "force_scale"}
        if unknown:
            raise ValueError(f"Unknown simple solver options: {sorted(unknown)}")
        values = {"stiffness": 1.0, "damping": 0.1, "force_scale": 1.0}
        values.update(options)
        for name, value in values.items():
            if not math.isfinite(float(value)) or (
                name != "force_scale" and float(value) < 0
            ):
                raise ValueError(f"Invalid simple solver option {name}={value!r}")
        self.model = model
        self.stiffness = float(values["stiffness"])
        self.damping = float(values["damping"])
        self.force_scale = torch.tensor(
            float(values["force_scale"]), device=str(model.device), requires_grad=True
        )
        self.refresh()

    def refresh(self) -> None:
        model = self.model
        types = model.joint_type.numpy()
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        moving = np.flatnonzero(types == int(newton.JointType.PRISMATIC))
        if (
            not len(moving)
            or not np.isin(
                types, [int(newton.JointType.FIXED), int(newton.JointType.PRISMATIC)]
            ).all()
            or len(children) != model.body_count
            or len(set(children.tolist())) != model.body_count
            or any(int(children[j]) in parents for j in moving)
            or not model.joint_enabled.numpy().all()
        ):
            raise ValueError(
                "simple supports independent prismatic leaf joints with fixed ancestors only"
            )
        if model.particle_count or getattr(model, "constraint_mimic_count", 0):
            raise ValueError("simple does not support particles or mimic constraints")
        moving_shapes = np.isin(model.shape_body.numpy(), children[moving])
        if np.any(
            model.shape_flags.numpy()[moving_shapes]
            & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ):
            raise ValueError(
                "simple does not support collision shapes on moving bodies"
            )
        if np.any(model.gravity.numpy()):
            raise ValueError("simple requires zero gravity")
        for name in ("joint_target_ke", "joint_target_kd", "joint_friction"):
            if np.any(getattr(model, name).numpy()):
                raise ValueError(f"simple does not support nonzero {name}")
        self.children = torch.as_tensor(
            children[moving], dtype=torch.long, device=str(model.device)
        )
        if np.any(
            model.body_flags.numpy()[children[moving]] & int(newton.BodyFlags.KINEMATIC)
        ):
            raise ValueError("simple requires dynamic slider bodies")
        mass = _tensor(model.body_mass)[self.children].detach().clone()
        if not bool(torch.isfinite(mass).all() and (mass > 0).all()):
            raise ValueError("simple requires finite positive slider mass")
        self.mass = mass + _tensor(model.joint_armature).detach().clone()
        if not bool(torch.isfinite(self.mass).all() and (self.mass > 0).all()):
            raise ValueError("simple requires finite positive effective mass")
        self.drag = _tensor(model.joint_damping).detach().clone() + self.damping
        self.lower = _tensor(model.joint_limit_lower).detach().clone()
        self.upper = _tensor(model.joint_limit_upper).detach().clone()
        self.effort_limit = _tensor(model.joint_effort_limit).detach().clone()
        self.velocity_limit = _tensor(model.joint_velocity_limit).detach().clone()
        child_frames = _tensor(model.joint_X_c)[
            torch.as_tensor(moving, device=str(model.device))
        ]
        self.local_axes = quat_apply(
            child_frames[:, 3:], _tensor(model.joint_axis)
        ).detach()

    def create_state(self, native_state: newton.State) -> FunctionalSolverState:
        return FunctionalSolverState(
            {
                name: _tensor(getattr(native_state, name)).detach().clone()
                for name in ("joint_q", "joint_qd", "body_q", "body_qd", "body_qdd")
                if getattr(native_state, name, None) is not None
            }
        )

    def parameters(self) -> Mapping[str, torch.Tensor]:
        return {"force_scale": self.force_scale}

    def step(
        self,
        state: FunctionalSolverState,
        control: torch.Tensor,
        parameters: Mapping[str, torch.Tensor],
        dt: float,
    ) -> FunctionalSolverState:
        q, qd = state.physical["joint_q"], state.physical["joint_qd"]
        if not isinstance(control, torch.Tensor) or control.shape != q.shape:
            raise ValueError(
                "simple control must be a tensor shaped (joint_dof_count,)"
            )
        if control.device != q.device or control.dtype != q.dtype:
            raise ValueError("simple control must match the state device and dtype")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("simple dt must be positive and finite")
        force = (parameters["force_scale"] * control).clamp(
            -self.effort_limit, self.effort_limit
        )
        acceleration = (force - self.stiffness * q - self.drag * qd) / self.mass
        velocity = (qd + dt * acceleration).clamp(
            -self.velocity_limit, self.velocity_limit
        )
        proposed = q + dt * velocity
        position = proposed.clamp(self.lower, self.upper)
        velocity = torch.where(
            proposed != position, torch.zeros_like(velocity), velocity
        )
        acceleration = (velocity - qd) / dt
        # Advance the existing pose by the coordinate delta so explicit root
        # placement and randomized states are preserved across transitions.
        previous_pose = state.physical["body_q"]
        axes = quat_apply(previous_pose[self.children, 3:], self.local_axes)
        pose = previous_pose.index_copy(
            0,
            self.children,
            torch.cat(
                (
                    previous_pose[self.children, :3] + axes * (position - q)[:, None],
                    previous_pose[self.children, 3:],
                ),
                dim=1,
            ),
        )
        spatial = torch.zeros(
            (self.model.body_count, 6), device=q.device, dtype=q.dtype
        )
        zeros = torch.zeros_like(axes)
        body_qd = spatial.index_copy(
            0, self.children, torch.cat((axes * velocity[:, None], zeros), dim=1)
        )
        body_qdd = spatial.index_copy(
            0,
            self.children,
            torch.cat((axes * acceleration[:, None], zeros), dim=1),
        )
        return FunctionalSolverState(
            {
                "joint_q": position,
                "joint_qd": velocity,
                "body_q": pose,
                "body_qd": body_qd,
                "body_qdd": body_qdd,
            }
        )

    def commit(self, state: FunctionalSolverState) -> None:
        # There is no recurrent history. DexSim owns physical publication.
        pass

    def reset_history(
        self,
        state: FunctionalSolverState,
        reset_state: FunctionalSolverState,
        world_mask: wp.array | None,
    ) -> tuple[Mapping[str, Any], None]:
        return {}, None


class _NativeSliderSolver(SolverBase):
    def __init__(self, model: newton.Model, dynamics: _SliderDynamics) -> None:
        super().__init__(model)
        self.dynamics = dynamics

    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
        *,
        observables: SolverBase.Observables | None = None,
    ) -> None:
        self.validate_observables(observables, contacts)
        with _torch_scope(self.model.device), torch.no_grad():
            source = self.dynamics.create_state(state_in)
            force = _tensor(self.model.joint_f if control is None else control.joint_f)
            result = self.dynamics.step(source, force, self.dynamics.parameters(), dt)
            for name, value in result.physical.items():
                destination = getattr(state_out, name, None)
                if destination is not None:
                    _tensor(destination).copy_(value)

    def notify_model_changed(self, flags: newton.ModelFlags | int) -> None:
        with _torch_scope(self.model.device):
            self.dynamics.refresh()


class SimpleSolverProvider:
    """Create a minimal, differentiable spring-slider Newton extension.

    Each moving body must be a dynamic prismatic leaf attached to the world or
    fixed ancestors. Visual shapes and static ground are allowed; moving-body
    collisions, particles, mimic
    constraints, gravity, target drives and friction are rejected. Joint mass,
    armature, damping and effort/velocity/position limits are respected. Limits
    use hard clamping; gradients at their boundaries are piecewise-defined.

    Options are ``stiffness`` (N/m, default 1), ``damping`` (N s/m, default 0.1)
    and trainable ``force_scale`` (default 1). Native and functional execution
    share semi-implicit Euler: first update velocity, then position. This is an
    eager educational solver, not a general robot/contact solver. Parameters
    belong to one finalized model and must be checkpointed across rebuilds.
    """

    capabilities = NewtonSolverCapabilities(
        reset=True,
        cuda_graph=False,
        external_wrenches=False,
        supported_control_fields=("joint_f",),
        runtime_controls=False,
        native_body_accelerations=True,
    )

    def register_custom_attributes(
        self,
        builder: newton.ModelBuilder,
        options: Mapping[str, Any],
    ) -> None:
        """Use standard Newton attributes; no custom registration is needed.

        Args:
            builder: Builder owned by DexSim before model finalization.
            options: Solver options validated during creation.
        """

    def create(
        self,
        context: NewtonSolverContext,
        options: Mapping[str, Any],
    ) -> NewtonSolverExtension:
        """Validate the model and create native and functional views of one solver.

        Args:
            context: Finalized model and resources borrowed from DexSim.
            options: Spring stiffness, damping and initial force scale.

        Returns:
            Model-scoped extension with an independent functional trajectory API.

        Raises:
            ValueError: The model or options exceed the documented supported scope.
        """
        with _torch_scope(context.device):
            dynamics = _SliderDynamics(context.model, options)
        return NewtonSolverExtension(
            native_solver=_NativeSliderSolver(context.model, dynamics),
            capabilities=self.capabilities,
            functional=dynamics,
        )
