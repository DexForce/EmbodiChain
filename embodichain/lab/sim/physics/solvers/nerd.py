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

"""Named NeRD solver using a shared native/functional state transition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import math

import newton
from newton.solvers import SolverBase
import numpy as np
import torch
import warp as wp

from dexsim.engine.newton_physics import (
    FunctionalSolverState,
    NewtonSolverCapabilities,
    NewtonSolverContext,
    NewtonSolverExtension,
)
from embodichain.compute.dynamics import NeRDStateCodec, append_history, reset_history
from embodichain.learning.models.nerd import load_model

from ._kinematics import differentiable_fk, _stream_scope
from ._nerd_binding import NeRDBinding, _PHYSICS_SIGNATURE_SCHEMA

__all__ = ["NeRDProvider", "NeuralRobotDynamicSolver"]

_PHYSICAL_FIELDS = (
    "joint_q",
    "joint_qd",
    "body_q",
    "body_qd",
    "particle_q",
    "particle_qd",
)


class NeuralRobotDynamicSolver(SolverBase):
    """Advance homogeneous Cartpole/Ant worlds with a portable NeRD export.

    Args:
        context: Model-owned construction resources supplied by DexSim.
        checkpoint: Export directory containing weights and deployment manifest.
        trainable: Whether the resident network parameters require gradients.
            Functional callers may alternatively supply explicit parameter tensors.
    """

    def __init__(
        self, context: NewtonSolverContext, checkpoint: str, *, trainable: bool = False
    ) -> None:
        super().__init__(context.model)
        self.network, self.manifest = load_model(
            checkpoint, device=wp.device_to_torch(context.device)
        )
        if any(
            value.is_floating_point() and value.dtype != torch.float32
            for value in self.network.state_dict().values()
        ):
            raise TypeError(
                "NeRD Newton execution requires float32 model weights and buffers."
            )
        self.network.requires_grad_(trainable)
        codec_cfg = self.manifest.get("codec")
        if not isinstance(codec_cfg, dict):
            raise ValueError("NeRD export requires an explicit codec manifest.")
        self.codec = NeRDStateCodec(**codec_cfg)
        self.binding = NeRDBinding(context.model, self.codec.robot)
        self._dt = float(self.manifest.get("dt", 0.0))
        self._check_dt(context.dt)
        if self.manifest.get("joint_names") != self.binding.joint_names:
            raise ValueError("NeRD exported joint order does not match the scene.")
        if self.manifest.get("physics_signature_schema") != _PHYSICS_SIGNATURE_SCHEMA:
            raise ValueError(
                "Unsupported NeRD physics signature schema; regenerate data and exports "
                "with the current physical contract instead of relabeling an old model."
            )
        if self.manifest.get("physics_signature") != self.binding.physics_signature:
            raise ValueError(
                "NeRD exported physics signature does not match the scene."
            )
        if not np.allclose(self.codec.base_com, self.binding.base_com, atol=1e-7):
            raise ValueError("NeRD codec COM does not match the root body.")
        if self.manifest.get("contact_schema") != "nerd-static-plane-v1":
            raise ValueError("Unsupported NeRD contact feature schema.")
        expected_gravity = self.manifest.get("gravity")
        if expected_gravity is None or not torch.allclose(
            self.binding.gravity,
            self.binding.gravity.new_tensor(expected_gravity).expand_as(
                self.binding.gravity
            ),
        ):
            raise ValueError("NeRD scene gravity differs from the training manifest.")
        for name in ("joint_target_ke", "joint_target_kd"):
            values = getattr(self.model, name, None)
            if values is not None and np.any(values.numpy() != 0):
                raise ValueError(
                    "NeRD effort deployment requires zero native position/velocity drive gains."
                )
        self.history_length = self.network.cfg.history_length
        self._history = {
            name: torch.zeros(
                (self.binding.num_envs, self.history_length, width),
                device=self.binding.device,
            )
            for name, width in self.network.cfg.input_dims.items()
        }
        self._lengths = torch.zeros(
            self.binding.num_envs, dtype=torch.long, device=self.binding.device
        )
        self._closed = False
        self._invalid = False
        initial = self.initial_functional_state(self.model.state())
        q, qd = self.binding.read(initial.physical)
        features = self.codec.encode(
            q,
            qd,
            torch.zeros_like(qd),
            self.binding.contacts(initial.physical["body_q"]),
            self.binding.gravity_dir,
        )
        if {k: v.shape[-1] for k, v in features.items()} != dict(
            self.network.cfg.input_dims
        ):
            raise ValueError(
                "NeRD network inputs do not match the scene feature schema."
            )

    def _check_dt(self, dt: float) -> None:
        if self._dt <= 0 or not math.isclose(
            float(dt), self._dt, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"NeRD timestep must equal trained dt={self._dt}, got {dt}."
            )

    def initial_functional_state(self, native_state: Any) -> FunctionalSolverState:
        """Clone current native state and resident history into an owned snapshot."""
        physical = {
            name: wp.to_torch(getattr(native_state, name)).detach().clone()
            for name in _PHYSICAL_FIELDS
            if getattr(native_state, name, None) is not None
        }
        return FunctionalSolverState(
            physical=physical,
            history={k: v.detach().clone() for k, v in self._history.items()},
            valid_history_lengths=self._lengths.detach().clone(),
        )

    def functional_step(
        self,
        state: FunctionalSolverState,
        control: Mapping[str, torch.Tensor],
        parameters: Mapping[str, torch.Tensor] | None,
        dt: float,
    ) -> FunctionalSolverState:
        """Return a new state while preserving all caller-owned graphs."""
        if self._closed:
            raise RuntimeError("NeRD solver is closed.")
        if self._invalid:
            raise RuntimeError(
                "NeRD physical model was invalidated; rebuild with a matching export."
            )
        self._check_dt(dt)
        if set(control) != {"joint_f"}:
            raise ValueError("NeRD requires explicit joint_f effort controls only.")
        force = control["joint_f"]
        if (
            force.shape != (self.model.joint_dof_count,)
            or force.dtype != torch.float32
            or force.device != self.binding.device
        ):
            raise ValueError(
                "joint_f must use flattened float32 model-order DOFs on the model device."
            )
        if self.codec.robot == "ant" and bool(
            torch.any(force[self.binding.qd_indices[:, :6]] != 0)
        ):
            raise ValueError(
                "NeRD Ant cannot model external generalized forces on its free base."
            )
        if state.valid_history_lengths is None:
            raise ValueError(
                "NeRD functional state requires per-world history lengths."
            )
        q, qd = self.binding.read(state.physical)
        # q/qd are the independent physical coordinates. Recompute Ant contact
        # geometry so the first step also differentiates through an initial q
        # leaf supplied by the caller, not an unrelated detached body snapshot.
        body_q = state.physical["body_q"]
        if self.codec.robot == "ant":
            body_q, _ = differentiable_fk(
                self.model, state.physical["joint_q"], state.physical["joint_qd"]
            )
        contacts = self.binding.contacts(body_q)
        features = self.codec.encode(
            q, qd, force[self.binding.qd_indices], contacts, self.binding.gravity_dir
        )
        history, lengths = append_history(
            state.history, state.valid_history_lengths, features
        )
        if parameters is None:
            prediction = self.network.evaluate(history, lengths).squeeze(1)
        else:
            output = torch.func.functional_call(
                self.network, parameters, (history,), strict=False
            )
            prediction = output.gather(
                1, (lengths - 1)[:, None, None].expand(-1, 1, output.shape[-1])
            ).squeeze(1)
        next_q, next_qd = self.codec.decode(
            q, qd, prediction, current_embedding=features["states_embedding"]
        )
        physical = dict(state.physical)
        physical["joint_q"] = state.physical["joint_q"].scatter(
            0, self.binding.q_indices.flatten(), next_q.flatten()
        )
        physical["joint_qd"] = state.physical["joint_qd"].scatter(
            0, self.binding.qd_indices.flatten(), next_qd.flatten()
        )
        physical["body_q"], physical["body_qd"] = differentiable_fk(
            self.model, physical["joint_q"], physical["joint_qd"]
        )
        return FunctionalSolverState(physical, history, lengths)

    def commit_functional_state(self, state: FunctionalSolverState) -> None:
        """Adopt detached history after an explicitly published functional step."""
        self._history = {k: v.detach().clone() for k, v in state.history.items()}
        self._lengths = state.valid_history_lengths.detach().clone()

    def reset_functional_history(
        self,
        state: FunctionalSolverState,
        reset_state: FunctionalSolverState,
        world_mask: Any | None,
    ):
        """Clear selected histories without detaching unselected trajectories."""
        del reset_state
        mask = self._selected_rows(world_mask)
        return reset_history(state.history, state.valid_history_lengths, mask)

    def _selected_rows(self, world_mask: Any | None) -> torch.Tensor:
        if world_mask is None:
            return torch.ones(
                self.binding.num_envs, dtype=torch.bool, device=self.binding.device
            )
        mask = (
            wp.to_torch(world_mask) if isinstance(world_mask, wp.array) else world_mask
        )
        return mask[self.binding.world_ids]

    def step(
        self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float
    ) -> None:
        """Advance the native double-buffer path with the same neural transition."""
        del contacts
        with torch.no_grad(), _stream_scope(self.binding.device):
            physical = {
                name: wp.to_torch(getattr(state_in, name))
                for name in _PHYSICAL_FIELDS
                if getattr(state_in, name, None) is not None
            }
            current = FunctionalSolverState(physical, self._history, self._lengths)
            result = self.functional_step(
                current, {"joint_f": wp.to_torch(control.joint_f)}, None, dt
            )
            for name, value in result.physical.items():
                wp.copy(
                    getattr(state_out, name),
                    wp.from_torch(
                        value.contiguous(), dtype=getattr(state_out, name).dtype
                    ),
                )
            self._history, self._lengths = (
                dict(result.history),
                result.valid_history_lengths,
            )

    def reset(
        self, state: Any, world_mask: Any | None = None, flags: Any | None = None
    ) -> None:
        """Clear selected resident history and synchronize FK after state reset."""
        del flags
        with torch.no_grad(), _stream_scope(self.binding.device):
            self._history, self._lengths = reset_history(
                self._history, self._lengths, self._selected_rows(world_mask)
            )
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)

    def notify_model_changed(self, flags: Any) -> None:
        """Reject changes invalidating a trained robot's physical contract."""
        del flags
        try:
            binding = NeRDBinding(self.model, self.codec.robot)
            if (
                binding.physics_signature != self.binding.physics_signature
                or not torch.equal(binding.gravity, self.binding.gravity)
            ):
                raise ValueError(
                    "Physical model changed; retrain or load a matching NeRD export."
                )
        except Exception:
            # Native mutation precedes Newton's notification. A caught error
            # must not permit later stepping with incompatible learned physics.
            self._invalid = True
            raise

    def close(self) -> None:
        """Release resident history after the owner stops stepping this solver."""
        self._closed = True
        self._history = {}


class NeRDProvider:
    """Entry-point provider for the registered ``nerd`` Newton solver."""

    capabilities = NewtonSolverCapabilities(
        differentiable=True,
        reset=True,
        contact_reports=False,
        reduced_coordinates=True,
        functional_contacts=False,
        external_wrenches=False,
        supported_control_fields=("joint_f",),
        runtime_controls=False,
    )

    def register_custom_attributes(
        self, builder: Any, options: Mapping[str, Any]
    ) -> None:
        """NeRD uses standard model state and requires no custom builder attributes."""
        del builder
        if set(options) - {"checkpoint", "trainable"}:
            raise ValueError("NeRD options support checkpoint and trainable only.")
        if not isinstance(options.get("checkpoint"), str) or not options["checkpoint"]:
            raise ValueError("NeRD requires an exported checkpoint directory.")
        if not isinstance(options.get("trainable", False), bool):
            raise TypeError("NeRD trainable must be boolean.")

    def create(
        self, context: NewtonSolverContext, options: Mapping[str, Any]
    ) -> NewtonSolverExtension:
        """Build a native and functional adapter bound to the finalized scene."""
        solver = NeuralRobotDynamicSolver(context, **options)
        return NewtonSolverExtension(
            native_solver=solver,
            capabilities=self.capabilities,
            functional_step=solver.functional_step,
            initial_functional_state=solver.initial_functional_state,
            named_parameters=lambda: dict(solver.network.named_parameters()),
            close=solver.close,
            commit_functional_state=solver.commit_functional_state,
            reset_functional_history=solver.reset_functional_history,
        )
