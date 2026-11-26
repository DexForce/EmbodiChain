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

"""Bounded host-side target commands for an exclusively controlled World."""

from __future__ import annotations

from copy import deepcopy
import math
from secrets import token_hex
from typing import Any, TYPE_CHECKING

import torch

from .attachment import AttachedSimulationManager
from .robot_state import read_robot_state

if TYPE_CHECKING:
    from dexsim.runtime import SessionContext

__all__ = ["WorldControl"]


class WorldControl:
    """Execute one control interval at a time on the host's owner thread.

    Install only on a dedicated manual host with an exclusive writer ingress.
    Each receipt supplies the revision needed to admit a subsequent command.
    A failed native write/step may have partial effects and fences this host.
    """

    def __init__(
        self,
        context: SessionContext,
        manager: AttachedSimulationManager,
        *,
        physics_dt: float,
        steps_per_control: int,
    ) -> None:
        manager.validate()
        if not context.world.is_physics_manually_update():
            raise ValueError("Control requires a manually stepped World")
        if context is not manager._context:
            raise ValueError("Controller must own the bound session")
        if (
            not math.isfinite(physics_dt)
            or physics_dt <= 0
            or type(steps_per_control) is not int
            or not 1 <= steps_per_control <= 1024
        ):
            raise ValueError(
                "Expected positive dt and 1..1024 physics intervals per command"
            )
        if context.scene.backend == "newton":
            diagnostics = context.scene.get_solver_diagnostics() or {}
            if diagnostics.get("solver_type") != "mujoco_warp" or not math.isclose(
                physics_dt, diagnostics.get("dt", 0), rel_tol=1e-6
            ):
                raise ValueError(
                    "Control requires a matching Newton MuJoCo-Warp timestep"
                )
        self.manager = manager
        self.context = context
        self.physics_dt = physics_dt
        self.steps_per_control = steps_per_control
        self.control_step = self.physics_steps = 0
        self._revision = token_hex(16)
        self._receipts: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self._state = "ready"

    def status(self) -> dict[str, Any]:
        """Describe the binding, ledger revision, and control-boundary counters."""
        try:
            self.manager.validate()
            binding_current = True
        except RuntimeError:
            # Preserve access to past receipts even if the host replaced its Scene.
            binding_current = False
        return dict(
            binding_current=binding_current,
            session_id=self.manager.session_id,
            topology_revision=self.manager.topology_revision,
            revision=self._revision if not self._receipts else None,
            state=self._state,
            physics_dt=self.physics_dt,
            steps_per_control=self.steps_per_control,
            physics_steps=self.physics_steps,
            control_step=self.control_step,
        )

    def receipt(self, command_id: str) -> dict[str, Any]:
        """Recover the exact previous result without stepping."""
        return deepcopy(self._receipts[command_id][1])

    def stop(self) -> dict[str, Any]:
        """Stop dispatch at the owner-thread boundary after any current interval."""
        if self._state != "effect_unknown":
            self._state = "stopped"
        return self.status()

    def target_and_step(
        self,
        *,
        command_id: str,
        session_id: str,
        topology_revision: int,
        revision: str,
        robot_uid: str,
        control_part: str,
        target: list[float],
        units: list[str],
        control_interval_s: float,
        tolerance: float = 1e-3,
    ) -> dict[str, Any]:
        """Set measured-group position targets and advance exactly one interval."""
        request = dict(
            session_id=session_id,
            topology_revision=topology_revision,
            revision=revision,
            robot_uid=robot_uid,
            control_part=control_part,
            target=target,
            units=units,
            control_interval_s=control_interval_s,
            tolerance=tolerance,
        )
        if not isinstance(command_id, str) or not command_id or len(command_id) > 128:
            raise ValueError("Expected a nonempty command_id of at most 128 characters")
        if command_id in self._receipts:
            if request != self._receipts[command_id][0]:
                raise ValueError("Command ID already used with different arguments")
            return self.receipt(command_id)
        if len(self._receipts) >= 4096:
            raise RuntimeError("This dedicated session has reached its command limit")
        state = self.status()
        if (
            not state["binding_current"]
            or self._state != "ready"
            or (session_id, topology_revision, revision)
            != (
                state["session_id"],
                state["topology_revision"],
                self._revision,
            )
        ):
            raise ValueError(
                "Control is stopped, uncertain or the supplied binding/revision is stale"
            )
        if not self.context.world.is_physics_manually_update() or not math.isclose(
            control_interval_s, self.physics_dt * self.steps_per_control, rel_tol=1e-9
        ):
            raise ValueError("Expected the dedicated host's manual control interval")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Expected a finite nonnegative attainment tolerance")
        robot = self.manager.get_robot(robot_uid)
        path = self.manager.object_path(robot_uid)

        def observe():
            return read_robot_state(
                self.context,
                object_ref=path,
                robot_cfg=robot.cfg,
                control_part=control_part,
            )

        before = observe()
        if (
            units != before["units"]
            or len(target) != len(units)
            or not all(type(v) in (int, float) and math.isfinite(v) for v in target)
        ):
            raise ValueError(
                "Target must match the ordered scalar joint units and contain finite values"
            )
        if set(before["dof_ids"]).intersection(robot.mimic_ids):
            raise ValueError("A mimic follower cannot receive an independent target")
        value = torch.tensor([target], dtype=torch.float32, device=robot.device)
        limits = robot.get_qpos_limits(name=control_part)
        if not bool(
            (
                (value >= limits[..., 0])
                & (value <= limits[..., 1])
                & torch.isfinite(value)
            ).all()
        ):
            raise ValueError("Target exceeds a live joint limit")
        self._state = "effect_unknown"
        entry = dict(
            command_id=command_id,
            target=list(target),
            target_accepted=None,
            target_attained=None,
            measured_before=before,
            measured_after=None,
            **self.status(),
        )
        self._receipts[command_id] = (deepcopy(request), entry)
        try:
            robot.set_qpos(
                value, name=control_part, target=True, clamp_target_qpos=False
            )
            accepted = robot.get_qpos(name=control_part, target=True)
            entry["target_accepted"] = bool(
                torch.allclose(accepted, value, rtol=0, atol=1e-6)
            )
            if not entry["target_accepted"]:
                raise RuntimeError(
                    "Native target readback differs from the requested target"
                )
            for _ in range(self.steps_per_control):
                self.context.world.update(self.physics_dt)
                self.physics_steps += 1
            self.control_step += 1
            after = observe()
            if before["step_index"] is not None and (
                after["step_index"] != before["step_index"] + self.steps_per_control
                or not math.isclose(
                    after["simulation_time_s"] - before["simulation_time_s"],
                    control_interval_s,
                    abs_tol=1e-7,
                )
            ):
                raise RuntimeError(
                    "Host clock disagrees with the commanded control interval"
                )
            entry["measured_after"] = after
            entry["target_attained"] = all(
                abs(a - b) <= tolerance
                for a, b in zip(after["qpos"], target, strict=True)
            )
            self._state = "ready"
        except Exception as exc:
            entry["error"] = str(exc)
        self._revision = token_hex(16)
        entry.update(self.status())
        entry["revision"] = self._revision
        entry["clock_source"] = (
            "native_scene"
            if before["step_index"] is not None
            else "exclusive_host_update_counter"
        )
        return self.receipt(command_id)
