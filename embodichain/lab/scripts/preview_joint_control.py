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

"""Simulation-thread articulation controls used by asset preview frontends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from embodichain.lab.visualization import (
    JointControlCommand,
    JointControlSpec,
    JointControlState,
)
from embodichain.lab.visualization._utils import to_numpy_array
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Articulation
    from embodichain.lab.visualization import VisualizationRuntime

__all__ = ["ArticulationPreviewController"]


_ROTATIONAL_STEP = np.deg2rad(1.0)
_PRISMATIC_STEP = 0.001
_SUPPORTED_JOINT_TYPES = {"revolute", "continuous", "prismatic"}


@dataclass
class _JointBinding:
    articulation: Articulation
    spec: JointControlSpec
    desired_value: float
    applied_sequence: int = 0


def _joint_type_name(value: object) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name).lower()
    return str(value).rsplit(".", maxsplit=1)[-1].lower()


class ArticulationPreviewController:
    """Expose scalar articulation joints and hold commanded preview poses.

    Viser callbacks enqueue immutable commands. :meth:`update` drains and
    validates those commands, then applies all desired joint values on the
    simulation thread. The controller intentionally writes both current and
    target state so preview behavior does not depend on an asset's drive type.

    Args:
        articulations: Articulations loaded by ``preview-asset``.
        runtime: Active Viser runtime receiving the exported controls.
    """

    def __init__(
        self,
        articulations: Sequence[Articulation],
        runtime: VisualizationRuntime,
    ) -> None:
        self._runtime = runtime
        self._bindings: dict[str, _JointBinding] = {}
        self._articulation_bindings: list[
            tuple[Articulation, tuple[_JointBinding, ...]]
        ] = []
        for articulation_index, articulation in enumerate(articulations):
            bindings = self._build_articulation_bindings(
                articulation,
                articulation_index,
            )
            if bindings:
                self._articulation_bindings.append((articulation, bindings))
                self._bindings.update(
                    (binding.spec.control_id, binding) for binding in bindings
                )

    @property
    def has_controls(self) -> bool:
        """Return whether at least one supported independent joint was found."""
        return bool(self._bindings)

    @staticmethod
    def _finite_limits(
        lower: float,
        upper: float,
        *,
        uid: str,
        joint_name: str,
        joint_type: str,
    ) -> tuple[float | None, float | None] | None:
        if np.isnan(lower) or np.isnan(upper):
            logger.log_warning(
                f"Skipping joint {uid!r}/{joint_name!r}: invalid position "
                f"limits [{lower}, {upper}]."
            )
            return None
        if joint_type == "continuous":
            return None, None
        if lower > upper:
            logger.log_warning(
                f"Skipping joint {uid!r}/{joint_name!r}: invalid position "
                f"limits [{lower}, {upper}]."
            )
            return None
        finite_lower = lower if np.isfinite(lower) else None
        finite_upper = upper if np.isfinite(upper) else None
        if finite_lower is not None and finite_upper is not None:
            if finite_lower == finite_upper:
                return None
        return finite_lower, finite_upper

    def _build_articulation_bindings(
        self,
        articulation: Articulation,
        articulation_index: int,
    ) -> tuple[_JointBinding, ...]:
        uid = str(
            getattr(getattr(articulation, "cfg", None), "uid", None)
            or f"articulation_{articulation_index}"
        )
        dof = int(articulation.dof)
        joint_names = tuple(articulation.joint_names)
        if len(joint_names) != dof:
            logger.log_warning(
                f"Skipping articulation {uid!r} joint controls: its {dof} DOFs "
                f"do not map one-to-one to {len(joint_names)} active joint names."
            )
            return ()

        qpos = to_numpy_array(articulation.get_qpos(), np.float32, copy=False)
        limits = to_numpy_array(
            articulation.get_qpos_limits(env_ids=[0]),
            np.float32,
            copy=False,
        )
        if qpos.shape[0] < 1 or qpos.shape[1:] != (dof,):
            raise ValueError(
                f"Articulation {uid!r} qpos must have shape (N, {dof}), "
                f"received {qpos.shape}."
            )
        if limits.shape != (1, dof, 2):
            raise ValueError(
                f"Articulation {uid!r} limits must have shape (1, {dof}, 2), "
                f"received {limits.shape}."
            )

        entity = articulation._entities[0]
        active_joint_ids = tuple(
            int(joint_id)
            for joint_id in getattr(
                articulation,
                "active_joint_ids",
                range(dof),
            )
        )
        bindings: list[_JointBinding] = []
        for joint_id in active_joint_ids:
            joint_name = joint_names[joint_id]
            joint_info = entity.get_joint_info(joint_name)
            joint_type = _joint_type_name(joint_info.joint_type)
            if joint_type not in _SUPPORTED_JOINT_TYPES:
                logger.log_warning(
                    f"Skipping joint {uid!r}/{joint_name!r}: joint type "
                    f"{joint_type!r} is not a supported scalar preview control."
                )
                continue

            initial_value = float(qpos[0, joint_id])
            if not np.isfinite(initial_value):
                logger.log_warning(
                    f"Skipping joint {uid!r}/{joint_name!r}: its initial qpos "
                    "is not finite."
                )
                continue
            limit_pair = self._finite_limits(
                float(limits[0, joint_id, 0]),
                float(limits[0, joint_id, 1]),
                uid=uid,
                joint_name=joint_name,
                joint_type=joint_type,
            )
            if limit_pair is None:
                continue
            lower, upper = limit_pair
            if lower is not None:
                initial_value = max(initial_value, lower)
            if upper is not None:
                initial_value = min(initial_value, upper)
            if lower is not None and upper is not None:
                span_step = (upper - lower) / 100.0
            else:
                span_step = float("inf")
            default_step = (
                _PRISMATIC_STEP if joint_type == "prismatic" else _ROTATIONAL_STEP
            )
            step = min(default_step, span_step)
            spec = JointControlSpec(
                control_id=f"articulation:{uid}/env:0/joint:{joint_id}",
                articulation_uid=uid,
                env_id=0,
                joint_id=joint_id,
                joint_name=joint_name,
                joint_type=joint_type,
                lower=lower,
                upper=upper,
                step=step,
                initial_value=initial_value,
            )
            bindings.append(
                _JointBinding(
                    articulation=articulation,
                    spec=spec,
                    desired_value=initial_value,
                )
            )
        return tuple(bindings)

    def joint_control_specs(self) -> tuple[JointControlSpec, ...]:
        """Return the static Viser controls in stable asset/joint order."""
        return tuple(binding.spec for binding in self._bindings.values())

    def joint_control_states(self) -> tuple[JointControlState, ...]:
        """Read authoritative qpos values on the simulation thread."""
        states: list[JointControlState] = []
        qpos_by_articulation: dict[int, np.ndarray] = {}
        for binding in self._bindings.values():
            key = id(binding.articulation)
            if key not in qpos_by_articulation:
                qpos_by_articulation[key] = to_numpy_array(
                    binding.articulation.get_qpos(),
                    np.float32,
                    copy=False,
                )
            value = float(qpos_by_articulation[key][0, binding.spec.joint_id])
            states.append(
                JointControlState(
                    control_id=binding.spec.control_id,
                    value=value,
                    applied_sequence=binding.applied_sequence,
                )
            )
        return tuple(states)

    @staticmethod
    def _clamp_value(spec: JointControlSpec, value: float) -> float:
        if spec.lower is not None:
            value = max(value, spec.lower)
        if spec.upper is not None:
            value = min(value, spec.upper)
        return value

    def _apply_commands(self, commands: Sequence[JointControlCommand]) -> int:
        accepted = 0
        exporter = self._runtime.exporter
        for command in commands:
            if (
                command.run_id != exporter.run_id
                or command.scene_revision != exporter.scene_revision
            ):
                continue
            binding = self._bindings.get(command.control_id)
            if binding is None:
                continue
            binding.desired_value = self._clamp_value(
                binding.spec,
                command.value,
            )
            binding.applied_sequence = command.sequence
            accepted += 1
        return accepted

    def _hold_desired_positions(self) -> None:
        for articulation, bindings in self._articulation_bindings:
            joint_ids = [binding.spec.joint_id for binding in bindings]
            qpos = torch.tensor(
                [[binding.desired_value for binding in bindings]],
                dtype=torch.float32,
                device=articulation.device,
            )
            zeros = torch.zeros_like(qpos)
            env_ids = [0]
            articulation.set_qpos(
                qpos,
                joint_ids=joint_ids,
                env_ids=env_ids,
                target=False,
            )
            articulation.set_qpos(
                qpos,
                joint_ids=joint_ids,
                env_ids=env_ids,
                target=True,
            )
            articulation.set_qvel(
                zeros,
                joint_ids=joint_ids,
                env_ids=env_ids,
                target=False,
            )
            articulation.set_qvel(
                zeros,
                joint_ids=joint_ids,
                env_ids=env_ids,
                target=True,
            )
            articulation.set_qf(
                zeros,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

    def update(self) -> int:
        """Consume browser commands and hold desired poses before one sim step.

        Returns:
            Number of valid commands accepted during this update.
        """
        accepted = self._apply_commands(self._runtime.drain_joint_control_commands())
        self._hold_desired_positions()
        return accepted
