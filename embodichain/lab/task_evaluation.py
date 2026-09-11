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


"""Bounded TaskSpec observation evaluation, without a simulation lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any
import torch

from embodichain.compute.task_predicates import axis_tilt
from embodichain.task_spec import canonical_template, validate_task_template

__all__ = ["UprightTaskEvaluator"]


class UprightTaskEvaluator:
    """Evaluate one local-+Z upright goal, optionally initially not upright.

    This deliberately rejects process, capability, duration and other predicate
    requirements. It does not infer attachment or support from object posture,
    and owns no stepping, retry, reset, data submission or persistence.

    Args:
        template: Validated TaskSpec v0.1 record with a single upright goal.
    """

    def __init__(self, template: Mapping[str, Any]) -> None:
        source = validate_task_template(template)
        value = canonical_template(source)
        goal = value["goal"]
        if (
            len(value["roles"]) != 1
            or value["roles"]["role_0"]["kind"] not in {"object", "container"}
            or value["roles"]["role_0"].get("capabilities")
            or value["invariants"]
            or value["requirements"]
            or value.get("temporal")
            or len(goal) != 1
            or goal[0].get("predicate") != "upright"
        ):
            raise ValueError(
                "Only bounded single-object upright TaskSpec is supported."
            )
        self.max_tilt = float(goal[0]["max_tilt"]["value"])
        if not 0 < self.max_tilt < math.pi / 2:
            raise ValueError("Upright max_tilt must be between zero and pi/2.")
        negative = {"op": "not", "args": [goal[0]]}
        if value["init"] not in ([], [negative]):
            raise ValueError(
                "Only bounded initially-not-upright constraints are supported."
            )
        self.require_initially_fallen = bool(value["init"])
        self.template_hash = source["semantic_hash"]

    def evaluate(self, poses: torch.Tensor, *, scope: str) -> dict[str, Any]:
        """Evaluate a detached simultaneous pose batch in metres, scene Z-up.

        Missing, nonfinite, or nonrigid rows are unavailable, including under
        negation. Results describe an instantaneous goal, not a stable window.

        Args:
            poses: Simultaneous observed transforms, shaped ``(N, 4, 4)``.
            scope: Either ``initial`` or ``task_goal``.

        Returns:
            Detached JSON-compatible statuses, angles and checker identity.
        """
        if scope not in {"initial", "task_goal"}:
            raise ValueError("Expected initial or task_goal evaluation scope.")
        if not isinstance(poses, torch.Tensor) or not poses.is_floating_point():
            raise ValueError("Observed poses must be a floating point tensor.")
        if poses.ndim != 3 or poses.shape[-2:] != (4, 4) or not len(poses):
            raise ValueError("Observed poses must have shape (N, 4, 4).")
        pose = poses.detach().clone()
        rotation = pose[:, :3, :3]
        valid = torch.isfinite(pose).all(dim=-1).all(dim=-1)
        valid &= (
            torch.isclose(
                rotation.transpose(-1, -2) @ rotation,
                torch.eye(3, device=pose.device, dtype=pose.dtype),
                atol=1e-4,
                rtol=0,
            )
            .all(dim=-1)
            .all(dim=-1)
        )
        valid &= torch.isclose(
            torch.linalg.det(rotation), pose.new_tensor(1.0), atol=1e-4, rtol=0
        )
        valid &= torch.isclose(
            pose[:, 3, :], pose.new_tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5, rtol=0
        ).all(-1)
        tilt = axis_tilt(pose, pose.new_tensor([0.0, 0.0, 1.0]))
        upright = tilt <= self.max_tilt
        accepted = upright
        if scope == "initial":
            accepted = (
                ~upright if self.require_initially_fallen else torch.ones_like(upright)
            )
        return {
            "schema_version": "taskspec/upright_observation/v0.1",
            "template_hash": self.template_hash,
            "checker": {"name": "upright_pose", "version": "1"},
            "predicate": {"name": "upright", "version": "1"},
            "scope": scope,
            "frame": "scene_z_up",
            "unit": "m",
            "max_tilt_rad": self.max_tilt,
            "status": [
                "unavailable" if not ok else "pass" if passed else "failed"
                for ok, passed in zip(valid.tolist(), accepted.tolist(), strict=True)
            ],
            "goal_satisfied": [
                bool(ok and satisfied)
                for ok, satisfied in zip(valid.tolist(), upright.tolist(), strict=True)
            ],
            "tilt_rad": [
                float(angle) if ok else None
                for ok, angle in zip(valid.tolist(), tilt.tolist(), strict=True)
            ],
        }
