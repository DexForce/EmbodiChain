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
"""Private source passive-joint-damping adaptation for native task qualification."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import warp as wp
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from newton import ModelFlags

from embodichain.utils.device_utils import current_warp_stream

if TYPE_CHECKING:
    from embodichain.lab.sim import SimulationManager

__all__: list[str] = []


@torch.no_grad()
def _apply_source_joint_damping(
    manager: SimulationManager, values: dict[str, float]
) -> None:
    """Restore source passive damping separately from actuator velocity gains."""
    result = manager.spawn_result
    if result is None or not manager.is_newton_backend:
        raise RuntimeError("Source joint damping requires prepared Newton physics")
    if not values or any(not math.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("Passive damping must contain finite nonnegative values")
    backend = get_newton_backend(result.world)
    if backend is None or backend.solver_type != "mujoco_warp":
        raise ValueError("Source damping currently qualifies MuJoCo-Warp only")
    model = backend.model
    starts = model.joint_qd_start.numpy()
    indices, coefficients = [], []
    counts = dict.fromkeys(values, 0)
    for joint, label in enumerate(model.joint_label):
        name = label.rsplit("/", 1)[-1]
        if name not in values:
            continue
        if starts[joint + 1] - starts[joint] != 1:
            raise ValueError("Source passive damping expects scalar hinge joints")
        indices.append(int(starts[joint]))
        coefficients.append(values[name])
        counts[name] += 1
    if any(count != manager.num_envs for count in counts.values()):
        raise ValueError(f"Expected one source joint per Arena: {counts}")
    device = torch.device(str(model.device))
    with wp.ScopedStream(current_warp_stream(device)):
        wp.to_torch(model.joint_damping)[torch.tensor(indices, device=device)] = (
            torch.tensor(coefficients, device=device)
        )
        backend._runtime.notify_model_changed(
            ModelFlags.JOINT_DOF_PROPERTIES, reason="native source passive damping"
        )
