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

"""Read qualified physical parameters without inventing simulator defaults."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager
__all__ = ["observed_gravity"]


def observed_gravity(simulation: SimulationManager) -> tuple[float, float, float]:
    """Read actual configured gravity for source-qualified Default/B1 consumers.

    Args:
        simulation: Borrowed existing simulation and physical configuration.

    Returns:
        Finite nonzero world gravity, preserving configured magnitude and direction.
        Other backends/batch sizes remain unsupported by this migrated path.
    """
    if simulation.physics_backend != "default" or simulation.num_envs != 1:
        raise ValueError("Observed source gravity requires Default/B1.")
    config = getattr(getattr(simulation, "sim_config", None), "physics_cfg", None)
    gravity = np.asarray(getattr(config, "gravity", None), dtype=float)
    if (
        gravity.shape != (3,)
        or not np.isfinite(gravity).all()
        or np.linalg.norm(gravity) <= 0
    ):
        raise ValueError(
            "Observed gravity requires a finite nonzero configured vector."
        )
    return tuple(float(value) for value in gravity)
