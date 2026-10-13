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

"""Optional SimulationManager verification for generated URDF artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["verify_urdf_in_simulation"]


def verify_urdf_in_simulation(
    output_path: Path,
    backend: str,
    seed: int | None,
    *,
    simulation: Any | None = None,
) -> dict[str, Any]:
    """Load one generated URDF, step once, and destroy the temporary world.

    Args:
        output_path: Generated URDF path owned by the assembly adapter.
        backend: Simulation backend name (``default`` or ``newton``).
        seed: Optional world seed.
        simulation: Optional existing ``SimulationManagerBackend``.  When it
            is omitted, the verifier owns a temporary backend instance.

    Returns:
        JSON-compatible robot and world metadata from the verification.

    Raises:
        ValueError: If the backend cannot load or step the generated model.
    """
    from .backend import SimulationManagerBackend

    owns_simulation = simulation is None
    if simulation is None:
        simulation = SimulationManagerBackend()
    world_id: str | None = None
    try:
        world = simulation.create_world(backend=backend, seed=seed)
        world_id = str(world["world_id"])
        state = simulation.load_scene(
            world_id,
            task_id=None,
            scene={
                "physics": backend,
                "robot": {
                    "uid": "urdf_assembly",
                    "fpath": str(output_path),
                },
            },
        )
        stepped = simulation.step_simulation(world_id, steps=1)
        return {
            "status": "succeeded",
            "backend": backend,
            "seed": seed,
            "robot_id": "urdf_assembly",
            "robots": simulation.list_robots(),
            "scene_revision": state["scene_revision"],
            "step_result": {"scene_revision": stepped["scene_revision"]},
        }
    finally:
        try:
            if world_id is not None:
                simulation.destroy_world(world_id)
        except Exception:  # noqa: BLE001 - cleanup must not skip backend close.
            pass
        finally:
            if owns_simulation:
                simulation.close()
