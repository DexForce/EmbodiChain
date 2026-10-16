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

"""Build and qualify a tabletop recipe without image or model services.

Run ``python scripts/tutorials/scene/tabletop_recipe.py --seed 7``. The example
uses existing primitive configuration types as its asset catalog; it creates
no task environment, robot, or dataset. The caller owns physics and cleanup.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from pathlib import Path

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.visualization import VisualizationCfg
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.scene import (
    LayoutProblem,
    SceneFragment,
    SceneRecipe,
    SceneView,
    SettlingCfg,
    capture_state,
    check_layout,
    compare_states,
    declare_fragment,
    generate,
    restore_state,
    settle_scene,
)


def make_recipe() -> SceneRecipe:
    """Assemble a work surface, two five-piece bins, six parts and a keep-out."""

    def box(
        uid: str,
        size: tuple[float, float, float],
        position: tuple[float, float, float],
        *,
        dynamic: bool = False,
    ) -> RigidObjectCfg:
        return RigidObjectCfg(
            uid=uid,
            shape=CubeCfg(size=list(size)),
            init_pos=position,
            body_type="dynamic" if dynamic else "static",
        )

    entities = [box("work_surface", (2.0, 1.5, 0.06), (0.0, 0.0, 0.75))]
    seeds = {}
    half = {}
    fixed = {}
    bounds = {}
    for uid, x in (("bin_a", -0.65), ("bin_b", 0.65)):
        y = 0.4
        entities.append(box(uid, (0.3, 0.3, 0.02), (x, y, 0.79)))
        for side, dx in (("left", -0.14), ("right", 0.14)):
            entities.append(
                box(f"{uid}__{side}", (0.02, 0.3, 0.15), (x + dx, y, 0.865))
            )
        for side, dy in (("front", -0.14), ("back", 0.14)):
            entities.append(
                box(f"{uid}__{side}", (0.26, 0.02, 0.15), (x, y + dy, 0.865))
            )
        seeds[uid] = fixed[uid] = (x, y)
        half[uid] = (0.15, 0.15)
    for index in range(6):
        uid = f"part_{index}"
        entities.append(box(uid, (0.06, 0.06, 0.06), (0.0, 0.0, 0.84), dynamic=True))
        seeds[uid] = (0.0, 0.0)
        half[uid] = (0.03, 0.03)
        bounds[uid] = ((-0.95, -0.70), (0.95, 0.70))
    seeds["keep_out"] = fixed["keep_out"] = (0.0, 0.25)
    half["keep_out"] = (0.2, 0.2)
    return SceneRecipe(
        "tabletop:primitive-assets/v1",
        SceneFragment(entities),
        LayoutProblem("work_surface", seeds, half, bounds, fixed),
    )


def run(seed: int) -> dict[str, object]:
    """Generate, settle, measure and restore one caller-owned B=1 scene."""
    recipe = make_recipe()
    generated = generate(recipe, seed=seed)
    if generated.fragment is None or generated.variant is None:
        raise RuntimeError(generated.layout.message)
    cfg = SimulationManagerCfg(
        headless=True,
        device="cpu",
        num_envs=1,
        visualization=VisualizationCfg(backend="none"),
    )
    cfg.render_cfg.renderer = "no-render"
    sim = SimulationManager(cfg)
    try:
        declare_fragment(sim, generated.fragment)
        sim.prepare()
        view = SceneView(sim, env_ids=(0,))
        stability = settle_scene(
            view,
            lambda steps: sim.update(step=steps),
            SettlingCfg(mode="convergence", max_steps=600, consecutive_steps=10),
        )
        poses = {
            entity.uid: view.pose(entity.uid)[0].detach().cpu().numpy()
            for entity in generated.fragment.entities
        }
        # Re-measure actual box footprints after settling: even tiny rotations
        # invalidate an authored fixed-orientation AABB.
        actual_aabbs = {}
        for entity in generated.fragment.entities:
            if entity.uid.startswith("part_"):
                pose = poses[entity.uid]
                half = np.abs(pose[:3, :3]) @ (
                    np.asarray(entity.shape.size) * np.asarray(entity.body_scale) / 2
                )
                actual_aabbs[entity.uid] = np.stack(
                    [pose[:2, 3] - half[:2], pose[:2, 3] + half[:2]]
                )
        measured = check_layout(recipe, poses, actual_aabbs_xy_by_uid=actual_aabbs)
        snapshot = capture_state(view)
        sim.update(step=5)
        restore_state(view, snapshot)
        restoration = compare_states(snapshot, capture_state(view))
        result = {
            "variant_id": generated.variant.variant_id,
            "layout_status": generated.layout.status,
            "stable": stability.passed,
            "measured_xy_layout_valid": measured.passed,
            "state_restored": restoration.passed,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "message": check.message,
                    "metrics": dict(check.metrics),
                }
                for report in (stability, measured, restoration)
                for check in report.checks
            ],
        }
        if not (stability.passed and measured.passed and restoration.passed):
            raise RuntimeError(json.dumps(result, indent=2))
        return result
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
