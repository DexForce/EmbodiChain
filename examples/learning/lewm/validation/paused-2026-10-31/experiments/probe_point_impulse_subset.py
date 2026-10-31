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

"""Private paired protocol for the PGS position-correction regression."""

from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg
from embodichain_tasks.manipulation.ogbench_cube import OGBenchCubeEnv, OGBenchCubeCfg

ROOT = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["baseline", "point"])
    p.add_argument("--indices", type=int, nargs="+", default=[5, 7, 12, 32, 34, 41])
    args = p.parse_args()
    from cube_point_impulse_trial import PointImpulseCube

    Env = PointImpulseCube if args.mode == "point" else OGBenchCubeEnv
    source = np.load(ROOT / "evaluation/mujoco/trace.npz")
    ids = np.array(args.indices)
    env = Env(
        OGBenchCubeCfg(
            reference_root=str(ROOT / "alignment-point-response-pgs"),
            num_envs=len(ids),
            enable_images=False,
            sim_cfg=SimulationManagerCfg(
                device="cuda:3",
                gpu_id=3,
                physics_dt=0.002,
                physics_cfg=DefaultPhysicsCfg(enable_tgs=False, cache_material=True),
                headless=True,
                enable_default_plane=False,
                render_cfg=RenderCfg(renderer="no-render"),
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                startup_summary="off",
            ),
        )
    )
    try:
        env.reset(
            seed=42,
            options={
                "qpos": source["qpos"][0, ids],
                "qvel": source["qvel"][0, ids],
                "goal_pos": source["goals"][ids],
            },
        )
        alive = np.ones(len(ids), bool)
        success = np.zeros(len(ids), bool)
        first = np.full(len(ids), -1)
        frozen = env.cube.get_local_pose().cpu().numpy()[:, :3].copy()
        trace = [frozen.copy()]
        driver = []
        for step in range(50):
            actions = source["actions"][step, ids].copy()
            actions[~np.isfinite(actions).all(-1)] = 0
            obs, _, _, _, info = env.step(torch.as_tensor(actions, device=env.device))
            position = env.cube.get_local_pose().cpu().numpy()[:, :3]
            frozen[alive] = position[alive]
            trace.append(frozen.copy())
            hit = info["success"].cpu().numpy() & alive
            success |= hit
            first[hit] = step + 1
            alive &= ~hit
            q = env.robot.get_qpos()[:, env._joint_ids].cpu().numpy()
            driver.append(q[:, 6] - q[:, 10])
        src = np.array(
            json.loads((ROOT / "evaluation/mujoco/report.json").read_text())[
                "episode_successes"
            ]
        )[ids]
        e = np.linalg.norm(np.array(trace) - source["qpos"][:, ids, 14:17], axis=-1)
        report = {
            "scope": "private6-query diagnosis, same raw-source actions; not formal50 evaluation",
            "point_trial": getattr(env, "point_trial", None),
            "mode": args.mode,
            "indices": ids.tolist(),
            "success": success.tolist(),
            "first_success_step": first.tolist(),
            "source_success": src.tolist(),
            "outcome_agreement": int((success == src).sum()),
            "cube_error_median_p95_max": np.quantile(e, [0.5, 0.95, 1]).tolist(),
            "driver_error_median_max": np.quantile(
                abs(np.array(driver)), [0.5, 1]
            ).tolist(),
        }
        path = ROOT / "alignment" / f"point-impulse-subset-{args.mode}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        np.savez(
            path.with_suffix(".npz"),
            cube=np.array(trace),
            driver_error=np.array(driver),
        )
        print(report, flush=True)
    finally:
        env.close()
        SimulationManager.flush_cleanup_queue()
