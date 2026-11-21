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

"""Check reconstructed native pad force against independently reported normals."""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, RenderCfg
from embodichain_tasks.manipulation.ogbench_cube import OGBenchCubeEnv, OGBenchCubeCfg
from dexsim.scene import SpawnedArticulation

root = Path("/root/sources/EmbodiChain/outputs/lewm/official-cube")
source = np.load(root / "alignment/expert-probe-reference.npz")
env = OGBenchCubeEnv(
    OGBenchCubeCfg(
        reference_root=str(root / "alignment-point-response-pgs"),
        num_envs=1,
        enable_images=False,
        sim_cfg=SimulationManagerCfg(
            physics_cfg=DefaultPhysicsCfg(cache_material=True, enable_tgs=False),
            device="cuda:6",
            gpu_id=6,
            physics_dt=0.002,
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
    print("ACTUAL SCENE", env.sim.physics.get_scene().dump_scene_config(), flush=True)
    env.reset(
        seed=42,
        options={
            "qpos": source["qpos"][0],
            "qvel": source["qvel"][0],
            "goal_pos": source["goal"],
        },
    )
    scene = env.sim.spawn_result
    handle = next(
        h for h in scene.handles.values() if isinstance(h, SpawnedArticulation)
    )
    query = scene.create_contact_query(
        [(handle, "ur5e/robotiq/right_pad")],
        match="any",
        capacity=256,
        device=env.device,
        frame="arena",
    )
    actor = query.selected_actor_ids[0]
    records = []
    for step, action in enumerate(source["actions"][:45]):
        obs, *_ = env.step(torch.tensor(action[None], device=env.device))
        data = query.fetch()
        count = data.count
        contacts = data.data[:count].cpu().numpy()
        ids = data.actor_ids[:count].cpu().numpy()
        wrench = obs["right_pad_external_wrench"][0].cpu().numpy()
        row = {
            "control_step": step + 1,
            "wrench_source_order": wrench.tolist(),
            "state_contact_scalar": float(obs["state"][0, 18]),
            "contacts": count,
            "qpos": env.robot.get_qpos().cpu().tolist(),
            "qvel": env.robot.get_qvel().cpu().tolist(),
            "cube_pose": obs["cube_pose"].cpu().tolist(),
        }
        if count:
            signs = np.where(ids[:, 0] == actor, -1.0, 1.0)
            normal = contacts[:, 3:6] * signs[:, None]
            normal_force = (normal * contacts[:, 9:10]).sum(0) / 0.002
            direction = normal[0]
            parallel = float(np.max(np.linalg.norm(normal - direction, axis=-1))) < 1e-4
            row.update(
                {
                    "normal_force_sum": normal_force.tolist(),
                    "normals_parallel": parallel,
                    "minimum_distance": float(contacts[:, 10].min()),
                    "reported_normal_projection": float(normal_force @ direction),
                    "rne_normal_projection": float(wrench[3:] @ direction),
                    "friction_report_valid": bool(data.friction_valid[:count].all()),
                }
            )
            if parallel:
                row["projection_relative_error"] = abs(
                    row["rne_normal_projection"] - row["reported_normal_projection"]
                ) / max(abs(row["reported_normal_projection"]), 1e-8)
        records.append(row)
        if count or step % 10 == 0:
            print(row, flush=True)
    (root / "alignment/pad-contact-wrench-contact-capability-noop.json").write_text(
        json.dumps(
            {
                "kind": "independent_normal_projection_not_full_tangential_contact_comparison",
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
finally:
    env.close()
    SimulationManager.flush_cleanup_queue()
