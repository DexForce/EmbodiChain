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

"""Measure P-02 through EmbodiChain public APIs using DexUni.

Run: python -m scripts.benchmark.tech_reports.physics.platforms.embodichain.p02_contact --help
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import time
import traceback

from scripts.benchmark.tech_reports.physics.experiments.p02_protocol import (
    add_arguments,
    begin_run,
    finish_run,
    scene_geometry,
    write_run,
)


def main() -> None:
    """Run one scene and step size, keeping every physical-step state."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--contact-grouping", action="store_true")
    parser.add_argument("--contact-history", action="store_true")
    parser.add_argument("--no-contact-snap", action="store_true")
    args = parser.parse_args()
    cfg, inputs, scene, meta = begin_run(args, "embodichain", Path(__file__))
    sim = None
    tic_total = time.perf_counter()
    try:
        import numpy as np
        import torch
        from scipy.spatial.transform import Rotation
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            NewtonPhysicsCfg,
            RenderCfg,
            DenoisingCfg,
            RigidObjectCfg,
            RigidBodyPhysicsCfg,
            MassPropertiesCfg,
            NewtonCollisionPropertiesCfg,
            RigidBodyMaterialCfg,
        )
        from embodichain.lab.sim.material import VisualMaterialCfg
        from embodichain.lab.sim.shapes import CubeCfg

        meta["module_origins"] = {
            n: importlib.util.find_spec(n).origin
            for n in ["embodichain", "embodichain_robots", "embodichain_tasks"]
        }
        assert all(
            str(args.source_repo.resolve()) in p
            for p in meta["module_origins"].values()
        ), meta["module_origins"]
        solver = cfg["dexuni_options"].copy()
        solver["vbd_options"] = solver["vbd_options"].copy()
        solver["collision_options"] = solver["collision_options"].copy()
        if args.iterations is not None:
            solver["vbd_options"]["iterations"] = args.iterations
        if args.contact_grouping:
            solver["vbd_options"]["rigid_contact_grouping"] = True
        if args.no_contact_snap:
            solver["vbd_options"].update(
                rigid_contact_stick_motion_eps=0.0,
                rigid_contact_stick_freeze_translation_eps=0.0,
                rigid_contact_stick_freeze_angular_eps=0.0,
            )
        if args.contact_history:
            solver["vbd_options"].update(
                rigid_contact_history=True,
                rigid_contact_stick_motion_eps=0.0,
                rigid_contact_stick_freeze_translation_eps=0.0,
                rigid_contact_stick_freeze_angular_eps=0.0,
            )
            solver["collision_options"]["contact_matching"] = "latest"
        sim_cfg = SimulationManagerCfg(
            headless=True,
            num_envs=1,
            startup_summary="full",
            dexsim_startup_info=True,
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=args.dt,
                gravity=[0, 0, -cfg["gravity_m_s2"]],
                num_substeps=1,
                use_cuda_graph=True,
                collision_cfg=None,
                solver_cfg=solver,
            ),
            render_cfg=RenderCfg(
                renderer="hybrid" if args.record else "no-render",
                spp=8,
                denoising=DenoisingCfg(
                    window="off", offscreen="optix" if args.record else "off"
                ),
            ),
        )
        meta["requested_sim_config"] = sim_cfg.to_dict()
        sim = SimulationManager(sim_cfg)
        geometry = scene_geometry(cfg, scene, inputs, args.episode_start)
        material = RigidBodyMaterialCfg(
            static_friction=scene["mu_static"],
            dynamic_friction=scene["mu_dynamic"],
            restitution=0,
        )
        collision = NewtonCollisionPropertiesCfg(
            margin=cfg["contact_margin_m"], gap=cfg["contact_gap_m"]
        )
        support = sim.add_rigid_object(
            RigidObjectCfg(
                uid="support",
                body_type="static",
                init_pos=tuple(geometry["support_pose"][:3]),
                init_rot=tuple(
                    Rotation.from_quat(geometry["support_pose"][3:]).as_euler(
                        "xyz", degrees=True
                    )
                ),
                shape=CubeCfg(
                    size=geometry["support_size"].tolist(),
                    visual_material=VisualMaterialCfg(
                        uid="support_color",
                        base_color=[*cfg["capture"]["surface_color"], 1.0],
                    ),
                ),
                attrs=RigidBodyPhysicsCfg(
                    material_props=material, collision_props=collision
                ),
            )
        )
        objects = []
        for i, pose in enumerate(geometry["body_poses"]):
            objects.append(
                sim.add_rigid_object(
                    RigidObjectCfg(
                        uid=f"block_{i:02d}",
                        init_pos=tuple(pose[:3]),
                        init_rot=tuple(
                            Rotation.from_quat(pose[3:]).as_euler("xyz", degrees=True)
                        ),
                        shape=CubeCfg(
                            size=geometry["body_size"].tolist(),
                            visual_material=VisualMaterialCfg(
                                uid=f"body_color_{i%3}",
                                base_color=[*cfg["capture"]["body_colors"][i % 3], 1.0],
                            ),
                        ),
                        attrs=RigidBodyPhysicsCfg(
                            mass_props=MassPropertiesCfg(
                                mass=cfg["mass_kg"],
                                inertia=geometry["body_inertia"].tolist(),
                                com_position=[0, 0, 0],
                            ),
                            material_props=material,
                            collision_props=collision,
                        ),
                    )
                )
            )
        recorder = None
        if args.record:
            from scripts.benchmark.tech_reports.physics.platforms.embodichain.p02_capture import (
                Capture,
            )

            recorder = Capture(
                sim, cfg, scene, inputs, args.episode_start, args.output, args.dt
            )
        sim.prepare()
        meta["solver"] = (
            f'DexUni / VBD {solver["vbd_options"]["iterations"]} iterations'
        )
        if args.contact_grouping:
            meta["solver"] += " / contact grouping"
        if args.contact_history:
            meta["solver"] += " / contact history (no stick snap)"
        if args.no_contact_snap:
            meta["solver"] += " / no contact snap"
        meta["resolved_solver"] = sim.physics.solver_type
        assert meta["resolved_solver"] == "dexuni"
        meta["contact_export"] = {
            "supported": sim.physics.supports_contact_sensor,
            "reason": "DexUni does not expose EmbodiChain ContactQuery; signed gaps and box intersections are computed offline from raw poses. No solver-force values are fabricated.",
        }
        assert not meta["contact_export"]["supported"]
        meta["material_semantics"] = (
            "Single Coulomb mu = dynamic_friction; separate static_friction is not consumed by Newton. Both contacting shapes share mu."
        )
        meta["background_ground"] = (
            "Mandatory plane z=0, mu=0.5, Newton margin/gap defaults 0.001 m. Test support is elevated; post-ramp-flight/ground stages are excluded from inclined-plane reference."
        )
        meta["body_readback"] = []
        for obj in objects:
            mass = obj.get_mass().cpu().numpy()
            inertia = obj.get_inertia().cpu().numpy()
            friction = obj.get_friction().cpu().numpy()
            np.testing.assert_allclose(mass, cfg["mass_kg"], rtol=1e-6, atol=0)
            np.testing.assert_allclose(
                inertia,
                np.broadcast_to(geometry["body_inertia"], inertia.shape),
                rtol=1e-6,
                atol=0,
            )
            np.testing.assert_allclose(friction, scene["mu_dynamic"], rtol=1e-6, atol=0)
            entry = dict(
                uid=obj.uid,
                mass_kg=mass.tolist(),
                inertia_kg_m2=inertia.tolist(),
                friction=friction.tolist(),
                damping=obj.get_damping().cpu().tolist(),
            )
            try:
                entry["collision_shapes"] = repr(obj.get_collision_shapes())
            except (RuntimeError, NotImplementedError) as exc:
                entry["collision_geometry_readback_unavailable"] = str(exc)
            meta["body_readback"].append(entry)
        meta["support_pose_readback"] = support.get_local_pose().cpu().tolist()
        write_run(args.output, meta)
        steps = round(args.duration / args.dt)
        times = np.arange(steps + 1, dtype=np.float64) * args.dt
        durations = []
        for episode in range(args.episode_start, args.episode_start + args.count):
            sim.reset_objects_state()
            geometry = scene_geometry(cfg, scene, inputs, episode)
            for obj, pose in zip(objects, geometry["body_poses"]):
                obj.set_local_pose(
                    torch.as_tensor(pose[None], dtype=torch.float32, device=sim.device)
                )
                obj.set_velocity(
                    lin_vel=torch.zeros((1, 3), device=sim.device),
                    ang_vel=torch.zeros((1, 3), device=sim.device),
                )
            state = np.lib.format.open_memmap(
                args.output / f"episode_{episode:02d}_state.npy",
                mode="w+",
                dtype=np.float64,
                shape=(steps + 1, len(objects), 13),
            )
            state[:] = np.nan
            torch.cuda.synchronize()
            tic = time.perf_counter()
            for k in range(steps + 1):
                if k:
                    sim.update(step=1, render_final_step=False)
                state[k] = (
                    torch.cat([obj.body_state for obj in objects], dim=0).cpu().numpy()
                )
                if not np.isfinite(state[k]).all():
                    raise FloatingPointError(
                        f"Nonfinite state episode={episode} step={k}"
                    )
                if recorder is not None:
                    recorder.capture(k, state[k])
                if k % 400 == 0 or k == steps:
                    state.flush()
                    meta["last_sample"] = {
                        "episode": episode,
                        "step": k,
                        "time_s": float(times[k]),
                    }
                    write_run(args.output, meta)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - tic)
            np.savez_compressed(
                args.output / f"episode_{episode:02d}.npz",
                t=times,
                state=state,
                requested_initial_poses=geometry["body_poses"],
                **{k: v for k, v in geometry.items() if k != "body_poses"},
            )
            meta["completed_episodes"] += 1
            meta["diagnostic_wall_s_per_episode"] = durations
            write_run(args.output, meta)
            print(
                f"P02 DexUni {args.scene} dt={args.dt} initial={episode} final_xyz={state[-1,:,:3].tolist()} wall={durations[-1]:.2f}s",
                flush=True,
            )
        meta["cuda_graph_status"] = sim.physics.cuda_graph_status
        if recorder is not None:
            meta["capture"] = recorder.finish()
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["failure"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        meta["wall_s_including_startup"] = time.perf_counter() - tic_total
        finish_run(args.output, meta)
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
