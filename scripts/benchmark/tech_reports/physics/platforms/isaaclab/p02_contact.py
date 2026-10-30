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

"""Measure P-02 using Isaac Lab GPU PhysX, including raw normal contacts.

Run: python -m scripts.benchmark.tech_reports.physics.platforms.isaaclab.p02_contact --help
"""

from __future__ import annotations

import argparse
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
    """Run one P-02 scene with per-step poses, velocities and contact records."""
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--solver", choices=["TGS", "PGS"], default="TGS")
    parser.add_argument("--position-iterations", type=int, default=4)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.record:
        args.enable_cameras = True
    cfg, inputs, scene, meta = begin_run(args, "isaaclab", Path(__file__))
    app = None
    tic_total = time.perf_counter()
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Sdf
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab_physx.physics import PhysxCfg

        physics = PhysxCfg(
            solver_type=1 if args.solver == "TGS" else 0,
            enable_external_forces_every_iteration=True,
            enable_stabilization=False,
            min_position_iteration_count=args.position_iterations,
            max_position_iteration_count=args.position_iterations,
            min_velocity_iteration_count=args.velocity_iterations,
            max_velocity_iteration_count=args.velocity_iterations,
        )
        sim = SimulationContext(
            SimulationCfg(
                dt=args.dt,
                device=args.device,
                gravity=(0, 0, -cfg["gravity_m_s2"]),
                physics=physics,
                visualizer_cfgs=[],
                use_newton_actuators=False,
                log_dir=str(args.output / "sim_logs"),
            )
        )
        stage = sim.stage
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        geometry = scene_geometry(cfg, scene, inputs, args.episode_start)
        materials = {}
        for name, ms, md in [
            ("test", scene["mu_static"], scene["mu_dynamic"]),
            ("ground", 0.5, 0.5),
        ]:
            mat = UsdShade.Material.Define(stage, f"/World/Materials/{name}")
            props = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            props.CreateStaticFrictionAttr(ms)
            props.CreateDynamicFrictionAttr(md)
            props.CreateRestitutionAttr(0)
            native = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
            native.CreateFrictionCombineModeAttr("average")
            native.CreateRestitutionCombineModeAttr("average")
            materials[name] = mat

        def add_box(
            path: str,
            pose: np.ndarray,
            size: np.ndarray,
            dynamic: bool,
            material: str,
            color: list[float],
        ) -> None:
            """Author SI geometry and explicit mass/COM/inertia on a rigid body."""
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            transform = UsdGeom.Xformable(prim)
            transform.AddTranslateOp().Set(Gf.Vec3d(*map(float, pose[:3])))
            transform.AddOrientOp().Set(
                Gf.Quatf(float(pose[6]), Gf.Vec3f(*map(float, pose[3:6])))
            )
            if dynamic:
                UsdPhysics.RigidBodyAPI.Apply(prim)
                mass = UsdPhysics.MassAPI.Apply(prim)
                mass.CreateMassAttr(cfg["mass_kg"])
                mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
                mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*geometry["body_inertia"]))
                mass.CreatePrincipalAxesAttr(Gf.Quatf(1))
                props = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                props.CreateLinearDampingAttr(0)
                props.CreateAngularDampingAttr(0)
                sleep_threshold = cfg["physx"].get("sleep_threshold", 0)
                if sleep_threshold is not None:
                    props.CreateSleepThresholdAttr(sleep_threshold)
                props.CreateStabilizationThresholdAttr(0)
                props.CreateMaxLinearVelocityAttr(1000)
                props.CreateMaxAngularVelocityAttr(10000)
                props.CreateSolverPositionIterationCountAttr(args.position_iterations)
                props.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0)
            cube = UsdGeom.Cube.Define(stage, path + "/shape")
            cube.CreateSizeAttr(1)
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr(
                0.002
                if material == "ground"
                else cfg["contact_margin_m"] + cfg["contact_gap_m"]
            )
            collision.CreateRestOffsetAttr(
                0.001 if material == "ground" else cfg["contact_margin_m"]
            )
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                materials[material], materialPurpose="physics"
            )

        add_box(
            "/World/support",
            geometry["support_pose"],
            geometry["support_size"],
            False,
            "test",
            cfg["capture"]["surface_color"],
        )
        add_box(
            "/World/background_ground",
            np.array([0, 0, -0.05, 0, 0, 0, 1]),
            np.array([1000, 1000, 0.1]),
            False,
            "ground",
            cfg["capture"]["background_color"],
        )
        objects = []
        sensors = []
        for i, pose in enumerate(geometry["body_poses"]):
            path = f"/World/block_{i:02d}"
            add_box(
                path,
                pose,
                geometry["body_size"],
                True,
                "test",
                cfg["capture"]["body_colors"][i % 3],
            )
            objects.append(RigidObject(RigidObjectCfg(prim_path=path, spawn=None)))
            if not cfg.get("record_contact_forces", True):
                continue
            sensors.append(
                ContactSensor(
                    ContactSensorCfg(
                        prim_path=path,
                        update_period=0,
                        filter_prim_paths_expr=[
                            "/World/support/shape",
                            *[
                                f"/World/block_{j:02d}"
                                for j in range(len(geometry["body_poses"]))
                                if j != i
                            ],
                            "/World/background_ground/shape",
                        ],
                        track_contact_points=True,
                        track_friction_forces=True,
                        max_contact_data_count_per_prim=256,
                    )
                )
            )
        recorder = None
        if args.record:
            from scripts.benchmark.tech_reports.physics.platforms.isaaclab.p02_capture import (
                Capture,
            )

            recorder = Capture(
                sim, cfg, scene, inputs, args.episode_start, args.output, args.dt
            )
        sim.reset()
        meta["solver"] = (
            f"PhysX {args.solver} {args.position_iterations}/{args.velocity_iterations}"
        )
        meta["effective_config"] = sim.cfg.to_dict()
        meta["physics_scene_attributes"] = {
            str(prim.GetPath()): {
                str(a.GetName()): a.Get() for a in prim.GetAttributes()
            }
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Scene)
        }
        meta["sleep_configuration"] = {
            "requested_threshold": cfg["physx"].get("sleep_threshold", 0),
            "body_threshold_readback": {
                str(prim.GetPath()): PhysxSchema.PhysxRigidBodyAPI(prim)
                .GetSleepThresholdAttr()
                .Get()
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI)
            },
        }
        meta["body_readback"] = []
        for obj in objects:
            mass = obj.data.body_mass.torch.cpu().numpy()
            inertia = obj.data.body_inertia.torch.cpu().numpy()
            np.testing.assert_allclose(mass, cfg["mass_kg"], rtol=1e-6, atol=0)
            np.testing.assert_allclose(
                inertia,
                np.broadcast_to(
                    np.diag(geometry["body_inertia"]).reshape(-1), inertia.shape
                ),
                rtol=1e-6,
                atol=0,
            )
            materials_readback = obj.root_view.get_material_properties().numpy().copy()
            np.testing.assert_allclose(
                materials_readback.reshape(-1, 3),
                np.array([[scene["mu_static"], scene["mu_dynamic"], 0]]),
                rtol=1e-6,
                atol=0,
            )
            meta["body_readback"].append(
                dict(
                    mass_kg=mass.tolist(),
                    material_mu_static_dynamic_restitution=materials_readback.tolist(),
                    inertia_kg_m2=inertia.tolist(),
                    com=obj.data.body_com_pose_b.torch.cpu().tolist(),
                )
            )
        meta["contact_export"] = {
            "enabled": bool(sensors),
            "scope": (
                "Force/impulse export disabled in state-only size sweep"
                if not sensors
                else "Per-step contact export"
            ),
            "capacity_per_sensor": 256,
            "normal_impulse_dt_argument_s": 1.0,
            "average_force_window_s": args.dt,
            "point_separation_includes_offsets": True,
            "filter_identity": "Raw backend filter index retained; no inferred actor identity. Independent box geometry is used for penetration.",
        }
        meta["background_ground"] = (
            "z=0 top of 1000 x 1000 m static slab, mu_s=mu_d=0.5, rest_offset=0.001 m; excluded after ramp exit from analytic comparisons."
        )
        stage.Export(str(args.output / "scene.usda"))
        write_run(args.output, meta)
        steps = round(args.duration / args.dt)
        times = np.arange(steps + 1, dtype=np.float64) * args.dt
        durations = []
        for episode in range(args.episode_start, args.episode_start + args.count):
            geometry = scene_geometry(cfg, scene, inputs, episode)
            for obj, pose in zip(objects, geometry["body_poses"]):
                obj.write_root_pose_to_sim_index(
                    root_pose=torch.as_tensor(
                        pose[None], dtype=torch.float32, device=sim.device
                    )
                )
                obj.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=sim.device)
                )
                obj.reset()
            for sensor in sensors:
                sensor.reset()
            state = np.lib.format.open_memmap(
                args.output / f"episode_{episode:02d}_state.npy",
                mode="w+",
                dtype=np.float64,
                shape=(steps + 1, len(objects), 13),
            )
            state[:] = np.nan
            normal_forces = np.full((steps + 1, len(objects), 3), np.nan)
            friction_forces = np.full_like(normal_forces, np.nan)
            contact_chunks = []
            contact_counts = np.zeros((steps + 1, len(objects)), dtype=np.int32)
            torch.cuda.synchronize()
            tic = time.perf_counter()
            for k in range(steps + 1):
                if k:
                    sim.step(render=False)
                    for obj in objects:
                        obj.update(args.dt)
                    for sensor in sensors:
                        sensor.update(args.dt, force_recompute=True)
                state[k] = (
                    torch.cat([obj.data.root_state_w.torch for obj in objects], dim=0)
                    .cpu()
                    .numpy()
                )
                if not np.isfinite(state[k]).all():
                    raise FloatingPointError(
                        f"Nonfinite state episode={episode} step={k}"
                    )
                if k:
                    for bi, sensor in enumerate(sensors):
                        normal_forces[k, bi] = (
                            sensor.data.net_normal_forces_w.torch[0, 0].cpu().numpy()
                        )
                        friction_forces[k, bi] = (
                            sensor.data.friction_force_matrix_w.torch[0, 0]
                            .sum(0)
                            .cpu()
                            .numpy()
                        )
                        impulse, point, normal, separation, counts, starts = (
                            sensor.contact_view.get_contact_data(dt=1.0)
                        )
                        impulse, point, normal, separation, counts, starts = [
                            a.numpy().copy()
                            for a in (
                                impulse,
                                point,
                                normal,
                                separation,
                                counts,
                                starts,
                            )
                        ]
                        counts = counts.reshape(-1)
                        starts = starts.reshape(-1)
                        total = int(counts.sum())
                        assert total < 256, f"Contact capacity reached: {total}"
                        contact_counts[k, bi] = total
                        for fi, (n, start) in enumerate(zip(counts, starts)):
                            if n:
                                sl = slice(int(start), int(start + n))
                                contact_chunks.append(
                                    np.column_stack(
                                        (
                                            np.full(int(n), k),
                                            np.full(int(n), bi),
                                            np.full(int(n), fi),
                                            impulse[sl].reshape(-1),
                                            point[sl].reshape(-1, 3),
                                            normal[sl].reshape(-1, 3),
                                            separation[sl].reshape(-1),
                                        )
                                    )
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
            contacts = (
                np.concatenate(contact_chunks) if contact_chunks else np.empty((0, 11))
            )
            np.savez_compressed(
                args.output / f"episode_{episode:02d}.npz",
                t=times,
                state=state,
                requested_initial_poses=geometry["body_poses"],
                **(
                    dict(
                        normal_forces=normal_forces,
                        friction_forces=friction_forces,
                        contact_counts=contact_counts,
                        contacts=contacts,
                    )
                    if sensors
                    else {}
                ),
                **{k: v for k, v in geometry.items() if k != "body_poses"},
            )
            meta["completed_episodes"] += 1
            meta["diagnostic_wall_s_per_episode"] = durations
            meta["contact_record_columns"] = [
                "step",
                "body_index",
                "filter_index",
                "normal_impulse_N_s",
                "point_x",
                "point_y",
                "point_z",
                "normal_x",
                "normal_y",
                "normal_z",
                "separation_m",
            ]
            write_run(args.output, meta)
            print(
                f'P02 {meta["solver"]} {args.scene} dt={args.dt} initial={episode} final_xyz={state[-1,:,:3].tolist()} contacts={len(contacts)} wall={durations[-1]:.2f}s',
                flush=True,
            )
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
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
