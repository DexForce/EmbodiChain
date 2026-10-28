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

"""Run one P-01 case/time-step group with Isaac Lab GPU PhysX."""

from __future__ import annotations

import os

import argparse
import hashlib
import importlib.metadata
import json
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(
    os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[6])
).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=["freefall", "pendulum"])
    parser.add_argument("--dt", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--position-iterations", type=int, default=4)
    parser.add_argument("--velocity-iterations", type=int, default=1)
    parser.add_argument("--solver", choices=["TGS", "PGS"], default="TGS")
    parser.add_argument(
        "--external-forces-every-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--theta-deg", type=float)
    parser.add_argument("--body-frame", choices=["pivot", "com"], default="pivot")
    parser.add_argument("--probe-body-velocity", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--record",
        action="store_true",
        help="Capture native RTX frames in a separate recording run.",
    )
    parser.add_argument(
        "--phase",
        choices=["pilot", "main", "diagnostic", "capture", "investigation"],
        default="main",
    )
    parser.add_argument("--common-view", type=Path)
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    assert args.common_view is None or args.record
    if args.record:
        assert args.phase in {"pilot", "capture"} and args.count == 1
        assert args.body_frame == "pivot"
        args.enable_cameras = True
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"Run already exists: {out}")
    config = json.loads((PROJECT / "configs/P-01/isaaclab.json").read_text())
    inputs = json.loads((PROJECT / config["initial_states"]).read_text())
    assert 1 <= args.count <= 10 and args.dt > 0 and args.duration > 0
    assert args.theta_deg is None or (args.case == "pendulum" and args.count == 1)
    assert args.position_iterations >= 1 and args.velocity_iterations >= 0
    snapshots = out / "config"
    snapshots.mkdir()
    sources = [
        Path(__file__),
        PROJECT / "configs/P-01/isaaclab.json",
        PROJECT / config["initial_states"],
        PROJECT / "specs/p-01-dynamics-spec.md",
    ]
    if args.phase == "investigation":
        sources.append(PROJECT / "configs/P-01/iteration-study.json")
    if args.record:
        sources.append(Path(__file__).with_name("p01_capture.py"))
    if args.common_view is not None:
        sources.append(args.common_view)
    hashes = {}
    for path in sources:
        shutil.copy2(path, snapshots / path.name)
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    now = lambda: datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    meta = dict(
        run_id=out.name,
        experiment="P-01",
        phase=args.phase,
        case=args.case,
        dt_s=args.dt,
        duration_s=args.duration,
        initial_count=args.count,
        position_iterations=args.position_iterations,
        velocity_iterations=args.velocity_iterations,
        external_forces_every_iteration=args.external_forces_every_iteration,
        body_frame=args.body_frame,
        theta_override_deg=args.theta_deg,
        probe_body_velocity=args.probe_body_velocity,
        num_envs=1,
        command=[sys.executable, *sys.argv],
        started_at=now(),
        status="running",
        solver=f"PhysX {args.solver}",
        device=args.device,
        hashes=hashes,
        python=sys.version,
        completed_episodes=0,
    )
    meta["packages"] = {
        name: importlib.metadata.version(name)
        for name in [
            "isaaclab",
            "isaaclab-physx",
            "isaacsim",
            "torch",
            "warp-lang",
            "numpy",
        ]
    }
    meta["isaaclab_commit"] = subprocess.check_output(
        ["git", "-C", "/home/oem/code/IsaacLab", "rev-parse", "HEAD"], text=True
    ).strip()
    meta["isaaclab_worktree_status"] = subprocess.check_output(
        ["git", "-C", "/home/oem/code/IsaacLab", "status", "--porcelain"], text=True
    )
    meta["gpu_inventory"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    app = None
    started = time.perf_counter()
    try:
        app = AppLauncher(args).app
        import numpy as np
        import torch
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
        from isaaclab.assets import (
            Articulation,
            ArticulationCfg,
            RigidObject,
            RigidObjectCfg,
        )
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg

        physics = PhysxCfg(
            solver_type=1 if args.solver == "TGS" else 0,
            enable_external_forces_every_iteration=args.external_forces_every_iteration,
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
                gravity=(0.0, 0.0, -config["gravity_m_s2"]),
                physics=physics,
                visualizer_cfgs=[],
                use_newton_actuators=False,
                log_dir=str(out / "sim_logs"),
            )
        )
        stage = sim.stage
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        mass, length, radius = (
            config["mass_kg"],
            config["pendulum_length_m"],
            config["bob_radius_m"],
        )

        def body(path: str, com: tuple[float, float, float], inertia: float) -> None:
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            props = UsdPhysics.MassAPI.Apply(prim)
            props.CreateMassAttr(mass)
            props.CreateCenterOfMassAttr(Gf.Vec3f(*com))
            props.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia, inertia, inertia))
            props.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
            native = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            native.CreateLinearDampingAttr(0.0)
            native.CreateAngularDampingAttr(0.0)
            native.CreateSleepThresholdAttr(0.0)
            native.CreateStabilizationThresholdAttr(0.0)
            native.CreateMaxLinearVelocityAttr(1000.0)
            native.CreateMaxAngularVelocityAttr(10000.0)
            native.CreateSolverPositionIterationCountAttr(args.position_iterations)
            native.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)

        if args.case == "freefall":
            body("/World/Body", (0.0, 0.0, 0.0), mass * config["cube_edge_m"] ** 2 / 6)
            shape = UsdGeom.Cube.Define(stage, "/World/Body/visual")
            shape.CreateSizeAttr(config["cube_edge_m"])
            obj = RigidObject(RigidObjectCfg(prim_path="/World/Body", spawn=None))
        else:
            UsdGeom.Xform.Define(stage, "/World/Pendulum")
            body("/World/Pendulum/base", (0.0, 0.0, 0.0), 0.001)
            com_z = -length if args.body_frame == "pivot" else 0.0
            body("/World/Pendulum/bob", (0.0, 0.0, com_z), 0.4 * mass * radius**2)
            for path in ["/World/Pendulum/base", "/World/Pendulum/bob"]:
                UsdGeom.Xformable(stage.GetPrimAtPath(path)).AddTranslateOp().Set(
                    Gf.Vec3d(
                        0.0,
                        0.0,
                        1.0
                        - (
                            length
                            if path.endswith("/bob") and args.body_frame == "com"
                            else 0.0
                        ),
                    )
                )
            shape = UsdGeom.Sphere.Define(stage, "/World/Pendulum/bob/visual")
            shape.CreateRadiusAttr(radius)
            shape.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, com_z))
            fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Pendulum/fixed")
            fixed.CreateBody1Rel().SetTargets(["/World/Pendulum/base"])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 1.0))
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Pendulum/pivot")
            joint.CreateBody0Rel().SetTargets(["/World/Pendulum/base"])
            joint.CreateBody1Rel().SetTargets(["/World/Pendulum/bob"])
            joint.CreateAxisAttr("Y")
            joint.CreateLocalPos0Attr(Gf.Vec3f(0.0))
            joint.CreateLocalPos1Attr(
                Gf.Vec3f(0.0, 0.0, length if args.body_frame == "com" else 0.0)
            )
            jprops = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
            jprops.CreateJointFrictionAttr(0.0)
            jprops.CreateArmatureAttr(0.0)
            UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
            art = PhysxSchema.PhysxArticulationAPI.Apply(fixed.GetPrim())
            art.CreateSleepThresholdAttr(0.0)
            art.CreateStabilizationThresholdAttr(0.0)
            art.CreateSolverPositionIterationCountAttr(args.position_iterations)
            art.CreateSolverVelocityIterationCountAttr(args.velocity_iterations)
            obj = Articulation(
                ArticulationCfg(
                    prim_path="/World/Pendulum",
                    spawn=None,
                    actuators={},
                    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                )
            )
        recorder = None
        if args.record:
            from scripts.benchmark.tech_reports.physics.platforms.isaaclab.p01_capture import (
                NativeCapture,
            )

            recorder = NativeCapture(sim, args.case, out, args.dt, args.common_view)
        sim.reset()
        meta["backend_class"] = type(obj).__module__ + "." + type(obj).__name__
        meta["effective_config"] = sim.cfg.to_dict()
        meta["physics_scene_attributes"] = {
            str(prim.GetPath()): {
                str(a.GetName()): a.Get() for a in prim.GetAttributes()
            }
            for prim in stage.Traverse()
            if prim.IsA(UsdPhysics.Scene)
        }
        meta["effective_body_mass"] = obj.data.body_mass.torch.cpu().tolist()
        meta["effective_body_inertia"] = obj.data.body_inertia.torch.cpu().tolist()
        meta["effective_body_com"] = obj.data.body_com_pose_b.torch.cpu().tolist()
        if args.case == "pendulum":
            meta["joint_names"] = obj.joint_names
            meta["body_names"] = obj.body_names
            meta["fixed_base"] = obj.is_fixed_base
            assert obj.num_joints == 1 and obj.is_fixed_base
            meta["effective_joint_friction"] = (
                obj.data.joint_friction_coeff.torch.cpu().tolist()
            )
            meta["effective_joint_armature"] = (
                obj.data.joint_armature.torch.cpu().tolist()
            )
            meta["effective_joint_stiffness"] = (
                obj.data.joint_stiffness.torch.cpu().tolist()
            )
            meta["effective_joint_damping"] = (
                obj.data.joint_damping.torch.cpu().tolist()
            )
            assert np.allclose(meta["effective_joint_friction"], 0.0)
            assert np.allclose(meta["effective_joint_armature"], 0.0)
            assert np.allclose(meta["effective_joint_stiffness"], 0.0)
            assert np.allclose(meta["effective_joint_damping"], 0.0)
            assert np.allclose(
                meta["effective_body_inertia"][0][1],
                np.diag([0.4 * mass * radius**2] * 3).reshape(-1),
                rtol=1e-6,
                atol=0.0,
            )
            assert np.allclose(meta["effective_body_com"][0][1][:3], [0.0, 0.0, com_z])
        assert np.allclose(meta["effective_body_mass"], mass, rtol=1e-6, atol=0.0)
        stage.Export(str(out / "scene.usda"))
        steps = round(args.duration / args.dt)
        assert abs(steps * args.dt - args.duration) < 1e-9
        t = np.arange(steps + 1, dtype=np.float64) * args.dt
        episode_times = []
        for episode in range(args.count):
            if args.case == "freefall":
                initial = inputs["freefall_z_m"][episode]
                pose = obj.data.default_root_pose.torch.clone()
                pose[:, :3] = torch.tensor([[0.0, 0.0, initial]], device=sim.device)
                obj.write_root_pose_to_sim_index(root_pose=pose)
                obj.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=sim.device)
                )
            else:
                initial = (
                    inputs["pendulum_theta_rad"][episode]
                    if args.theta_deg is None
                    else np.deg2rad(args.theta_deg)
                )
                obj.write_joint_position_to_sim_index(
                    position=torch.tensor(
                        [[initial]], device=sim.device, dtype=torch.float32
                    )
                )
                obj.write_joint_velocity_to_sim_index(
                    velocity=torch.zeros((1, 1), device=sim.device)
                )
            obj.reset()
            states = np.empty((steps + 1, 2), dtype=np.float64)
            com_trace = (
                np.empty((steps + 1, 3), dtype=np.float64)
                if args.case == "pendulum"
                else None
            )
            body_velocity = (
                np.empty((steps + 1, 6), dtype=np.float64)
                if args.probe_body_velocity
                else None
            )
            if sim.device.startswith("cuda"):
                torch.cuda.synchronize()
            tic = time.perf_counter()
            for k in range(steps + 1):
                if k:
                    sim.step(render=False)
                    obj.update(args.dt)
                if args.case == "freefall":
                    states[k] = [
                        obj.data.root_pos_w.torch[0, 2].item(),
                        obj.data.root_lin_vel_w.torch[0, 2].item(),
                    ]
                else:
                    states[k] = [
                        obj.data.joint_pos.torch[0, 0].item(),
                        obj.data.joint_vel.torch[0, 0].item(),
                    ]
                    com_trace[k] = (
                        obj.data.body_com_pos_w.torch[0, obj.body_names.index("bob")]
                        .cpu()
                        .numpy()
                    )
                if body_velocity is not None:
                    assert args.case == "pendulum"
                    body_velocity[k] = (
                        obj.data.body_com_vel_w.torch[0, obj.body_names.index("bob")]
                        .cpu()
                        .numpy()
                    )
                if recorder is not None:
                    recorder.capture(k, states[k])
            if sim.device.startswith("cuda"):
                torch.cuda.synchronize()
            episode_times.append(time.perf_counter() - tic)
            assert np.isfinite(states).all(), f"Nonfinite state in episode {episode}"
            arrays = dict(t=t, state=states, initial=np.array(initial))
            if com_trace is not None:
                arrays["com_position"] = com_trace
            if body_velocity is not None:
                arrays["body_com_velocity"] = body_velocity
            np.savez_compressed(out / f"episode_{episode:02d}.npz", **arrays)
            meta["completed_episodes"] = episode + 1
            (out / "run.json").write_text(
                json.dumps(meta, indent=2, default=str) + "\n"
            )
            print(
                f"P01 {args.case} dt={args.dt} episode={episode} initial={states[0]} final={states[-1]}",
                flush=True,
            )
        meta["diagnostic_wall_s_per_episode"] = episode_times
        if recorder is not None:
            meta["native_capture"] = recorder.finish()
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["failure"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        meta["ended_at"] = now()
        meta["wall_s_including_startup"] = time.perf_counter() - started
        meta["process_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        )
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
