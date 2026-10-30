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

"""P-01 through EmbodiChain public APIs, with an immutable directory per run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
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
    parser.add_argument("--case", choices=["freefall", "pendulum"], required=True)
    parser.add_argument("--dt", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--phase", choices=["pilot", "main", "supplement", "capture"], default="main"
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--common-view", type=Path)
    args = parser.parse_args()
    assert args.common_view is None or args.record
    cfg_path = PROJECT / "configs/P-01/embodichain-dexuni.json"
    cfg = json.loads(cfg_path.read_text())
    inputs_path = PROJECT / cfg["initial_states"]
    inputs = json.loads(inputs_path.read_text())
    assert 1 <= args.count <= 10 and args.dt > 0 and args.duration > 0
    assert not args.record or (args.count == 1 and args.phase == "capture")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(f"Existing run: {out}")
    snapshot = out / "config"
    snapshot.mkdir()
    sources = [
        Path(__file__),
        cfg_path,
        inputs_path,
        PROJECT / "assets/P-01/pendulum.urdf",
        PROJECT / "specs/p-01-dynamics-spec.md",
    ]
    if args.common_view is not None:
        sources += [
            args.common_view,
            PROJECT / "assets/P-01/pendulum-display.urdf",
            Path(__file__).with_name("p01_common_capture.py"),
        ]
    hashes = {}
    for path in sources:
        shutil.copy2(path, snapshot / path.name)
        hashes[str(path.relative_to(PROJECT))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    now = lambda: datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    meta = dict(
        run_id=out.name,
        experiment="P-01",
        platform="EmbodiChain",
        solver="DexUni",
        phase=args.phase,
        case=args.case,
        dt_s=args.dt,
        duration_s=args.duration,
        initial_count=args.count,
        num_envs=1,
        num_substeps=cfg["num_substeps"],
        solver_dt_s=args.dt / cfg["num_substeps"],
        device="cuda:0",
        command=[sys.executable, *sys.argv],
        started_at=now(),
        status="running",
        completed_episodes=0,
        hashes=hashes,
        python=sys.version,
        pythonpath=os.environ.get("PYTHONPATH"),
        performance_benchmark=False,
        ground_note="EmbodiChain creates a mandatory ground; all test-body colliders are disabled. No ground contact can occur.",
    )
    repo = Path(cfg["repository"])
    meta["repository_commit"] = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    meta["repository_status"] = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )
    meta["source_version"] = (repo / "VERSION").read_text().strip()
    meta["packages"] = {
        name: importlib.metadata.version(name)
        for name in [
            "embodichain",
            "dexsim-engine",
            "newton",
            "warp-lang",
            "torch",
            "numpy",
            "mujoco",
            "mujoco-warp",
        ]
    }
    meta["module_origins"] = {
        name: importlib.util.find_spec(name).origin
        for name in ["embodichain", "embodichain_robots", "embodichain_tasks"]
    }
    assert all(str(repo) in path for path in meta["module_origins"].values()), meta[
        "module_origins"
    ]
    meta["gpu_inventory"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    started = time.perf_counter()
    sim = None
    try:
        import numpy as np
        import torch
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.cfg import (
            ArticulationCfg,
            ArticulationRootPropertiesCfg,
            CollisionPropertiesCfg,
            DenoisingCfg,
            MassPropertiesCfg,
            NewtonJointDrivePropertiesCfg,
            NewtonPhysicsCfg,
            RenderCfg,
            RigidBodyPhysicsCfg,
            RigidObjectCfg,
        )
        from embodichain.lab.sim.shapes import CubeCfg
        from embodichain.lab.sim.material import VisualMaterialCfg

        solver_cfg = cfg["solver_cfg"].copy()
        solver_cfg["mujoco_options"] = dict(solver_cfg["mujoco_options"])
        solver_cfg["mujoco_options"]["wind"] = tuple(
            solver_cfg["mujoco_options"]["wind"]
        )
        if args.case == "pendulum":
            solver_cfg["mujoco_options"]["save_to_mjcf"] = str(
                out / "effective_model.xml"
            )
        else:
            # DexUni's pure VBD path rejects unused articulation solver options.
            solver_cfg.pop("mujoco_options")
        sim_cfg = SimulationManagerCfg(
            headless=True,
            num_envs=1,
            width=640,
            height=480,
            startup_summary="full",
            dexsim_startup_info=True,
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=args.dt,
                gravity=[0.0, 0.0, -cfg["gravity_m_s2"]],
                num_substeps=cfg["num_substeps"],
                use_cuda_graph=True,
                collision_cfg=None,
                solver_cfg=solver_cfg,
            ),
            render_cfg=RenderCfg(
                renderer="hybrid" if args.record else "no-render",
                denoising=DenoisingCfg(
                    window="off", offscreen="optix" if args.common_view else "off"
                ),
                spp=8 if args.common_view else 1,
            ),
        )
        meta["requested_sim_config"] = sim_cfg.to_dict()
        sim = SimulationManager(sim_cfg)
        m, length, radius = (
            cfg[k] for k in ["mass_kg", "pendulum_length_m", "bob_radius_m"]
        )
        if args.case == "freefall":
            obj = sim.add_rigid_object(
                RigidObjectCfg(
                    uid="falling_cube",
                    shape=CubeCfg(
                        size=[cfg["cube_edge_m"]] * 3,
                        visual_material=(
                            VisualMaterialCfg(
                                uid="test_body",
                                base_color=json.loads(args.common_view.read_text())[
                                    "body_color"
                                ],
                            )
                            if args.common_view
                            else None
                        ),
                    ),
                    init_pos=(0.0, 0.0, 1.0),
                    attrs=RigidBodyPhysicsCfg(
                        mass_props=MassPropertiesCfg(
                            mass=m,
                            inertia=[m * cfg["cube_edge_m"] ** 2 / 6] * 3,
                            com_position=[0.0] * 3,
                        ),
                        collision_props=CollisionPropertiesCfg(collision_enabled=False),
                    ),
                )
            )
        else:
            obj = sim.add_articulation(
                ArticulationCfg(
                    uid="pendulum",
                    fpath=str(
                        PROJECT
                        / (
                            "assets/P-01/pendulum-display.urdf"
                            if args.common_view
                            else "assets/P-01/pendulum.urdf"
                        )
                    ),
                    init_pos=(0.0, 0.0, 1.0),
                    asset_physics_mode="overlay",
                    build_pk_chain=False,
                    attrs=RigidBodyPhysicsCfg(
                        collision_props=CollisionPropertiesCfg(collision_enabled=False)
                    ),
                    root_props=ArticulationRootPropertiesCfg(
                        fixed_base=True,
                        self_collision_enabled=False,
                        newton_gravity_compensation=0.0,
                    ),
                    joint_drive_props=NewtonJointDrivePropertiesCfg(
                        target_mode="none",
                        drive_type="none",
                        stiffness=0.0,
                        damping=0.0,
                        friction=0.0,
                        armature=0.0,
                        passive_damping=0.0,
                    ),
                )
            )
        camera, recorder = None, None
        if args.common_view is not None:
            from scripts.benchmark.tech_reports.physics.platforms.embodichain.p01_common_capture import (
                CommonCapture,
            )

            recorder = CommonCapture(sim, args.case, out, args.dt, args.common_view)
        elif args.record:
            from embodichain.lab.sim.sensors import CameraCfg

            sim.set_ground_plane_visibility(False)
            camera = sim.add_sensor(
                CameraCfg(
                    uid="report_camera",
                    width=640,
                    height=480,
                    intrinsics=(480.0, 480.0, 320.0, 240.0),
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=(0.0, -1.9, 0.55),
                        target=(0.0, 0.0, 0.55),
                        up=(0.0, 0.0, 1.0),
                    ),
                )
            )
            (out / "native_camera").mkdir()
        sim.prepare()
        meta["resolved_solver"] = sim.physics.solver_type
        assert meta["resolved_solver"] == "dexuni"
        meta["mass_kg_readback"] = obj.get_mass().cpu().tolist()
        meta["inertia_kg_m2_readback"] = obj.get_inertia().cpu().tolist()
        if args.case == "pendulum":
            meta["com_pose_readback"] = obj.get_com_pose().cpu().tolist()
            meta["link_names"] = obj.link_names
            meta["joint_names"] = obj.joint_names
            meta["joint_drive_readback"] = {
                key: value.cpu().tolist()
                for key, value in zip(
                    [
                        "stiffness",
                        "damping",
                        "max_effort",
                        "max_velocity",
                        "friction",
                        "armature",
                    ],
                    obj.get_joint_drive(),
                )
            }
            meta["joint_target_mode_readback"] = obj.get_joint_target_mode()
            assert obj.dof == 1
            assert np.allclose(obj.get_mass("bob").cpu(), m, rtol=1e-6, atol=0)
            assert np.allclose(
                obj.get_inertia("bob").cpu(), 0.4 * m * radius**2, rtol=1e-6, atol=0
            )
            assert np.allclose(obj.get_com_pose("bob").cpu()[0, 0, :3], [0, 0, -length])
            assert np.allclose(meta["joint_target_mode_readback"], 0)
            for key in ["stiffness", "damping", "friction", "armature"]:
                assert np.allclose(meta["joint_drive_readback"][key], 0)
        else:
            assert np.allclose(meta["mass_kg_readback"], m, rtol=1e-6, atol=0)
            assert np.allclose(
                meta["inertia_kg_m2_readback"],
                m * cfg["cube_edge_m"] ** 2 / 6,
                rtol=1e-6,
                atol=0,
            )
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
        steps = round(args.duration / args.dt)
        assert abs(steps * args.dt - args.duration) < 1e-10
        t = np.arange(steps + 1, dtype=np.float64) * args.dt
        np.save(out / "time_s.npy", t)
        episode_times, frames = [], []
        for episode in range(args.count):
            sim.reset_objects_state()
            initial = inputs[
                "freefall_z_m" if args.case == "freefall" else "pendulum_theta_rad"
            ][episode]
            if args.case == "freefall":
                obj.set_local_pose(
                    torch.tensor(
                        [[0, 0, initial, 0, 0, 0, 1]],
                        dtype=torch.float32,
                        device=sim.device,
                    )
                )
                obj.set_velocity(
                    lin_vel=torch.zeros((1, 3), device=sim.device),
                    ang_vel=torch.zeros((1, 3), device=sim.device),
                )
            else:
                obj.set_qpos(
                    torch.tensor([[initial]], dtype=torch.float32, device=sim.device),
                    target=False,
                )
                obj.set_qvel(torch.zeros((1, 1), device=sim.device), target=False)
            # Memory-mapped raw arrays retain completed samples even after interruption.
            state = np.lib.format.open_memmap(
                out / f"episode_{episode:02d}_state.npy",
                mode="w+",
                dtype=np.float64,
                shape=(steps + 1, 2),
            )
            pose = np.lib.format.open_memmap(
                out / f"episode_{episode:02d}_pose.npy",
                mode="w+",
                dtype=np.float64,
                shape=(steps + 1, 7),
            )
            com = np.lib.format.open_memmap(
                out / f"episode_{episode:02d}_com.npy",
                mode="w+",
                dtype=np.float64,
                shape=(steps + 1, 3),
            )
            state[:] = pose[:] = com[:] = np.nan
            torch.cuda.synchronize()
            tic = time.perf_counter()
            for k in range(steps + 1):
                if k:
                    sim.update(step=1, render_final_step=False)
                if args.case == "freefall":
                    pose[k] = obj.get_local_pose()[0].cpu().numpy()
                    state[k] = [pose[k, 2], obj.body_data.lin_vel[0, 2].item()]
                    com[k] = pose[k, :3]
                else:
                    state[k] = [
                        obj.get_qpos()[0, 0].item(),
                        obj.get_qvel()[0, 0].item(),
                    ]
                    pose[k] = obj.get_link_pose("bob")[0].cpu().numpy()
                    qx, qy, qz, qw = pose[k, 3:]
                    com[k] = pose[k, :3] - length * np.array(
                        [
                            2 * (qx * qz + qw * qy),
                            2 * (qy * qz - qw * qx),
                            1 - 2 * (qx * qx + qy * qy),
                        ]
                    )
                if recorder is not None:
                    recorder.capture(k, state[k])
                if (
                    camera is not None
                    and k > 0
                    and k % round(0.025 / args.dt) == 0
                    and (args.case == "pendulum" or t[k] <= 0.4 + 1e-12)
                ):
                    from PIL import Image

                    sim.render_camera_group([camera.group_id])
                    camera.update(fetch_only=True)
                    name = f"frame_{len(frames):05d}.png"
                    Image.fromarray(camera.get_data()["color"][0].cpu().numpy()).save(
                        out / "native_camera" / name
                    )
                    frames.append(
                        dict(
                            file=name,
                            step=k,
                            time_s=float(t[k]),
                            state=state[k].tolist(),
                        )
                    )
                if k % 1000 == 0 or k == steps:
                    state.flush()
                    pose.flush()
                    com.flush()
                    meta["last_sample"] = dict(
                        episode=episode, step=k, time_s=float(t[k])
                    )
                    (out / "run.json").write_text(
                        json.dumps(meta, indent=2, default=str) + "\n"
                    )
            torch.cuda.synchronize()
            episode_times.append(time.perf_counter() - tic)
            assert np.isfinite(state).all() and np.isfinite(pose).all()
            assert state[0, 1] == 0.0
            np.savez_compressed(
                out / f"episode_{episode:02d}.npz",
                t=t,
                state=state,
                initial=np.array(initial),
                body_pose=pose,
                com_position=com,
            )
            meta["completed_episodes"] = episode + 1
            print(
                f"P01 {args.case} dt={args.dt} episode={episode} initial={state[0]} final={state[-1]} wall={episode_times[-1]:.2f}s",
                flush=True,
            )
        meta["diagnostic_wall_s_per_episode"] = episode_times
        meta["cuda_graph_status"] = sim.physics.cuda_graph_status
        if camera is not None:
            (out / "native_camera/frames.json").write_text(
                json.dumps(dict(fps=40, playback_speed=1, frames=frames), indent=2)
                + "\n"
            )
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
        if sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
