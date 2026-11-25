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

"""L-03 common-USD model preflight in the actual benchmark environments.

Run with --platform, --manifest, --robot, --num-envs and a fresh --output.
This is a model/first-step probe, not a loading-performance sample.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import traceback

__all__ = ["main"]


def main() -> None:
    """Read the same frozen asset, prepare native replicas and save all states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform", choices=("isaaclab", "embodichain"), required=True
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--num-envs", type=int, choices=(1, 4), required=True)
    parser.add_argument(
        "--init-mode", choices=("joints", "pose", "full"), default="full"
    )
    parser.add_argument("--warmup-joint-state", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--explicit-cleanup", action="store_true")
    parser.add_argument("--overlap", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.steps >= 1
    assert not args.warmup_joint_state or args.platform == "isaaclab"
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(args.manifest.read_text())["robots"][args.robot]
    if args.overlap:
        assert args.num_envs == 4
        cfg["env_spacing"] = 0.0
    shutil.copy2(__file__, args.output / "source.py")
    (args.output / "input.json").write_text(json.dumps(cfg, indent=2) + "\n")
    meta = {
        "experiment": "L-03",
        "phase": "model_and_first_step_preflight",
        "performance_eligible": False,
        "platform": args.platform,
        "robot": args.robot,
        "num_envs": args.num_envs,
        "status": "running",
        "started_at": datetime.now().astimezone().isoformat(),
        "python": sys.executable,
        "command": sys.argv,
        "dt_s": 0.005,
        "gravity": [0, 0, 0],
        "rendering": False,
        "quaternion_order": "xyzw",
        "dexuni_sleep": True if args.platform == "embodichain" else None,
        "init_mode": args.init_mode,
        "warmup_joint_state": args.warmup_joint_state,
        "recorded_physics_steps": args.steps,
        "explicit_cleanup": args.explicit_cleanup,
        "overlap_isolation_probe": args.overlap,
        "source_snapshot_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }

    def save() -> None:
        (args.output / "run.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n"
        )

    save()
    app = None
    try:
        for item in cfg["files"]:
            assert (
                hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
                == item["sha256"]
            )
        if args.platform == "isaaclab":
            from isaaclab.app import AppLauncher

            app = AppLauncher(
                headless=True, visualizer="none", enable_cameras=False
            ).app
        import numpy as np
        import torch
        import warp as wp

        trace = {}
        if args.platform == "embodichain":
            from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
            from embodichain.lab.sim.cfg import (
                ArticulationCfg,
                ArticulationRootPropertiesCfg,
                JointDrivePropertiesCfg,
                NewtonPhysicsCfg,
                RenderCfg,
                UsdSourceCfg,
            )

            sim = SimulationManager(
                SimulationManagerCfg(
                    headless=True,
                    num_envs=args.num_envs,
                    arena_space=cfg["env_spacing"],
                    physics_cfg=NewtonPhysicsCfg(
                        device="cuda:0",
                        gravity=(0, 0, 0),
                        physics_dt=0.005,
                        num_substeps=1,
                        use_cuda_graph=True,
                        solver_cfg={
                            "solver_type": "dexuni",
                            "joint_mode": "dynamic",
                            "vbd_options": {
                                "iterations": 100,
                                "rigid_enable_sleep": True,
                            },
                            "collision_options": {"rigid_contact_max": 4096},
                        },
                    ),
                    render_cfg=RenderCfg(renderer="no-render"),
                )
            )
            robot = sim.add_articulation(
                ArticulationCfg(
                    uid="robot",
                    source=UsdSourceCfg(file_path=cfg["usd_path"]),
                    init_pos=cfg["initial_position"],
                    asset_physics_mode="overlay",
                    root_props=ArticulationRootPropertiesCfg(
                        fixed_base=cfg["fixed_base"], self_collision_enabled=False
                    ),
                    joint_drive_props=JointDrivePropertiesCfg(
                        drive_type="none",
                        target_mode="none",
                        stiffness=0.0,
                        damping=0.0,
                        friction=0.0,
                        armature=0.0,
                    ),
                )
            )
            meta["stage"] = "declared"
            save()
            sim.prepare()
            origins = sim.arena_offsets
            names = robot.joint_names
            links = robot.link_names
            q = torch.tensor(
                [[cfg["initial_qpos"][n] for n in names]], device=sim.device
            ).repeat(args.num_envs, 1)

            def sample() -> dict:
                return {
                    "qpos": robot.get_qpos().cpu().numpy().copy(),
                    "qvel": robot.get_qvel().cpu().numpy().copy(),
                    "link_pose": torch.stack(
                        [robot.get_link_pose(n) for n in links], dim=1
                    )
                    .cpu()
                    .numpy()
                    .copy(),
                    "root_pose": robot.get_local_pose().cpu().numpy().copy(),
                    "root_velocity": robot.body_data.root_vel.cpu().numpy().copy(),
                }

            trace["prepared"] = sample()
            robot.set_qpos(q, target=False)
            trace["joint_position_written"] = sample()
            robot.set_qvel(torch.zeros_like(q), target=False)
            trace["joint_velocity_written"] = sample()
            if args.init_mode in ("pose", "full"):
                root_pose = torch.tensor(
                    [cfg["initial_position"] + [0, 0, 0, 1]], device=sim.device
                ).repeat(args.num_envs, 1)
                robot.set_local_pose(root_pose)
                trace["root_pose_written"] = sample()
            if args.init_mode == "full":
                robot.set_root_velocity(
                    torch.zeros((args.num_envs, 6), device=sim.device)
                )
                trace["root_velocity_written"] = sample()
            initial = sample()
            for index in range(args.steps):
                sim.update(step=1, render_final_step=False)
                trace[f"step_{index + 1:04d}"] = sample()
            first = trace["step_0001"]
            arrays = {
                "mass": robot.get_mass().cpu().numpy(),
                "inertia_principal": robot.get_inertia().cpu().numpy(),
                "com_pose": robot.get_com_pose().cpu().numpy(),
                "joint_limits": robot.get_qpos_limits().cpu().numpy(),
                "env_origins": origins.cpu().numpy(),
            }
            from dexsim.engine.newton_physics.backend_registry import get_newton_backend

            backend = get_newton_backend(sim.get_world())
            model = backend.model
            for key in (
                "body_world",
                "body_mass",
                "body_q",
                "shape_type",
                "shape_scale",
                "shape_body",
                "shape_world",
                "shape_flags",
                "shape_transform",
                "joint_type",
                "joint_parent",
                "joint_child",
            ):
                arrays["native_" + key] = getattr(model, key).numpy()
            meta.update(
                instance_count=robot.num_instances,
                arena_names=[sim.get_env(i).get_name() for i in range(sim.num_envs)],
                native_body_count=model.body_count,
                native_joint_dof_count=model.joint_dof_count,
                native_joint_labels=list(model.joint_label),
                native_shape_labels=list(model.shape_label),
                vbd_active=backend.solver.vbd_solver is not None,
                effective_vbd_sleep=(
                    backend.solver.vbd_solver._rigid_sleep is not None
                    if backend.solver.vbd_solver is not None
                    else None
                ),
                articulation_solver=type(backend.solver.mujoco_solver).__name__,
                config=sim.sim_config.to_dict(),
            )
        else:
            import isaaclab.sim as su
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.assets import ArticulationCfg
            from isaaclab.actuators import ImplicitActuatorCfg
            from isaaclab_physx.physics import PhysxCfg

            sim = su.SimulationContext(
                su.SimulationCfg(
                    dt=0.005,
                    gravity=(0, 0, 0),
                    device="cuda:0",
                    physics=PhysxCfg(
                        solver_type=1,
                        min_position_iteration_count=4,
                        max_position_iteration_count=4,
                        min_velocity_iteration_count=1,
                        max_velocity_iteration_count=1,
                    ),
                    visualizer_cfgs=[],
                    use_newton_actuators=False,
                    log_dir=str(args.output / "sim_logs"),
                )
            )
            scene_cfg = InteractiveSceneCfg(
                num_envs=args.num_envs,
                env_spacing=cfg["env_spacing"],
                replicate_physics=True,
                filter_collisions=True,
            )
            scene_cfg.robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=su.UsdFileCfg(
                    usd_path=cfg["usd_path"],
                    articulation_props=su.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(cfg["initial_position"]), joint_pos=cfg["initial_qpos"]
                ),
                actuators={
                    "passive": ImplicitActuatorCfg(
                        joint_names_expr=[".*"],
                        stiffness=0.0,
                        damping=0.0,
                        friction=0.0,
                        armature=0.0,
                    )
                },
            )
            scene = InteractiveScene(scene_cfg)
            robot = scene["robot"]
            if args.warmup_joint_state:
                from pxr import PhysxSchema, UsdPhysics

                edits = []
                for prim in sim.stage.Traverse():
                    if prim.GetName() not in cfg["initial_qpos"]:
                        continue
                    if prim.IsA(UsdPhysics.RevoluteJoint):
                        axis, value = "angular", float(
                            np.rad2deg(cfg["initial_qpos"][prim.GetName()])
                        )
                    elif prim.IsA(UsdPhysics.PrismaticJoint):
                        axis, value = "linear", cfg["initial_qpos"][prim.GetName()]
                    else:
                        continue
                    state = PhysxSchema.JointStateAPI.Apply(prim, axis)
                    state.CreatePositionAttr(value)
                    state.CreateVelocityAttr(0.0)
                    edits.append(
                        {"path": str(prim.GetPath()), "axis": axis, "position": value}
                    )
                meta["diagnostic_stage_joint_state_edits"] = edits
                assert len(edits) == args.num_envs * len(cfg["initial_qpos"])
            meta["stage"] = "declared"
            save()
            sim.reset()
            scene.update(0)
            names = robot.joint_names
            links = robot.body_names
            origins = scene.env_origins
            q = torch.tensor(
                [[cfg["initial_qpos"][n] for n in names]], device=sim.device
            ).repeat(args.num_envs, 1)

            def sample() -> dict:
                poses = robot.data.body_link_pose_w.torch.clone()
                poses[..., :3] -= origins[:, None, :]
                root = robot.data.root_pose_w.torch.clone()
                root[:, :3] -= origins
                return {
                    "qpos": robot.data.joint_pos.torch.cpu().numpy().copy(),
                    "qvel": robot.data.joint_vel.torch.cpu().numpy().copy(),
                    "link_pose": poses.cpu().numpy(),
                    "root_pose": root.cpu().numpy(),
                    "root_velocity": robot.data.root_link_vel_w.torch.cpu()
                    .numpy()
                    .copy(),
                }

            trace["prepared"] = sample()
            robot.write_joint_position_to_sim_index(position=q)
            trace["joint_position_written"] = sample()
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            trace["joint_velocity_written"] = sample()
            if args.init_mode in ("pose", "full"):
                root_pose = torch.tensor(
                    [cfg["initial_position"] + [0, 0, 0, 1]], device=sim.device
                ).repeat(args.num_envs, 1)
                root_pose[:, :3] += origins
                robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
                trace["root_pose_written"] = sample()
            if args.init_mode == "full":
                robot.write_root_link_velocity_to_sim_index(
                    root_velocity=torch.zeros((args.num_envs, 6), device=sim.device)
                )
                trace["root_velocity_written"] = sample()
            initial = sample()
            for index in range(args.steps):
                sim.step(render=False)
                scene.update(0.005)
                trace[f"step_{index + 1:04d}"] = sample()
            first = trace["step_0001"]
            com = robot.data.body_com_pose_b.torch.clone()
            arrays = {
                "mass": robot.data.body_mass.torch.cpu().numpy(),
                "inertia_matrix": robot.data.body_inertia.torch.cpu().numpy(),
                "com_pose": com.cpu().numpy(),
                "joint_limits": robot.data.joint_pos_limits.torch.cpu().numpy(),
                "env_origins": origins.cpu().numpy(),
            }
            meta.update(
                instance_count=robot.num_instances,
                native_shapes_per_articulation=robot.root_view.max_shapes,
                arena_names=scene.env_prim_paths,
                effective_fixed_base=robot.is_fixed_base,
                config=sim.cfg.to_dict(),
            )
        from pxr import Usd, UsdPhysics

        source_stage = Usd.Stage.Open(cfg["usd_path"])
        colliders = []
        for prim in Usd.PrimRange(
            source_stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()
        ):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            colliders.append(
                {
                    "path": str(prim.GetPath()),
                    "type": str(prim.GetTypeName()),
                    "enabled": UsdPhysics.CollisionAPI(prim)
                    .GetCollisionEnabledAttr()
                    .Get(),
                    "approximation": (
                        str(
                            UsdPhysics.MeshCollisionAPI(prim)
                            .GetApproximationAttr()
                            .Get()
                        )
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                        else None
                    ),
                    "child_geometry": [
                        {
                            "path": str(p.GetPath()),
                            "type": str(p.GetTypeName()),
                            "collision_api": p.HasAPI(UsdPhysics.CollisionAPI),
                        }
                        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                        if p.GetTypeName()
                        in ("Mesh", "Cube", "Sphere", "Cylinder", "Capsule")
                    ],
                }
            )
        (args.output / "source-colliders.json").write_text(
            json.dumps(colliders, indent=2) + "\n"
        )
        wp.synchronize()
        torch.cuda.synchronize()
        np.savez(
            args.output / "states.npz",
            **{"initial_" + k: v for k, v in initial.items()},
            **{"first_" + k: v for k, v in first.items()},
        )
        np.savez(args.output / "model.npz", **arrays)
        np.savez(
            args.output / "initialization-trace.npz",
            **{
                f"{stage}_{key}": value
                for stage, state in trace.items()
                for key, value in state.items()
            },
        )
        meta["trace_stages"] = list(trace)
        meta.update(
            joint_names=names,
            link_names=links,
            first_state_shapes={k: list(v.shape) for k, v in first.items()},
            mass_per_env_kg=arrays["mass"].sum(axis=1).tolist(),
            max_joint_change=float(np.max(np.abs(first["qpos"] - initial["qpos"]))),
            max_link_translation_change_m=float(
                np.max(
                    np.linalg.norm(
                        first["link_pose"][..., :3] - initial["link_pose"][..., :3],
                        axis=-1,
                    )
                )
            ),
        )
        save()
        assert sorted(names) == cfg["expected_joint_names"], names
        assert meta["instance_count"] == args.num_envs
        assert len(set(meta["arena_names"])) == args.num_envs
        assert first["qpos"].shape == (args.num_envs, len(names))
        assert all(np.isfinite(a).all() for a in first.values())
        if args.platform == "embodichain" and args.explicit_cleanup:
            sim.destroy(exit_process=False)
            meta["cleanup_queued"] = True
        meta["status"] = "completed"
    except BaseException:
        meta["status"] = "failed"
        meta["failure"] = traceback.format_exc()
        raise
    finally:
        meta["ended_at"] = datetime.now().astimezone().isoformat()
        meta["module_origins"] = {
            n: getattr(sys.modules.get(n), "__file__", None)
            for n in ("isaaclab", "isaacsim", "embodichain", "dexsim", "newton", "warp")
        }
        meta["packages"] = {}
        for name in ("isaacsim", "isaaclab", "dexsim-engine", "newton", "warp-lang"):
            try:
                meta["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        save()
        print("L03_PREFLIGHT", meta["status"], flush=True)
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
    # main's native child references must unwind before public deferred cleanup.
    if "embodichain.lab.sim" in sys.modules:
        import gc
        from embodichain.lab.sim import SimulationManager

        gc.collect()
        SimulationManager.flush_cleanup_queue()
        print("L03_CLEANUP_RETURNED", flush=True)
