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

"""Probe a local Franka USD using each platform's public articulation API.

Run in the corresponding environment with --platform and a fresh --output.
This is an asset/drive preflight, not the P-03 grasp experiment.
"""

from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--platform", choices=["embodichain", "isaaclab"], required=True
    )
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--control-file",
        type=Path,
        help="Replay only the leader finger from prepared P-03 commands; hold the arm.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert not (args.output / "run.json").exists()
    shutil.copy2(__file__, args.output / "source.py")
    meta = dict(
        experiment="P-03",
        phase="asset-drive-preflight",
        status="running",
        platform=args.platform,
        started_at=datetime.now().astimezone().isoformat(),
        command=sys.argv,
        executable=sys.executable,
        asset=str(args.asset),
        asset_sha256=hashlib.sha256(args.asset.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(
            [
                "git",
                "-C",
                (
                    "/home/oem/code/EmbodiChain"
                    if args.platform == "embodichain"
                    else "/home/oem/code/IsaacLab"
                ),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip(),
    )

    def save() -> None:
        (args.output / "run.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n"
        )

    save()
    sim = None
    app = None
    try:
        import numpy as np
        import torch

        meta["packages"] = {
            name: importlib.metadata.version(name)
            for name in (
                ["dexsim-engine", "newton", "warp-lang", "torch"]
                if args.platform == "embodichain"
                else ["isaaclab", "isaacsim", "warp-lang", "torch"]
            )
        }
        leader_targets = None
        samples = 301
        if args.control_file:
            shutil.copy2(args.control_file, args.output / "commands.npz")
            commands = np.load(args.control_file)
            np.testing.assert_allclose(np.diff(commands["physics_t"]), 0.005)
            np.testing.assert_allclose(np.diff(commands["control_t"]), 0.02)
            leader_targets = commands["q_target"][0, :, 7]
            samples = len(commands["physics_t"])
            meta["control_file"] = str(args.control_file)
            meta["control_sha256"] = hashlib.sha256(
                args.control_file.read_bytes()
            ).hexdigest()
            meta["control_scope"] = (
                "Initial 0 leader finger only, ZOH at 20 ms. Arm holds the "
                "original preflight pose; no target object or external force."
            )
        meta["dt_s"] = 0.005
        meta["samples"] = samples
        prefix = "fr3" if args.asset.suffix == ".urdf" else "panda"
        initial = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]
        names = [f"{prefix}_joint{i}" for i in range(1, 8)] + [
            f"{prefix}_finger_joint1",
            f"{prefix}_finger_joint2",
        ]
        if args.platform == "embodichain":
            import dexsim.kit.usd.parser as usd_parser

            parser_path = Path(usd_parser.__file__)
            meta["usd_parser"] = {
                "path": str(parser_path),
                "sha256": hashlib.sha256(parser_path.read_bytes()).hexdigest(),
                "has_mimic_translation": hasattr(
                    usd_parser, "_apply_physics_mimic_joints"
                ),
            }
            from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
            from embodichain.lab.sim.cfg import (
                NewtonPhysicsCfg,
                RenderCfg,
                RobotCfg,
                UsdSourceCfg,
                JointDrivePropertiesCfg,
                ArticulationRootPropertiesCfg,
            )

            meta["module_origins"] = {
                n: importlib.util.find_spec(n).origin
                for n in [
                    "embodichain",
                    "embodichain_robots",
                    "embodichain_tasks",
                    "dexsim",
                ]
            }
            sim = SimulationManager(
                SimulationManagerCfg(
                    headless=True,
                    num_envs=1,
                    physics_cfg=NewtonPhysicsCfg(
                        device="cuda:0",
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
            robot = sim.add_robot(
                RobotCfg(
                    uid="franka",
                    **(
                        {"fpath": str(args.asset)}
                        if args.asset.suffix == ".urdf"
                        else {"source": UsdSourceCfg(file_path=str(args.asset))}
                    ),
                    init_pos=[0.15, 0.5, 0.75],
                    init_qpos=initial,
                    root_props=ArticulationRootPropertiesCfg(
                        fixed_base=True, self_collision_enabled=False
                    ),
                    joint_drive_props=JointDrivePropertiesCfg(
                        drive_type="force",
                        stiffness={
                            f"{prefix}_joint[1-7]": 120.0,
                            f"{prefix}_finger_joint1": 1000.0,
                            f"{prefix}_finger_joint2": 0.0,
                        },
                        damping={
                            f"{prefix}_joint[1-7]": 15.0,
                            f"{prefix}_finger_joint1": 20.0,
                            f"{prefix}_finger_joint2": 0.0,
                        },
                        max_effort={
                            f"{prefix}_joint[1-4]": 87.0,
                            f"{prefix}_joint[5-7]": 12.0,
                            f"{prefix}_finger_joint[1-2]": 40.0,
                        },
                        friction=0.0,
                        armature=0.0,
                    ),
                )
            )
            sim.prepare()
            meta["joint_names"] = robot.joint_names
            meta["link_names"] = robot.link_names
            meta["mass"] = robot.get_mass().cpu().tolist()
            meta["inertia"] = robot.get_inertia().cpu().tolist()
            meta["com"] = robot.get_com_pose().cpu().tolist()
            meta["drives"] = [str(x) for x in robot.get_joint_drive()]
            meta["mimic"] = {k: str(v) for k, v in vars(robot).items() if "mimic" in k}
            q = torch.tensor(
                [[dict(zip(names, initial))[n] for n in robot.joint_names]],
                device=sim.device,
            )
            robot.set_qpos(q, target=False)
            robot.set_qvel(torch.zeros_like(q), target=False)
            robot.set_qpos(q)
            save()
            rows = []
            leader_id = robot.joint_names.index(f"{prefix}_finger_joint1")
            for k in range(samples):
                if leader_targets is not None and k and (k - 1) % 4 == 0:
                    q[0, leader_id] = float(leader_targets[(k - 1) // 4])
                    robot.set_qpos(q)
                elif leader_targets is None and k == 100:
                    q[0, leader_id] = 0.015
                    robot.set_qpos(q)
                if k:
                    sim.update(step=1, render_final_step=False)
                rows.append(
                    torch.cat([robot.get_qpos(), robot.get_qvel()], dim=1).cpu().numpy()
                )
            np.save(args.output / "joint_state.npy", np.stack(rows))
            meta["hand_pose_final"] = (
                robot.get_link_pose(f"{prefix}_hand").cpu().tolist()
            )
            meta["finite"] = bool(np.isfinite(rows).all())
        else:
            from isaaclab.app import AppLauncher

            app = AppLauncher(headless=True, visualizer="none").app
            if args.asset.suffix == ".urdf":
                import omni.kit.app

                assert (
                    omni.kit.app.get_app()
                    .get_extension_manager()
                    .set_extension_enabled_immediate(
                        "isaacsim.asset.importer.urdf", True
                    )
                )
            from isaaclab.sim import (
                SimulationContext,
                SimulationCfg,
                UsdFileCfg,
                ArticulationRootPropertiesCfg,
                UrdfFileCfg,
            )
            from isaaclab_physx.physics import PhysxCfg
            from isaaclab.assets import Articulation, ArticulationCfg
            from isaaclab.actuators import ImplicitActuatorCfg

            sim = SimulationContext(
                SimulationCfg(
                    dt=0.005,
                    device="cuda:0",
                    physics=PhysxCfg(solver_type=1),
                    visualizer_cfgs=[],
                    use_newton_actuators=False,
                    log_dir=str(args.output / "sim_logs"),
                )
            )
            spawn = (
                UrdfFileCfg(
                    asset_path=str(args.asset),
                    fix_base=True,
                    merge_fixed_joints=False,
                    usd_dir=str(args.output / "converted"),
                    force_usd_conversion=True,
                    articulation_props=ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False
                    ),
                )
                if args.asset.suffix == ".urdf"
                else UsdFileCfg(
                    usd_path=str(args.asset),
                    articulation_props=ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False
                    ),
                )
            )
            robot = Articulation(
                ArticulationCfg(
                    prim_path="/World/Robot",
                    spawn=spawn,
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.15, 0.5, 0.75), joint_pos=dict(zip(names, initial))
                    ),
                    actuators={
                        "arm": ImplicitActuatorCfg(
                            joint_names_expr=[f"{prefix}_joint[1-7]"],
                            stiffness=120.0,
                            damping=15.0,
                            joint_effort_limit={
                                f"{prefix}_joint[1-4]": 87.0,
                                f"{prefix}_joint[5-7]": 12.0,
                            },
                            armature=0.0,
                        ),
                        "finger": ImplicitActuatorCfg(
                            joint_names_expr=[f"{prefix}_finger_joint1"],
                            stiffness=1000.0,
                            damping=20.0,
                            joint_effort_limit=40.0,
                        ),
                        "follower": ImplicitActuatorCfg(
                            joint_names_expr=[f"{prefix}_finger_joint2"],
                            stiffness=0.0,
                            damping=0.0,
                            joint_effort_limit=40.0,
                        ),
                    },
                )
            )
            sim.reset()
            robot.update(0.005)
            meta["joint_names"] = robot.joint_names
            meta["link_names"] = robot.body_names
            meta["mass"] = robot.data.body_mass.torch.cpu().tolist()
            meta["inertia"] = robot.data.body_inertia.torch.cpu().tolist()
            meta["com"] = robot.data.body_com_pose_b.torch.cpu().tolist()
            q = torch.tensor(
                [[dict(zip(names, initial))[n] for n in robot.joint_names]],
                device=sim.device,
            )
            robot.write_joint_position_to_sim_index(position=q)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
            robot.actuators.target_command.set_position_index(value=q)
            save()
            rows = []
            leader_id = robot.joint_names.index(f"{prefix}_finger_joint1")
            for k in range(samples):
                if leader_targets is not None and k and (k - 1) % 4 == 0:
                    q[0, leader_id] = float(leader_targets[(k - 1) // 4])
                    robot.actuators.target_command.set_position_index(value=q)
                elif leader_targets is None and k == 100:
                    q[0, leader_id] = 0.015
                    robot.actuators.target_command.set_position_index(value=q)
                if k:
                    robot.write_data_to_sim()
                    sim.step(render=False)
                    robot.update(0.005)
                rows.append(
                    torch.cat(
                        [robot.data.joint_pos.torch, robot.data.joint_vel.torch], dim=1
                    )
                    .cpu()
                    .numpy()
                )
            np.save(args.output / "joint_state.npy", np.stack(rows))
            meta["finite"] = bool(np.isfinite(rows).all())
        meta["status"] = "completed"
    except BaseException as exc:
        meta["status"] = "failed"
        meta["failure"] = repr(exc)
        traceback.print_exc()
        raise
    finally:
        meta["ended_at"] = datetime.now().astimezone().isoformat()
        save()
        if args.platform == "embodichain" and sim is not None:
            sim.destroy(exit_process=False)
            SimulationManager.flush_cleanup_queue()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
