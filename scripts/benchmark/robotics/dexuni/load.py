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
"""Measure static scale response and loaded grasp, carry and release.

Run: python -m scripts.benchmark.robotics.dexuni.load auto
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    MassPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
)
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.objects.articulation import Articulation
    from embodichain.lab.sim.objects.rigid_object import RigidObject

MASSES = (0.032, 0.063, 0.126)
FRICTIONS = (0.35, 1.0)
RAMP_FRAMES = (12, 36)
ROWS = [(m, f, r) for m in MASSES for f in FRICTIONS for r in RAMP_FRAMES]
DT = 1 / 60


def asset(path: Path, scale: bool) -> None:
    """Write the lift gantry or a calibrated vertically compliant scale."""
    root = ET.Element("robot", name=path.stem)
    ET.SubElement(root, "link", name="base")

    def slide(name, parent, xyz, axis, mass, size=None):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=".001",
            iyy=".001",
            izz=".001",
            ixy="0",
            ixz="0",
            iyz="0",
        )
        if size:
            geometry = ET.SubElement(ET.SubElement(link, "collision"), "geometry")
            ET.SubElement(geometry, "box", size=size)
        joint = ET.SubElement(root, "joint", name=name, type="prismatic")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz=xyz)
        ET.SubElement(joint, "axis", xyz=axis)
        ET.SubElement(
            joint, "limit", lower="-.03", upper=".2", effort="10", velocity="2"
        )

    if scale:
        slide("platform", "base", "0 0 -.01", "0 0 1", 0.10, ".4 .4 .02")
    else:
        slide("carry", "base", "0 0 0", "0 1 0", 0.1)
        slide("lift", "carry", "0 0 0", "0 0 1", 0.1)
        slide("left", "lift", "-.08 0 .03", "1 0 0", 0.012, ".02 .04 .03")
        slide("right", "lift", ".08 0 .03", "-1 0 0", 0.012, ".02 .04 .03")
    ET.ElementTree(root).write(path)


def measure_selected_reset(
    sim: SimulationManager,
    robot: Articulation,
    box: RigidObject,
    mode: str,
    graph: bool,
    out: Path,
) -> None:
    """Time partial reset and the subsequent update, verifying selected and live rows."""
    defaults = robot.get_qpos().clone()
    initial_pose = box.get_local_pose().clone()
    selected = list(range(0, sim.num_envs, 2))
    results = []
    for _ in range(5):
        command = torch.full_like(defaults, 0.018)
        robot.set_qpos(command, target=True)
        sim.update(step=12)
        before_q = robot.get_qpos().clone()
        before_v = robot.get_qvel().clone()
        before_box = box.get_local_pose().clone()
        wp.synchronize_device("cuda:0")
        start = time.perf_counter()
        sim.reset_objects_state(env_ids=selected)
        wp.synchronize_device("cuda:0")
        reset_s = time.perf_counter() - start
        status = sim.physics.cuda_graph_status
        after_q = robot.get_qpos()
        after_v = robot.get_qvel()
        after_box = box.get_local_pose()
        live_error = max(
            float((before_q[1::2] - after_q[1::2]).abs().max()),
            float((before_v[1::2] - after_v[1::2]).abs().max()),
            float((before_box[1::2] - after_box[1::2]).abs().max()),
        )
        reset_error = max(
            float((defaults[::2] - after_q[::2]).abs().max()),
            float((initial_pose[::2] - after_box[::2]).abs().max()),
        )
        wp.synchronize_device("cuda:0")
        start = time.perf_counter()
        sim.update(step=1)
        wp.synchronize_device("cuda:0")
        first_s = time.perf_counter() - start
        results.append(
            dict(
                live_error=live_error,
                reset_error=reset_error,
                reset_s=reset_s,
                first_update_s=first_s,
                graph_after_reset=status,
                graph_after_update=sim.physics.cuda_graph_status,
                moved=float((before_q[::2] - defaults[::2]).abs().max()),
                finite=bool(torch.isfinite(robot.get_qpos()).all()),
            )
        )
    diagnostics = sim._spawn_scene.builder.result.get_solver_diagnostics()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lifecycle_{mode}_{int(graph)}.json").write_text(
        json.dumps(
            dict(mode=mode, graph=graph, diagnostics=diagnostics, results=results),
            default=str,
        )
    )
    if not all(
        v["moved"] > 0.001
        and v["finite"]
        and max(v["live_error"], v["reset_error"]) < 1e-5
        for v in results
    ):
        raise AssertionError("Selected reset qualification failed; see recorded errors")


def run(
    mode: str,
    directory: Path,
    out: Path,
    material_friction: float | None = None,
    lifecycle_graph: bool | None = None,
) -> None:
    """Compare the same smooth, time-bounded motion for each physical row."""
    scenarios = [
        row for row in ROWS if material_friction is None or row[1] == material_friction
    ]
    batch = len(scenarios)
    solver = {"solver_type": "mujoco_warp" if mode == "mj" else "dexuni"}
    if mode in ("one_way", "two_way"):
        solver["coupling"] = mode
    sim = SimulationManager(
        SimulationManagerCfg(
            num_envs=batch,
            arena_space=1.0,
            headless=True,
            enable_ground_plane=False,
            render_cfg=RenderCfg(renderer="no-render"),
            physics_cfg=NewtonPhysicsCfg(
                device="cuda:0",
                physics_dt=DT,
                num_substeps=10,
                use_cuda_graph=True if lifecycle_graph is None else lifecycle_graph,
                collision_cfg=None,
                solver_cfg=solver,
                dexuni_mujoco_joints_per_arena=8 if mode == "all_mj" else None,
            ),
        )
    )
    material = RigidBodyPhysicsCfg(
        collision_props=NewtonCollisionPropertiesCfg(margin=0.0, gap=0.0),
        material_props=NewtonRigidBodyMaterialCfg(
            ke=2e4,
            kd=100.0,
            dynamic_friction=1.0 if material_friction is None else material_friction,
        ),
    )

    def add(name, scale):
        path = directory / (name + ".urdf")
        asset(path, scale)
        return sim.add_articulation(
            ArticulationCfg(
                uid=name,
                fpath=str(path),
                build_pk_chain=False,
                asset_physics_mode="overlay",
                attrs=material,
                root_props=ArticulationRootPropertiesCfg(
                    newton_gravity_compensation=1.0
                ),
                joint_drive_props=NewtonJointDrivePropertiesCfg(
                    target_mode="position_velocity",
                    stiffness=1000.0,
                    damping=20.0,
                    passive_damping=50.0,
                    armature=0.3,
                    max_effort=10.0,
                ),
            )
        )

    try:
        robot = add("gantry", False)
        scale = add("scale", True)
        material = material.copy()
        material.mass_props = MassPropertiesCfg(mass=MASSES[0])
        box = sim.add_rigid_object(
            RigidObjectCfg(
                uid="box",
                shape=CubeCfg(size=(0.05,) * 3),
                init_pos=(0.0, 0.0, 0.025),
                attrs=material,
            )
        )
        sim.prepare()
        if lifecycle_graph is not None:
            measure_selected_reset(sim, robot, box, mode, lifecycle_graph, out)
            return
        # Targets are (finger travel, lift, carry) in metres. The half-cosine
        # interpolant has zero endpoint velocity and a known peak acceleration.
        phases = {
            "weigh": (0.0, 0.0, 0.0),
            "close": (0.07, 0.0, 0.0),
            "lift": (0.07, 0.08, 0.0),
            "carry": (0.07, 0.08, 0.08),
            "release": (0.0, 0.08, 0.08),
        }
        measurements = []
        order = ["left", "right", "lift", "carry"]
        joint_order = [order.index(n) for n in robot.joint_names]
        commands = {}
        previous = np.zeros(3)
        ramps = np.array([ramp for _, _, ramp in scenarios])
        frames = 90
        for phase, target in phases.items():
            phase_commands = []
            for frame in range(1, frames + 1):
                u = np.minimum(1.0, frame / ramps)
                blend = (1.0 - np.cos(np.pi * u))[:, None] / 2.0
                positions = previous + blend * (np.asarray(target) - previous)
                phase_commands.append(
                    torch.as_tensor(
                        positions[:, [0, 0, 1, 2]][:, joint_order].copy(),
                        device=sim.device,
                        dtype=torch.float32,
                    )
                )
            commands[phase] = phase_commands
            previous = np.asarray(target)
        for rep in range(3):
            sim.reset_objects_state()
            box.set_mass(torch.tensor([row[0] for row in scenarios], device=sim.device))
            box.set_friction(
                torch.tensor([row[1] for row in scenarios], device=sim.device)
            )
            sim.update(step=6)
            diagnostics = sim._spawn_scene.builder.result.get_solver_diagnostics()
            expected = "coupled" if mode in ("one_way", "two_way") else "pure_mujoco"
            if mode != "mj" and diagnostics["features"]["backend"] != expected:
                raise ValueError(diagnostics)
            assert diagnostics["cuda_graph_status"] == "captured"
            origin = box.get_local_pose()[:, :3].cpu().numpy().copy()
            rows = []
            for phase, actions in commands.items():
                for first in range(0, frames, 15):
                    wp.synchronize_device("cuda:0")
                    start = time.perf_counter()
                    for cmd in actions[first : first + 15]:
                        robot.set_qpos(cmd, target=True)
                        sim.update(step=1)
                    wp.synchronize_device("cuda:0")
                    elapsed = time.perf_counter() - start
                    # Under static equilibrium, with platform gravity compensation,
                    # the spring supports the payload, F = -k*q.
                    q = (
                        robot.get_qpos()
                        .cpu()
                        .numpy()[:, [robot.joint_names.index(n) for n in order]]
                    )
                    xyz = box.get_local_pose()[:, :3].cpu().numpy()
                    spring = -1000 * scale.get_qpos().cpu().numpy()[:, 0]
                    rows.append(
                        dict(
                            phase=phase,
                            frame=first + 15,
                            seconds=elapsed,
                            q=q.tolist(),
                            delta=(xyz - origin).tolist(),
                            xyz=xyz.tolist(),
                            scale_force_N=spring.tolist(),
                        )
                    )
            measurements.append(rows)
        out.mkdir(exist_ok=True, parents=True)
        suffix = "" if material_friction is None else f"_mu{material_friction:g}"
        (out / f"load_{mode}{suffix}.json").write_text(
            json.dumps(
                dict(
                    mode=mode,
                    dt=DT,
                    rows=scenarios,
                    common_material_friction=material_friction,
                    diagnostics=diagnostics,
                    repeats=measurements,
                ),
                default=str,
            )
        )
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("auto", "all_mj", "mj", "one_way", "two_way"))
    parser.add_argument("--out", type=Path, default=Path("outputs/dexuni"))
    parser.add_argument("--material-friction", type=float, choices=FRICTIONS)
    parser.add_argument("--lifecycle-graph", choices=(0, 1), type=int)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        run(
            args.mode,
            Path(directory),
            args.out,
            args.material_friction,
            None if args.lifecycle_graph is None else bool(args.lifecycle_graph),
        )
