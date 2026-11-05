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
"""Reference C02 in the existing Isaac Lab / Isaac Sim native Newton runtime.

Run: /home/oem/app/IsaacLab/.venv/bin/python <this file> --visualizer none
No DexSim SDK is imported into Isaac Lab. Robot-only prescribed FK and
unaltered native VBD advance the two dynamic goods and the free paper bag.
"""

from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from typing import Any, ClassVar
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
import newton
import warp as wp
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom, Gf

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.physics import PhysicsManager
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
)
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m02 import (
    NewtonM02VBDManager,
    _M02VBDSolverCfg,
    _KinematicReplay,
    _numpy,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m06 import (
    _write_kinematic_joints,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.c01_bread import (
    _write_soft,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_replay import (
    link_matrices,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_controller import (
    ReferenceController,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c02_replay import (
    OUTPUT,
    load,
    freeze,
    record,
    seed_paper,
    audit_model,
    table_pieces,
    fixed_link_expansion,
    sha,
    write_json,
)

__all__ = ["main"]


class RobotReplay(_KinematicReplay):
    """Refresh robot bodies only, preserving integrated can/carton states."""

    def step(self, state_in: newton.State, state_out: newton.State) -> None:
        wp.launch(
            _write_kinematic_joints,
            self.joint_count,
            inputs=[
                self.targets,
                self.counter,
                self.q_indices,
                self.qd_indices,
                self.substeps,
                self.frame_dt,
                state_in.joint_q,
                state_in.joint_qd,
                state_out.joint_q,
                state_out.joint_qd,
            ],
            device=self.model.device,
        )
        for state in (state_in, state_out):
            newton.eval_fk(
                self.model,
                state.joint_q,
                state.joint_qd,
                state,
                body_flag_filter=int(newton.BodyFlags.KINEMATIC),
            )


class NewtonC02VBDManager(NewtonM02VBDManager):
    """Use native VBD for dynamic goods, with one-way robot contact."""

    task: ClassVar[dict[str, Any]] = {}
    hands: ClassVar[list[str]] = []

    @classmethod
    def _restore_robot_colliders(cls, builder: newton.ModelBuilder) -> None:
        """Replace converted convex pieces with the authored URDF surfaces.

        Imported USD visuals remain visible. Its convex collision decomposition
        is disabled; every original collision mesh is added at the corresponding
        retained body/fixed-link transform, with zero added body mass.
        """
        for i, body in enumerate(builder.shape_body):
            if body >= 0 and "/Robot/" in builder.body_label[body]:
                builder.shape_flags[i] &= int(newton.ShapeFlags.VISIBLE)
        assets = cls.task["output"] / "inputs/w1_v030"
        urdf = assets / "robot.urdf"
        names = cls.task["manifest"]["link_state_order"]
        available = [label.rsplit("/", 1)[-1] for label in builder.body_label]
        body_ids, offsets = fixed_link_expansion(urdf, names, available)
        records = []
        arrays = {}
        for link in ET.parse(urdf).getroot().findall("link"):
            name = link.attrib["name"]
            for j, collision in enumerate(link.findall("collision")):
                geometry = collision.find("geometry/mesh")
                assert geometry is not None
                source = assets / geometry.attrib["filename"]
                mesh = trimesh.load(source, force="mesh")
                mesh.apply_scale(np.fromstring(geometry.get("scale", "1 1 1"), sep=" "))
                local = np.eye(4)
                origin = collision.find("origin")
                if origin is not None:
                    local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
                    local[:3, :3] = Rotation.from_euler(
                        "xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
                    ).as_matrix()
                index = names.index(name)
                transform = offsets[index] @ local
                vertices = np.asarray(mesh.vertices, dtype=np.float32)
                faces = np.asarray(mesh.faces, dtype=np.int32)
                native = newton.Mesh(vertices, faces.reshape(-1))
                hand = name in cls.hands
                if hand:
                    native.build_sdf(target_voxel_size=0.001, margin=0.05)
                cfg = newton.ModelBuilder.ShapeConfig(
                    density=0.0,
                    ke=4e4,
                    kd=80.0,
                    mu=1.2,
                    margin=0.0015 if hand else 0.0,
                    gap=0.002,
                    is_visible=False,
                    force_sdf=hand,
                )
                shape = builder.add_shape_mesh(
                    body_ids[index],
                    xform=wp.transform(
                        transform[:3, 3],
                        Rotation.from_matrix(transform[:3, :3]).as_quat(),
                    ),
                    mesh=native,
                    cfg=cfg,
                    label=f"/World/envs/env_0/Robot/AuthoredCollision/{name}/{j}",
                )
                records.append(
                    dict(
                        shape=shape,
                        source_link=name,
                        body_link=available[body_ids[index]],
                        source_file=str(source.relative_to(cls.task["output"])),
                        source_sha256=sha(source),
                        finger_sdf=hand,
                    )
                )
                arrays[f"vertices_{shape}"] = vertices
                arrays[f"faces_{shape}"] = faces
                arrays[f"local_transform_{shape}"] = transform
        assert len(records) == 29 and sum(r["finger_sdf"] for r in records) == 4
        np.savez_compressed(
            cls.task["output"] / "analysis/authored_robot_collision_meshes.npz",
            **arrays,
        )
        write_json(
            cls.task["output"] / "analysis/authored_robot_colliders.json",
            dict(
                imported_collision_pieces_disabled=True, added_mass=0.0, shapes=records
            ),
        )

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        cls._restore_robot_colliders(builder)
        builder.particle_mass = cls.task["initial"]["particle_mass"].tolist()
        builder.tri_materials = cls.task["initial"]["tri_materials"].tolist()
        builder.edge_bending_properties = cls.task["initial"][
            "edge_bending_properties"
        ].tolist()
        builder.particle_radius = cls.task["initial"]["particle_radius"].tolist()
        for i, label in enumerate(builder.body_label):
            if "/Robot/" in label:
                builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        for i, body in enumerate(builder.shape_body):
            label = builder.shape_label[i]
            name = builder.body_label[body].rsplit("/", 1)[-1] if body >= 0 else ""
            robot = body >= 0 and "/Robot/" in builder.body_label[body]
            hand = robot and name in cls.hands
            builder.shape_material_ke[i] = 4e4
            builder.shape_material_kd[i] = 80.0
            builder.shape_material_mu[i] = 1.2 if robot or body >= 0 else 0.65
            builder.shape_margin[i] = 0.0015 if hand else 0.0
            builder.shape_gap[i] = 0.002
            if (
                hand
                and (
                    int(builder.shape_flags[i])
                    & int(newton.ShapeFlags.COLLIDE_PARTICLES)
                )
                and builder.shape_type[i] == int(newton.GeoType.MESH)
            ):
                mesh = builder.shape_source[i]
                if mesh.sdf is None:
                    mesh.build_sdf(target_voxel_size=0.001, margin=0.05)
                builder.shape_force_sdf[i] = True
        builder.color(include_bending=True, balance_colors=True)

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: Any) -> Any:
        np.savez_compressed(
            cls.task["output"] / "analysis/before_rest_seed.npz",
            **{
                k: getattr(model, k).numpy()
                for k in (
                    "particle_q",
                    "particle_mass",
                    "particle_inv_mass",
                    "tri_indices",
                    "tri_poses",
                    "tri_areas",
                    "tri_materials",
                    "edge_indices",
                    "edge_rest_angle",
                    "edge_bending_properties",
                )
            },
        )
        seed_paper(model, cls.task)
        cls.replay = RobotReplay(model, solver_cfg)
        return NewtonVBDManager._create_solver.__func__(cls, model, solver_cfg)

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: Any,
        substep_dt: float,
    ) -> None:
        """Publish VBD-integrated free goods into Lab's public joint views."""
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        newton.eval_fk(
            cls.replay.model,
            state_1.joint_q,
            state_1.joint_qd,
            state_1,
            body_flag_filter=int(newton.BodyFlags.KINEMATIC),
        )
        newton.eval_ik(
            cls.replay.model,
            state_1,
            state_1.joint_q,
            state_1.joint_qd,
            body_flag_filter=int(newton.BodyFlags.DYNAMIC),
        )

    @classmethod
    def _initialize_contacts(cls) -> None:
        super()._initialize_contacts()
        m = cls.backend.model
        p = cls._collision_pipeline
        bodies = m.shape_body.numpy()
        selected = []
        flags = m.shape_flags.numpy()
        for i, label in enumerate(m.shape_label):
            if not int(flags[i]) & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                continue
            body = int(bodies[i])
            name = m.body_label[body].rsplit("/", 1)[-1] if body >= 0 else ""
            if (name in cls.hands and "/Robot/" in m.body_label[body]) or any(
                "/" + n + "/" in label or label.rsplit("/", 1)[-1] == n
                for n in ("table", "leg0", "leg1", "leg2", "leg3", "can", "carton")
            ):
                selected.append(i)
        assert selected and any(
            "can" in m.shape_label[i].lower() for i in selected
        ), selected
        for i in selected:
            if int(m.shape_type.numpy()[i]) in (
                int(newton.GeoType.MESH),
                int(newton.GeoType.CONVEX_MESH),
            ):
                assert m.shape_source[i].sdf is not None, (
                    i,
                    m.shape_label[i],
                    "full-surface SDF missing",
                )
        assert len(selected) == 11, selected
        p.enable_rigid_soft_full_surface_contact = True
        for field, count in (
            ("soft_face_rigid_pairs", m.tri_count),
            ("soft_edge_rigid_pairs", m.edge_count),
        ):
            pairs = np.column_stack(
                (np.repeat(np.arange(count), len(selected)), np.tile(selected, count))
            ).astype(np.int32)
            setattr(p, field, wp.array(pairs, dtype=wp.vec2i, device=m.device))
        write_json(
            cls.task["output"] / "analysis/full_surface_shapes.json",
            {str(i): m.shape_label[i] for i in selected},
        )


@configclass
class SolverCfg(_M02VBDSolverCfg):
    """Original substep/sweep budget with stock native VBD capabilities."""

    class_type: type = NewtonC02VBDManager
    friction_epsilon: float = 0.001
    rigid_contact_hard: bool = False
    rigid_contact_history: bool = False


def main() -> None:
    """Build a public Isaac Lab scene and record its actual free dynamics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, default=OUTPUT)
    p.add_argument("--out", type=Path)
    p.add_argument("--until", type=float)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--substeps", type=int, default=6)
    p.add_argument("--iterations", type=int, default=12)
    p.add_argument(
        "--trajectory", type=Path, help="Explicit robot-only TCP correction JSON"
    )
    add_launcher_args(p)
    args = p.parse_args()
    inputs = load(args.reference)
    out = (args.out or OUTPUT / "platforms/isaaclab").resolve()
    freeze(inputs, out, Path(__file__))
    if args.trajectory is not None:
        shutil.copy2(args.trajectory, out / "inputs/controller_trajectory.json")
    inputs["output"] = out
    assets = out / "inputs"
    joints = inputs["joints"]
    bag = inputs["bag"]
    materials = inputs["materials"]
    shared = OUTPUT.parent / "m04_bimanual_tshirt_single_fold/inputs/isaaclab"
    meta = json.loads((shared / "w1_usd_path.json").read_text())
    for rel, h in meta["sha256"].items():
        assert sha(shared / "w1_usd" / rel) == h, rel
    source_robot = OUTPUT.parent / "m04_bimanual_tshirt_single_fold/inputs/w1_v030"
    for f in (assets / "w1_v030").rglob("*"):
        if f.is_file():
            assert sha(f) == sha(source_robot / f.relative_to(assets / "w1_v030")), f
    shutil.copytree(shared / "w1_usd", assets / "w1_usd")
    write_json(assets / "w1_usd_provenance.json", meta)
    # Reuse the installed scene's USD material authoring, then replace its
    # C01-specific UV mapping and texture with the original C02 paper map.
    _write_soft(
        dict(
            mesh={
                "bag_vertices": materials["particle_positions"],
                "bag_triangles": bag["faces"],
            },
            asset={"vertices": bag["vertices"]},
        ),
        assets / "bag.usda",
        "bag",
    )
    from pxr import UsdShade, Sdf

    stage = Usd.Stage.Open(str(assets / "bag.usda"))
    surface = UsdGeom.Mesh(stage.GetPrimAtPath("/Soft/mesh"))
    rest = bag["vertices"]
    uv = np.column_stack(
        (
            np.arctan2(rest[:, 1] / 0.32, rest[:, 0] / 0.13) / (2 * np.pi) + 0.5,
            rest[:, 2] / 0.25,
        )
    )
    UsdGeom.PrimvarsAPI(surface).GetPrimvar("st").Set(
        [Gf.Vec2f(*map(float, v)) for v in uv]
    )
    stage.GetPrimAtPath("/Soft/Looks/Material/Texture").GetAttribute("inputs:file").Set(
        Sdf.AssetPath(str(assets / "kraft.png"))
    )
    stage.GetPrimAtPath("/Soft/Looks/Material/Surface").GetAttribute(
        "inputs:roughness"
    ).Set(0.95)
    stage.GetRootLayer().Save()
    for name in ("m02.py", "m06.py", "c01_bread.py"):
        shutil.copy2(
            Path(__file__).with_name(name), out / "source" / ("isaaclab_" + name)
        )
    shutil.copy2(
        Path(__file__).parents[3] / "experiments/dexuni/m04_replay.py",
        out / "source/m04_replay.py",
    )
    NewtonC02VBDManager.task = inputs
    NewtonC02VBDManager.hands = [
        link.attrib["name"]
        for link in ET.parse(assets / "w1_v030/robot.urdf").getroot().findall("link")
        if "finger" in link.attrib["name"].lower()
        or link.attrib["name"].lower() in ("link7", "link8")
    ]
    physics = NewtonCfg(
        solver_cfg=SolverCfg(
            replay_targets=joints["targets"],
            replay_joint_names=joints["joint_names"].tolist(),
            replay_dt=1 / 60,
            replay_substeps=args.substeps,
            iterations=args.iterations,
            integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.0012,
            particle_self_contact_margin=0.003,
            particle_collision_detection_interval=0,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=0.0015,
            rigid_body_contact_buffer_size=4096,
            rigid_body_particle_contact_buffer_size=8192,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=2e5, soft_contact_kd=10.0, soft_contact_mu=0.6
        ),
        default_shape_cfg=NewtonShapeCfg(ke=4e4, kd=80, mu=1.2, margin=0, gap=0.002),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=0.004,
            soft_contact_max=131072,
            enable_rigid_soft_full_surface_contact=False,
        ),
        num_substeps=args.substeps,
        collision_decimation=0,
        use_cuda_graph=True,
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={},
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "w1_usd/robot/robot.usda"),
                variants={"Physics": "mujoco"},
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(-0.03, 0, 0),
                joint_pos=dict(
                    zip(joints["joint_names"].tolist(), joints["targets"][0].tolist())
                ),
            ),
        )
        bag = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bag",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "bag.usda"),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    density=0.18, particle_radius=0.0012
                ),
            ),
        )

    for name, center, size in table_pieces():
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            (0.52, 0.55, 0.56)
                            if name == "table"
                            else (0.22, 0.24, 0.25)
                        )
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )
    for name, center, color in [
        ("can", (0.43, -0.36, 0.98), (0.85, 0.34, 0.10)),
        ("carton", (0.43, -0.25, 0.98), (0.20, 0.48, 0.28)),
    ]:
        kwargs = dict(
            mass_props=sim_utils.MassPropertiesCfg(density=280),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        spawn = (
            sim_utils.CylinderCfg(radius=0.0325, height=0.118, axis="Z", **kwargs)
            if name == "can"
            else sim_utils.CuboidCfg(size=(0.048, 0.060, 0.118), **kwargs)
        )
        setattr(
            SceneCfg,
            name,
            RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(pos=center),
            ),
        )
    focal = 640 / np.tan(np.radians(20))
    if not args.no_video:
        SceneCfg.floor = AssetBaseCfg(
            prim_path="/World/FloorVisual",
            spawn=sim_utils.CuboidCfg(
                size=(10, 10, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.30, 0.32, 0.34)
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.005)),
        )
        SceneCfg.light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600)
        )
        SceneCfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(focal / 1280 * 20.955),
                horizontal_aperture=20.955,
                clipping_range=(0.01, 10),
            ),
            renderer_cfg=OVRTXRendererCfg(),
        )
    controller = ReferenceController(out)
    try:
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=1 / 60,
                    device=args.device,
                    gravity=(0, 0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(
                SceneCfg(num_envs=1, env_spacing=3, replicate_physics=True)
            )
            sim.reset()
            replay = NewtonC02VBDManager.replay
            replay.counter.zero_()
            model = NewtonManager.get_model()
            robot = scene["robot"]
            ids = [robot.joint_names.index(n) for n in joints["joint_names"]]
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    joints["targets"][:1], device=sim.device, dtype=torch.float32
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            audit = audit_model(model, inputs, out)
            audit["solver_differences"] = [
                f"Installed native Newton has no DexUni DAT, multilevel correction or Chebyshev acceleration; plain VBD {args.iterations} sweeps,{args.substeps} substeps",
                "Original 29 URDF collision meshes replace disabled imported USD convex pieces; four finger SDFs provisioned, noncolliding visuals excluded from full-surface pairs",
                "Robot FK writes only KINEMATIC bodies; can and carton remain dynamic",
                "Original paper tensors and masses seeded once before solver construction",
            ]
            names = inputs["manifest"]["link_state_order"]
            order, fixed_offsets = fixed_link_expansion(
                assets / "w1_v030/robot.urdf", names, robot.body_names
            )
            hands = [
                names.index(n) for n in ("left_gripper_base", "right_gripper_base")
            ]
            camera = None if args.no_video else scene["camera"]
            if camera is not None:
                camera.set_world_poses_from_view(
                    eyes=torch.tensor([[2.75, -0.45, 2.20]], device=sim.device),
                    targets=torch.tensor([[0.44, -0.03, 1.08]], device=sim.device),
                )

            def readings() -> dict[str, np.ndarray]:
                poses = _numpy(robot.data.body_link_pose_w)[0]
                matrices = link_matrices(poses)[order] @ fixed_offsets
                links = np.column_stack(
                    (
                        Rotation.from_matrix(matrices[:, :3, :3]).as_quat(),
                        matrices[:, :3, 3],
                    )
                )
                snack = np.stack(
                    [
                        _numpy(scene[n].data.root_link_pose_w)[0]
                        for n in ("can", "carton")
                    ]
                )
                snack_pose = np.column_stack((snack[:, 3:], snack[:, :3]))
                return dict(
                    time=np.asarray(PhysicsManager.get_simulation_time()),
                    qpos=_numpy(robot.data.joint_pos)[0, ids],
                    bag=_numpy(scene["bag"].data.nodal_pos_w)[0],
                    links=links,
                    snack_pose=snack_pose,
                    tcp=(matrices[hands] @ np.array((0, 0, 0.125, 1)))[:, :3],
                    tcp_targets=controller.targets.copy(),
                    joint_command=controller.current_command.copy(),
                )

            command_frame = 0
            command_buffer = wp.zeros(24, dtype=wp.float32, device=model.device)

            def step() -> None:
                nonlocal command_frame
                q = controller.command(
                    PhysicsManager.get_simulation_time(),
                    _numpy(scene["bag"].data.nodal_pos_w)[0],
                )
                command_frame += 1
                command_buffer.assign(q)
                wp.copy(
                    replay.targets.flatten(),
                    command_buffer,
                    dest_offset=command_frame * 24,
                    count=24,
                )
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(1 / 60)

            def capture() -> np.ndarray:
                sim.render()
                return _numpy(camera.data.output["rgb"])[0, ..., :3]

            def diagnostics() -> dict[str, Any]:
                np.savez_compressed(
                    out / "analysis/final_materials.npz",
                    **{
                        k: getattr(model, k).numpy()
                        for k in (
                            "particle_mass",
                            "particle_inv_mass",
                            "tri_materials",
                            "tri_areas",
                            "edge_rest_angle",
                            "edge_bending_properties",
                        )
                    },
                )
                count = int(replay.counter.numpy()[0])
                assert count % args.substeps == 0
                return dict(
                    num_substeps=args.substeps,
                    vbd=dict(
                        iterations=NewtonC02VBDManager._solver.iterations,
                        particle_enable_self_contact=True,
                    ),
                    cuda_graph_status=(
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    sim_time=PhysicsManager.get_simulation_time(),
                    step_index=count // args.substeps,
                    newton_version=newton.__version__,
                    warp_version=wp.__version__,
                    isaaclab_version=importlib.metadata.version("isaaclab"),
                    differences=audit["solver_differences"],
                )

            record(
                "Isaac Lab / Isaac Sim",
                inputs,
                out,
                step,
                readings,
                None if camera is None else capture,
                diagnostics,
                audit,
                args.until,
            )
    finally:
        controller.close()
        NewtonC02VBDManager.replay = None


if __name__ == "__main__":
    main()
