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
"""C01 single bread in Isaac Lab with its installed native Newton VBD.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/c01_bread.py
The user explicitly requested no milk on this platform. No gas cavity or right
arm cycle is created. --pilot-performance records one full no-camera pilot;
default previews remain untimed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, ClassVar

import newton
import numpy as np
import torch
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.physics import PhysicsManager
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.utils import configclass, instantiate
from isaaclab_newton.physics import (
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonManager,
    NewtonShapeCfg,
    NewtonSoftContactCfg,
)
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg

from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m02 import (
    NewtonM02VBDManager,
    _M02VBDSolverCfg,
    _numpy,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_common import (
    OUTPUT,
    worktop_pieces,
    rack_foot,
    rack_parts,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_bread_replay import (
    prepare,
    record,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_replay import (
    sha256,
    write_json,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_replay import (
    link_matrices,
)

__all__ = ["main"]


@wp.kernel
def _film_coefficients(
    faces: wp.array2d(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    drag: wp.array(dtype=wp.mat33),
    start: int,
):
    face = wp.tid()
    a, b, c = faces[face, 0], faces[face, 1], faces[face, 2]
    cross = wp.cross(positions[b] - positions[a], positions[c] - positions[a])
    double_area = wp.length(cross)
    if double_area > 1.0e-12:
        normal = cross / double_area
        projection = wp.outer(normal, normal)
        tangent = wp.identity(n=3, dtype=float) - projection
        for corner in range(3):
            node = faces[face, corner]
            speed = wp.dot(velocities[node], normal)
            tv = velocities[node] - speed * normal
            matrix = (1.2 * double_area / 12.0) * (
                1.2 * wp.abs(speed) * projection + 0.02 * wp.length(tv) * tangent
            )
            wp.atomic_add(drag, node - start, matrix)


@wp.kernel
def _film_force(
    velocity: wp.array(dtype=wp.vec3),
    invmass: wp.array(dtype=float),
    drag: wp.array(dtype=wp.mat33),
    forces: wp.array(dtype=wp.vec3),
    start: int,
    dt: float,
):
    i = wp.tid()
    node = start + i
    if invmass[node] > 0.0:
        matrix = drag[i]
        v = (
            wp.inverse(wp.identity(n=3, dtype=float) + dt * invmass[node] * matrix)
            * velocity[node]
        )
        forces[node] += -matrix * v


class NewtonC01BreadVBDManager(NewtonM02VBDManager):
    """Adapt prescribed kinematics and original film air drag to stock VBD."""

    task: ClassVar[dict[str, Any]] = {}
    bag_start: ClassVar[int] = 0
    drag: ClassVar[Any] = None
    faces: ClassVar[Any] = None

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        """Set the same rigid materials, SDF meshes and kinematic robot."""
        contact = cls.task["scene"]["contact"]
        for i, body in enumerate(builder.shape_body):
            label = builder.shape_label[i]
            ke, kd, mu, margin, gap = (
                contact["table_ke"],
                100.0,
                contact["table_mu"],
                contact["margin"],
                contact["gap"],
            )
            if body >= 0:
                link = builder.body_label[body].rsplit("/", 1)[-1]
                ke, mu = contact["robot_ke"], contact["finger_mu"]
                if link in ("left_finger1", "left_finger2"):
                    ke = contact["left_finger_ke"]
                elif link in ("right_finger1", "right_finger2"):
                    ke = contact["right_finger_ke"]
                if builder.shape_type[i] == int(newton.GeoType.MESH):
                    builder.shape_force_sdf[i] = True
            elif "rack_" in label and "rack_foot" not in label:
                ke, kd, mu, margin, gap = 2e5, 30.0, 0.35, 0.001, None
            builder.shape_material_ke[i] = ke
            builder.shape_material_kd[i] = kd
            builder.shape_material_mu[i] = mu
            builder.shape_margin[i] = margin
            # Newton uses an explicit nonnegative gap; the native rack's unset
            # gap resolves to zero in this staged reference.
            builder.shape_gap[i] = 0.0 if gap is None else gap
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        builder.color(include_bending=True, balance_colors=True)

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: Any) -> Any:
        """Allocate bag-only passive drag buffers before graph capture."""
        result = super()._create_solver(model, solver_cfg)
        points = model.particle_q.numpy()
        expected = cls.task["mesh"]["bag_vertices"]
        starts = np.flatnonzero(
            np.all(np.isclose(points, expected[0], rtol=0, atol=1e-7), axis=1)
        )
        matches = [
            int(i)
            for i in starts
            if np.allclose(points[i : i + len(expected)], expected, rtol=0, atol=1e-7)
        ]
        if len(matches) != 1:
            raise ValueError("Cannot locate exact frozen bag particle order")
        cls.bag_start = matches[0]
        cls.faces = wp.array(
            cls.task["mesh"]["bag_triangles"] + cls.bag_start,
            dtype=wp.int32,
            device=model.device,
        )
        cls.drag = wp.zeros(len(expected), dtype=wp.mat33, device=model.device)
        return result

    @classmethod
    def _initialize_contacts(cls) -> None:
        """Select the same static full-surface targets; fingers remain per-node."""
        super()._initialize_contacts()
        pipeline = cls._collision_pipeline
        model = cls.backend.model
        targets = cls.task["scene"]["full_surface_contact_targets"]
        selected = {
            i
            for i, label in enumerate(model.shape_label)
            if any(t == label.rsplit("/", 1)[-1] or f"/{t}/" in label for t in targets)
        }
        if len(selected) != len(targets):
            raise ValueError(
                f"Full-surface support selection differs: {selected}; labels={model.shape_label}"
            )
        # This task needs full-surface contacts only on analytic static supports.
        # Native Newton's global switch validates every robot mesh before a
        # subset can be selected. Provision the explicit one-world pairs before
        # graph capture, keeping its existing native edge/face contact kernels.
        pipeline.enable_rigid_soft_full_surface_contact = True
        for name, count in (
            ("soft_face_rigid_pairs", model.tri_count),
            ("soft_edge_rigid_pairs", model.edge_count),
        ):
            pairs = np.column_stack(
                (
                    np.repeat(np.arange(count), len(selected)),
                    np.tile(sorted(selected), count),
                )
            ).astype(np.int32)
            setattr(
                pipeline, name, wp.array(pairs, dtype=wp.vec2i, device=model.device)
            )

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: Any,
        substep_dt: float,
    ) -> None:
        """Apply the reference quadratic air drag, then stock VBD substep."""
        model = cls.replay.model
        cls.drag.zero_()
        wp.launch(
            _film_coefficients,
            cls.faces.shape[0],
            inputs=[
                cls.faces,
                state_0.particle_q,
                state_0.particle_qd,
                cls.drag,
                cls.bag_start,
            ],
            device=model.device,
        )
        wp.launch(
            _film_force,
            cls.drag.shape[0],
            inputs=[
                state_0.particle_qd,
                model.particle_inv_mass,
                cls.drag,
                state_0.particle_f,
                cls.bag_start,
                substep_dt,
            ],
            device=model.device,
        )
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)


@configclass
class BreadSolverCfg(_M02VBDSolverCfg):
    """Keep the installed unmodified VBD solver and original regularization."""

    class_type: type = NewtonC01BreadVBDManager
    friction_epsilon: float = 0.001


def _write_soft(inputs: dict[str, Any], path: Path, name: str) -> None:
    """Author exact world-space cloth or tets with the existing texture."""
    mesh = inputs["mesh"]
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Soft")
    stage.SetDefaultPrim(root.GetPrim())
    points = mesh[name + "_vertices"]
    if name == "bag":
        surface = UsdGeom.Mesh.Define(stage, "/Soft/mesh")
        triangles = mesh["bag_triangles"]
        surface.CreateFaceVertexCountsAttr([3] * len(triangles))
        surface.CreateFaceVertexIndicesAttr(triangles.ravel().tolist())
        surface.CreateDoubleSidedAttr(True)
        surface.CreateSubdivisionSchemeAttr("none")
        local = np.asarray(inputs["asset"]["vertices"])
        uv = np.column_stack(((local[:, 0] + 0.16) / 0.32, local[:, 2] / 0.40))
        texture, roughness = "bag_print.png", 0.27
    else:
        surface = UsdGeom.TetMesh.Define(stage, "/Soft/tet")
        surface.GetTetVertexIndicesAttr().Set(
            [Gf.Vec4i(*map(int, t)) for t in mesh["bread_tetrahedra"]]
        )
        uv = np.column_stack(
            (
                (points[:, 0] - points[:, 0].min()) / np.ptp(points[:, 0]),
                (points[:, 2] - points[:, 2].min()) / np.ptp(points[:, 2]),
            )
        )
        texture, roughness = "bread_crust.png", 0.82
    surface.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in points])
    UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([Gf.Vec2f(*map(float, p)) for p in uv])
    material = UsdShade.Material.Define(stage, "/Soft/Looks/Material")
    shader = UsdShade.Shader.Define(stage, "/Soft/Looks/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    tex = UsdShade.Shader.Define(stage, "/Soft/Looks/Material/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(path.parent / texture))
    )
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, "/Soft/Looks/Material/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(root.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def main() -> None:
    """Build the one-bread scene and record an untimed complete episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=OUTPUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--until", type=float)
    parser.add_argument("--pilot-performance", action="store_true")
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.until is not None and (args.out is None or args.until <= 0):
        parser.error("A positive truncated diagnostic requires its own --out")
    os.environ.setdefault("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "0")
    if args.pilot_performance:
        if (
            args.out is not None
            or args.until is not None
            or args.reference.resolve() != OUTPUT.resolve()
        ):
            parser.error("Pilot uses the complete current demo and its reserved output")
        args.no_video = True
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_pilot import (
            pilot_directory,
        )

        output = pilot_directory("isaaclab")
    else:
        output = (args.out or OUTPUT / "platforms/isaaclab").resolve()
    inputs = prepare(args.reference, output, Path(__file__))
    cfg, mesh, joints = inputs["scene"], inputs["mesh"], inputs["joints"]
    contact, cloth, bread_material = cfg["contact"], cfg["bag_material"], cfg["bread"]
    assets = output / "inputs"
    # Copy the already verified conversion; never change protected M04 assets.
    shared = OUTPUT.parent / "m04_bimanual_tshirt_single_fold/inputs/isaaclab"
    meta = json.loads((shared / "w1_usd_path.json").read_text())
    for p, h in meta["sha256"].items():
        if sha256(shared / "w1_usd" / p) != h:
            raise ValueError(f"Converted W1 changed: {p}")
    source_robot = OUTPUT.parent / "m04_bimanual_tshirt_single_fold/inputs/w1_v030"
    for p in (assets / "w1_v030").rglob("*"):
        if p.is_file() and sha256(p) != sha256(
            source_robot / p.relative_to(assets / "w1_v030")
        ):
            raise ValueError(f"Converted robot source differs from C01: {p}")
    shutil.copytree(shared / "w1_usd", assets / "w1_usd")
    robot_usd = assets / "w1_usd/robot/robot.usda"
    write_json(assets / "w1_usd_provenance.json", meta)
    _write_soft(inputs, assets / "bag.usda", "bag")
    _write_soft(inputs, assets / "bread.usda", "bread")
    for name in ("m02.py", "m06.py"):
        shutil.copy2(
            Path(__file__).with_name(name), output / "source" / f"isaaclab_{name}"
        )
    NewtonC01BreadVBDManager.task = inputs
    selfcontact = {
        k: v
        for k, v in contact["self_contact"].items()
        if k != "particle_enable_truncation_cache"
    }
    physics = NewtonCfg(
        solver_cfg=BreadSolverCfg(
            replay_targets=joints["targets"],
            replay_joint_names=joints["joint_names"].tolist(),
            replay_dt=1 / 60,
            replay_substeps=16,
            iterations=24,
            integrate_with_external_rigid_solver=True,
            particle_collision_detection_interval=-1,
            **selfcontact,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=2000, soft_contact_kd=0.25, soft_contact_mu=0.4
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=contact["robot_ke"], kd=100, mu=2, margin=0.0005, gap=0
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=0.004,
            enable_rigid_soft_full_surface_contact=False,
            soft_contact_max=65536,
        ),
        num_substeps=16,
        collision_decimation=0,
        use_cuda_graph=True,
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={},
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(robot_usd), variants={"Physics": "mujoco"}
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos=dict(
                    zip(joints["joint_names"].tolist(), joints["targets"][0].tolist())
                )
            ),
        )
        bag = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bag",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "bag.usda"),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    **{
                        k: cloth[k]
                        for k in (
                            "density",
                            "particle_radius",
                            "tri_ke",
                            "tri_ka",
                            "tri_kd",
                            "edge_ke",
                            "edge_kd",
                        )
                    }
                ),
            ),
        )
        bread = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Bread",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(assets / "bread.usda"),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonDeformableBodyMaterialCfg(
                    density=bread_material["volume_density"],
                    k_mu=bread_material["k_mu"],
                    k_lambda=bread_material["k_lambda"],
                    k_damp=bread_material["k_damp"],
                    particle_radius=bread_material["particle_radius"],
                ),
            ),
        )

    for name, center, size in [*worktop_pieces(), ("rack_foot", *rack_foot())]:
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=tuple(size),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=UsdPhysicsRigidBodyMaterialCfg(
                        static_friction=0.9, dynamic_friction=0.9
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.5, 0.5, 0.5)
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(center)),
            ),
        )
    parts = rack_parts(inputs["asset"]["rack_rail_centers_local"])
    for name, a, b, radius in parts:
        direction = b - a
        quat = Rotation.align_vectors(
            [direction / np.linalg.norm(direction)], [[0, 0, 1]]
        )[0].as_quat()
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CapsuleCfg(
                    radius=radius,
                    height=float(np.linalg.norm(direction)),
                    axis="Z",
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.55, 0.57, 0.58)
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=tuple((a + b) / 2), rot=tuple(quat)
                ),
            ),
        )
    cam = inputs["manifest"]["camera"]
    focal = 1280 / 2 / (np.tan(np.deg2rad(cam["hfov"]) / 2) * 720 / 1280)
    if not args.no_video:
        SceneCfg.floor = AssetBaseCfg(
            prim_path="/World/FloorVisual",
            spawn=sim_utils.CuboidCfg(
                size=(8, 8, 0.01),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.55, 0.56, 0.54)
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
            replay = NewtonC01BreadVBDManager.replay
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
            for name in ("bag", "bread"):
                actual = _numpy(scene[name].data.nodal_pos_w)[0]
                if not np.allclose(actual, mesh[name + "_vertices"], rtol=0, atol=1e-7):
                    raise ValueError(f"Initial {name} order or geometry differs")
            if (
                model.particle_count != 1349
                or model.tet_count != 750
                or np.any(model.particle_inv_mass.numpy() <= 0)
            ):
                raise ValueError("Unexpected milk/pinned/missing particles or tets")
            # Read back topology and Lamé material, rather than trusting config.
            q = model.particle_q.numpy()
            bread_start = int(
                np.flatnonzero(
                    np.all(
                        np.isclose(q, mesh["bread_vertices"][0], rtol=0, atol=1e-7),
                        axis=1,
                    )
                )[0]
            )
            if not np.array_equal(
                model.tet_indices.numpy() - bread_start, mesh["bread_tetrahedra"]
            ):
                raise ValueError("Bread tet topology differs")
            if not np.allclose(model.tet_materials.numpy(), [15000, 30000, 1]):
                raise ValueError("Bread tet materials differ")
            audit = dict(
                particles=1349,
                tets=750,
                bag_nodes=1097,
                bread_nodes=252,
                pinned_nodes=0,
                active_goods=["bread"],
                milk_particles=0,
                gas_cavities=0,
                source="Public Isaac Lab scene and native Newton VBD; original passive film drag only",
                camera_focal_pixels=float(focal),
                capsule_parts=[
                    dict(
                        name=n,
                        center=(a + b) / 2,
                        radius=r,
                        half_height=np.linalg.norm(b - a) / 2,
                    )
                    for n, a, b, r in parts
                ],
            )
            pipeline = NewtonC01BreadVBDManager._collision_pipeline
            selected = set(pipeline.soft_face_rigid_pairs.numpy()[:, 1]) | set(
                pipeline.soft_edge_rigid_pairs.numpy()[:, 1]
            )
            audit["full_surface_shapes"] = {
                str(i): model.shape_label[i] for i in sorted(selected)
            }
            audit["global_particle_contact"] = {
                name: float(getattr(model, name))
                for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu")
            }
            if not np.allclose(
                list(audit["global_particle_contact"].values()), [2000, 0.25, 0.4]
            ):
                raise ValueError("Actual global particle contact differs")
            audit["contact_adaptation"] = (
                "Native support-only edge/face candidates; robot contacts use native mesh/BVH fallback. No solver code changes."
            )
            if any("finger" in model.shape_label[i].lower() for i in selected):
                raise ValueError("Finger entered the full-surface candidates")
            analysis = output / "analysis"
            analysis.mkdir()
            fields = (
                "particle_q",
                "particle_inv_mass",
                "particle_radius",
                "tri_indices",
                "tri_materials",
                "tet_indices",
                "tet_materials",
                "edge_indices",
                "edge_rest_angle",
                "edge_bending_properties",
                "shape_transform",
                "shape_scale",
                "shape_type",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu",
                "shape_margin",
                "shape_gap",
                "shape_body",
                "shape_flags",
            )
            np.savez_compressed(
                analysis / "initial_model.npz",
                **{k: getattr(model, k).numpy() for k in fields},
                shape_labels=np.asarray(model.shape_label),
                soft_face_rigid_pairs=pipeline.soft_face_rigid_pairs.numpy(),
                soft_edge_rigid_pairs=pipeline.soft_edge_rigid_pairs.numpy(),
            )
            hand_ids = [
                robot.body_names.index(f"{side}_gripper_base")
                for side in ("left", "right")
            ]
            tcp_local = np.r_[[0, 0, 0.108], 1.0]
            # The exact TCP offset is frozen in the accepted W1 utility.
            import ast

            tree = ast.parse(
                (output / "source/reference/dexuni_w1_two_way_utils.py").read_text()
            )
            node = next(
                n
                for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TCP" for t in n.targets)
            )
            tcp_local = np.r_[ast.literal_eval(node.value.args[0]), 1.0]
            camera = None
            if not args.no_video:
                camera = scene["camera"]
                camera.set_world_poses_from_view(
                    eyes=torch.tensor([cam["eye"]], device=sim.device),
                    targets=torch.tensor([cam["target"]], device=sim.device),
                )

            def readings() -> dict[str, np.ndarray]:
                """Read actual physical clock and untouched geometry."""
                matrices = link_matrices(_numpy(robot.data.body_link_pose_w)[0])
                return dict(
                    time=np.asarray(PhysicsManager.get_simulation_time()),
                    qpos=_numpy(robot.data.joint_pos)[0, ids],
                    bag=_numpy(scene["bag"].data.nodal_pos_w)[0],
                    bread=_numpy(scene["bread"].data.nodal_pos_w)[0],
                    tcp=(matrices[hand_ids] @ tcp_local)[:, :3],
                )

            def step() -> None:
                """Advance one 1/60-second outer step through Isaac Lab."""
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(1 / 60)

            def capture() -> np.ndarray:
                """Render the current physical state from the centered camera."""
                sim.render()
                return _numpy(camera.data.output["rgb"])[0, ..., :3]

            def diagnostics() -> dict[str, Any]:
                """Save actual solver, clock and graph state without timing."""
                count = int(replay.counter.numpy()[0])
                assert count % 16 == 0
                return dict(
                    sim_time=PhysicsManager.get_simulation_time(),
                    step_index=count // 16,
                    actual_substeps=count,
                    num_substeps=16,
                    iterations=NewtonC01BreadVBDManager._solver.iterations,
                    cuda_graph_status=(
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    solver="Installed native Newton SolverVBD",
                    newton_version=newton.__version__,
                    warp_version=wp.__version__,
                    isaaclab_version=importlib.metadata.version("isaaclab"),
                    particle_collision_detection_interval=-1,
                    particle_truncation_cache="not available in native Newton",
                    versions_scope="Different from two-good EC/native scene; no cross-scene performance comparison",
                    camera_intrinsics=(
                        None
                        if camera is None
                        else _numpy(camera.data.intrinsic_matrices)[0]
                    ),
                )

            if args.pilot_performance:
                from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_pilot import (
                    record_episode,
                )

                record_episode(
                    "isaaclab",
                    inputs,
                    output,
                    step,
                    readings,
                    lambda: wp.synchronize_device(str(sim.device)),
                    diagnostics,
                    audit,
                )
            else:
                record(
                    inputs,
                    output,
                    step,
                    readings,
                    None if args.no_video else capture,
                    diagnostics,
                    audit,
                    until=args.until,
                )
    finally:
        NewtonC01BreadVBDManager.replay = None
        NewtonC01BreadVBDManager.drag = None
        NewtonC01BreadVBDManager.faces = None


if __name__ == "__main__":
    main()
