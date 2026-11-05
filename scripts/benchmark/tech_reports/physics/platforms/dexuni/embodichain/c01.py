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
"""Build accepted C01 with EmbodiChain, including the actual milk gas cavity.

Run with the recorded project SDK/Newton Python paths; see C01's REPORT.md.
Default previews are untimed. --pilot-performance records one no-camera pilot.
--no-video --out <scratch> supports physical diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
import dexsim
import newton
import warp as wp
from dexsim.engine.newton_physics import (
    FilmDragControl,
    KinematicJointTrajectoryControl,
)
from dexsim.engine.newton_physics.backend_registry import get_newton_backend
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationCfg,
    ArticulationRootPropertiesCfg,
    LightCfg,
    NewtonCollisionPropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    RenderCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    SurfaceDeformableObjectCfg,
    SurfaceDeformablePhysicsCfg,
    UsdSourceCfg,
    VisualObjectCfg,
    VolumeDeformableObjectCfg,
    VolumeDeformablePhysicsCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_common import (
    OUTPUT,
    rack_foot,
    rack_parts,
    worktop_pieces,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_replay import (
    freeze_inputs,
    load_inputs,
    record_preview,
    write_json,
)

__all__ = ["main"]


def _rack_usd(path: Path, parts: list) -> None:
    """Author exact capsule primitives for the public USD-source import API."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    material = UsdShade.Material.Define(stage, "/Looks/metal")
    shader = UsdShade.Shader.Define(stage, "/Looks/metal/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    from pxr import Sdf

    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.62, 0.65, 0.67)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for name, a, b, radius in parts:
        capsule = UsdGeom.Capsule.Define(stage, f"/{name}")
        capsule.CreateRadiusAttr(float(radius))
        capsule.CreateHeightAttr(float(np.linalg.norm(b - a)))
        capsule.CreateAxisAttr(UsdGeom.Tokens.z)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def main() -> None:
    """Declare the physical scene, audit its initialized model, and record it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=OUTPUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--until", type=float)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--pilot-performance", action="store_true")
    parser.add_argument("--measure", type=int, choices=range(1, 6))
    args = parser.parse_args()
    if args.until is not None and args.out is None:
        parser.error("Truncated diagnostics need a separate --out")
    if args.measure is not None:
        if (
            args.pilot_performance
            or args.out is not None
            or args.until is not None
            or args.reference.resolve() != OUTPUT.resolve()
        ):
            parser.error("Formal measurement requires the full current demo")
        args.no_video = True
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_measurement import (
            measurement_directory,
        )

        output = measurement_directory(args.measure)
    elif args.pilot_performance:
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

        output = pilot_directory("embodichain")
    else:
        output = args.out or OUTPUT / "platforms/embodichain"
    inputs = load_inputs(args.reference)
    freeze_inputs(inputs, output, Path(__file__))
    native = inputs["manifest"]
    scene, mesh, joints = native["scene"], inputs["mesh"], inputs["joints"]
    contact, camera_spec = scene["contact"], native["camera"]
    width, height = camera_spec["resolution"]
    parts = rack_parts(inputs["asset"]["rack_rail_centers_local"])
    rack_file = output / "inputs/rack_capsules.usda"
    _rack_usd(rack_file, parts)
    full = {f"arena_0/{name}": None for name in scene["full_surface_contact_targets"]}
    solver = dict(
        solver_type="dexuni",
        mujoco_articulation_paths=("arena_0/w1",),
        joint_mode="kinematic",
        contact_mode="full",
        coupling="one_way",
        full_surface_contact_targets=full,
        vbd_options=dict(
            iterations=scene["solver"]["iterations"],
            friction_epsilon=contact["friction_epsilon"],
            pneumatic_enable_incremental_volume=False,
            enable_cuda_fast_path=True,
            rigid_body_contact_buffer_size=4096,
            rigid_body_particle_contact_buffer_size=8192,
            rigid_contact_hard=False,
            rigid_contact_history=False,
            **contact["self_contact"],
        ),
        collision_options=dict(
            soft_contact_max=65536,
            rigid_contact_max=4096,
            soft_contact_margin=contact["soft_contact_margin"],
            include_static_kinematic_pairs=False,
            enable_rigid_soft_full_surface_contact=True,
            enable_cuda_fast_path=True,
            contact_matching="disabled",
        ),
    )
    sim = None
    robot = bag = bread = pouch = camera = backend = pipeline = model = gas = None
    try:
        sim = SimulationManager(
            SimulationManagerCfg(
                num_envs=1,
                headless=True,
                device="cuda:0",
                width=width,
                height=height,
                enable_entity_gizmo=False,
                robot_ik_gizmo=None,
                render_cfg=RenderCfg(
                    renderer="no-render" if args.no_video else "rt", spp=1
                ),
                physics_cfg=NewtonPhysicsCfg(
                    device="cuda:0",
                    physics_dt=native["effective"]["dt"],
                    num_substeps=scene["solver"]["substeps"],
                    use_cuda_graph=True,
                    collision_cfg=None,
                    solver_cfg=solver,
                ),
            )
        )

        def attrs(
            ke: float,
            kd: float,
            mu: float,
            margin: float = contact["margin"],
            gap: float | None = contact["gap"],
        ) -> RigidBodyPhysicsCfg:
            return RigidBodyPhysicsCfg(
                collision_props=NewtonCollisionPropertiesCfg(margin=margin, gap=gap),
                material_props=NewtonRigidBodyMaterialCfg(
                    ke=ke, kd=kd, dynamic_friction=mu
                ),
            )

        robot = sim.add_articulation(
            ArticulationCfg(
                uid="w1",
                fpath=str(output / "inputs/w1_v030/robot.urdf"),
                asset_physics_mode="overlay",
                build_pk_chain=False,
                attrs=attrs(contact["robot_ke"], 100.0, contact["finger_mu"]),
                root_props=ArticulationRootPropertiesCfg(
                    fixed_base=True,
                    newton_mimic_stabilizer=False,
                    newton_force_sdf=True,
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, index: KinematicJointTrajectoryControl(
                f"{arena}/w1",
                joints["targets"],
                fps=60,
                joint_names=joints["joint_names"].tolist(),
                initialize_pose=True,
            )
        )
        for name, center, size in [*worktop_pieces(), ("rack_foot", *rack_foot())]:
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    shape=CubeCfg(
                        size=size,
                        visual_material=VisualMaterialCfg(
                            uid=name,
                            base_color=(
                                [0.18, 0.2, 0.21, 1.0]
                                if name == "rack_foot"
                                else [0.5, 0.5, 0.5, 1.0]
                            ),
                        ),
                    ),
                    body_type="static",
                    init_pos=tuple(center),
                    attrs=attrs(contact["table_ke"], 100.0, contact["table_mu"]),
                )
            )
        for name, a, b, radius in parts:
            direction = b - a
            pose = np.eye(4)
            pose[:3, :3] = Rotation.align_vectors(
                [direction / np.linalg.norm(direction)], [[0, 0, 1]]
            )[0].as_matrix()
            pose[:3, 3] = 0.5 * (a + b)
            sim.add_rigid_object(
                RigidObjectCfg(
                    uid=name,
                    source=UsdSourceCfg(
                        file_path=str(rack_file), source_prim_path=f"/{name}"
                    ),
                    body_type="static",
                    asset_physics_mode="overlay",
                    init_local_pose=pose,
                    attrs=attrs(2e5, 30.0, scene["rack_mu"], 0.001, None),
                )
            )
        # EC owns an uncontacted 1000 m default ground. Render the reference's
        # gray floor separately; all deformable nodes stay >0.5 m above ground.
        if not args.no_video:
            sim.add_visual_object(
                VisualObjectCfg(
                    uid="floor_visual",
                    shape=CubeCfg(
                        size=[8.0, 8.0, 0.01],
                        visual_material=VisualMaterialCfg(
                            uid="floor", base_color=[0.55, 0.56, 0.54, 1.0]
                        ),
                    ),
                    init_pos=(0, 0, -0.005),
                )
            )
        bag_local = np.asarray(inputs["asset"]["vertices"])
        bag_uv = np.column_stack(
            ((bag_local[:, 0] + 0.16) / 0.32, bag_local[:, 2] / 0.40)
        )
        bm = scene["bag_material"]
        bag = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="bag",
                shape=MeshCfg(
                    vertices=mesh["bag_vertices"],
                    triangles=mesh["bag_triangles"],
                    uv_coords=bag_uv,
                    visual_material=VisualMaterialCfg(
                        uid="film",
                        base_color=[1.0, 1.0, 1.0, 0.9],
                        base_color_texture=str(output / "inputs/bag_print.png"),
                        roughness=0.27,
                    ),
                ),
                particle_radius=bm["particle_radius"],
                attrs=SurfaceDeformablePhysicsCfg(
                    density=bm["density"],
                    surface_props={
                        k: bm[k]
                        for k in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                    contact_material=scene["bag_particle_contact"],
                ),
            )
        )
        sim.add_runtime_control(
            lambda arena, index: FilmDragControl(f"{arena}/bag", **scene["film_drag"])
        )
        br = scene["bread"]
        mu, lam = br["k_mu"], br["k_lambda"]
        youngs = mu * (3 * lam + 2 * mu) / (lam + mu)
        poissons = lam / (2 * (lam + mu))
        bread = sim.add_deformable_object(
            VolumeDeformableObjectCfg(
                uid="bread",
                shape=MeshCfg(
                    vertices=mesh["bread_vertices"],
                    compute_uv=True,
                    visual_material=VisualMaterialCfg(
                        uid="bread",
                        base_color=[1.0, 1.0, 1.0, 1.0],
                        base_color_texture=str(output / "inputs/bread_crust.png"),
                        roughness=0.82,
                    ),
                ),
                tetrahedra=mesh["bread_tetrahedra"],
                particle_radius=br["particle_radius"],
                attrs=VolumeDeformablePhysicsCfg(
                    youngs=youngs,
                    poissons=poissons,
                    density=br["volume_density"],
                    elasticity_damping=br["k_damp"],
                    contact_material=br["contact"],
                ),
            )
        )
        pm = scene["pouch"]
        pouch_local = mesh["pouch_vertices"] - mesh["pouch_vertices"].mean(0)
        pouch = sim.add_deformable_object(
            SurfaceDeformableObjectCfg(
                uid="pouch",
                shape=MeshCfg(
                    vertices=mesh["pouch_vertices"],
                    triangles=mesh["pouch_triangles"],
                    uv_coords=pouch_local[:, :2] / np.array((0.066, 0.092)) + 0.5,
                    visual_material=VisualMaterialCfg(
                        uid="pouch",
                        base_color=[1.0, 1.0, 1.0, 1.0],
                        base_color_texture=str(output / "inputs/milk_pouch.png"),
                        roughness=0.38,
                    ),
                ),
                particle_radius=pm["particle_radius"],
                validate_mesh=True,
                attrs=SurfaceDeformablePhysicsCfg(
                    density=pm["surface_density"],
                    surface_props={
                        k: pm[k]
                        for k in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                    contact_material=pm["contact"],
                    pneumatic={**pm["gas"], "mode": "isothermal"},
                ),
            )
        )
        for side in ("left", "right"):
            sim.register_contact_material_schedule(
                "w1",
                {"stiffness": ((0.0, contact[f"{side}_finger_ke"]),)},
                link_names=(f"{side}_finger1", f"{side}_finger2"),
            )
        # Match the native camera's actual horizontal-FOV API call, not just
        # the field named hfov (the native worker passes its computed vfov).
        actual_fov = 2 * np.arctan(
            np.tan(np.deg2rad(camera_spec["hfov"]) / 2) * height / width
        )
        focal = width / 2 / np.tan(actual_fov / 2)
        if not args.no_video:
            sim.add_light(
                LightCfg(
                    uid="key_light",
                    intensity=50.0,
                    color=(1, 1, 1),
                    init_pos=(1.5, -1.0, 3.0),
                    light_type="point",
                    enable_shadow=False,
                )
            )
            camera = sim.add_sensor(
                CameraCfg(
                    uid="preview",
                    width=width,
                    height=height,
                    near=0.01,
                    far=10.0,
                    intrinsics=(focal, focal, width / 2, height / 2),
                    enable_depth=False,
                    enable_mask=False,
                    extrinsics=CameraCfg.ExtrinsicsCfg(
                        eye=camera_spec["eye"],
                        target=camera_spec["target"],
                        up=(0, 0, 1),
                    ),
                )
            )
        sim.prepare()
        sim.set_ground_plane_visibility(False)
        ids = [robot.joint_names.index(n) for n in joints["joint_names"]]
        initial = torch.as_tensor(
            joints["targets"][:1], dtype=torch.float32, device=sim.device
        )
        robot.set_qpos(initial, joint_ids=ids, target=False)
        robot.set_qpos(initial, joint_ids=ids, target=True)
        robot.clear_dynamics()
        for obj in (bag, bread, pouch):
            for mat in obj.get_visual_material_inst():
                if mat is not None:
                    mat.mat.set_double_sided(True)
        if camera is not None:
            camera.set_intrinsics(
                torch.tensor(
                    [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
                    dtype=torch.float32,
                    device=sim.device,
                )
            )
        backend = get_newton_backend(sim.get_world())
        pipeline = backend.solver.backend.pipeline
        model = backend.solver.model
        particles = {"bag": bag, "bread": bread, "pouch": pouch}
        particle_audit = {}
        for name, obj in particles.items():
            handle = sim.spawn_result.get_particle_set(f"arena_0/{name}")
            positions = obj.data.nodal_pos_w[0].cpu().numpy()
            inverse = handle.get_inverse_masses().numpy()
            if not np.allclose(positions, mesh[f"{name}_vertices"], atol=1e-6, rtol=0):
                raise RuntimeError(f"C01 initial {name} node order/position changed")
            if np.any(inverse <= 0) or not np.isfinite(inverse).all():
                raise RuntimeError(f"C01 {name} has fixed/invalid nodes")
            particle_audit[name] = dict(
                nodes=len(inverse), mass_kg=float((1 / inverse).sum()), pinned_nodes=0
            )
        gas = model.pneumatic
        gas_audit = {
            n: getattr(gas, n).numpy()
            for n in (
                "mode",
                "rest_volume",
                "reference_absolute_pressure",
                "ambient_pressure",
                "bulk_damping",
                "max_absolute_pressure",
            )
        }
        for name in (
            "reference_absolute_pressure",
            "ambient_pressure",
            "bulk_damping",
            "max_absolute_pressure",
        ):
            if not np.allclose(gas_audit[name], pm["gas"][name], rtol=1e-6):
                raise RuntimeError(f"C01 effective gas {name} differs")
        if not np.array_equal(gas_audit["mode"], [0]) or not np.allclose(
            gas_audit["rest_volume"], mesh["pouch_rest_volume"], rtol=1e-5
        ):
            raise RuntimeError("C01 cavity mode/volume differs")
        faces = pipeline.soft_face_rigid_pairs.numpy()
        edges = pipeline.soft_edge_rigid_pairs.numpy()
        selected = set(faces[:, 1]) | set(edges[:, 1])
        surface_audit = {}
        for name in scene["full_surface_contact_targets"]:
            found = {
                i for i, label in enumerate(pipeline.model.shape_label) if name in label
            }
            if not found or not found <= selected:
                raise RuntimeError(f"Missing C01 full-surface support: {name}")
            surface_audit[name] = sorted(found)
        finger_shapes = {
            i
            for i, label in enumerate(pipeline.model.shape_label)
            if any(
                n in label
                for n in (
                    "left_finger1",
                    "left_finger2",
                    "right_finger1",
                    "right_finger2",
                )
            )
        }
        if finger_shapes & selected:
            raise RuntimeError("C01 fingers must use per-node contact")
        # Read the initialized Newton arrays, not just the requested configs.
        material_ids = model.dexuni.particle_material_id.numpy()
        contact_table = model.dexuni.contact_materials.numpy()
        expected_contact = np.array([2000.0, 0.25, 0.4])
        if (material_ids < 0).any() or not np.allclose(
            contact_table[material_ids], expected_contact, rtol=1e-6
        ):
            raise RuntimeError("C01 effective particle contact material differs")
        tet_materials = model.tet_materials.numpy()
        if not np.allclose(tet_materials, [mu, lam, br["k_damp"]], rtol=1e-6):
            raise RuntimeError("C01 effective bread tetrahedron material differs")
        tri_indices = model.tri_indices.numpy().reshape(-1, 3)
        tri_materials = model.tri_materials.numpy()
        offset = 0
        for name, properties in (("bag", bm), ("bread", br), ("pouch", pm)):
            count = len(mesh[f"{name}_vertices"])
            mask = ((tri_indices >= offset) & (tri_indices < offset + count)).all(1)
            if name != "bread" and not np.allclose(
                tri_materials[mask, :3],
                [properties[k] for k in ("tri_ke", "tri_ka", "tri_kd")],
                rtol=1e-6,
            ):
                raise RuntimeError(f"C01 effective {name} membrane material differs")
            offset += count
        shape_types = model.shape_type.numpy()
        shape_scales = model.shape_scale.numpy()
        shape_poses = model.shape_transform.numpy()
        rack_audit = {}
        for name, a, b, radius in parts:
            indices = surface_audit[name]
            if len(indices) != 1:
                raise RuntimeError(f"C01 {name} must remain one capsule")
            index = indices[0]
            if (
                shape_types[index] != int(newton.GeoType.CAPSULE)
                or not np.allclose(
                    shape_scales[index, :2],
                    [radius, np.linalg.norm(b - a) / 2],
                    atol=1e-7,
                )
                or not np.allclose(shape_poses[index, :3], (a + b) / 2, atol=1e-6)
            ):
                raise RuntimeError(f"C01 initialized capsule geometry differs: {name}")
            rack_audit[name] = dict(
                shape_index=index, scale=shape_scales[index], pose=shape_poses[index]
            )
        (output / "analysis").mkdir()
        np.savez_compressed(
            output / "analysis/initial_model.npz",
            particle_material_ids=material_ids,
            contact_materials=contact_table,
            particle_inverse_masses=model.particle_inv_mass.numpy(),
            tetrahedra=model.tet_indices.numpy(),
            tetrahedron_materials=tet_materials,
            triangles=tri_indices,
            triangle_materials=tri_materials,
            bending_materials=model.edge_bending_properties.numpy(),
            shape_types=shape_types,
            shape_scales=shape_scales,
            shape_poses=shape_poses,
            shape_labels=np.asarray(model.shape_label),
            shape_ke=model.shape_material_ke.numpy(),
            shape_kd=model.shape_material_kd.numpy(),
            shape_mu=model.shape_material_mu.numpy(),
        )
        audit = dict(
            particles=particle_audit,
            pneumatic=gas_audit,
            particle_contact_materials_verified=contact_table,
            tetrahedron_materials_verified=np.unique(tet_materials, axis=0),
            membrane_materials_verified=True,
            exact_capsules_verified=rack_audit,
            full_surface_support_shapes=surface_audit,
            fingers_per_node_verified=True,
            model_counts=dict(
                particles=model.particle_count,
                tets=model.tet_count,
                triangles=model.tri_count,
                shapes=model.shape_count,
            ),
            dexsim_version=dexsim.__version__,
            dexsim_module=dexsim.__file__,
            newton_version=newton.__version__,
            newton_module=newton.__file__,
            warp_version=wp.__version__,
            render_camera=dict(
                actual_hfov_degrees=float(np.rad2deg(actual_fov)),
                focal_pixels=float(focal),
                eye=camera_spec["eye"],
                target=camera_spec["target"],
            ),
            disclosed_differences=[
                "EC uncontacted default ground is 1000m with its default material; reference visible gray8m floor reproduced as visual only.",
                "Explicit-tet EC visual UVs generated by mesh load options; physical vertices/tets unchanged.",
            ],
        )
        tcp_local = np.array((0.0, 0.0, 0.125, 1.0))

        def readings() -> dict[str, np.ndarray]:
            tcp = np.stack(
                [
                    (
                        robot.get_link_pose(f"{side}_gripper_base", to_matrix=True)[0]
                        .cpu()
                        .numpy()
                        @ tcp_local
                    )[:3]
                    for side in ("left", "right")
                ]
            )
            return dict(
                time=np.asarray(sim.spawn_result.get_simulation_time()),
                qpos=robot.get_qpos()[0, ids].cpu().numpy(),
                tcp=tcp,
                **{
                    n: o.data.nodal_pos_w[0].cpu().numpy() for n, o in particles.items()
                },
            )

        def step() -> None:
            sim.update(
                physics_dt=native["effective"]["dt"], step=1, render_final_step=False
            )

        def capture() -> np.ndarray:
            with sim.render_frame():
                sim.sync_render_state()
                camera.update()
                return camera.get_data()["color"][0, ..., :3].cpu().numpy().copy()

        def diagnostics() -> dict[str, Any]:
            value = dict(sim.spawn_result.get_solver_diagnostics())
            ke = model.shape_material_ke.numpy()
            finger_ke = {}
            for side in ("left", "right"):
                indices = [
                    i
                    for i, label in enumerate(model.shape_label)
                    if any(f"{side}_finger{n}" in label for n in (1, 2))
                ]
                if not indices or not np.allclose(
                    ke[indices], contact[f"{side}_finger_ke"]
                ):
                    raise RuntimeError(f"C01 effective {side} finger contact differs")
                finger_ke[side] = ke[indices].tolist()
            value.update(
                sim_time=float(sim.spawn_result.get_simulation_time()),
                step_index=int(backend.step_index),
                dexsim_version=dexsim.__version__,
                newton_version=newton.__version__,
                outer_steps_per_command=1,
                effective_finger_stiffness=finger_ke,
            )
            return value

        if args.pilot_performance or args.measure is not None:
            from scripts.benchmark.tech_reports.physics.experiments.dexuni.c01_pilot import (
                record_episode,
            )

            record_episode(
                "embodichain",
                inputs,
                output,
                step,
                readings,
                lambda: wp.synchronize_device(str(sim.device)),
                diagnostics,
                audit,
                **(
                    dict(
                        protocol_path=OUTPUT / "measurement/protocol.json",
                        formal=True,
                        repeat=args.measure,
                    )
                    if args.measure is not None
                    else {}
                ),
            )
        else:
            record_preview(
                inputs,
                output,
                step,
                readings,
                None if args.no_video else capture,
                diagnostics,
                audit,
                until=args.until,
            )
    except Exception as error:
        if not (output / "manifest.json").exists():
            write_json(
                output / "manifest.json",
                dict(
                    task="C01",
                    platform="EmbodiChain",
                    status="preparation_failed",
                    error=repr(error),
                    performance_measurements_started=False,
                ),
            )
        raise
    finally:
        robot = bag = bread = pouch = camera = backend = pipeline = model = gas = None
        if sim is not None:
            sim.destroy(exit_process=False)


if __name__ == "__main__":
    main()
