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
"""Replay approved four-layer M03 with Isaac Lab and upstream Newton VBD.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m03.py
Stock VBD uses 20 ordinary sweeps; --iterations selects a diagnostic budget.
No performance samples; --no-video records one full diagnostic episode.
"""

from __future__ import annotations
import argparse
import time
import json
import os
from pathlib import Path
import sys
from typing import Any, ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parents[7]))
from isaaclab.app import add_launcher_args, launch_simulation
import numpy as np
import torch
import newton
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
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
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg
from scipy.spatial.transform import Rotation
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m02 import (
    NewtonM02VBDManager,
    _M02VBDSolverCfg,
    _numpy,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_common import (
    OUTPUT,
    CAMERAS,
    load_inputs,
    boxes,
    audit_layers,
    record_preview,
    digest,
)

__all__ = ["main"]


class NewtonM03VBDManager(NewtonM02VBDManager):
    """Retain stock VBD and named graph replay with exact finger materials."""

    task_contact: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        """Prescribe robot motion and match native six-color cloth scheduling."""
        for i, body in enumerate(builder.shape_body):
            if body >= 0 and builder.body_label[body].rsplit("/", 1)[-1] in tuple(
                f"{side}_finger{k}" for side in ("left", "right") for k in (1, 2)
            ):
                builder.shape_material_ke[i] = cls.task_contact["gripper_ke"]
                builder.shape_material_kd[i] = cls.task_contact["gripper_kd"]
                builder.shape_material_mu[i] = cls.task_contact["gripper_friction"]
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        builder.color(include_bending=True, balance_colors=True)


@configclass
class M03VBDSolverCfg(_M02VBDSolverCfg):
    """Stock VBD with native friction smoothing and contact settings."""

    class_type: type = NewtonM03VBDManager
    friction_epsilon: float = 0.01


def write_cloth_usd(
    vertices: np.ndarray, triangles: np.ndarray, color: tuple, path: Path
) -> None:
    """Author an independent dynamic cloth in the exact frozen node order."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Cloth")
    stage.SetDefaultPrim(root.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/Cloth/mesh")
    surface.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in vertices])
    surface.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    surface.GetFaceVertexIndicesAttr().Set(triangles.reshape(-1).tolist())
    surface.GetDoubleSidedAttr().Set(True)
    surface.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    surface.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
    material = UsdShade.Material.Define(stage, "/Cloth/Looks/Fabric")
    shader = UsdShade.Shader.Define(stage, "/Cloth/Looks/Fabric/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def main() -> None:
    """Build exact scene assets and record untimed full states and dual views."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    os.environ.setdefault("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "0")
    init_started = time.perf_counter()
    if args.measure:
        args.no_video = True
    inputs = load_inputs()
    root = inputs["output_root"]
    inputs["stock_iterations"] = args.iterations
    cfg = inputs["config"]
    sol = cfg["solver"]
    contact = cfg["contact"]
    cloth = cfg["cloth"]
    output = root / "preview/isaaclab"
    output.mkdir(parents=True, exist_ok=True)
    if args.measure:
        from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_measurement import (
            measurement_directory,
        )

        output = measurement_directory("isaaclab", args.repeat)
    assets = OUTPUT / "inputs/isaaclab"
    meta = json.loads((assets / "w1_usd_path.json").read_text())
    usd_root = assets / "w1_usd"
    for p, expected in meta["sha256"].items():
        if digest(usd_root / p) != expected:
            raise ValueError(f"Converted W1 asset changed: {p}")
    NewtonM03VBDManager.task_contact = contact
    physics = NewtonCfg(
        solver_cfg=M03VBDSolverCfg(
            replay_targets=inputs["targets"],
            replay_joint_names=inputs["names"],
            replay_dt=sol["dt"],
            replay_substeps=sol["num_substeps"],
            iterations=args.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=sol["self_contact_radius"],
            particle_self_contact_margin=sol["self_contact_margin"],
            particle_rest_shape_contact_exclusion_radius=sol[
                "rest_shape_exclusion_radius"
            ],
            particle_collision_detection_interval=-1,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=64,
            particle_topological_contact_filter_threshold=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=contact["particle_ke"],
            soft_contact_kd=contact["particle_kd"],
            soft_contact_mu=contact["particle_friction"],
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=contact["shape_ke"],
            kd=contact["shape_kd"],
            mu=contact["table_friction"],
            margin=0.001,
            gap=0.002,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=contact["soft_contact_margin"],
            enable_rigid_soft_full_surface_contact=False,
        ),
        num_substeps=sol["num_substeps"],
        collision_decimation=0,
        use_cuda_graph=sol["use_cuda_graph"],
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={},
            spawn=sim_utils.UsdFileCfg(
                usd_path=meta["usd_path"], variants={"Physics": "mujoco"}
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(cfg["robot"]["base_position"]),
                joint_pos=dict(zip(inputs["names"], inputs["targets"][0].tolist())),
            ),
        )
        if not args.no_video:
            light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0)
            )

    for k in range(4):
        path = assets / f"cloth_{k}.usda"
        if args.measure and not path.exists():
            raise ValueError("Frozen cloth USD is missing")
        if not args.measure:
            write_cloth_usd(
                inputs["mesh"]["initial_positions"][k],
                inputs["mesh"]["triangles"],
                tuple(cloth["colors"][k]),
                path,
            )
        setattr(
            SceneCfg,
            f"cloth_{k}",
            DeformableObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cloth_" + str(k),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(path),
                    deformable_props=NewtonDeformableBodyPropertiesCfg(),
                    physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                        density=cloth["surface_density"],
                        particle_radius=cloth["particle_radius"],
                        **{
                            key: cloth[key]
                            for key in (
                                "tri_ke",
                                "tri_ka",
                                "tri_kd",
                                "edge_ke",
                                "edge_kd",
                            )
                        },
                    ),
                ),
            ),
        )
    static_material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=contact["table_friction"],
        dynamic_friction=contact["table_friction"],
    )
    for name, center, size, color, collision in boxes(cfg):
        if not collision and args.no_video:
            continue
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=(
                        sim_utils.CollisionPropertiesCfg() if collision else None
                    ),
                    physics_material=static_material if collision else None,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )
    if not args.no_video:
        for name, ((w, h), eye, target) in CAMERAS.items():
            aperture = 20.955
            focal = aperture * (h / w) / 2 / np.tan(np.deg2rad(48) / 2)
            setattr(
                SceneCfg,
                name,
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/" + name,
                    update_period=0.0,
                    height=h,
                    width=w,
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(focal),
                        horizontal_aperture=aperture,
                        clipping_range=(0.01, 10.0),
                    ),
                    renderer_cfg=OVRTXRendererCfg(),
                ),
            )
    try:
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=sol["dt"],
                    device=args.device,
                    gravity=(0, 0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(
                SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
            )
            sim.reset()
            replay = NewtonM03VBDManager.replay
            replay.counter.zero_()
            model = NewtonManager.get_model()
            robot = scene["robot"]
            layers = [scene[f"cloth_{k}"] for k in range(4)]
            ids = [robot.joint_names.index(n) for n in inputs["names"]]
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    inputs["targets"][:1], device=sim.device, dtype=torch.float32
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            initial = np.stack([_numpy(layer.data.nodal_pos_w)[0] for layer in layers])
            # Resolve global particle ranges from actual initialized coordinates,
            # rather than assuming asset insertion order.
            global_positions = model.particle_q.numpy()
            indices = []
            for positions in initial:
                distances = np.linalg.norm(
                    global_positions[:, None, :] - positions[None, :, :], axis=2
                )
                chosen = distances.argmin(0)
                if (
                    len(set(chosen.tolist())) != 961
                    or distances[chosen, np.arange(961)].max() > 1e-6
                ):
                    raise ValueError("Cloth model particle mapping differs")
                indices.append(chosen)
            indices = np.stack(indices)
            if len(set(indices.reshape(-1).tolist())) != 3844:
                raise ValueError("Cloth layers share particles")
            inverse = model.particle_inv_mass.numpy()[indices]
            model_arrays = {
                name: getattr(model, name).numpy()
                for name in (
                    "particle_q",
                    "particle_mass",
                    "particle_radius",
                    "tri_indices",
                    "tri_materials",
                    "edge_rest_angle",
                    "edge_bending_properties",
                    "shape_material_ke",
                    "shape_material_kd",
                    "shape_material_mu",
                    "shape_margin",
                    "shape_gap",
                    "body_flags",
                )
                if hasattr(getattr(model, name, None), "numpy")
            }
            np.savez_compressed(
                output / "model.npz",
                shape_label=np.asarray([str(s) for s in model.shape_label]),
                body_label=np.asarray([str(s) for s in model.body_label]),
                shape_body=model.shape_body.numpy(),
                **model_arrays,
            )
            audit = audit_layers(
                output, inputs, initial, inverse, model_particle_indices=indices
            )
            cameras = []
            if not args.no_video:
                for name, (resolution, eye, target) in CAMERAS.items():
                    camera = scene[name]
                    camera.set_world_poses_from_view(
                        eyes=torch.tensor([eye], device=sim.device),
                        targets=torch.tensor([target], device=sim.device),
                    )
                    cameras.append(camera)
            bodies = [
                robot.body_names.index(f"{side}_gripper_base")
                for side in ("left", "right")
            ]

            def readings() -> dict[str, np.ndarray]:
                pose = _numpy(robot.data.body_link_pose_w)[0]
                return dict(
                    time=np.asarray(PhysicsManager.get_simulation_time()),
                    qpos=_numpy(robot.data.joint_pos)[0, ids],
                    particles=np.stack(
                        [_numpy(layer.data.nodal_pos_w)[0] for layer in layers]
                    ),
                    velocities=np.stack(
                        [_numpy(layer.data.nodal_vel_w)[0] for layer in layers]
                    ),
                    tcp=np.stack(
                        [
                            pose[i, :3]
                            + Rotation.from_quat(pose[i, 3:]).apply(
                                cfg["robot"]["tcp_offset"]
                            )
                            for i in bodies
                        ]
                    ),
                )

            def step(frame: int) -> None:
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sol["dt"])
                if (
                    not args.measure
                    and int(replay.counter.numpy()[0]) != frame * sol["num_substeps"]
                ):
                    raise ValueError(f"Replay substep clock mismatch at {frame}")

            def capture() -> list[np.ndarray]:
                sim.render()
                return [
                    _numpy(camera.data.output["rgb"])[0, ..., :3] for camera in cameras
                ]

            def diagnostics() -> dict:
                v = NewtonM03VBDManager._solver
                assert (
                    v.iterations == args.iterations
                    and len(model.particle_color_groups) == 6
                )
                return dict(
                    outer_steps_per_command=1,
                    simulated_elapsed_s=PhysicsManager.get_simulation_time(),
                    actual_substeps=int(replay.counter.numpy()[0]),
                    num_substeps=physics.num_substeps,
                    solver_dt=NewtonManager.get_solver_dt(),
                    cuda_graph_status=(
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    vbd_iterations=v.iterations,
                    vbd_preset=None,
                    solver="upstream Newton SolverVBD; kinematic replay through VBD manager",
                    joint_mode="kinematic",
                    coupling="one_way",
                    particle_color_groups=len(model.particle_color_groups),
                    particle_coloring_include_bending=True,
                    particle_coloring_balance_colors=True,
                    particle_enable_self_contact=(
                        v.particle_enable_self_contact
                        if hasattr(v, "particle_enable_self_contact")
                        else True
                    ),
                    friction_epsilon=0.01,
                    newton_version=newton.__version__,
                    warp_version=wp.__version__,
                    cameras={
                        camera.cfg.prim_path: {
                            "intrinsics": _numpy(camera.data.intrinsic_matrices)[0]
                        }
                        for camera in cameras
                    },
                )

            if args.measure:
                from scripts.benchmark.tech_reports.physics.experiments.dexuni.m03_measurement import (
                    record_episode,
                )

                record_episode(
                    "isaaclab",
                    args.repeat,
                    inputs,
                    output,
                    step,
                    readings,
                    lambda: wp.synchronize_device(args.device),
                    diagnostics,
                    audit,
                    time.perf_counter() - init_started,
                )
            else:
                record_preview(
                    "isaaclab",
                    inputs,
                    step,
                    readings,
                    None if args.no_video else capture,
                    diagnostics,
                    audit,
                    Path(__file__),
                )

    finally:
        NewtonM03VBDManager.replay = None


if __name__ == "__main__":
    main()
