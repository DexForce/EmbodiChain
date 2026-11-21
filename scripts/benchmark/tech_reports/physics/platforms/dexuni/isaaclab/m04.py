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
"""Replay approved M04 with Isaac Lab and its upstream Newton VBD solver.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m04.py
Only prescribed kinematics, materials and passive drag are adapted; VBD is stock.
Use --pilot-performance for a separate single no-camera timing probe.
Truncated diagnostics require --out <scratch>.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, ClassVar
import numpy as np
import torch
import warp as wp
import newton
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

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
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg
from scripts.benchmark.tech_reports.physics.platforms.dexuni.isaaclab.m02 import (
    NewtonM02VBDManager,
    _M02VBDSolverCfg,
    _numpy,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
    OUTPUT,
    supports,
    digest,
)
from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_replay import (
    CAMERA,
    load_inputs,
    finger_schedule,
    shape_damping,
    texture_uv,
    link_matrices,
    audit_cloth,
    record_preview,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_pilot import (
    pilot_directory,
    record_episode,
)

from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_measurement import (
    measurement_directory,
)

__all__ = ["main"]


@wp.kernel
def _finger_friction(
    counter: wp.array(dtype=wp.int32),
    times: wp.array(dtype=wp.float64),
    values: wp.array(dtype=float),
    shape_ids: wp.array(dtype=wp.int32),
    substep_dt: wp.float64,
    mu: wp.array(dtype=float),
):
    t = wp.float64(counter[0]) * substep_dt
    value = values[0]
    for i in range(times.shape[0]):
        if t >= times[i]:
            value = values[i]
    mu[shape_ids[wp.tid()]] = value


@wp.kernel
def _particle_drag(
    velocity: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    rate: float,
    force: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    force[i] = force[i] - rate * mass[i] * velocity[i]


class NewtonM04VBDManager(NewtonM02VBDManager):
    """Keep upstream VBD and adapt prescribed input schedules inside its graph."""

    task_config: ClassVar[dict[str, Any]] = {}
    friction_buffers: ClassVar[tuple[Any, ...] | None] = None

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        c = cls.task_config["contact"]
        fingers = tuple(f"{s}_finger{k}" for s in ("left", "right") for k in (1, 2))
        # Imported visual shapes stay noncolliding; preserve every collision mesh.
        for i, body in enumerate(builder.shape_body):
            robot = body >= 0
            ke = c["robot_ke"] if robot else c["support_ke"]
            builder.shape_material_ke[i] = ke
            builder.shape_material_kd[i] = shape_damping(ke)
            builder.shape_material_mu[i] = c["open_mu"] if robot else c["table_mu"]
            builder.shape_margin[i] = c["margin"]
            builder.shape_gap[i] = c["gap"]
            if robot and builder.shape_type[i] == int(newton.GeoType.MESH):
                builder.shape_force_sdf[i] = True
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        # The demo explicitly uses flat rest curvature even at stitched seams.
        builder.edge_rest_angle[:] = [
            cls.task_config["cloth"]["edge_rest_angle"]
        ] * len(builder.edge_rest_angle)
        builder.color(include_bending=True, balance_colors=True)

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: Any) -> Any:
        result = super()._create_solver(model, solver_cfg)
        fingers = tuple(f"{s}_finger{k}" for s in ("left", "right") for k in (1, 2))
        shape_body = model.shape_body.numpy()
        ids = [
            i
            for i, b in enumerate(shape_body)
            if b >= 0 and model.body_label[b].rsplit("/", 1)[-1] in fingers
        ]
        if not ids:
            raise ValueError("M04 finger collision shapes were not imported")
        track = np.asarray(finger_schedule(cls.task_config))
        cls.friction_buffers = (
            wp.array(track[:, 0], dtype=wp.float64, device=model.device),
            wp.array(track[:, 1], dtype=float, device=model.device),
            wp.array(ids, dtype=wp.int32, device=model.device),
        )
        return result

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: Any,
        substep_dt: float,
    ) -> None:
        model = cls.replay.model
        times, values, ids = cls.friction_buffers
        wp.launch(
            _finger_friction,
            len(ids),
            inputs=[
                cls.replay.counter,
                times,
                values,
                ids,
                wp.float64(substep_dt),
                model.shape_material_mu,
            ],
            device=model.device,
        )
        wp.launch(
            _particle_drag,
            model.particle_count,
            inputs=[
                state_0.particle_qd,
                model.particle_mass,
                cls.task_config["cloth"]["air_drag_rate"],
                state_0.particle_f,
            ],
            device=model.device,
        )
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)


@configclass
class _M04SolverCfg(_M02VBDSolverCfg):
    """Stock VBD sweeps and approved Coulomb regularization."""

    class_type: type = NewtonM04VBDManager
    friction_epsilon: float = 0.01


def _write_cloth(inputs: dict[str, Any], path: Path) -> None:
    """Author the approved world-space mesh and FabricGray render material."""
    mesh = inputs["mesh"]
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Cloth")
    stage.SetDefaultPrim(root.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/Cloth/mesh")
    surface.CreatePointsAttr(
        [Gf.Vec3f(*map(float, p)) for p in mesh["initial_positions"]]
    )
    surface.CreateFaceVertexCountsAttr([3] * len(mesh["triangles"]))
    surface.CreateFaceVertexIndicesAttr(mesh["triangles"].reshape(-1).tolist())
    surface.CreateDoubleSidedAttr(True)
    surface.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([Gf.Vec2f(*map(float, p)) for p in texture_uv(mesh["vertices"])])
    material = UsdShade.Material.Define(stage, "/Cloth/Looks/FabricGray")
    shader = UsdShade.Shader.Define(stage, "/Cloth/Looks/FabricGray/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    tex = UsdShade.Shader.Define(stage, "/Cloth/Looks/FabricGray/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(OUTPUT / "inputs/shirt/FabricGray/FabricGray_Color.png"))
    )
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    reader = UsdShade.Shader.Define(stage, "/Cloth/Looks/FabricGray/UV")
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
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def _rerender(
    output: Path,
    sim: Any,
    scene: Any,
    robot: Any,
    garment: Any,
    joint_ids: list[int],
    readings: Any,
    capture: Any,
) -> None:
    """Republish actual recorded geometry after a render-only material correction."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.m04_common import (
        phase,
        write_json,
    )

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "completed":
        raise ValueError("Render replay requires a complete physical recording")
    state_hash = digest(output / "states.npz")
    with np.load(output / "states.npz") as data:
        states = {key: data[key] for key in data.files}
    frames = np.flatnonzero(
        (np.arange(len(states["time"])) % 2 == 0) & (states["time"] >= 4.0)
    )
    initial_time = PhysicsManager.get_simulation_time()
    font = ImageFont.truetype("DejaVuSans.ttf", 20)
    errors = dict(particles=0.0, qpos=0.0, link_poses=0.0)
    with imageio.get_writer(
        str(output / "video.mp4"),
        fps=30,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            robot.write_joint_position_to_sim_index(
                position=torch.as_tensor(
                    states["qpos"][frame : frame + 1],
                    device=sim.device,
                    dtype=torch.float32,
                ),
                joint_ids=joint_ids,
            )
            garment.write_nodal_pos_to_sim_index(
                torch.as_tensor(
                    states["particles"][frame : frame + 1],
                    device=sim.device,
                    dtype=torch.float32,
                )
            )
            scene.update(0.0)
            actual = readings()
            for key in errors:
                error = float(np.abs(actual[key] - states[key][frame]).max())
                errors[key] = max(errors[key], error)
                if error > 2e-6:
                    raise RuntimeError(
                        f"Render replay {key} mismatch at {frame}: {error}"
                    )
            t = float(states["time"][frame])
            image = Image.fromarray(capture())
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, image.width, 42), fill="black")
            draw.text(
                (12, 10),
                f"isaaclab | M04 T-shirt fold | t={t-4:.2f}s | {phase(t,manifest['config'])}",
                font=font,
                fill="white",
            )
            writer.append_data(np.asarray(image))
            if any(abs(t - key) < 1e-7 for key in (4, 14, 18, 26, 28, 32)):
                image.save(output / f"stage_{t:05.2f}.jpg")
            if frame % 120 == 0:
                print(f"render replay frame={frame}/1920 t={t:.2f}", flush=True)
    assert PhysicsManager.get_simulation_time() == initial_time
    assert digest(output / "states.npz") == state_hash
    write_json(
        output / "render_replay.json",
        dict(
            source_states_sha256=state_hash,
            source_states_unchanged=True,
            physical_steps=0,
            video_frames=len(frames),
            max_actual_writeback_error=errors,
            correction="FabricGray UV repeat wrapping; no physical geometry interpolation or smoothing",
            physical_frame_indices=frames,
            physical_times_s=states["time"][frames],
        ),
    )
    manifest["render_replay"] = "render_replay.json"
    write_json(manifest_path, manifest)


def main() -> None:
    """Build the approved physical scene, replay and record an untimed preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot-performance", action="store_true")
    mode.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(1, 6))
    parser.add_argument(
        "--rerender",
        action="store_true",
        help="Render saved physical states only; never advance physics",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--until", type=float)
    parser.add_argument("--iterations", type=int, default=20)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.until is not None and args.out is None:
        parser.error("Truncated diagnostics require --out <scratch>")
    os.environ.setdefault("ISAAC_LAB_OVRTX_READ_GPU_TRANSFORMS", "0")
    if args.pilot_performance:
        if (
            args.out is not None
            or args.until is not None
            or args.rerender
            or args.iterations != 20
        ):
            parser.error(
                "Pilot uses the full accepted episode and its separate fixed output"
            )
        args.no_video = True
    if args.measure:
        if (
            args.repeat is None
            or args.out is not None
            or args.until is not None
            or args.rerender
            or args.iterations != 20
        ):
            parser.error(
                "Measurement requires --repeat 1..5 and the full accepted configuration"
            )
        args.no_video = True
    elif args.repeat is not None:
        parser.error("--repeat requires --measure")
    inputs = load_inputs()
    cfg, mesh, commands = inputs["config"], inputs["mesh"], inputs["commands"]
    sol, c, cloth = cfg["solver"], cfg["contact"], cfg["cloth"]
    output = (
        pilot_directory("isaaclab")
        if args.pilot_performance
        else (
            measurement_directory("isaaclab", args.repeat)
            if args.measure
            else args.out or OUTPUT / "preview/isaaclab"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    assets = OUTPUT / "inputs/isaaclab"
    meta = json.loads((assets / "w1_usd_path.json").read_text())
    for p, h in meta["sha256"].items():
        if digest(assets / "w1_usd" / p) != h:
            raise ValueError(f"Converted W1 changed: {p}")
    cloth_usd = assets / "cloth.usda"
    if not cloth_usd.exists():
        _write_cloth(inputs, cloth_usd)
    NewtonM04VBDManager.task_config = cfg
    physics = NewtonCfg(
        solver_cfg=_M04SolverCfg(
            replay_targets=commands["targets"],
            replay_joint_names=inputs["names"],
            replay_dt=sol["dt"],
            replay_substeps=sol["num_substeps"],
            iterations=args.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=sol["self_contact_radius"],
            particle_self_contact_margin=sol["self_contact_margin"],
            particle_rest_shape_contact_exclusion_radius=0.03,
            particle_collision_detection_interval=-1,
            particle_topological_contact_filter_threshold=1,
        ),
        soft_contact_cfg=NewtonSoftContactCfg(
            soft_contact_ke=c["particle_ke"],
            soft_contact_kd=c["particle_kd"],
            soft_contact_mu=c["particle_mu"],
        ),
        default_shape_cfg=NewtonShapeCfg(
            ke=c["robot_ke"],
            kd=shape_damping(c["robot_ke"]),
            mu=c["open_mu"],
            margin=c["margin"],
            gap=c["gap"],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=c["soft_margin"],
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
                usd_path=str(assets / "w1_usd/robot/robot.usda"),
                variants={"Physics": "mujoco"},
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(cfg["robot"]["base_position"]),
                joint_pos=dict(zip(inputs["names"], commands["targets"][0].tolist())),
            ),
        )
        garment = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cloth",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(cloth_usd),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    density=cloth["surface_density"],
                    particle_radius=cloth["particle_radius"],
                    **{
                        key: cloth[key]
                        for key in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                ),
            ),
        )
        if not args.no_video:
            light = AssetBaseCfg(
                prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600.0)
            )

    material = UsdPhysicsRigidBodyMaterialCfg(
        static_friction=c["table_mu"], dynamic_friction=c["table_mu"]
    )
    for name, center, size in supports(cfg):
        color = (0.53, 0.48, 0.39) if name == "table" else (0.30, 0.31, 0.33)
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=material,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
            ),
        )
    width, height = CAMERA["resolution"]
    if not args.no_video:
        aperture = 20.955
        focal = aperture / 2 / np.tan(np.deg2rad(CAMERA["horizontal_fov_deg"]) / 2)
        SceneCfg.camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            update_period=0.0,
            height=height,
            width=width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(focal),
                horizontal_aperture=aperture,
                clipping_range=(0.01, 10),
            ),
            renderer_cfg=OVRTXRendererCfg(),
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
            replay = NewtonM04VBDManager.replay
            replay.counter.zero_()
            model = NewtonManager.get_model()
            robot, garment = scene["robot"], scene["garment"]
            ids = [robot.joint_names.index(n) for n in inputs["names"]]
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    commands["targets"][:1], device=sim.device, dtype=torch.float32
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            initial = _numpy(garment.data.nodal_pos_w)[0]
            # Single garment: preserve direct global order, with an explicit check.
            if not np.allclose(model.particle_q.numpy(), initial, atol=1e-7, rtol=0):
                raise ValueError("M04 global particle order differs from cloth asset")
            inverse = model.particle_inv_mass.numpy()
            audit = audit_cloth(inputs, initial, inverse)
            fields = (
                "particle_mass",
                "tri_indices",
                "tri_materials",
                "edge_rest_angle",
                "edge_bending_properties",
                "shape_material_ke",
                "shape_material_kd",
                "shape_material_mu",
                "shape_margin",
                "shape_gap",
                "shape_body",
                "shape_flags",
            )
            if not args.rerender:
                np.savez_compressed(
                    output / "initial_model.npz",
                    particles=initial,
                    inverse_mass=inverse,
                    shape_label=np.asarray(model.shape_label),
                    body_label=np.asarray(model.body_label),
                    **{k: getattr(model, k).numpy() for k in fields},
                )
            links = list(robot.body_names)
            hands = [
                links.index(f"{side}_{name}")
                for side in ("left", "right")
                for name in ("gripper_base", "finger1", "finger2")
            ]
            tcp_local = np.r_[cfg["robot"]["tcp_offset"], 1]
            camera = None
            if not args.no_video:
                camera = scene["camera"]
                camera.set_world_poses_from_view(
                    eyes=torch.tensor([CAMERA["eye"]], device=sim.device),
                    targets=torch.tensor([CAMERA["target"]], device=sim.device),
                )

            def readings() -> dict[str, np.ndarray]:
                matrices = link_matrices(_numpy(robot.data.body_link_pose_w)[0])
                hand = matrices[hands]
                return dict(
                    time=np.asarray(PhysicsManager.get_simulation_time()),
                    qpos=_numpy(robot.data.joint_pos)[0, ids],
                    particles=_numpy(garment.data.nodal_pos_w)[0],
                    velocities=_numpy(garment.data.nodal_vel_w)[0],
                    tcp=(hand[[0, 3]] @ tcp_local)[:, :3],
                    hand_poses=hand,
                    link_poses=matrices,
                )

            def step() -> None:
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sol["dt"])

            def capture() -> np.ndarray:
                sim.render()
                return _numpy(camera.data.output["rgb"])[0, ..., :3]

            def diagnostics() -> dict[str, Any]:
                v = NewtonM04VBDManager._solver
                assert v.iterations == args.iterations
                return dict(
                    cuda_graph_status=(
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    actual_substeps=int(replay.counter.numpy()[0]),
                    num_substeps=physics.num_substeps,
                    dt=sol["dt"],
                    solver_dt=NewtonManager.get_solver_dt(),
                    joint_mode="kinematic",
                    coupling="one_way",
                    vbd_iterations=v.iterations,
                    vbd_preset=None,
                    solver="upstream Newton SolverVBD",
                    particle_color_groups=len(model.particle_color_groups),
                    particle_enable_self_contact=True,
                    friction_epsilon=v.friction_epsilon,
                    air_drag_rate=cloth["air_drag_rate"],
                    finger_friction_keyframes=finger_schedule(cfg),
                    versions=dict(newton=newton.__version__, warp=wp.__version__),
                    camera_intrinsics=(
                        None
                        if camera is None
                        else _numpy(camera.data.intrinsic_matrices)[0].tolist()
                    ),
                )

            if args.rerender:
                _rerender(output, sim, scene, robot, garment, ids, readings, capture)
                return

            if args.pilot_performance or args.measure:
                record_episode(
                    "isaaclab",
                    inputs,
                    output,
                    step,
                    readings,
                    lambda: wp.synchronize_device(args.device),
                    diagnostics,
                    audit,
                    links,
                    repeat=args.repeat if args.measure else None,
                )
                return

            record_preview(
                "isaaclab",
                inputs,
                output,
                step,
                readings,
                None if args.no_video else capture,
                diagnostics,
                audit,
                links,
                until=args.until,
            )
    finally:
        NewtonM04VBDManager.replay = None
        NewtonM04VBDManager.friction_buffers = None


if __name__ == "__main__":
    main()
