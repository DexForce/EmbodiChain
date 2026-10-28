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
"""Preview the approved kinematic M06 fold through Isaac Lab and Newton VBD.

Run: /home/oem/app/IsaacLab/.venv/bin/python scripts/benchmark/tech_reports/physics/platforms/dexuni/isaaclab/m06.py
The configured VBD manager extension supplies substep-interpolated rigid poses;
Newton's VBD solver computes the cloth dynamics. --measure records a formal episode.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime as dt
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, ClassVar

import numpy as np
import torch
import warp as wp
import newton
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation

from isaaclab.app import add_launcher_args, launch_simulation
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
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
    VBDSolverCfg,
)
from isaaclab_newton.physics.vbd_manager import NewtonVBDManager
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import (
    NewtonSurfaceDeformableBodyMaterialCfg,
)
from isaaclab_ov.renderers import OVRTXRendererCfg

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))
from scripts.benchmark.tech_reports.physics.platforms.dexuni.dexsim.m06 import (
    DEFAULT_OUTPUT as TASK,
    JOINTS,
    digest,
    observations,
    write_json,
)
from scripts.benchmark.tech_reports.physics.platforms.dexuni.embodichain.m06 import (
    load_inputs,
)

__all__ = ["main"]


@wp.kernel
def _write_kinematic_joints(
    targets: wp.array2d[float],
    counter: wp.array[int],
    q_indices: wp.array[int],
    qd_indices: wp.array[int],
    substeps: int,
    frame_dt: float,
    q_in: wp.array[float],
    qd_in: wp.array[float],
    q_out: wp.array[float],
    qd_out: wp.array[float],
) -> None:
    """Write both endpoints and velocities of one constant-speed substep."""
    joint = wp.tid()
    tick = counter[0]
    frame = wp.min(tick // substeps, targets.shape[0] - 2)
    fraction = float(tick % substeps) / float(substeps)
    q0 = targets[frame, joint]
    delta = targets[frame + 1, joint] - q0
    q_in[q_indices[joint]] = q0 + fraction * delta
    q_out[q_indices[joint]] = q0 + (fraction + 1.0 / float(substeps)) * delta
    velocity = delta / frame_dt
    qd_in[qd_indices[joint]] = velocity
    qd_out[qd_indices[joint]] = velocity


@wp.kernel
def _advance_clock(counter: wp.array[int]) -> None:
    """Advance the graph-resident replay clock after a completed substep."""
    counter[0] = counter[0] + 1


class _KinematicReplay:
    """Fixed-size motion buffers captured with the Isaac Lab solver graph."""

    def __init__(self, model: newton.Model, config: Any) -> None:
        self.model = model
        self.substeps = config.replay_substeps
        self.frame_dt = config.replay_dt
        starts, dofs = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        coord, velocity = {}, {}
        for i, label in enumerate(model.joint_label):
            if starts[i + 1] > starts[i]:
                name = label.rsplit("/", 1)[-1]
                if name in JOINTS:
                    if (
                        name in coord
                        or starts[i + 1] - starts[i] != 1
                        or dofs[i + 1] - dofs[i] != 1
                    ):
                        raise ValueError(f"Unexpected M06 joint layout: {label}")
                    coord[name], velocity[name] = int(starts[i]), int(dofs[i])
        self.q_indices = wp.array(
            [coord[n] for n in JOINTS], dtype=wp.int32, device=model.device
        )
        self.qd_indices = wp.array(
            [velocity[n] for n in JOINTS], dtype=wp.int32, device=model.device
        )
        self.targets = wp.array(
            config.replay_targets, dtype=wp.float32, device=model.device
        )
        self.counter = wp.zeros(1, dtype=wp.int32, device=model.device)

    def step(self, state_in: newton.State, state_out: newton.State) -> None:
        """Publish robot poses at the beginning and end of this substep."""
        wp.launch(
            _write_kinematic_joints,
            len(JOINTS),
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
        newton.eval_fk(self.model, state_in.joint_q, state_in.joint_qd, state_in)
        newton.eval_fk(self.model, state_out.joint_q, state_out.joint_qd, state_out)


class NewtonM06VBDManager(NewtonVBDManager):
    """Use the standard manager dispatch with kinematic robot integration."""

    replay: ClassVar[_KinematicReplay | None] = None
    task_contact: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _prepare_builder_for_finalize(cls, builder: newton.ModelBuilder) -> None:
        """Apply approved flat rest angles and constant gripper materials."""
        builder.edge_rest_angle[:] = [0.0] * len(builder.edge_rest_angle)
        for i, body in enumerate(builder.shape_body):
            if body >= 0 and builder.body_label[body].rsplit("/", 1)[-1] in (
                "fr3_hand",
                "fr3_leftfinger",
                "fr3_rightfinger",
            ):
                builder.shape_material_ke[i] = cls.task_contact["gripper_ke"]
                builder.shape_material_kd[i] = cls.task_contact["gripper_kd"]
                builder.shape_material_mu[i] = cls.task_contact["gripper_friction"]
        # Robot motion is prescribed; flags also communicate that to contacts.
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        super()._prepare_builder_for_finalize(builder)

    @classmethod
    def _create_solver(cls, model: newton.Model, solver_cfg: VBDSolverCfg) -> Any:
        """Construct upstream VBD and the fixed kinematic replay buffers."""
        cls.replay = _KinematicReplay(model, solver_cfg)
        return super()._create_solver(model, solver_cfg)

    @classmethod
    def _step_solver(
        cls,
        state_0: newton.State,
        state_1: newton.State,
        control: newton.Control,
        contacts: Any,
        substep_dt: float,
    ) -> None:
        """Refresh kinematics and one contact set before upstream VBD."""
        cls.replay.step(state_0, state_1)
        cls._collision_pipeline.collide(state_0, contacts)
        super()._step_solver(state_0, state_1, control, contacts, substep_dt)
        wp.launch(
            _advance_clock,
            1,
            inputs=[cls.replay.counter],
            device=cls.replay.model.device,
        )

    @classmethod
    def _simulate_physics_only(cls) -> None:
        """Preserve the VBD manager's BVH, force, swap and sensor lifecycle."""
        if cls.backend.model.particle_count > 0:
            cls._solver.rebuild_bvh(cls.backend.state_0)
        cls._stage_body_forces()
        cls._run_solver_substeps(cls._contacts)
        for callback in cls._post_step_callbacks:
            callback()
        cls._update_sensors(cls._contacts)

    @classmethod
    def _simulate_full(cls) -> None:
        """The kinematic scene has no actuator integration."""
        cls._simulate_physics_only()


@configclass
class _M06VBDSolverCfg(VBDSolverCfg):
    """Bind the kinematic manager through Isaac Lab's normal configuration."""

    class_type: type = NewtonM06VBDManager
    replay_targets: Any = None
    replay_dt: float = 1 / 60
    replay_substeps: int = 10


def write_cloth_usd(
    config: dict[str, Any], mesh: dict[str, np.ndarray], path: Path
) -> None:
    """Author the approved physical vertex order, UVs and fabric appearance."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    root = UsdGeom.Xform.Define(stage, "/Shirt")
    stage.SetDefaultPrim(root.GetPrim())
    surface = UsdGeom.Mesh.Define(stage, "/Shirt/mesh")
    surface.GetPointsAttr().Set([Gf.Vec3f(*map(float, p)) for p in mesh["vertices"]])
    surface.GetFaceVertexCountsAttr().Set([3] * len(mesh["triangles"]))
    surface.GetFaceVertexIndicesAttr().Set(mesh["triangles"].reshape(-1).tolist())
    surface.GetDoubleSidedAttr().Set(True)
    surface.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    uv = UsdGeom.PrimvarsAPI(surface).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    uv.Set([Gf.Vec2f(*map(float, p)) for p in mesh["uv"]])
    material = UsdShade.Material.Define(stage, "/Shirt/Looks/Fabric")
    shader = UsdShade.Shader.Define(stage, "/Shirt/Looks/Fabric/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    reader = UsdShade.Shader.Define(stage, "/Shirt/Looks/Fabric/UV")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, "/Shirt/Looks/Fabric/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        str(TASK / "inputs/mianbu.png")
    )
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
        Gf.Vec4f(0.85, 0.23, 0.16, 1)
    )
    for axis in ("wrapS", "wrapT"):
        texture.CreateInput(axis, Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()


def physics_config(
    config: dict[str, Any], trajectory: dict[str, np.ndarray]
) -> NewtonCfg:
    """Keep the approved budgets, material mixing and self-contact policy."""
    solver, contact = config["solver"], config["contact"]
    NewtonM06VBDManager.task_contact = contact
    return NewtonCfg(
        solver_cfg=_M06VBDSolverCfg(
            replay_targets=trajectory["joint_targets"],
            replay_dt=solver["dt"],
            replay_substeps=solver["num_substeps"],
            iterations=solver["vbd_iterations"],
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=solver["self_contact_radius"],
            particle_self_contact_margin=solver["self_contact_margin"],
            particle_collision_detection_interval=-1,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=40,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.005,
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
            gap=0.001,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            broad_phase="nxn",
            soft_contact_margin=contact["soft_contact_margin"],
            enable_rigid_soft_full_surface_contact=False,
        ),
        num_substeps=solver["num_substeps"],
        collision_decimation=0,
        use_cuda_graph=True,
    )


def scene_config(
    config: dict[str, Any],
    mesh: dict[str, np.ndarray],
    trajectory: dict[str, np.ndarray],
    output: Path,
    video: bool,
) -> InteractiveSceneCfg:
    """Create the robot, table, garment and cameras with Isaac Lab assets."""
    usd = output / "shirt.usda"
    write_cloth_usd(config, mesh, usd)
    cloth = config["cloth"]

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            actuators={},
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(
                    TASK
                    / "inputs/isaaclab/franka/fr3_franka_hand.dexsim/fr3_franka_hand.dexsim.usda"
                ),
                variants={"Physics": "mujoco"},
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(config["robot"]["base_position"]),
                joint_pos={
                    name: float(value)
                    for name, value in zip(JOINTS, trajectory["joint_targets"][0])
                },
            ),
        )
        shirt = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Shirt",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(usd),
                deformable_props=NewtonDeformableBodyPropertiesCfg(),
                physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                    density=cloth["density"],
                    particle_radius=cloth["particle_radius"],
                    **{
                        k: cloth[k]
                        for k in ("tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
                    },
                ),
            ),
            init_state=DeformableObjectCfg.InitialStateCfg(pos=tuple(cloth["center"])),
        )

    for name, center, size in (
        ("table", (0, -0.5, 0.10), (0.82, 0.82, 0.20)),
        ("robot_stand", (-0.52, -0.5, 0.06), (0.22, 0.22, 0.12)),
        ("ground", (0, 0, -0.01), (20, 20, 0.02)),
    ):
        setattr(
            SceneCfg,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                init_state=AssetBaseCfg.InitialStateCfg(pos=center),
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(
                            (0.42, 0.48, 0.52)
                            if name != "ground"
                            else (0.65, 0.65, 0.65)
                        ),
                        roughness=0.9,
                    ),
                ),
            ),
        )
    if video:
        SceneCfg.light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=600)
        )
        for name, width in (("perspective", 960), ("overhead", 720)):
            aperture = 20.955
            focal_px = 720 / 2 / np.tan(np.deg2rad(48) / 2)
            focal_mm = aperture * focal_px / width
            setattr(
                SceneCfg,
                "camera_" + name,
                CameraCfg(
                    prim_path="{ENV_REGEX_NS}/Camera_" + name,
                    width=width,
                    height=720,
                    data_types=["rgb"],
                    update_period=0,
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=float(focal_mm),
                        horizontal_aperture=aperture,
                        clipping_range=(0.01, 10),
                    ),
                    renderer_cfg=OVRTXRendererCfg(),
                ),
            )
    return SceneCfg(num_envs=1, env_spacing=2, replicate_physics=True)


def numpy(value: Any) -> np.ndarray:
    """Copy an Isaac Lab tensor or Warp wrapper to host storage."""
    return (value.torch if hasattr(value, "torch") else value).cpu().numpy().copy()


def configure_cameras(scene: Any, device: str) -> list[Any]:
    """Use explicit OpenGL look-at poses, including the overhead up vector."""
    cameras = []
    for name, eye, target, up in (
        ("perspective", (1.25, 0.65, 1.35), (-0.15, -0.5, 0.36), (0, 0, 1)),
        ("overhead", (0, -0.5, 1.4), (0, -0.5, 0.2), (0, 1, 0)),
    ):
        eye, target, up = np.asarray(eye), np.asarray(target), np.asarray(up)
        forward = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        rotation = np.column_stack((right, np.cross(right, forward), -forward))
        camera = scene["camera_" + name]
        camera.set_world_poses(
            positions=torch.tensor(eye[None, :], device=device, dtype=torch.float32),
            orientations=torch.tensor(
                Rotation.from_matrix(rotation).as_quat()[None, :],
                device=device,
                dtype=torch.float32,
            ),
            convention="opengl",
        )
        cameras.append(camera)
    return cameras


def write_report(output: Path, manifest: dict[str, Any]) -> None:
    """Write one preview report while leaving performance unmeasured."""
    o = manifest["observations"]
    (output / "REPORT.md").write_text(f"""# M06 Isaac Lab 预览

运动学 Franka，原始衣服拓扑与已确认材料，左向右折一次，TCP 在桌面上方 95 mm 松爪。Isaac Lab 管理场景和图重放，Newton VBD 积分自由布料。

## Time & Memory

| 平台 | 性能耗时 | 显存 | 范围 |
| --- | --- | --- | --- |
| Isaac Lab / Newton VBD | 未测量 | 未测量 | 场景预览 |

## Success & Other Metrics

| 状态有限 | 时长 | 左袖抬升 | 向右位移 | Graph | 人工确认 |
| --- | --- | --- | --- | --- | --- |
| {o['all_states_finite']} | {o['actual_end_time']:.3f} s | {o['left_sleeve_peak_lift_m']*1000:.1f} mm | {o['left_sleeve_final_rightward_motion_m']*1000:.1f} mm | {manifest['cuda_graph_status']} | 待观看 |

## Leaderboard

| 平台 | 结论 |
| --- | --- |
| Isaac Lab / Newton VBD | 尚未测量性能或排名 |

[视频](isaaclab.mp4) · [状态](states.npz) · [运行记录](manifest.json)。袖口量是几何描述，不是精确对齐或无贯穿证明。
""")


def run(args: argparse.Namespace) -> None:
    """Execute and validate the complete kinematic fold before recording."""
    config, mesh, trajectory = load_inputs()
    output = args.out_dir.resolve()
    if TASK not in output.parents or output.parts[len(TASK.parts)] not in {
        "preview",
        "history",
    }:
        raise ValueError("Write only under M06 preview/ or history/")
    if (output / "manifest.json").exists():
        raise FileExistsError("Preserve the existing attempt before rerunning")
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    shutil.copy2(__file__, output / "source/m06.py")
    manifest: dict[str, Any] = {
        "task": "M06",
        "platform": "isaaclab",
        "status": "running",
        "scope": "scene_preview_only",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.executable,
        "source_sha256": digest(Path(__file__)),
        "config_sha256": digest(TASK / "inputs/config.json"),
        "trajectory_sha256": digest(TASK / "inputs/trajectory.npz"),
        "versions": {"newton": newton.__version__, "warp": wp.__version__},
        "joint_mode": "kinematic",
        "performance_measured": False,
        "human_scene_approved": False,
        "solver": "Isaac Lab NewtonVBDManager extension + upstream SolverVBD",
        "requested": config["solver"],
    }
    write_json(output / "manifest.json", manifest)
    states: dict[str, list[np.ndarray]] = {
        k: [] for k in ("time", "particles", "qpos", "tcp")
    }
    frame_times = []
    try:
        physics = physics_config(config, trajectory)
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=config["solver"]["dt"],
                    device=args.device,
                    gravity=(0, 0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(
                scene_config(config, mesh, trajectory, output, not args.no_video)
            )
            sim.reset()
            replay = NewtonM06VBDManager.replay
            replay.counter.zero_()
            model = NewtonManager.get_model()
            if not np.allclose(model.edge_rest_angle.numpy(), 0):
                raise RuntimeError(
                    "The cloth rest angle must match the approved flat setting"
                )
            if np.any(model.particle_inv_mass.numpy() <= 0):
                raise RuntimeError("All M06 cloth nodes must remain dynamic")
            np.savez_compressed(
                output / "model.npz",
                **{
                    name: getattr(model, name).numpy()
                    for name in (
                        "particle_q",
                        "particle_mass",
                        "particle_inv_mass",
                        "particle_flags",
                        "particle_radius",
                        "tri_indices",
                        "tri_materials",
                        "edge_indices",
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
                },
            )
            robot, shirt = scene["robot"], scene["shirt"]
            ids = [robot.joint_names.index(name) for name in JOINTS]
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    trajectory["joint_targets"][:1],
                    device=sim.device,
                    dtype=torch.float32,
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            hand = robot.body_names.index(config["robot"]["tcp_body"])
            cameras = configure_cameras(scene, args.device) if not args.no_video else []
            last = len(trajectory["time"]) - 1
            if args.steps is not None:
                last = min(last, args.steps)
            with ExitStack() as stack:
                writer = None
                if cameras:
                    import imageio.v2 as imageio
                    from PIL import Image, ImageDraw, ImageFont

                    font = ImageFont.truetype("DejaVuSans.ttf", 21)
                    writer = stack.enter_context(
                        imageio.get_writer(
                            str(output / "isaaclab.mp4"),
                            fps=30,
                            codec="libx264",
                            quality=8,
                            macro_block_size=2,
                        )
                    )
                for frame in range(last + 1):
                    if frame:
                        scene.write_data_to_sim()
                        sim.step(render=False)
                        scene.update(config["solver"]["dt"])
                    actual = PhysicsManager.get_simulation_time()
                    if not np.isclose(actual, trajectory["time"][frame], atol=1e-9):
                        raise RuntimeError(
                            f"Actual physics time differs at frame {frame}"
                        )
                    if (
                        int(replay.counter.numpy()[0])
                        != frame * config["solver"]["num_substeps"]
                    ):
                        raise RuntimeError(f"Replay counter differs at frame {frame}")
                    particles = numpy(shirt.data.nodal_pos_w)[0]
                    qpos = numpy(robot.data.joint_pos)[0, ids]
                    body_pose = numpy(robot.data.body_link_pose_w)[0, hand]
                    tcp = body_pose[:3] + Rotation.from_quat(body_pose[3:]).apply(
                        config["robot"]["tcp_offset"]
                    )
                    if frame == 0 and not np.allclose(
                        particles, mesh["initial_positions"], atol=1e-6
                    ):
                        raise RuntimeError(
                            "The initial physical cloth node order/positions changed"
                        )
                    if not all(np.isfinite(x).all() for x in (particles, qpos, tcp)):
                        raise RuntimeError(f"Nonfinite state at frame {frame}")
                    if not np.allclose(
                        qpos, trajectory["joint_targets"][frame], atol=1e-5
                    ):
                        write_json(
                            output / "tracking_failure.json",
                            {
                                "frame": frame,
                                "actual": qpos.tolist(),
                                "expected": trajectory["joint_targets"][frame].tolist(),
                                "state_q": NewtonManager.get_state_0()
                                .joint_q.numpy()
                                .tolist(),
                            },
                        )
                        raise RuntimeError(
                            f"Kinematic joint tracking failed at frame {frame}"
                        )
                    if frame % 6 == 0 or frame == last:
                        for key, value in (
                            ("time", actual),
                            ("qpos", qpos),
                            ("tcp", tcp),
                            ("particles", particles),
                        ):
                            states[key].append(np.array(value, copy=True))
                    if cameras and frame > 0 and frame % 2 == 0:
                        sim.render()
                        panel = Image.fromarray(
                            np.concatenate(
                                [
                                    numpy(camera.data.output["rgb"])[0, ..., :3].astype(
                                        np.uint8
                                    )
                                    for camera in cameras
                                ],
                                axis=1,
                            )
                        )
                        draw = ImageDraw.Draw(panel)
                        draw.rectangle((0, 0, 1680, 45), fill="black")
                        phase = trajectory["phase_names"][
                            trajectory["phase_ids"][frame]
                        ]
                        draw.text(
                            (12, 11),
                            f"Isaac Lab | M06 Franka T-shirt LEFT -> RIGHT | t={actual:.2f}s | {phase}",
                            font=font,
                            fill="white",
                        )
                        writer.append_data(np.asarray(panel))
                        frame_times.append(float(actual))
                        if frame == 2 or frame == last - last % 2:
                            panel.save(
                                output / f"{'first' if frame==2 else 'last'}_frame.jpg"
                            )
                    if frame % 120 == 0:
                        print(
                            f"frame={frame}/{last} t={actual:.2f} sleeve={particles[mesh['left_ids']].mean(0).round(4)}",
                            flush=True,
                        )
            arrays = {key: np.stack(value) for key, value in states.items()}
            observation = observations(arrays, mesh)
            desired = np.stack(
                [
                    np.interp(arrays["time"], trajectory["time"], col)
                    for col in trajectory["desired_tcp"].T
                ],
                axis=-1,
            )
            error = float(np.linalg.norm(arrays["tcp"] - desired, axis=-1).max())
            if error > 0.0005:
                raise RuntimeError(f"Recorded TCP error {error} m exceeds IK tolerance")
            if last == len(trajectory["time"]) - 1 and (
                observation["left_sleeve_peak_lift_m"] < 0.05
                or observation["left_sleeve_final_rightward_motion_m"] < 0.15
            ):
                raise RuntimeError(
                    "Insufficient physical sleeve motion to reproduce the approved fold"
                )
            graph = "captured" if NewtonManager._graph is not None else "not captured"
            if graph != "captured":
                raise RuntimeError("Expected a captured CUDA Graph")
            manifest.update(
                status="completed",
                observations=observation,
                full_trajectory=last == len(trajectory["time"]) - 1,
                video_frames=len(frame_times),
                actual_outer_steps=last,
                actual_substeps=int(replay.counter.numpy()[0]),
                cuda_graph_status=graph,
                max_sampled_tcp_error_m=error,
                all_nodes_dynamic=True,
                release_tcp_height_above_table_m=config["motion"]["left_place"][2]
                - 0.2,
                actual_solver_dt=NewtonManager.get_solver_dt(),
                vbd_iterations=config["solver"]["vbd_iterations"],
            )
            write_json(
                output / "frame_times.json", {"fps": 30, "actual_time": frame_times}
            )
            write_report(output, manifest)
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        if states["time"]:
            np.savez_compressed(
                output / "states.npz",
                **{key: np.stack(value) for key, value in states.items()},
            )
        manifest["ended_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_json(output / "manifest.json", manifest)
        NewtonM06VBDManager.replay = None
    print(f"Isaac Lab scene ready: {output}", flush=True)


def measure(args: argparse.Namespace) -> None:
    """Record one authorized full no-camera Newton VBD episode."""
    from scripts.benchmark.tech_reports.physics.experiments.dexuni.m06_measurement import (
        measurement_directory,
        record_episode,
    )

    if args.steps is not None:
        raise ValueError("Formal M06 requires the full episode")
    config, mesh, trajectory = load_inputs()
    output = measurement_directory("isaaclab", args.repeat)
    start = time.perf_counter()
    physics = physics_config(config, trajectory)
    try:
        with launch_simulation(physics, args):
            sim = sim_utils.SimulationContext(
                sim_utils.SimulationCfg(
                    dt=config["solver"]["dt"],
                    device=args.device,
                    gravity=(0, 0, -9.81),
                    physics=physics,
                )
            )
            scene = instantiate(scene_config(config, mesh, trajectory, output, False))
            sim.reset()
            replay = NewtonM06VBDManager.replay
            replay.counter.zero_()
            robot, shirt = scene["robot"], scene["shirt"]
            ids = [robot.joint_names.index(name) for name in JOINTS]
            hand = robot.body_names.index(config["robot"]["tcp_body"])
            robot.write_joint_position_to_sim_index(
                position=torch.tensor(
                    trajectory["joint_targets"][:1],
                    device=sim.device,
                    dtype=torch.float32,
                ),
                joint_ids=ids,
            )
            scene.write_data_to_sim()
            scene.update(0.0)
            model = NewtonManager.get_model()
            if np.any(model.particle_inv_mass.numpy() <= 0) or not np.allclose(
                model.edge_rest_angle.numpy(), 0
            ):
                raise RuntimeError("Frozen dynamic cloth/rest angles changed")
            np.savez_compressed(
                output / "initial_model.npz",
                positions=numpy(shirt.data.nodal_pos_w)[0],
                inverse_mass=model.particle_inv_mass.numpy(),
                triangles=model.tri_indices.numpy(),
                tri_materials=model.tri_materials.numpy(),
                edge_bending_properties=model.edge_bending_properties.numpy(),
                edge_rest_angle=model.edge_rest_angle.numpy(),
                particle_radius=model.particle_radius.numpy(),
            )
            init_s = time.perf_counter() - start

            def advance() -> None:
                """Advance one native platform outer step with graph-resident replay."""
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(config["solver"]["dt"])

            def readings() -> dict[str, np.ndarray]:
                """Read actual backend time and actual robot/cloth state."""
                pose = numpy(robot.data.body_link_pose_w)[0, hand]
                return {
                    "time": np.asarray(PhysicsManager.get_simulation_time()),
                    "particles": numpy(shirt.data.nodal_pos_w)[0],
                    "qpos": numpy(robot.data.joint_pos)[0, ids],
                    "tcp": pose[:3]
                    + Rotation.from_quat(pose[3:]).apply(config["robot"]["tcp_offset"]),
                }

            if not np.allclose(
                readings()["particles"], mesh["initial_positions"], atol=1e-6
            ):
                raise RuntimeError("Frozen cloth initial positions changed")

            def diagnostics() -> dict[str, Any]:
                """Read the actual graph, solver budget and device replay counter."""
                substeps = int(replay.counter.numpy()[0])
                return {
                    "cuda_graph_status": (
                        "captured"
                        if NewtonManager._graph is not None
                        else "not captured"
                    ),
                    "outer_steps": substeps // config["solver"]["num_substeps"],
                    "actual_substeps": substeps,
                    "num_substeps": physics.num_substeps,
                    "simulated_elapsed_s": PhysicsManager.get_simulation_time(),
                    "vbd_iterations": NewtonM06VBDManager._solver.iterations,
                    "particle_color_groups": len(model.particle_color_groups),
                    "solver_dt": NewtonManager.get_solver_dt(),
                    "joint_mode": "kinematic",
                    "solver": "upstream Newton SolverVBD",
                    "vbd_preset": None,
                }

            record_episode(
                "isaaclab",
                output,
                config,
                mesh,
                trajectory,
                advance,
                readings,
                lambda: wp.synchronize_device(args.device),
                diagnostics,
                init_s,
            )
    finally:
        NewtonM06VBDManager.replay = None


def main() -> None:
    """Launch the requested Isaac Lab scene preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=TASK / "preview/isaaclab")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--repeat", type=int, choices=range(1, 6), default=1)
    add_launcher_args(parser)
    parser.set_defaults(visualizer=[])
    args = parser.parse_args()
    if args.measure:
        measure(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
