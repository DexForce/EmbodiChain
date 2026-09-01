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

"""Fold a live Newton cloth T-shirt with a trajectory-driven DexForce W1."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ArticulationRootPropertiesCfg,
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    CollisionPropertiesCfg,
    JointDrivePropertiesCfg,
    JointDynamicsPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonPhysicsCfg,
    NewtonRigidBodyMaterialCfg,
    NewtonRigidBodyPhysicsCfg,
    RigidBodyMaterialCfg,
    RigidBodyPhysicsCfg,
    RigidObjectCfg,
    RobotCfg,
    RenderCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import ClothObject, RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg, MeshCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.utils import logger

DEFAULT_DT = 1.0 / 60.0
DEFAULT_TRAJECTORY_TIME_SCALE = 4.0
NUM_SUBSTEPS = 9
SOLVER_ITERATIONS = 20
FPS_LOG_INTERVAL = 120

ASSET_DATASET = "DeformableDemoData"
ANNIVERSARY_MATERIAL = "anniversary"
ANNIVERSARY_VISUAL_OBJ = "shirt_with_front_anniversary_decal_fold_atlas.obj"
ANNIVERSARY_TEXTURE = "shirt_with_front_anniversary_decal_fold_atlas.png"
TEXTURE_FILE_MAP = {
    "mianbu": "mianbu.png",
    "shabu": "shabu.png",
    "mabu": "mabu.png",
    "pige": "pige.png",
    "jinduan": "jinduan.png",
    "niuzai": "niuzai.png",
    "wenli": "wenli.png",
}

TABLE_POSITION = (0.55, 0.0, 1.15)
TABLE_SIZE = (0.52, 1.24, 0.05)
GROUND_POSITION = (0.0, 0.0, -0.01)
GROUND_SIZE = (8.0, 8.0, 0.02)
SHIRT_POSITION = (0.55, 0.0, 1.189)
SHIRT_SCALE = 0.0080 * 0.8
CONTACT_SCRIPT_TRANSITIONS = np.asarray(
    [4.21, 16.8, 19.0, 22.18, 27.4, 31.4],
    dtype=np.float32,
)
CONTACT_SCRIPT_TO_SIMULATION_OFFSET = 1.2


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Physics frames to run; defaults to the complete cached trajectory.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Outer EmbodiChain physics timestep in seconds.",
    )
    parser.add_argument(
        "--trajectory-time-scale",
        type=float,
        default=DEFAULT_TRAJECTORY_TIME_SCALE,
        help="Playback-speed multiplier for the cached W1 trajectory.",
    )
    parser.add_argument(
        "--static-w1",
        action="store_true",
        help="Keep W1 at the first trajectory pose for scene inspection.",
    )
    parser.add_argument(
        "--disable-w1-collision",
        action="store_true",
        help="Disable particle collision on all W1 links for debugging.",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Use direct Newton stepping and enable particle-friction scheduling.",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep after each frame to approximate wall-clock playback.",
    )
    parser.add_argument(
        "--cloth-material",
        choices=(ANNIVERSARY_MATERIAL, *TEXTURE_FILE_MAP),
        default=ANNIVERSARY_MATERIAL,
        help=(
            "Use the authored anniversary atlas or one of the tiled fabric "
            "textures packaged with the reference scene."
        ),
    )
    parser.add_argument(
        "--w1-urdf",
        type=Path,
        default=None,
        help="Optional W1 URDF override.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Optional cached W1 trajectory override.",
    )
    parser.set_defaults(device="cuda", physics="newton", renderer="rt")
    args = parser.parse_args()

    if args.physics != "newton":
        parser.error("T-shirt folding requires --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("Newton MJVBD cloth simulation requires a CUDA device.")
    if args.num_envs != 1:
        parser.error("This trajectory scene currently supports --num_envs 1.")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        parser.error("--dt must be finite and positive.")
    if (
        not math.isfinite(args.trajectory_time_scale)
        or args.trajectory_time_scale <= 0.0
    ):
        parser.error("--trajectory-time-scale must be finite and positive.")
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be non-negative.")
    return args


def resolve_assets(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path | None, Path | None]:
    """Resolve the W1, trajectory, simulation mesh, and selected visual assets."""
    asset_root = Path(get_data_path(f"{ASSET_DATASET}/fold_tshirt"))
    texture_root = asset_root.parent / "textures"

    urdf_path = asset_root / "W1-hand-obj" / "DexforceW1V021_visual_collision.urdf"
    trajectory_path = asset_root / "fold_tshirt.npz"
    shirt_asset_root = asset_root / "shirt_front_decal_asset"
    shirt_mesh_path = shirt_asset_root / "shirt_mesh.txt"
    ground_texture_path = texture_root / "ground.png"
    visual_mesh_path: Path | None = None
    texture_path: Path | None = None

    if args.w1_urdf is not None:
        urdf_path = args.w1_urdf.expanduser().resolve()
    if args.trajectory is not None:
        trajectory_path = args.trajectory.expanduser().resolve()
    if args.cloth_material == ANNIVERSARY_MATERIAL:
        visual_mesh_path = shirt_asset_root / ANNIVERSARY_VISUAL_OBJ
        texture_path = shirt_asset_root / ANNIVERSARY_TEXTURE
    else:
        texture_path = texture_root / TEXTURE_FILE_MAP[args.cloth_material]

    required = [
        urdf_path,
        trajectory_path,
        shirt_mesh_path,
        ground_texture_path,
    ]
    if visual_mesh_path is not None:
        required.extend(
            [
                visual_mesh_path,
                visual_mesh_path.with_suffix(".mtl"),
            ]
        )
    if texture_path is not None:
        required.append(texture_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "W1 fold asset package is incomplete; missing: " + ", ".join(missing)
        )
    return (
        urdf_path,
        trajectory_path,
        shirt_mesh_path,
        ground_texture_path,
        visual_mesh_path,
        texture_path,
    )


def load_shirt_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and normalize the reference shirt simulation mesh."""
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "shirt_vertices":
            section = "vertices"
            continue
        if line == "shirt_indices":
            section = "indices"
            continue
        if section is None or ":" not in line:
            raise RuntimeError(f"Unexpected shirt mesh line: {line!r}.")
        _, raw_values = line.split(":", 1)
        values = raw_values.split()
        if len(values) < 3:
            raise RuntimeError(f"Incomplete shirt mesh line: {line!r}.")
        if section == "vertices":
            vertices.append([float(value) for value in values[:3]])
        else:
            triangles.append([int(value) for value in values[:3]])

    vertex_array = np.asarray(vertices, dtype=np.float32)
    triangle_array = np.asarray(triangles, dtype=np.int32)
    if vertex_array.ndim != 2 or vertex_array.shape[1:] != (3,) or not len(vertices):
        raise RuntimeError(f"{path} contains no valid shirt vertices.")
    if (
        triangle_array.ndim != 2
        or triangle_array.shape[1:] != (3,)
        or not len(triangles)
    ):
        raise RuntimeError(f"{path} contains no valid shirt triangles.")
    if np.any(triangle_array < 0) or np.any(triangle_array >= len(vertex_array)):
        raise RuntimeError(f"{path} contains out-of-range triangle indices.")

    minimum = vertex_array.min(axis=0)
    maximum = vertex_array.max(axis=0)
    vertex_array[:, :2] -= 0.5 * (minimum[:2] + maximum[:2])
    vertex_array[:, 2] -= minimum[2]
    return vertex_array * SHIRT_SCALE, triangle_array


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute area-weighted per-vertex normals."""
    normals = np.zeros_like(vertices)
    for triangle in triangles:
        normal = np.cross(
            vertices[triangle[1]] - vertices[triangle[0]],
            vertices[triangle[2]] - vertices[triangle[0]],
        )
        normals[triangle] += normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = lengths[:, 0] > 0.0
    normals[valid] /= lengths[valid]
    return normals


def planar_uv(vertices: np.ndarray) -> np.ndarray:
    """Generate the tiled planar UVs used by the fabric material variants."""
    uv_coords = vertices[:, :2].copy()
    uv_coords -= uv_coords.min(axis=0)
    uv_coords /= np.maximum(uv_coords.max(axis=0), 1.0e-8)
    uv_coords[:, 1] = 1.0 - uv_coords[:, 1]
    return uv_coords * 2.0


def load_trajectory(path: Path, time_scale: float) -> tuple[np.ndarray, float]:
    """Load the cached W1 public-qpos trajectory and compute playback FPS."""
    with np.load(path) as archive:
        trajectory = np.asarray(archive["robot_qpos"], dtype=np.float32)
        source_dt = float(np.asarray(archive["dt"]).reshape(-1)[0])
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 3:
        raise ValueError(
            f"Expected a non-empty trajectory with shape [frames, dof], got "
            f"{trajectory.shape}."
        )
    if not np.isfinite(trajectory).all():
        raise ValueError("W1 trajectory must contain only finite values.")
    if not math.isfinite(source_dt) or source_dt <= 0.0:
        raise ValueError("W1 trajectory dt must be finite and positive.")

    # The cache already follows the public Spawn qpos order. Keep all columns
    # intact while locking the three leg joints so the torso stays at table height.
    trajectory = np.ascontiguousarray(trajectory, dtype=np.float32).copy()
    trajectory[:, :3] = 0.0
    trajectory_fps = (1.0 / source_dt) * time_scale / DEFAULT_TRAJECTORY_TIME_SCALE
    return trajectory, trajectory_fps


def initialize_simulation(
    args: argparse.Namespace,
    *,
    use_cuda_graph: bool,
) -> SimulationManager:
    """Create the EmbodiChain manager with the reference MJVBD settings."""
    cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=args.headless,
        device=args.device,
        gpu_id=args.gpu_id,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        render_cfg=RenderCfg(renderer=args.renderer, spp=1),
        physics_cfg=NewtonPhysicsCfg(
            physics_dt=args.dt,
            device=args.device,
            num_substeps=NUM_SUBSTEPS,
            use_cuda_graph=use_cuda_graph,
            solver_cfg={
                "solver_type": "mjvbd",
                "iterations": SOLVER_ITERATIONS,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.002,
                "particle_self_contact_margin": 0.002,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.005,
                "particle_vertex_contact_buffer_size": 96,
                "particle_edge_contact_buffer_size": 128,
                "particle_collision_detection_interval": -1,
                "self_contact_bvh_rebuild_interval_frames": 15,
                "rigid_contact_max": 0,
                "step_rigid_bodies": False,
                "soft_contact_margin": 0.008,
                "soft_contact_ke": 3.0e5,
                "soft_contact_kd": 5.0e-2,
                "soft_contact_mu": 0.5,
            },
        ),
        visualization=visualization_cfg_from_args(args),
    )
    return SimulationManager(cfg)


def create_table(sim: SimulationManager) -> RigidObject:
    """Declare the static folding table."""
    return sim.add_rigid_object(
        RigidObjectCfg(
            uid="table",
            shape=CubeCfg(
                size=list(TABLE_SIZE),
                visual_material=VisualMaterialCfg(
                    uid="fold_table_material",
                    base_color=[0.35, 0.42, 0.48, 1.0],
                    roughness=0.7,
                    metallic=0.0,
                ),
            ),
            attrs=RigidBodyPhysicsCfg(
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                material_props=RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                ),
                newton_props=NewtonRigidBodyPhysicsCfg(
                    collision_props=NewtonCollisionPropertiesCfg(
                        has_particle_collision=True,
                    ),
                    material_props=NewtonRigidBodyMaterialCfg(
                        ke=5.0e5,
                        kd=1.0e-6,
                    ),
                ),
            ),
            body_type="static",
            init_pos=TABLE_POSITION,
        )
    )


def create_ground(sim: SimulationManager, texture_path: Path) -> RigidObject:
    """Declare the textured static ground used by the reference scene."""
    return sim.add_rigid_object(
        RigidObjectCfg(
            uid="ground",
            shape=CubeCfg(
                size=list(GROUND_SIZE),
                visual_material=VisualMaterialCfg(
                    uid="fold_ground_material",
                    base_color=[1.0, 1.0, 1.0, 1.0],
                    base_color_texture=str(texture_path),
                    roughness=0.65,
                    metallic=0.0,
                ),
            ),
            attrs=RigidBodyPhysicsCfg(
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                material_props=RigidBodyMaterialCfg(
                    static_friction=0.5,
                    dynamic_friction=0.5,
                ),
                newton_props=NewtonRigidBodyPhysicsCfg(
                    collision_props=NewtonCollisionPropertiesCfg(
                        has_particle_collision=True,
                    ),
                    material_props=NewtonRigidBodyMaterialCfg(
                        ke=5.0e5,
                        kd=1.0e-6,
                    ),
                ),
            ),
            body_type="static",
            init_pos=GROUND_POSITION,
        )
    )


def create_w1(
    sim: SimulationManager,
    urdf_path: Path,
    initial_qpos: np.ndarray,
    *,
    particle_collision_enabled: bool,
) -> Robot:
    """Declare the fixed-base W1 using its exact source URDF."""
    robot = sim.add_robot(
        RobotCfg(
            uid="w1",
            fpath=str(urdf_path),
            asset_physics_mode="overlay",
            articulation_props=ArticulationRootPropertiesCfg(
                fixed_base=True,
                self_collision_enabled=False,
            ),
            # Preserve source drive modes; the runtime control writes kinematic
            # joint state directly at every Newton substep.
            drive_pros=JointDrivePropertiesCfg(),
            joint_props=JointDynamicsPropertiesCfg(
                # Several hand joints author zero limits in the URDF, which
                # produce invalid MuJoCo actfrcrange values without this overlay.
                max_effort=180.0,
                max_velocity=4.0,
            ),
            attrs=RigidBodyPhysicsCfg(
                collision_props=CollisionPropertiesCfg(collision_enabled=True),
                material_props=RigidBodyMaterialCfg(
                    static_friction=0.25,
                    dynamic_friction=0.25,
                ),
                newton_props=NewtonRigidBodyPhysicsCfg(
                    collision_props=NewtonCollisionPropertiesCfg(
                        has_particle_collision=particle_collision_enabled,
                    ),
                    material_props=NewtonRigidBodyMaterialCfg(
                        ke=3.0e5,
                        kd=1.0e-4,
                    ),
                ),
            ),
            init_qpos=initial_qpos,
            build_pk_chain=False,
        )
    )
    if robot is None:
        raise RuntimeError("Failed to declare the DexForce W1 robot.")
    return robot


def create_shirt(
    sim: SimulationManager,
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    visual_mesh_path: Path | None,
    texture_path: Path | None,
) -> ClothObject:
    """Declare the low-resolution cloth and its independently bound visual mesh."""
    if visual_mesh_path is not None:
        # The OBJ carries seam-duplicated vertices, authored UVs, and its own
        # double-sided MTL. Leaving visual_material unset preserves that MTL.
        visual_shape = MeshCfg(fpath=str(visual_mesh_path))
    else:
        if texture_path is None:
            raise ValueError("A fabric texture is required without an atlas mesh.")
        visual_shape = MeshCfg(
            vertices=vertices,
            triangles=triangles,
            normals=vertex_normals(vertices, triangles),
            uv_coords=planar_uv(vertices),
            visual_material=VisualMaterialCfg(
                uid="fold_shirt_material",
                base_color=[1.0, 1.0, 1.0, 1.0],
                base_color_texture=str(texture_path),
                roughness=0.8,
                metallic=0.0,
            ),
        )

    return sim.add_cloth_object(
        ClothObjectCfg(
            uid="shirt",
            shape=MeshCfg(vertices=vertices, triangles=triangles),
            visual_shape=visual_shape,
            visual_binding_mode="nearest_vertex",
            init_pos=SHIRT_POSITION,
            init_rot=(0.0, 0.0, -90.0),
            particle_radius=0.008,
            physical_attr=ClothPhysicalAttributesCfg(
                density=200.0,
                tri_ke=1.5e3,
                tri_ka=1.5e3,
                tri_kd=1.0e-5,
                edge_ke=1.2,
                edge_kd=0.1,
            ),
        )
    )


def register_runtime_controls(
    sim: SimulationManager,
    trajectory: np.ndarray,
    trajectory_fps: float,
    trajectory_time_scale: float,
    *,
    static_w1: bool,
    use_cuda_graph: bool,
) -> None:
    """Register the folding trajectory and phase-dependent contact materials."""
    transition_times = (
        CONTACT_SCRIPT_TRANSITIONS + CONTACT_SCRIPT_TO_SIMULATION_OFFSET
    ) / trajectory_time_scale
    contact_times = np.concatenate([np.zeros(1, dtype=np.float32), transition_times])

    def friction_track(values: tuple[float, ...]) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(sample_time), value)
            for sample_time, value in zip(contact_times, values, strict=True)
        )

    particle_friction = friction_track((0.5, 1.2, 0.0, 0.5, 1.2, 0.0, 0.5))
    w1_friction = friction_track((0.25, 1.2, 0.0, 0.25, 1.2, 0.0, 0.25))
    table_friction = friction_track((0.5, 0.12, 0.5, 0.5, 0.12, 0.5, 0.5))

    # This global particle control is host-side. Graph mode keeps the initial
    # soft_contact_mu while the graph-compatible rigid schedules still run.
    if not use_cuda_graph:
        sim.register_particle_contact_material_schedule(
            {"dynamic_friction": particle_friction}
        )
    sim.register_contact_material_schedule(
        "w1",
        {"dynamic_friction": w1_friction},
    )
    sim.register_contact_material_schedule(
        "table",
        {"dynamic_friction": table_friction},
    )
    sim.register_contact_material_schedule(
        "ground",
        {"dynamic_friction": table_friction},
    )
    if not static_w1:
        sim.register_kinematic_joint_trajectory(
            "w1",
            trajectory[None],
            fps=trajectory_fps,
        )


def configure_window_camera(sim: SimulationManager) -> None:
    """Frame both W1 arms and the shirt on the table."""
    window = sim.get_world().get_windows()
    if window is not None:
        window.set_look_at(
            eye=np.asarray([1.15, -2.10, 1.65], dtype=np.float32),
            look_at=np.asarray([0.55, 0.0, 1.18], dtype=np.float32),
            up=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )


def main() -> None:
    """Build the EmbodiChain scene and play the complete folding trajectory."""
    args = parse_arguments()
    (
        urdf_path,
        trajectory_path,
        shirt_mesh_path,
        ground_texture_path,
        visual_mesh_path,
        texture_path,
    ) = resolve_assets(args)
    vertices, triangles = load_shirt_mesh(shirt_mesh_path)
    trajectory, trajectory_fps = load_trajectory(
        trajectory_path,
        args.trajectory_time_scale,
    )
    if args.steps is None:
        args.steps = int(math.ceil(len(trajectory) / trajectory_fps / args.dt))

    use_cuda_graph = not args.disable_cuda_graph
    sim = initialize_simulation(args, use_cuda_graph=use_cuda_graph)
    try:
        # Replace EmbodiChain's default grid plane with the reference wood floor.
        sim.set_ground_plane_visibility(False)
        create_ground(sim, ground_texture_path)
        create_table(sim)
        robot = create_w1(
            sim,
            urdf_path,
            trajectory[0],
            particle_collision_enabled=not args.disable_w1_collision,
        )
        shirt = create_shirt(
            sim,
            vertices,
            triangles,
            visual_mesh_path=visual_mesh_path,
            texture_path=texture_path,
        )
        register_runtime_controls(
            sim,
            trajectory,
            trajectory_fps,
            args.trajectory_time_scale,
            static_w1=args.static_w1,
            use_cuda_graph=use_cuda_graph,
        )
        sim.prepare()

        if trajectory.shape[1] != robot.dof:
            raise ValueError(
                f"W1 trajectory has {trajectory.shape[1]} columns, but the "
                f"articulation has {robot.dof} DOFs."
            )
        if not args.headless and sim.open_window():
            sim.set_emission_light([1.0, 1.0, 1.0], 90.0)
            configure_window_camera(sim)

        particle_count = shirt.get_default_nodal_state().shape[1]
        logger.log_info(
            "Running W1 T-shirt fold | "
            f"driver={'static' if args.static_w1 else 'kinematic-trajectory'} | "
            f"cuda_graph={use_cuda_graph} | "
            f"frames={len(trajectory)} | trajectory_fps={trajectory_fps:.3f} | "
            f"dof={robot.dof} | cloth_particles={particle_count} | "
            f"material={args.cloth_material}"
        )

        fps_window_start: float | None = None
        fps_window_steps = 0
        for frame in range(args.steps):
            frame_start = time.perf_counter()
            sim.update(step=1)
            if args.real_time:
                elapsed = time.perf_counter() - frame_start
                time.sleep(max(0.0, args.dt - elapsed))

            frame_end = time.perf_counter()
            if fps_window_start is None:
                # Exclude one-time Warp compilation and CUDA Graph capture.
                fps_window_start = frame_end
            else:
                fps_window_steps += 1
            if (frame + 1) % FPS_LOG_INTERVAL == 0 or frame == args.steps - 1:
                elapsed = frame_end - fps_window_start
                fps = (
                    fps_window_steps / elapsed
                    if elapsed > 0.0 and fps_window_steps > 0
                    else 0.0
                )
                logger.log_info(f"Frame {frame + 1}/{args.steps}, FPS={fps:.1f}")
                fps_window_start = frame_end
                fps_window_steps = 0
        logger.log_info("W1 T-shirt folding simulation complete.")
    except KeyboardInterrupt:
        logger.log_info("\nExit")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
