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

"""Twist two fixed cloth edges in opposite directions with Newton VBD."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from embodichain.data import get_data_path
from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    NewtonPhysicsCfg,
    RenderCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import ClothObject
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.utils import logger

ASSET_DATASET = "DeformableDemoData"
TEXTURE_FILE_MAP = {
    "mianbu": "mianbu.png",
    "shabu": "shabu.png",
    "mabu": "mabu.png",
    "pige": "pige.png",
    "jinduan": "jinduan.png",
    "niuzai": "niuzai.png",
}

FPS = 60
NUM_SUBSTEPS = 10
SOLVER_ITERATIONS = 4
DEFAULT_FRAMES = 1000
ROTATION_ANGULAR_VELOCITY = math.pi / 3.0
ROTATION_END_TIME = 30.0
MESH_SCALE = 0.01
GRID_SIZE = 50
CLOTH_POSITION = (0.0, 0.0, 0.75)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the cloth-twist demo."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_FRAMES,
        help="Number of 60 Hz simulation frames.",
    )
    parser.add_argument(
        "--cloth-material",
        choices=tuple(TEXTURE_FILE_MAP),
        default="jinduan",
        help="Texture preset packaged with the DexSim reference demo.",
    )
    parser.set_defaults(device="cuda", physics="newton")
    args = parser.parse_args()
    if args.physics != "newton":
        parser.error("Cloth requires --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("EmbodiChain cloth currently requires a CUDA device.")
    if args.num_envs != 1:
        parser.error("This cloth demo currently supports --num_envs 1.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    return args


def prepare_cloth_asset(
    material_name: str,
) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the reference assets and load an order-preserving mesh.

    Args:
        material_name: Texture preset selected on the command line.

    Returns:
        The texture path, scaled vertices, triangles, and per-vertex UVs.

    Raises:
        FileNotFoundError: If the downloaded reference package is incomplete.
        RuntimeError: If the source mesh does not have the expected topology.
    """
    asset_root = Path(get_data_path(f"{ASSET_DATASET}/cloth_twist")).parent
    source_mesh = asset_root / "cloth_twist" / "cloth_twist_square_cloth.obj"
    texture_path = asset_root / "textures" / TEXTURE_FILE_MAP[material_name]

    missing = [path for path in (source_mesh, texture_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cloth twist asset package is incomplete; missing: "
            + ", ".join(str(path) for path in missing)
        )

    vertices: list[list[float]] = []
    texcoords: list[list[float]] = []
    faces: list[list[int]] = []
    face_uvs: list[list[int]] = []
    for line in source_mesh.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            _, x, y, z = line.split()[:4]
            vertices.append([float(x), float(y), float(z)])
        elif line.startswith("vt "):
            _, u, v = line.split()[:3]
            texcoords.append([float(u), float(v)])
        elif line.startswith("f "):
            references = [item.split("/") for item in line.split()[1:]]
            if len(references) != 3:
                raise RuntimeError(f"{source_mesh} must contain triangle faces only.")
            faces.append([int(reference[0]) - 1 for reference in references])
            face_uvs.append(
                [
                    int(reference[1]) - 1 if len(reference) > 1 and reference[1] else -1
                    for reference in references
                ]
            )

    vertices_array = np.asarray(vertices, dtype=np.float32) * MESH_SCALE
    triangles = np.asarray(faces, dtype=np.int32)
    expected_vertices = GRID_SIZE * GRID_SIZE
    expected_faces = 2 * (GRID_SIZE - 1) * (GRID_SIZE - 1)
    if vertices_array.shape != (expected_vertices, 3) or triangles.shape != (
        expected_faces,
        3,
    ):
        raise RuntimeError(
            "cloth_twist_square_cloth.obj does not match the expected "
            f"{GRID_SIZE} x {GRID_SIZE} topology."
        )

    vertex_uvs = np.full((len(vertices_array), 2), np.nan, dtype=np.float32)
    for face, face_uv in zip(triangles, face_uvs, strict=True):
        for vertex_index, texcoord_index in zip(face, face_uv, strict=True):
            if texcoord_index < 0:
                continue
            uv = np.asarray(texcoords[texcoord_index], dtype=np.float32)
            if np.isnan(vertex_uvs[vertex_index, 0]):
                vertex_uvs[vertex_index] = uv
            elif not np.allclose(vertex_uvs[vertex_index], uv, atol=1.0e-6):
                raise RuntimeError(
                    f"{source_mesh} maps multiple UVs to vertex {vertex_index}."
                )
    vertex_uvs[np.isnan(vertex_uvs[:, 0])] = 0.0
    return texture_path, vertices_array, triangles, vertex_uvs


def build_twist_trajectory(
    vertices: np.ndarray,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-node flags and opposite edge-rotation offsets.

    Args:
        vertices: Scaled source vertices before the configured cloth pose.
        frame_count: Number of outer 60 Hz simulation frames.

    Returns:
        Shared node indices, per-node particle flags, and batched offsets.
    """
    left_edge = np.asarray(
        [GRID_SIZE - 1 + row * GRID_SIZE for row in range(GRID_SIZE)],
        dtype=np.int32,
    )
    right_edge = np.asarray(
        [row * GRID_SIZE for row in range(GRID_SIZE)],
        dtype=np.int32,
    )
    node_indices = np.concatenate((left_edge, right_edge))

    particle_flags = np.ones(len(vertices), dtype=np.int32)
    particle_flags[node_indices] = 0

    angle = np.pi / 2.0
    cloth_rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    posed_vertices = vertices @ cloth_rotation.T
    selected_positions = posed_vertices[node_indices]
    rotation_axes = np.asarray(
        [[0.0, 1.0, 0.0]] * len(left_edge) + [[0.0, -1.0, 0.0]] * len(right_edge),
        dtype=np.float32,
    )
    roots = (
        np.sum(selected_positions * rotation_axes, axis=1, keepdims=True)
        * rotation_axes
    )
    radial_vectors = selected_positions - roots
    axis_cross_radial = np.cross(rotation_axes, radial_vectors)
    axis_dot_radial = np.sum(
        rotation_axes * radial_vectors,
        axis=1,
        keepdims=True,
    )

    sample_count = frame_count * NUM_SUBSTEPS
    times = np.minimum(
        np.arange(sample_count, dtype=np.float32) / (FPS * NUM_SUBSTEPS),
        ROTATION_END_TIME,
    )
    theta = times * ROTATION_ANGULAR_VELOCITY
    cosine = np.cos(theta)[:, None, None]
    sine = np.sin(theta)[:, None, None]
    rotated_radial = (
        cosine * radial_vectors[None]
        + sine * axis_cross_radial[None]
        + (1.0 - cosine) * rotation_axes[None] * axis_dot_radial[None]
    )
    target_positions = roots[None] + rotated_radial
    offsets = target_positions - selected_positions[None]
    return node_indices, particle_flags, offsets[None].astype(np.float32)


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    """Create the zero-gravity Newton VBD simulation manager."""
    cfg = SimulationManagerCfg(
        width=1920,
        height=1080,
        headless=args.headless,
        device=args.device,
        gpu_id=args.gpu_id,
        num_envs=args.num_envs,
        arena_space=args.arena_space,
        render_cfg=RenderCfg(renderer=args.renderer),
        physics_cfg=NewtonPhysicsCfg(
            physics_dt=1.0 / FPS,
            device=args.device,
            gravity=(0.0, 0.0, 0.0),
            num_substeps=NUM_SUBSTEPS,
            use_cuda_graph=False,
            solver_cfg={
                "solver_type": "vbd",
                "iterations": SOLVER_ITERATIONS,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.002,
                "particle_self_contact_margin": 0.0035,
                "particle_enable_tile_solve": True,
                "soft_contact_ke": 1.0e3,
                "soft_contact_kd": 1.0e-1,
                "soft_contact_mu": 0.2,
            },
        ),
        visualization=visualization_cfg_from_args(args),
    )
    return SimulationManager(cfg)


def create_cloth(
    sim: SimulationManager,
    texture_path: Path,
    vertices: np.ndarray,
    triangles: np.ndarray,
    uv_coords: np.ndarray,
    particle_flags: np.ndarray,
) -> ClothObject:
    """Declare the textured cloth with reference VBD material parameters."""
    return sim.add_cloth_object(
        ClothObjectCfg(
            uid="twist_cloth",
            shape=MeshCfg(
                vertices=vertices,
                triangles=triangles,
                uv_coords=uv_coords,
                visual_material=VisualMaterialCfg(
                    uid="twist_cloth_material",
                    base_color=[1.0, 1.0, 1.0, 1.0],
                    base_color_texture=str(texture_path),
                    roughness=0.8,
                    metallic=0.0,
                ),
            ),
            init_pos=CLOTH_POSITION,
            init_rot=(0.0, 0.0, 90.0),
            particle_flags=particle_flags,
            physical_attr=ClothPhysicalAttributesCfg(
                density=0.2,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=2.0e-4,
                edge_ke=1.0e-3,
                edge_kd=1.0e-2,
            ),
        )
    )


def configure_window_camera(sim: SimulationManager) -> None:
    """Frame the vertical cloth in the native viewer."""
    window = sim.get_world().get_windows()
    if window is not None:
        window.set_look_at(
            eye=np.asarray([2.25, 0.0, CLOTH_POSITION[2]], dtype=np.float32),
            look_at=np.asarray(CLOTH_POSITION, dtype=np.float32),
            up=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )


def main() -> None:
    """Create the scene and execute the finite cloth-twist trajectory."""
    args = parse_arguments()
    texture_path, vertices, triangles, uv_coords = prepare_cloth_asset(
        args.cloth_material
    )
    node_indices, particle_flags, offsets = build_twist_trajectory(
        vertices,
        args.iterations,
    )
    sim = initialize_simulation(args)

    try:
        cloth = create_cloth(
            sim,
            texture_path,
            vertices,
            triangles,
            uv_coords,
            particle_flags,
        )
        sim.register_kinematic_nodal_trajectory(
            cloth.uid,
            node_indices,
            offsets,
            rebuild_self_contact_bvh=True,
        )
        sim.prepare()

        if not args.headless and sim.open_window():
            configure_window_camera(sim)

        particle_count = cloth.get_default_nodal_state().shape[1]
        logger.log_info(
            f"Running cloth twist for {args.iterations} frames at {FPS} Hz "
            f"with {particle_count} particles."
        )
        for frame in range(args.iterations):
            sim.update(step=1)
            if frame % 50 == 0:
                logger.log_info(
                    f"Frame {frame}/{args.iterations}, sim_time={frame / FPS:.2f}s"
                )
        logger.log_info("Cloth twist simulation complete.")
    except KeyboardInterrupt:
        logger.log_info("\nExit")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
