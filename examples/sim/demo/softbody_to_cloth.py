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

"""Drop a volumetric soft body onto a cloth sheet with Newton VBD."""

from __future__ import annotations

import argparse

import numpy as np

from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import (
    ClothObjectCfg,
    ClothPhysicalAttributesCfg,
    LightCfg,
    NewtonPhysicsCfg,
    RenderCfg,
    SoftObjectCfg,
    SoftbodyPhysicalAttributesCfg,
    SoftbodyVoxelAttributesCfg,
)
from embodichain.lab.sim.material import VisualMaterialCfg
from embodichain.lab.sim.objects import ClothObject, SoftObject
from embodichain.lab.sim.shapes import MeshCfg
from embodichain.lab.visualization import visualization_cfg_from_args
from embodichain.utils import logger

__all__ = [
    "configure_window_camera",
    "create_box_surface_mesh",
    "create_cloth",
    "create_cloth_grid_mesh",
    "create_soft_body",
    "initialize_simulation",
    "main",
    "parse_arguments",
    "run_simulation",
]

FPS = 60
NUM_SUBSTEPS = 3
SOLVER_ITERATIONS = 6
DEFAULT_ITERATIONS = 500

CLOTH_SIZE = 2.0
CLOTH_GRID_CELLS = 28
CLOTH_POSITION = (-1.0, -1.0, 1.0)
SOFT_BODY_SIZE = (0.6, 0.6, 0.3)
SOFT_BODY_POSITION = (0.0, 0.0, 2.0)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the coupled deformable demo.

    Returns:
        The validated command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_env_launcher_args_to_parser(parser)
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of outer 60 Hz simulation frames.",
    )
    parser.set_defaults(device="cuda", physics="newton")
    args = parser.parse_args()
    if args.physics != "newton":
        parser.error("Soft bodies and cloth require --physics newton.")
    if not str(args.device).startswith("cuda"):
        parser.error("EmbodiChain deformables currently require a CUDA device.")
    if args.iterations <= 0:
        parser.error("--iterations must be positive.")
    return args


def initialize_simulation(args: argparse.Namespace) -> SimulationManager:
    """Create the Newton VBD simulation manager.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The configured simulation manager.
    """
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
            num_substeps=NUM_SUBSTEPS,
            solver_cfg={
                "solver_type": "vbd",
                "iterations": SOLVER_ITERATIONS,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.01,
                "particle_self_contact_margin": 0.02,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.05,
                "particle_enable_tile_solve": True,
                "soft_contact_ke": 1.0e5,
                "soft_contact_kd": 1.0e-5,
                "soft_contact_mu": 1.0,
            },
        ),
        visualization=visualization_cfg_from_args(args),
    )
    sim = SimulationManager(cfg)
    sim.set_emission_light(color=(0.5, 0.5, 0.5), intensity=90.0)
    sim.add_light(
        LightCfg(
            uid="main_light",
            intensity=100.0,
            radius=10.0,
            init_pos=(-4.0, 3.0, 4.5),
        )
    )
    return sim


def create_box_surface_mesh(
    size: tuple[float, float, float],
    subdivisions: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a watertight, subdivided box surface.

    Shared edge and corner vertices keep the mesh suitable for soft-body
    tetrahedralization while providing enough render vertices to show its
    deformation.

    Args:
        size: Box dimensions along the X, Y, and Z axes.
        subdivisions: Number of cells along each box edge.

    Returns:
        Float32 vertices and int32 outward-facing triangle indices.

    Raises:
        ValueError: If the size or subdivision count is invalid.
    """
    size_array = np.asarray(size, dtype=np.float32)
    if size_array.shape != (3,) or not np.isfinite(size_array).all():
        raise ValueError("size must contain three finite values.")
    if np.any(size_array <= 0.0):
        raise ValueError("All box dimensions must be positive.")
    if subdivisions < 1:
        raise ValueError("subdivisions must be at least one.")

    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    vertex_indices: dict[tuple[int, int, int], int] = {}

    def vertex_index(lattice_index: tuple[int, int, int]) -> int:
        """Return a shared vertex index for one surface lattice point."""
        index = vertex_indices.get(lattice_index)
        if index is not None:
            return index
        coordinate = (
            np.asarray(lattice_index, dtype=np.float32) / subdivisions - 0.5
        ) * size_array
        index = len(vertices)
        vertices.append(coordinate)
        vertex_indices[lattice_index] = index
        return index

    # Each (constant axis, side, U axis, V axis) tuple has U x V pointing
    # outward, so both generated triangles have consistent winding.
    face_specs = (
        (2, subdivisions, 0, 1),
        (2, 0, 1, 0),
        (0, subdivisions, 1, 2),
        (0, 0, 2, 1),
        (1, subdivisions, 2, 0),
        (1, 0, 0, 2),
    )
    for constant_axis, side, u_axis, v_axis in face_specs:
        for v_index in range(subdivisions):
            for u_index in range(subdivisions):
                corners: list[int] = []
                for u_offset, v_offset in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    lattice = [0, 0, 0]
                    lattice[constant_axis] = side
                    lattice[u_axis] = u_index + u_offset
                    lattice[v_axis] = v_index + v_offset
                    corners.append(vertex_index(tuple(lattice)))
                triangles.append((corners[0], corners[1], corners[2]))
                triangles.append((corners[0], corners[2], corners[3]))

    return (
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(triangles, dtype=np.int32),
    )


def create_cloth_grid_mesh(
    size: float = CLOTH_SIZE,
    cells: int = CLOTH_GRID_CELLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the horizontal cloth grid and its two fixed edge selections.

    Args:
        size: Cloth width and depth in metres.
        cells: Number of grid cells along each axis.

    Returns:
        Vertices, triangles, and fixed-node indices.

    Raises:
        ValueError: If the size or cell count is invalid.
    """
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("size must be finite and positive.")
    if cells < 1:
        raise ValueError("cells must be at least one.")

    coordinates = np.linspace(0.0, size, cells + 1, dtype=np.float32)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    vertices = np.stack(
        (xx.reshape(-1), yy.reshape(-1), np.zeros(xx.size, dtype=np.float32)),
        axis=1,
    )

    grid_indices = np.arange((cells + 1) ** 2, dtype=np.int32).reshape(
        cells + 1, cells + 1
    )
    lower_left = grid_indices[:-1, :-1].reshape(-1)
    lower_right = grid_indices[:-1, 1:].reshape(-1)
    upper_left = grid_indices[1:, :-1].reshape(-1)
    upper_right = grid_indices[1:, 1:].reshape(-1)
    triangles = np.concatenate(
        (
            np.stack((lower_left, lower_right, upper_right), axis=1),
            np.stack((lower_left, upper_right, upper_left), axis=1),
        ),
        axis=0,
    ).astype(np.int32, copy=False)
    fixed_indices = np.flatnonzero(
        np.isclose(vertices[:, 0], 0.0) | np.isclose(vertices[:, 0], size)
    ).astype(np.int32, copy=False)
    return (
        np.ascontiguousarray(vertices),
        np.ascontiguousarray(triangles),
        np.ascontiguousarray(fixed_indices),
    )


def create_soft_body(sim: SimulationManager) -> SoftObject:
    """Declare the falling soft box with the reference material parameters.

    Args:
        sim: Simulation manager that owns the scene declaration.

    Returns:
        The declared volume-deformable facade.
    """
    vertices, triangles = create_box_surface_mesh(SOFT_BODY_SIZE)
    return sim.add_soft_object(
        SoftObjectCfg(
            uid="falling_soft_body",
            shape=MeshCfg(
                vertices=vertices,
                triangles=triangles,
                visual_material=VisualMaterialCfg(
                    uid="soft_material",
                    base_color=[0.9, 0.5, 0.2, 1.0],
                    roughness=0.8,
                ),
            ),
            init_pos=SOFT_BODY_POSITION,
            voxel_attr=SoftbodyVoxelAttributesCfg(
                simulation_mesh_resolution=10,
            ),
            physical_attr=SoftbodyPhysicalAttributesCfg(
                # This converts exactly to k_mu=8e3 and k_lambda=8e3.
                youngs=2.0e4,
                poissons=0.25,
                density=1.5e2,
                elasticity_damping=6.0e-4,
            ),
        )
    )


def create_cloth(sim: SimulationManager) -> ClothObject:
    """Declare the cloth sheet with both X edges fixed.

    Args:
        sim: Simulation manager that owns the scene declaration.

    Returns:
        The declared surface-deformable facade.
    """
    vertices, triangles, fixed_indices = create_cloth_grid_mesh()
    particle_flags = np.ones(len(vertices), dtype=np.int32)
    particle_flags[fixed_indices] = 0
    return sim.add_cloth_object(
        ClothObjectCfg(
            uid="cloth_sheet",
            shape=MeshCfg(
                vertices=vertices,
                triangles=triangles,
                visual_material=VisualMaterialCfg(
                    uid="cloth_material",
                    base_color=[0.4, 0.6, 0.9, 1.0],
                    roughness=0.8,
                ),
            ),
            init_pos=CLOTH_POSITION,
            particle_radius=0.05,
            particle_flags=particle_flags,
            physical_attr=ClothPhysicalAttributesCfg(
                density=5.0e-4,
                tri_ke=1.0e5,
                tri_ka=1.0e5,
                tri_kd=1.0e-5,
                edge_ke=0.01,
                edge_kd=1.0e-2,
            ),
        )
    )


def configure_window_camera(sim: SimulationManager) -> None:
    """Frame the falling soft body and suspended cloth in the viewer.

    Args:
        sim: Prepared simulation manager with an open native window.
    """
    window = sim.get_world().get_windows()
    if window is not None:
        window.set_look_at(
            eye=np.asarray([3.0, 3.0, 4.0], dtype=np.float32),
            look_at=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            up=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        )


def run_simulation(
    sim: SimulationManager,
    soft_body: SoftObject,
    cloth: ClothObject,
    iterations: int,
) -> None:
    """Advance the coupled scene for a finite number of frames.

    Args:
        sim: Prepared simulation manager.
        soft_body: Falling volumetric deformable.
        cloth: Suspended surface deformable.
        iterations: Number of outer simulation frames.
    """
    logger.log_info(f"Running soft body to cloth for {iterations} frames at {FPS} Hz.")
    logger.log_info(
        f"Soft body: {soft_body.get_default_nodal_state().shape[1]} particles, "
        f"{soft_body.get_surface_triangles().shape[1]} surface triangles."
    )
    logger.log_info(
        f"Cloth: {cloth.get_default_nodal_state().shape[1]} particles, "
        f"{cloth.get_surface_triangles().shape[1]} triangles."
    )

    for frame in range(iterations):
        sim.update(step=1)
        if frame % 50 == 0 or frame + 1 == iterations:
            soft_height = float(
                soft_body.get_current_nodal_position()[..., 2].mean().item()
            )
            cloth_height = float(
                cloth.get_current_nodal_position()[..., 2].mean().item()
            )
            logger.log_info(
                f"Frame {frame + 1}/{iterations}, "
                f"sim_time={(frame + 1) / FPS:.2f}s, "
                f"soft_mean_z={soft_height:.3f}m, "
                f"cloth_mean_z={cloth_height:.3f}m"
            )
    logger.log_info("Soft body to cloth simulation complete.")


def main() -> None:
    """Build and run the coupled soft-body/cloth scene."""
    args = parse_arguments()
    sim = initialize_simulation(args)

    try:
        soft_body = create_soft_body(sim)
        cloth = create_cloth(sim)
        sim.prepare()

        if not args.headless and sim.open_window():
            configure_window_camera(sim)
        run_simulation(sim, soft_body, cloth, args.iterations)
    except KeyboardInterrupt:
        logger.log_info("\nExit")
    finally:
        sim.destroy()


if __name__ == "__main__":
    main()
