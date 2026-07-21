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
"""Generate a cuRobo V2 robot configuration YAML from an EmbodiChain ``Robot``.

The :func:`generate_curobo_robot_yaml` helper pulls the robot's URDF path and
each link's collision mesh (vertices/faces) from the simulator, fits collision
spheres to every link mesh with cuRobo's sphere-fitting library, and writes a
complete cuRobo robot configuration YAML (mirroring ``franka.yml``) that can be
passed to :class:`~embodichain.lab.sim.planners.curobo_planner.CuroboRobotProfileCfg.robot_config_path`.

The cuRobo planner adapter calls this automatically (with on-disk caching) when
a profile leaves ``robot_config_path`` unset; see
:class:`~embodichain.lab.sim.planners.curobo_planner.CuroboAutoGenCfg`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

__all__ = ["generate_curobo_robot_yaml"]


def generate_curobo_robot_yaml(
    robot: Robot,
    control_part: str,
    output_path: str,
    *,
    tool_frame: str | None = None,
    fit_type: str = "morphit",
    num_spheres: int | None = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    iterations: int = 200,
    collision_sphere_buffer: float = 0.0,
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
    device: str = "cuda:0",
) -> str:
    """Fit collision spheres to each robot link's mesh and write a cuRobo robot YAML.

    Extracts the URDF path and per-link vertices/faces from ``robot``, fits
    collision spheres to every link mesh with cuRobo's :func:`fit_spheres_to_mesh`,
    and writes a complete cuRobo robot configuration YAML (mirroring
    ``franka.yml``) that can be passed to
    ``CuroboRobotProfileCfg.robot_config_path``.

    .. attention::
        Requires a CUDA GPU and cuRobo installed (sphere fitting runs on GPU).
        Link meshes from ``robot.get_link_vert_face`` are assumed to be in the
        link-local rest frame -- the convention cuRobo collision spheres use,
        since cuRobo applies each link's transform via FK at runtime.

    Args:
        robot: The EmbodiChain robot to generate a config for.
        control_part: Control-part name whose joints stay active; every other
            actuated joint is pinned via ``lock_joints``.
        output_path: Destination YAML file path.
        tool_frame: cuRobo tool frame (a URDF link name) to plan to. If ``None``,
            defaults to the last link of the control part.
        fit_type: cuRobo sphere-fit strategy - ``"morphit"`` (default, best),
            ``"voxel"`` (faster), or ``"surface"`` (crude, fixed radius).
        num_spheres: Per-link sphere count. If ``None``, cuRobo auto-estimates
            it from the link's bounding-box volume.
        sphere_density: Multiplier on the auto sphere count (ignored when
            ``num_spheres`` is set).
        surface_radius: Fixed radius used only by the ``surface`` strategy.
        iterations: Adam iterations for the ``morphit`` strategy.
        collision_sphere_buffer: Padding added to every sphere's radius (m).
        max_acceleration: cspace maximum acceleration.
        max_jerk: cspace maximum jerk.
        device: CUDA device for sphere fitting.

    Returns:
        The ``output_path`` that was written.

    Raises:
        ImportError: If cuRobo or trimesh is not installed.
        RuntimeError: If CUDA is unavailable or no spheres could be fitted.
    """
    import os

    import trimesh
    import yaml

    from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
    from curobo._src.geom.sphere_fit.types import SphereFitType
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("generate_curobo_robot_yaml requires a CUDA GPU.")
    fit_type_map = {
        "morphit": SphereFitType.MORPHIT,
        "voxel": SphereFitType.VOXEL,
        "surface": SphereFitType.SURFACE,
    }
    if fit_type not in fit_type_map:
        raise ValueError(
            f"fit_type must be one of {list(fit_type_map)}, got {fit_type!r}."
        )
    fit_type_enum = fit_type_map[fit_type]
    device_cfg = DeviceCfg(device=device)

    urdf_path = robot.cfg.fpath
    link_vert_dict: dict = {}
    link_face_dict: dict = {}
    for link_name in robot.get_link_names() or []:
        verts, faces = robot.get_link_vert_face(link_name)
        link_vert_dict[link_name] = verts
        link_face_dict[link_name] = faces

    # 1. Parse the URDF kinematic tree (no meshes) for base_link + parent map.
    #    ``robot.root_link_name`` is avoided because it touches an uninitialized
    #    ``entities`` attribute on some Robot instances; cuRobo's parser resolves
    #    the root link directly from the URDF.
    base_link: str | None = None
    urdf_parent_map: dict[str, str | None] = {}
    mimic_joints: set[str] = set()
    try:
        parser = UrdfRobotParser(urdf_path, load_meshes=False, build_scene_graph=True)
        parser.build_link_parent()
        base_link = parser.root_link
        mimic_joints = set(parser.get_mimic_joint_map().keys())
        # Build the full parent map for every URDF link so self_collision_ignore
        # can walk multiple hops (the parent of a non-collision link still
        # connects two collision links, e.g. fr3_link8 between fr3_link7 and
        # fr3_hand).
        for link_name in parser.get_link_names_from_urdf():
            try:
                urdf_parent_map[link_name] = parser.get_link_parameters(
                    link_name
                ).parent_link_name
            except Exception:  # noqa: BLE001  (e.g. root link has no parent entry)
                urdf_parent_map[link_name] = None
    except Exception as exc:  # noqa: BLE001
        logger.log_warning(f"Could not parse URDF kinematic tree ({exc}).")
    if base_link is None:
        link_names_fb = robot.get_link_names() or []
        base_link = getattr(robot.cfg, "base_link_name", None) or (
            link_names_fb[0] if link_names_fb else "base_link"
        )

    # 2. Fit collision spheres per link from the simulator meshes.
    collision_spheres: dict[str, list[dict]] = {}
    for link_name, verts in link_vert_dict.items():
        faces = link_face_dict[link_name]
        if verts is None or faces is None or verts.numel() == 0 or faces.numel() == 0:
            continue
        verts_np = torch.as_tensor(verts).detach().to(torch.float32).cpu().numpy()
        faces_np = torch.as_tensor(faces).detach().to(torch.int64).cpu().numpy()
        mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
        if len(mesh.vertices) == 0:
            continue
        mesh.fill_holes()
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)
        trimesh.repair.fix_winding(mesh)
        try:
            fit_result = fit_spheres_to_mesh(
                mesh,
                num_spheres=num_spheres,
                sphere_density=sphere_density,
                surface_radius=surface_radius,
                fit_type=fit_type_enum,
                iterations=iterations,
                device_cfg=device_cfg,
            )
        except Exception as exc:  # noqa: BLE001
            logger.log_warning(f"Sphere fitting failed for link {link_name!r}: {exc}")
            continue
        if fit_result.num_spheres == 0:
            continue
        collision_spheres[link_name] = [
            {"center": list(c), "radius": float(r)}
            for c, r in zip(
                fit_result.centers.detach().cpu().tolist(),
                fit_result.radii.detach().cpu().tolist(),
            )
        ]

    if not collision_spheres:
        raise RuntimeError(
            "No collision spheres could be fitted from the robot's link meshes."
        )
    collision_link_names = list(collision_spheres.keys())

    # 3. self_collision_ignore: ignore link pairs within two kinematic hops
    #    (parent/grandparent, children/grandchildren, siblings). cuRobo's curated
    #    profiles (e.g. franka.yml) ignore adjacent-plus-near links because their
    #    spheres physically overlap near joints; a neighbor-only matrix leaves
    #    those pairs colliding and makes reachable start poses fail validation.
    self_collision_ignore: dict[str, list[str]] = {}
    if urdf_parent_map:
        children_map: dict[str, list[str]] = {}
        for link_name, parent in urdf_parent_map.items():
            if parent is not None:
                children_map.setdefault(parent, []).append(link_name)
        collision_set = set(collision_link_names)

        def _two_hop_neighbors(link: str) -> set[str]:
            neighbors: set[str] = set()
            parent = urdf_parent_map.get(link)
            if parent is not None:
                neighbors.add(parent)
                grandparent = urdf_parent_map.get(parent)
                if grandparent is not None:
                    neighbors.add(grandparent)
                for sibling in children_map.get(parent, []):
                    if sibling != link:
                        neighbors.add(sibling)
            for child in children_map.get(link, []):
                neighbors.add(child)
                for grandchild in children_map.get(child, []):
                    neighbors.add(grandchild)
            return neighbors

        for link_name in collision_link_names:
            self_collision_ignore[link_name] = [
                n for n in _two_hop_neighbors(link_name) if n in collision_set
            ]

    # 4. cspace from the robot's joints + init qpos. Mimic joints are excluded -
    #    cuRobo drives them from their active joint and rejects them in cspace.
    joint_names = list(robot.joint_names)
    init_qpos = list(robot.cfg.init_qpos) if robot.cfg.init_qpos is not None else []
    if len(init_qpos) != len(joint_names):
        logger.log_warning(
            "init_qpos length does not match joint_names; using current qpos."
        )
        try:
            init_qpos = robot.get_qpos()[0].detach().cpu().tolist()
        except Exception:  # noqa: BLE001
            init_qpos = [0.0] * len(joint_names)
    cspace_pairs = [
        (jname, float(val))
        for jname, val in zip(joint_names, init_qpos)
        if jname not in mimic_joints
    ]
    cspace = {
        "joint_names": [j for j, _ in cspace_pairs],
        "default_joint_position": [v for _, v in cspace_pairs],
        "max_acceleration": float(max_acceleration),
        "max_jerk": float(max_jerk),
        "cspace_distance_weight": [1.0] * len(cspace_pairs),
        "null_space_weight": [1.0] * len(cspace_pairs),
    }

    # 5. lock_joints: actuated joints outside the control part, pinned to init values.
    #    Mimic joints are already excluded from cspace_pairs (see step 4).
    control_joints = set((robot.control_parts or {}).get(control_part, []))
    lock_joints: dict[str, float] = {
        jname: val for jname, val in cspace_pairs if jname not in control_joints
    }

    # 6. tool_frames default to the last link of the control part.
    if tool_frame is None:
        part_links = robot.get_control_part_link_names(control_part)
        if not part_links:
            raise RuntimeError(
                f"Control part {control_part!r} has no links; specify tool_frame."
            )
        tool_frame = part_links[-1]

    # 7. Assemble and write the YAML, mirroring franka.yml's schema.
    data = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "base_link": base_link,
                "urdf_path": urdf_path,
                "asset_root_path": os.path.dirname(urdf_path),
                "tool_frames": [tool_frame],
                "collision_link_names": collision_link_names,
                "collision_spheres": collision_spheres,
                "collision_sphere_buffer": float(collision_sphere_buffer),
                "mesh_link_names": collision_link_names,
                "self_collision_buffer": {ln: 0.0 for ln in collision_link_names},
                "self_collision_ignore": self_collision_ignore,
                "lock_joints": lock_joints,
                "cspace": cspace,
                "use_global_cumul": True,
            }
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
    return output_path
