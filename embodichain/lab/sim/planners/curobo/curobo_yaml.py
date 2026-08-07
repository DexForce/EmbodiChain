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
"""Generate cuRobo V2 configuration YAMLs from EmbodiChain simulator objects.

The :func:`generate_curobo_robot_yaml` helper pulls the robot's URDF path and
each link's collision mesh (vertices/faces) from the simulator, fits collision
spheres to every link mesh with DexSim's sphere-fitting library, and writes a
complete cuRobo V2 robot configuration YAML. The cuRobo planner adapter calls
this automatically (with on-disk caching) on the first plan; see
:class:`~embodichain.lab.sim.planners.curobo.curobo_planner.CuroboAutoGenCfg`.

:func:`generate_curobo_world_yaml` builds the cuRobo collision-world YAML from
live :class:`~embodichain.lab.sim.objects.RigidObject` meshes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch

from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot

__all__ = [
    "generate_curobo_robot_yaml",
    "generate_curobo_world_yaml",
    "visualize_curobo_collision_models",
    "visualize_curobo_robot_collision_model",
    "visualize_curobo_world_collision_model",
]


_ROBOT_MAX_CONVEX_HULL_NUM = 2
_OBSTACLE_MAX_CONVEX_HULL_NUM = 16


def _parse_mimic_joint_names(urdf_path: str) -> set[str]:
    """Return the names of URDF joints that mimic another joint.

    cuRobo's URDF parser folds each ``<mimic>`` joint into its active joint: the
    mimic joint's body takes the active joint's name, so the mimic joint has no
    independent entry in the kinematics tree. cuRobo therefore rejects mimic
    joints in ``cspace`` and ``lock_joints`` - locking one raises ``KeyError``
    because cuRobo finds no body for it. They must be excluded from both, so
    cuRobo drives them from their active joint instead.

    cuRobo's ``UrdfRobotParser`` exposes no public accessor for mimic joints (a
    prior ``get_mimic_joint_map`` call did not exist on this cuRobo version and
    silently left the set empty), so they are read directly from the URDF XML -
    the same ``<mimic>`` tags cuRobo itself parses.

    Args:
        urdf_path: Path to the robot URDF file.

    Returns:
        The set of joint names declared with a ``<mimic>`` child element. Empty
        if the URDF cannot be parsed (a warning is logged).
    """
    import xml.etree.ElementTree as ET

    mimic_joints: set[str] = set()
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:  # noqa: BLE001
        logger.log_warning(f"Could not parse mimic joints from URDF ({exc}).")
        return mimic_joints
    for joint in root.findall("joint"):
        if joint.find("mimic") is not None:
            name = joint.get("name")
            if name is not None:
                mimic_joints.add(name)
    return mimic_joints


def _to_open3d_legacy_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    o3d: Any,
) -> Any:
    """Create a legacy Open3D triangle mesh from tensor-like geometry."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        torch.as_tensor(vertices).detach().to(torch.float64).cpu().numpy()
    )
    mesh.triangles = o3d.utility.Vector3iVector(
        torch.as_tensor(faces).detach().to(torch.int32).cpu().reshape(-1, 3).numpy()
    )
    mesh.compute_vertex_normals()
    return mesh


def _to_open3d_tensor_mesh(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    o3d: Any,
) -> Any:
    """Create the Open3D tensor mesh expected by DexSim's ``sphere_fit``."""
    return o3d.t.geometry.TriangleMesh.from_legacy(
        _to_open3d_legacy_mesh(vertices, faces, o3d)
    )


def generate_curobo_robot_yaml(
    robot: Robot,
    control_part: str,
    output_path: str,
    *,
    tool_frame: str | None = None,
    urdf_path: str | None = None,
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
    collision spheres to every link mesh with DexSim's :func:`sphere_fit`,
    and writes a complete cuRobo V2 robot configuration YAML that the cuRobo
    planner loads as its robot model.

    .. attention::
        Requires a CUDA GPU, DexSim, Open3D, and cuRobo (sphere fitting runs on GPU).
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
        urdf_path: URDF to generate the cuRobo model from. If ``None`` (default),
            uses ``robot.cfg.fpath`` -- the *assembled* URDF that includes every
            mounted component (arm + gripper). Pass this explicitly when the
            caller already resolved the URDF (the planner does, so the on-disk
            cache key and the generation use the same file). Must be the full
            assembled URDF, not a solver's sub-chain URDF, or gripper links are
            silently dropped from the collision model.
        num_spheres: Per-link sphere count. If ``None``, DexSim auto-estimates
            it from the link's bounding-box volume.
        sphere_density: Multiplier on the auto sphere count (ignored when
            ``num_spheres`` is set).
        surface_radius: Fixed radius used by MorphIt's surface fallback.
        iterations: Adam iterations for MorphIt.
        collision_sphere_buffer: Padding added to every sphere's radius (m).
        max_acceleration: cspace maximum acceleration.
        max_jerk: cspace maximum jerk.
        device: CUDA device for sphere fitting.

    Returns:
        The ``output_path`` that was written.

    Raises:
        ImportError: If DexSim or Open3D is not installed.
        RuntimeError: If CUDA is unavailable or no spheres could be fitted.
    """
    import os

    import open3d as o3d
    import yaml

    from dexsim.kit.meshproc import SphereFitType, sphere_fit
    from curobo._src.robot.parser.parser_urdf import UrdfRobotParser

    if not torch.cuda.is_available():
        raise RuntimeError("generate_curobo_robot_yaml requires a CUDA GPU.")
    urdf_path = urdf_path or robot.cfg.fpath
    link_vert_dict: dict = {}
    link_face_dict: dict = {}
    for link_name in robot.get_link_names() or []:
        verts, faces = robot.get_link_vert_face(link_name)
        link_vert_dict[link_name] = verts
        link_face_dict[link_name] = faces

    # 1. Parse the URDF kinematic tree (no meshes) for the base link.
    #    ``robot.root_link_name`` is avoided because it touches an uninitialized
    #    ``entities`` attribute on some Robot instances; cuRobo's parser resolves
    #    the root link directly from the URDF.
    #    Mimic joints are detected from the URDF XML (not cuRobo's parser, which
    #    exposes no mimic accessor) so they can be excluded from cspace/lock_joints
    #    below - cuRobo folds them into their active joint and raises
    #    KeyError if they are locked.
    mimic_joints: set[str] = _parse_mimic_joint_names(urdf_path)
    base_link: str | None = None
    try:
        parser = UrdfRobotParser(urdf_path, load_meshes=False, build_scene_graph=True)
        parser.build_link_parent()
        base_link = parser.root_link
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
        mesh = _to_open3d_tensor_mesh(verts, faces, o3d)
        try:
            is_success, centers, radii = sphere_fit(
                mesh,
                num_spheres=num_spheres,
                sphere_density=sphere_density,
                surface_radius=surface_radius,
                fit_type=SphereFitType.MORPHIT,
                iterations=iterations,
                max_convex_hull_num=_ROBOT_MAX_CONVEX_HULL_NUM,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.log_warning(f"Sphere fitting failed for link {link_name!r}: {exc}")
            continue
        if not is_success:
            continue
        collision_spheres[link_name] = [
            {"center": list(c), "radius": float(r)}
            for c, r in zip(
                centers.detach().cpu().tolist(),
                radii.detach().cpu().tolist(),
            )
        ]

    if not collision_spheres:
        raise RuntimeError(
            "No collision spheres could be fitted from the robot's link meshes."
        )
    collision_link_names = list(collision_spheres.keys())

    # 3. cspace from the robot's joints + init qpos. Mimic joints are excluded -
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

    # 4. lock_joints: actuated joints outside the control part, pinned to init values.
    #    Mimic joints are already excluded from cspace_pairs (see step 3).
    control_joints = set((robot.control_parts or {}).get(control_part, []))
    lock_joints: dict[str, float] = {
        jname: val for jname, val in cspace_pairs if jname not in control_joints
    }

    # 5. tool_frames default to the last link of the control part.
    if tool_frame is None:
        part_links = robot.get_control_part_link_names(control_part)
        if not part_links:
            raise RuntimeError(
                f"Control part {control_part!r} has no links; specify tool_frame."
            )
        tool_frame = part_links[-1]

    # 6. Assemble and write the YAML, mirroring franka.yml's schema.
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


# =============================================================================
# World (obstacle) YAML generation from RigidObject meshes
# =============================================================================


_REPRESENTATIONS = ("cuboid", "mesh", "sphere")


def _mesh_to_obstacle_entry(
    name: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pose: torch.Tensor,
    *,
    representation: str = "cuboid",
    num_spheres: int | None = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    iterations: int = 200,
    collision_sphere_buffer: float = 0.0,
    device: str = "cuda:0",
) -> list[tuple[str, str, dict]]:
    """Convert one mesh + pose into cuRobo world-YAML obstacle entry/entries.

    Pure tensor helper (no simulator import for ``cuboid``/``mesh``) so it is
    unit-testable without CUDA. ``sphere`` lazily imports DexSim + Open3D.

    Args:
        name: Obstacle name (cuRobo key under ``cuboid``/``mesh``/``sphere``).
        vertices: Mesh vertices ``(V, 3)`` in the object's local frame.
        faces: Triangle indices ``(F, 3)`` (any integer dtype).
        pose: Object pose as ``(x, y, z, qw, qx, qy, qz)`` ``(7,)`` or a
            homogeneous ``(4, 4)`` matrix, expressed in the cuRobo world/base
            frame (the same frame static collision YAMLs are authored in).
        representation: ``"cuboid"`` (local-frame AABB -> OBB via ``pose``,
            default), ``"mesh"`` (exact triangle mesh), or ``"sphere"`` (fit
            spheres with DexSim's :func:`sphere_fit`).
        num_spheres: Per-mesh sphere count; ``None`` auto-estimates (sphere only).
        sphere_density: Multiplier on the auto sphere count (sphere only).
        surface_radius: Fixed radius for MorphIt's surface fallback (sphere only).
        iterations: Adam iterations for MorphIt (sphere only).
        collision_sphere_buffer: Padding added to each fitted radius (sphere only).
        device: CUDA device for sphere fitting (sphere only).

    Returns:
        A list of ``(top_level_key, obstacle_name, fields)`` tuples. ``cuboid``/
        ``mesh`` return one entry; ``sphere`` returns one entry per fitted sphere.

    Raises:
        ValueError: If ``representation`` is unsupported, ``pose`` is malformed,
            or the mesh has no geometry for the requested representation.
        RuntimeError: If ``"sphere"`` is requested without CUDA.
        ImportError: If ``"sphere"`` is requested without DexSim/Open3D.
    """
    if representation not in _REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {_REPRESENTATIONS}, got {representation!r}."
        )

    vertices = (
        torch.as_tensor(vertices, dtype=torch.float32).detach().to("cpu").reshape(-1, 3)
    )
    faces = torch.as_tensor(faces).detach().to("cpu")
    pose = torch.as_tensor(pose, dtype=torch.float32).detach().to("cpu")
    if pose.shape == (4, 4):
        position = pose[:3, 3]
        quaternion = quat_from_matrix(pose[:3, :3])  # wxyz
        pose = torch.cat([position, quaternion])
    if pose.shape != (7,):
        raise ValueError(
            f"pose must be (7,) [x,y,z,qw,qx,qy,qz] or (4, 4), got {tuple(pose.shape)}."
        )

    if representation == "mesh":
        if vertices.numel() == 0 or faces.numel() == 0:
            raise ValueError(
                f"object {name!r} has no mesh geometry for the 'mesh' representation."
            )
        return [
            (
                "mesh",
                name,
                {
                    "vertices": vertices.tolist(),
                    "faces": faces.reshape(-1).to(torch.int64).tolist(),
                    "pose": pose.tolist(),
                },
            )
        ]

    if representation == "cuboid":
        if vertices.numel() == 0:
            raise ValueError(
                f"object {name!r} has no vertices for the 'cuboid' representation."
            )
        # Local-frame AABB, emitted as an OBB via the object pose: cuRobo's
        # Cuboid is centered at ``pose[:3]`` with ``dims`` along the pose axes.
        vmin = vertices.amin(dim=0)
        vmax = vertices.amax(dim=0)
        dims = vmax - vmin
        center_local = (vmin + vmax) / 2.0
        rotation = matrix_from_quat(pose[3:7])  # (3, 3), wxyz
        center_world = rotation @ center_local + pose[:3]
        cuboid_pose = torch.cat([center_world, pose[3:7]])
        return [("cuboid", name, {"dims": dims.tolist(), "pose": cuboid_pose.tolist()})]

    # representation == "sphere": fit spheres in the local frame, then transform
    # centers into the cuRobo world/base frame (Sphere obstacles have no pose/FK).
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError(
            f"object {name!r} has no mesh geometry for the 'sphere' representation."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The 'sphere' representation requires CUDA for DexSim MorphIt fitting."
        )

    import open3d as o3d

    from dexsim.kit.meshproc import SphereFitType, sphere_fit

    mesh = _to_open3d_tensor_mesh(vertices, faces, o3d)
    is_success, centers, fitted_radii = sphere_fit(
        mesh,
        num_spheres=num_spheres,
        sphere_density=sphere_density,
        surface_radius=surface_radius,
        fit_type=SphereFitType.MORPHIT,
        iterations=iterations,
        max_convex_hull_num=_OBSTACLE_MAX_CONVEX_HULL_NUM,
        device=device,
    )
    if not is_success:
        raise RuntimeError(f"No spheres could be fitted for object {name!r}.")

    centers_local = centers.detach().to("cpu").reshape(-1, 3).to(torch.float32)
    radii = fitted_radii.detach().to("cpu").reshape(-1).to(torch.float32) + float(
        collision_sphere_buffer
    )
    rotation = matrix_from_quat(pose[3:7])
    centers_world = centers_local @ rotation.T + pose[:3]
    entries: list[tuple[str, str, dict]] = []
    for i in range(centers_world.shape[0]):
        entries.append(
            (
                "sphere",
                f"{name}_{i}",
                {
                    "position": centers_world[i].tolist(),
                    "radius": float(radii[i].item()),
                },
            )
        )
    return entries


def generate_curobo_world_yaml(
    rigid_objects: Sequence[RigidObject],
    output_path: str,
    *,
    representation: str = "cuboid",
    env_id: int = 0,
    num_spheres: int | None = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    iterations: int = 200,
    collision_sphere_buffer: float = 0.0,
    device: str = "cuda:0",
) -> str:
    """Generate a cuRobo V2 scene (world) YAML from a sequence of ``RigidObject``.

    Each object's mesh (``get_vertices`` / ``get_triangles``) and world pose
    (``get_local_pose``) are converted into cuRobo obstacle entries under a single
    top-level key (``cuboid`` / ``mesh`` / ``sphere``). The cuRobo planner loads
    the resulting YAML as its collision world.

    .. attention::
        Poses are written in the cuRobo world/base frame - the same convention as
        a hand-authored static collision YAML. When the robot base is offset from
        the simulator world origin, rebase the object poses first, or register the
        obstacle name in ``CuroboWorldCfg.dynamic_obstacle_names`` and update its
        pose at plan time via
        :meth:`~embodichain.lab.sim.planners.curobo.curobo_planner.CuroboPlanner.update_dynamic_obstacles`.

    Args:
        rigid_objects: ``RigidObject`` instances to bake into the collision world.
        output_path: Destination YAML file path.
        representation: ``"cuboid"`` (default, AABB->OBB, no CUDA), ``"mesh"``
            (exact triangle mesh, no CUDA), or ``"sphere"`` (DexSim MorphIt
            sphere fit, requiring CUDA + DexSim + Open3D).
        env_id: Environment instance index to read geometry/pose from (the static
            world is shared, so env 0 is representative).
        num_spheres: Per-object sphere count; ``None`` auto-estimates (sphere only).
        sphere_density: Multiplier on the auto sphere count (sphere only).
        surface_radius: Fixed radius for MorphIt's surface fallback (sphere only).
        iterations: Adam iterations for MorphIt (sphere only).
        collision_sphere_buffer: Padding added to each fitted radius (sphere only).
        device: CUDA device for sphere fitting (sphere only).

    Returns:
        The ``output_path`` that was written.

    Raises:
        ValueError: If ``rigid_objects`` is empty or a representation/pose is
            invalid.
    """
    import os

    import yaml

    rigid_objects = list(rigid_objects)
    if not rigid_objects:
        raise ValueError("rigid_objects must contain at least one RigidObject.")

    data: dict[str, dict[str, object]] = {}
    used_names: set[str] = set()
    for idx, obj in enumerate(rigid_objects):
        name = getattr(obj, "uid", None) or f"obstacle_{idx}"
        if name in used_names:
            raise ValueError(
                f"Duplicate obstacle name {name!r}; RigidObject uids must be unique."
            )
        used_names.add(name)

        vertices = obj.get_vertices(env_ids=[env_id], scale=True)[0]
        faces = obj.get_triangles(env_ids=[env_id])[0]
        pose = obj.get_local_pose(to_matrix=False)[env_id]

        if vertices is None or faces is None or vertices.numel() == 0:
            logger.log_warning(
                f"RigidObject {name!r} has no mesh geometry; skipping collision export."
            )
            continue

        entries = _mesh_to_obstacle_entry(
            name,
            vertices,
            faces,
            pose,
            representation=representation,
            num_spheres=num_spheres,
            sphere_density=sphere_density,
            surface_radius=surface_radius,
            iterations=iterations,
            collision_sphere_buffer=collision_sphere_buffer,
            device=device,
        )
        for top_key, obstacle_name, fields in entries:
            data.setdefault(top_key, {})[obstacle_name] = fields

    if not data:
        raise ValueError(
            "No collision obstacles could be generated from the given RigidObjects."
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
    return output_path


# =============================================================================
# Cached collision-model visualization
# =============================================================================


def _collision_visualization_geometries(
    meshes: list[tuple[str, Any]],
    centers: torch.Tensor,
    radii: torch.Tensor,
    *,
    sphere_name: str,
    sphere_color: list[float],
    mesh_color: list[float],
) -> list[dict[str, Any]]:
    """Build Open3D draw entries in the style of DexSim's ``sphere_fit_visual``."""
    import open3d as o3d

    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_color = mesh_color

    geometries = [
        {"name": name, "geometry": mesh, "material": mesh_material}
        for name, mesh in meshes
    ]

    spheres_mesh = o3d.geometry.TriangleMesh()
    centers_np = centers.detach().cpu().numpy().reshape(-1, 3)
    radii_np = radii.detach().cpu().numpy().reshape(-1)
    for center, radius in zip(centers_np, radii_np):
        sphere = o3d.geometry.TriangleMesh.create_sphere(float(radius))
        sphere.translate(center)
        spheres_mesh += sphere
    spheres_mesh.compute_vertex_normals()

    sphere_material = o3d.visualization.rendering.MaterialRecord()
    sphere_material.shader = "defaultLitSSR"
    sphere_material.base_color = sphere_color
    sphere_material.base_roughness = 0.05
    sphere_material.base_reflectance = 0.0
    sphere_material.base_clearcoat = 1.0
    sphere_material.thickness = 1.0
    sphere_material.transmission = 0.2
    sphere_material.absorption_distance = 10.0
    sphere_material.absorption_color = sphere_color[:3]
    geometries.append(
        {
            "name": sphere_name,
            "geometry": spheres_mesh,
            "material": sphere_material,
        }
    )
    return geometries


def visualize_curobo_robot_collision_model(
    robot: Robot,
    robot_yaml_path: str,
    env_id: int = 0,
    *,
    draw: bool = True,
) -> list[dict[str, Any]]:
    """Visualize a robot's live link meshes and cached collision spheres.

    Sphere centers and radii are always loaded from ``robot_yaml_path``. Each
    link-local cached center is transformed by the link's live simulator pose
    from :meth:`Articulation.get_link_pose`, making frame errors directly
    visible against the corresponding world-space mesh.

    Args:
        robot: Live simulator robot.
        robot_yaml_path: Cached auto-generated cuRobo robot YAML.
        env_id: Simulator environment instance to visualize.
        draw: Open an Open3D window immediately. ``False`` returns draw entries
            for composition with another collision model.

    Returns:
        Open3D geometry dictionaries suitable for :func:`open3d.visualization.draw`.
    """
    import open3d as o3d
    import yaml

    with open(robot_yaml_path, encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    kinematics = data["robot_cfg"]["kinematics"]
    cached_spheres = kinematics.get("collision_spheres", {})
    sphere_buffer = float(kinematics.get("collision_sphere_buffer", 0.0))

    meshes: list[tuple[str, Any]] = []
    world_centers: list[torch.Tensor] = []
    radii: list[float] = []
    for link_name, link_spheres in cached_spheres.items():
        vertices, faces = robot.get_link_vert_face(link_name)
        if vertices is None or faces is None or vertices.numel() == 0:
            continue
        link_pose = torch.as_tensor(
            robot.get_link_pose(link_name, env_ids=[env_id], to_matrix=True)[0],
            dtype=torch.float32,
        ).cpu()
        mesh = _to_open3d_legacy_mesh(vertices, faces, o3d)
        mesh.transform(link_pose.numpy())
        meshes.append((f"robot_mesh/{link_name}", mesh))

        centers_local = torch.as_tensor(
            [sphere["center"] for sphere in link_spheres], dtype=torch.float32
        ).reshape(-1, 3)
        centers_world = centers_local @ link_pose[:3, :3].T + link_pose[:3, 3]
        world_centers.extend(centers_world.unbind())
        radii.extend(float(sphere["radius"]) + sphere_buffer for sphere in link_spheres)

    if not world_centers:
        raise ValueError(
            f"Robot cache {robot_yaml_path!r} contains no collision spheres."
        )
    geometries = _collision_visualization_geometries(
        meshes,
        torch.stack(world_centers),
        torch.tensor(radii, dtype=torch.float32),
        sphere_name="robot_spheres",
        sphere_color=[0.0, 0.2, 0.8, 0.5],
        mesh_color=[0.5, 0.5, 0.5, 1.0],
    )
    if draw:
        o3d.visualization.draw(geometries, title="cuRobo robot collision model")
    return geometries


def visualize_curobo_world_collision_model(
    rigid_objects: Sequence[RigidObject],
    world_yaml_path: str,
    env_id: int = 0,
    *,
    draw: bool = True,
) -> list[dict[str, Any]]:
    """Visualize live obstacle meshes and spheres loaded from a cached world YAML.

    Args:
        rigid_objects: Live simulator obstacles represented by the cache.
        world_yaml_path: Cached auto-generated cuRobo world YAML. It must use
            the ``sphere`` representation.
        env_id: Simulator environment instance whose live meshes are shown.
        draw: Open an Open3D window immediately. ``False`` returns draw entries
            for composition with the robot collision model.

    Returns:
        Open3D geometry dictionaries suitable for :func:`open3d.visualization.draw`.
    """
    import open3d as o3d
    import yaml

    with open(world_yaml_path, encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    sphere_entries = data.get("sphere", {}) if isinstance(data, dict) else {}
    if not sphere_entries:
        raise ValueError(
            f"World cache {world_yaml_path!r} contains no sphere representation."
        )

    meshes: list[tuple[str, Any]] = []
    for idx, obj in enumerate(rigid_objects):
        name = getattr(obj, "uid", None) or f"obstacle_{idx}"
        vertices = obj.get_vertices(env_ids=[env_id], scale=True)[0]
        faces = obj.get_triangles(env_ids=[env_id])[0]
        if vertices is None or faces is None or vertices.numel() == 0:
            continue
        pose = torch.as_tensor(
            obj.get_local_pose(to_matrix=True)[env_id], dtype=torch.float32
        ).cpu()
        mesh = _to_open3d_legacy_mesh(vertices, faces, o3d)
        mesh.transform(pose.numpy())
        meshes.append((f"obstacle_mesh/{name}", mesh))

    centers = torch.as_tensor(
        [entry["position"] for entry in sphere_entries.values()], dtype=torch.float32
    ).reshape(-1, 3)
    radii = torch.as_tensor(
        [entry["radius"] for entry in sphere_entries.values()], dtype=torch.float32
    ).reshape(-1)
    geometries = _collision_visualization_geometries(
        meshes,
        centers,
        radii,
        sphere_name="obstacle_spheres",
        sphere_color=[0.8, 0.15, 0.0, 0.5],
        mesh_color=[0.45, 0.55, 0.45, 1.0],
    )
    if draw:
        o3d.visualization.draw(geometries, title="cuRobo obstacle collision model")
    return geometries


def visualize_curobo_collision_models(
    robot: Robot,
    robot_yaml_path: str,
    rigid_objects: Sequence[RigidObject] | None = None,
    world_yaml_path: str | None = None,
    env_id: int = 0,
) -> None:
    """Draw cached robot and obstacle collision spheres in one Open3D window."""
    import open3d as o3d

    geometries = visualize_curobo_robot_collision_model(
        robot, robot_yaml_path, env_id, draw=False
    )
    if rigid_objects and world_yaml_path is not None:
        geometries.extend(
            visualize_curobo_world_collision_model(
                rigid_objects, world_yaml_path, env_id, draw=False
            )
        )
    o3d.visualization.draw(geometries, title="cuRobo collision models")
