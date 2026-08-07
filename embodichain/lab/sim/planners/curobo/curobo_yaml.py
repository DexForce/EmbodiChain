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
spheres to every link mesh with dexsim's MorphIt :func:`sphere_fit`, and writes
a complete cuRobo V2 robot configuration YAML. The cuRobo planner adapter calls
this automatically (with on-disk caching) on the first plan; see
:class:`~embodichain.lab.sim.planners.curobo.curobo_planner.CuroboAutoGenCfg`.

:func:`generate_curobo_world_yaml` builds the cuRobo collision-world YAML from
live :class:`~embodichain.lab.sim.objects.RigidObject` meshes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot

__all__ = ["generate_curobo_robot_yaml", "generate_curobo_world_yaml"]


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


# Robot links are near-convex, so 4 VisACD convex parts are enough and keep the
# per-link sphere count (and thus planning time) low. Obstacles may be concave,
# so allow up to 16 parts for a tighter sphere fit.
_ROBOT_LINK_MAX_CONVEX_HULL_NUM = 4
_OBSTACLE_MAX_CONVEX_HULL_NUM = 16


def _fit_collision_spheres(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    max_convex_hull_num: int,
    num_spheres: int | None,
    sphere_density: float,
    iterations: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fit collision spheres to a mesh with dexsim's MorphIt ``sphere_fit``.

    Repairs the mesh (fill holes / fix winding & normals) via trimesh, builds an
    ``o3d.t.geometry.TriangleMesh``, then calls
    :func:`dexsim.kit.meshproc.sphere_fit` with ``SphereFitType.MORPHIT`` and the
    requested ``max_convex_hull_num`` (4 for robot links, 16 for obstacles).

    Args:
        vertices: Mesh vertices ``(V, 3)`` (any float dtype, any device).
        faces: Triangle indices ``(F, 3)`` (any integer dtype).
        max_convex_hull_num: Cap on VisACD convex parts before per-part fitting.
        num_spheres: Explicit sphere count; ``None`` auto-estimates from
            ``sphere_density``.
        sphere_density: Multiplier on the auto-estimated sphere count.
        iterations: MorphIt Adam iterations.
        device: Torch device for the fit (e.g. ``"cuda:0"``).

    Returns:
        ``(centers (N, 3), radii (N,))`` CPU float32 tensors, or ``None`` when no
        spheres could be fitted.
    """
    import open3d as o3d
    import trimesh
    from dexsim.kit.meshproc import SphereFitType, sphere_fit

    verts_np = torch.as_tensor(vertices).detach().to(torch.float32).cpu().numpy()
    faces_np = torch.as_tensor(faces).detach().to(torch.int64).cpu().numpy()
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    if len(mesh.vertices) == 0:
        return None
    mesh.fill_holes()
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_winding(mesh)

    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(
        np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    )
    o3d_mesh.triangle.indices = o3d.core.Tensor(
        np.ascontiguousarray(mesh.faces, dtype=np.int32)
    )

    is_success, centers, radii = sphere_fit(
        o3d_mesh,
        num_spheres=num_spheres,
        sphere_density=sphere_density,
        fit_type=SphereFitType.MORPHIT,
        iterations=iterations,
        max_convex_hull_num=max_convex_hull_num,
        device=device,
    )
    if not is_success or centers.shape[0] == 0:
        return None
    return (
        centers.detach().to("cpu").reshape(-1, 3).to(torch.float32),
        radii.detach().to("cpu").reshape(-1).to(torch.float32),
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
    iterations: int = 200,
    collision_sphere_buffer: float = 0.0,
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
    device: str = "cuda:0",
) -> str:
    """Fit collision spheres to each robot link's mesh and write a cuRobo robot YAML.

    Extracts the URDF path and per-link vertices/faces from ``robot``, fits
    collision spheres to every link mesh with dexsim's MorphIt
    :func:`sphere_fit` (``max_convex_hull_num=4``), and writes a complete cuRobo
    V2 robot configuration YAML that the cuRobo planner loads as its robot model.

    .. attention::
        Requires a CUDA GPU and dexsim installed (MorphIt sphere fitting runs on
        GPU). Link meshes from ``robot.get_link_vert_face`` are assumed to be in
        the link-local rest frame -- the convention cuRobo collision spheres use,
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
        num_spheres: Per-link sphere count. If ``None``, dexsim auto-estimates it
            from the link's bounding-box volume.
        sphere_density: Multiplier on the auto sphere count (ignored when
            ``num_spheres`` is set).
        iterations: Adam iterations for the MorphIt fit.
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

    import yaml

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

    # 1. Parse the URDF kinematic tree (no meshes) for base_link + parent map.
    #    ``robot.root_link_name`` is avoided because it touches an uninitialized
    #    ``entities`` attribute on some Robot instances; cuRobo's parser resolves
    #    the root link directly from the URDF.
    #    Mimic joints are detected from the URDF XML (not cuRobo's parser, which
    #    exposes no mimic accessor) so they can be excluded from cspace/lock_joints
    #    in step 4/5 - cuRobo folds them into their active joint and raises
    #    KeyError if they are locked.
    mimic_joints: set[str] = _parse_mimic_joint_names(urdf_path)
    base_link: str | None = None
    urdf_parent_map: dict[str, str | None] = {}
    try:
        parser = UrdfRobotParser(urdf_path, load_meshes=False, build_scene_graph=True)
        parser.build_link_parent()
        base_link = parser.root_link
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
        try:
            fit_result = _fit_collision_spheres(
                verts,
                faces,
                max_convex_hull_num=_ROBOT_LINK_MAX_CONVEX_HULL_NUM,
                num_spheres=num_spheres,
                sphere_density=sphere_density,
                iterations=iterations,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.log_warning(f"Sphere fitting failed for link {link_name!r}: {exc}")
            continue
        if fit_result is None:
            continue
        centers, radii = fit_result
        collision_spheres[link_name] = [
            {"center": list(c), "radius": float(r)}
            for c, r in zip(centers.tolist(), radii.tolist())
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
    iterations: int = 200,
    collision_sphere_buffer: float = 0.0,
    device: str = "cuda:0",
) -> list[tuple[str, str, dict]]:
    """Convert one mesh + pose into cuRobo world-YAML obstacle entry/entries.

    Pure tensor helper (no simulator / cuRobo import for ``cuboid``/``mesh``) so
    it is unit-testable without CUDA. ``sphere`` lazily imports dexsim + trimesh
    and runs on CUDA.

    Args:
        name: Obstacle name (cuRobo key under ``cuboid``/``mesh``/``sphere``).
        vertices: Mesh vertices ``(V, 3)`` in the object's local frame.
        faces: Triangle indices ``(F, 3)`` (any integer dtype).
        pose: Object pose as ``(x, y, z, qw, qx, qy, qz)`` ``(7,)`` or a
            homogeneous ``(4, 4)`` matrix, expressed in the cuRobo world/base
            frame (the same frame static collision YAMLs are authored in).
        representation: ``"cuboid"`` (local-frame AABB -> OBB via ``pose``,
            default), ``"mesh"`` (exact triangle mesh), or ``"sphere"`` (fit
            spheres with dexsim's :func:`sphere_fit`, MorphIt,
            ``max_convex_hull_num=16``).
        num_spheres: Per-mesh sphere count; ``None`` auto-estimates (sphere only).
        sphere_density: Multiplier on the auto sphere count (sphere only).
        iterations: Adam iterations for the MorphIt fit (sphere only).
        collision_sphere_buffer: Padding added to each fitted radius (sphere only).
        device: CUDA device for sphere fitting (sphere only).

    Returns:
        A list of ``(top_level_key, obstacle_name, fields)`` tuples. ``cuboid``/
        ``mesh`` return one entry; ``sphere`` returns one entry per fitted sphere.

    Raises:
        ValueError: If ``representation`` is unsupported, ``pose`` is malformed,
            or the mesh has no geometry for the requested representation.
        RuntimeError: If ``"sphere"`` is requested without CUDA.
        ImportError: If ``"sphere"`` is requested without dexsim/trimesh.
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
            "The 'sphere' representation requires CUDA for sphere fitting."
        )

    fit_result = _fit_collision_spheres(
        vertices,
        faces,
        max_convex_hull_num=_OBSTACLE_MAX_CONVEX_HULL_NUM,
        num_spheres=num_spheres,
        sphere_density=sphere_density,
        iterations=iterations,
        device=device,
    )
    if fit_result is None:
        raise RuntimeError(f"No spheres could be fitted for object {name!r}.")

    centers_local, radii = fit_result
    radii = radii + float(collision_sphere_buffer)
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
            (exact triangle mesh, no CUDA), or ``"sphere"`` (dexsim MorphIt sphere
            fit, requires CUDA + dexsim + trimesh).
        env_id: Environment instance index to read geometry/pose from (the static
            world is shared, so env 0 is representative).
        num_spheres: Per-object sphere count; ``None`` auto-estimates (sphere only).
        sphere_density: Multiplier on the auto sphere count (sphere only).
        iterations: Adam iterations for the MorphIt fit (sphere only).
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
