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

:func:`generate_curobo_world_scene` builds mixed cuRobo collision data from live
:class:`~embodichain.lab.sim.objects.RigidObject` physical shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch
from dexsim.types import RigidBodyShape

from embodichain.lab.sim.objects.rigid_object import CollisionShapeDesc
from embodichain.utils import logger
from embodichain.utils.math import matrix_from_quat, quat_from_matrix

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import RigidObject, Robot

__all__ = [
    "generate_curobo_robot_yaml",
    "generate_curobo_world_scene",
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
    """Create an Open3D tensor triangle mesh from tensor-like geometry."""
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
# World collision generation from RigidObject physical shapes
# =============================================================================


def _voxel_grid_coordinates(
    grid_shape: tuple[int, int, int], voxel_size: float
) -> torch.Tensor:
    """Return voxel centers in cuRobo's X/Y/Z flattening order."""
    axes = [
        (torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0) * voxel_size
        for size in grid_shape
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)


def _convex_hulls_to_voxel_entry(
    name: str,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pose: torch.Tensor,
    *,
    voxel_size: float = 0.01,
    voxel_padding: float = 0.005,
) -> tuple[str, dict[str, object]]:
    """Decompose one mesh with VisACD and convert its union to an ESDF grid.

    The grid is centered at the object's local origin, so the voxel obstacle's
    pose stays identical to the source object's pose during dynamic updates.
    """
    vertices = (
        torch.as_tensor(vertices, dtype=torch.float32).detach().to("cpu").reshape(-1, 3)
    )
    faces = torch.as_tensor(faces).detach().to("cpu").reshape(-1, 3)
    if vertices.numel() == 0 or faces.numel() == 0:
        raise ValueError(f"object {name!r} has no mesh geometry for voxelization.")
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")
    if voxel_padding < 0.0:
        raise ValueError(f"voxel_padding must be non-negative, got {voxel_padding}.")

    pose = torch.as_tensor(pose, dtype=torch.float32).detach().to("cpu")
    if pose.shape == (4, 4):
        position = pose[:3, 3]
        quaternion = quat_from_matrix(pose[:3, :3])  # wxyz
        pose = torch.cat([position, quaternion])
    if pose.shape != (7,):
        raise ValueError(
            f"pose must be (7,) [x,y,z,qw,qx,qy,qz] or (4, 4), got {tuple(pose.shape)}."
        )

    import open3d as o3d

    from dexsim.kit.meshproc import convex_decomposition_visacd

    mesh = _to_open3d_tensor_mesh(vertices, faces, o3d)
    is_success, convex_hulls = convex_decomposition_visacd(
        mesh,
        max_convex_hull_num=_OBSTACLE_MAX_CONVEX_HULL_NUM,
        is_visual=False,
    )
    if not is_success or not convex_hulls:
        raise RuntimeError(f"VisACD decomposition failed for object {name!r}.")

    local_half_extent = torch.maximum(
        vertices.amin(dim=0).abs(), vertices.amax(dim=0).abs()
    )
    requested_dims = 2.0 * (local_half_extent + float(voxel_padding))
    grid_shape_tensor = torch.ceil(requested_dims / float(voxel_size)).to(torch.int64)
    grid_shape_tensor = torch.clamp(grid_shape_tensor, min=2)
    grid_shape = tuple(int(value) for value in grid_shape_tensor.tolist())
    dims = grid_shape_tensor.to(torch.float32) * float(voxel_size)
    query_points = _voxel_grid_coordinates(grid_shape, float(voxel_size))
    query_o3d = o3d.core.Tensor(query_points.numpy(), dtype=o3d.core.Dtype.Float32)

    union_sdf = torch.full((query_points.shape[0],), torch.inf, dtype=torch.float32)
    for hull in convex_hulls:
        hull_cpu = hull.cpu() if hasattr(hull, "cpu") else hull
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(hull_cpu)
        hull_sdf = torch.from_numpy(
            scene.compute_signed_distance(query_o3d).numpy()
        ).to(torch.float32)
        union_sdf = torch.minimum(union_sdf, hull_sdf)

    feature_tensor = union_sdf.reshape(grid_shape).to(torch.float16).contiguous()
    return name, {
        "pose": pose.tolist(),
        "dims": dims.tolist(),
        "voxel_size": float(voxel_size),
        "feature_tensor": feature_tensor,
    }


def _pose_matrix_to_list(pose: torch.Tensor) -> list[float]:
    """Convert a homogeneous pose matrix to cuRobo ``xyz+wxyz`` format."""
    pose = torch.as_tensor(pose, dtype=torch.float32).detach().cpu()
    return torch.cat([pose[:3, 3], quat_from_matrix(pose[:3, :3])]).tolist()


def _collision_shape_mesh(
    shape: CollisionShapeDesc,
    plane_dims: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a physical collision descriptor to a local triangle mesh."""
    if shape.vertices is not None and shape.triangles is not None:
        if shape.vertices.numel() and shape.triangles.numel():
            return shape.vertices, shape.triangles

    import trimesh

    if shape.shape_type == RigidBodyShape.BOX:
        assert shape.half_extents is not None
        mesh = trimesh.creation.box(extents=(2.0 * shape.half_extents).numpy())
    elif shape.shape_type == RigidBodyShape.PLANE:
        mesh = trimesh.creation.box(extents=plane_dims)
    elif shape.shape_type == RigidBodyShape.SPHERE:
        assert shape.radius is not None
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=shape.radius)
    elif shape.shape_type == RigidBodyShape.CAPSULE:
        assert shape.radius is not None and shape.half_height is not None
        mesh = trimesh.creation.capsule(
            radius=shape.radius, height=2.0 * shape.half_height
        )
    else:
        raise ValueError(
            f"Collision shape {shape.name!r} ({shape.shape_type.name}) does not "
            "expose a mesh usable by cuRobo."
        )
    return (
        torch.as_tensor(mesh.vertices, dtype=torch.float32),
        torch.as_tensor(mesh.faces, dtype=torch.int32),
    )


def _estimated_voxel_count(
    vertices: torch.Tensor,
    voxel_size: float,
    voxel_padding: float,
) -> int:
    """Estimate the dense ESDF allocation for a local collision mesh."""
    extents = vertices.amax(dim=0) - vertices.amin(dim=0) + 2.0 * voxel_padding
    shape = torch.clamp(torch.ceil(extents / voxel_size), min=2).to(torch.int64)
    return int(torch.prod(shape).item())


def _auto_collision_representation(
    shape: CollisionShapeDesc,
    *,
    is_dynamic: bool,
    voxel_size: float,
    voxel_padding: float,
    mesh_triangle_threshold: int,
    max_voxel_count: int,
    plane_dims: tuple[float, float, float],
) -> str:
    """Select a cuRobo representation from one physical shape descriptor."""
    native = {
        RigidBodyShape.BOX: "cuboid",
        RigidBodyShape.PLANE: "cuboid",
        RigidBodyShape.SPHERE: "sphere",
        RigidBodyShape.CAPSULE: "capsule",
        RigidBodyShape.CONVEX: "mesh",
        RigidBodyShape.SDF: "mesh",
    }
    if shape.shape_type in native:
        return native[shape.shape_type]
    if shape.shape_type != RigidBodyShape.MESH:
        raise ValueError(
            f"No automatic cuRobo representation for DexSim shape "
            f"{shape.shape_type.name}."
        )

    vertices, triangles = _collision_shape_mesh(shape, plane_dims)
    effective_triangle_threshold = mesh_triangle_threshold * (2 if is_dynamic else 1)
    if triangles.shape[0] <= effective_triangle_threshold:
        return "mesh"
    voxel_count = _estimated_voxel_count(vertices, voxel_size, voxel_padding)
    if voxel_count <= max_voxel_count:
        return "voxel"
    logger.log_warning(
        f"Keeping collision mesh {shape.name!r}: its estimated ESDF allocation "
        f"({voxel_count} voxels) exceeds max_voxel_count={max_voxel_count}."
    )
    return "mesh"


def _validate_forced_representation(
    representation: str,
    shape: CollisionShapeDesc,
) -> None:
    """Reject analytic overrides that do not match the physics shape."""
    required_type = {
        "cuboid": {RigidBodyShape.BOX, RigidBodyShape.PLANE},
        "sphere": {RigidBodyShape.SPHERE},
        "capsule": {RigidBodyShape.CAPSULE},
    }
    if (
        representation in required_type
        and shape.shape_type not in required_type[representation]
    ):
        raise ValueError(
            f"Cannot represent DexSim {shape.shape_type.name} shape "
            f"{shape.name!r} as {representation!r}."
        )


def generate_curobo_world_scene(
    rigid_objects: Sequence[RigidObject],
    *,
    env_id: int = 0,
    representation: str = "auto",
    overrides: dict[str, str] | None = None,
    dynamic_obstacle_names: Sequence[str] = (),
    voxel_size: float = 0.01,
    voxel_padding: float = 0.005,
    mesh_triangle_threshold: int = 5_000,
    max_voxel_count: int = 2_000_000,
    plane_dims: tuple[float, float, float] = (10.0, 10.0, 0.01),
) -> dict[str, dict[str, dict[str, object]]]:
    """Build a mixed cuRobo scene from DexSim physical collision shapes.

    ``auto`` preserves primitives, exports collision meshes directly, and uses
    ESDF for triangle meshes whose complexity exceeds ``mesh_triangle_threshold``
    when the estimated dense grid fits ``max_voxel_count``. Forced ``voxel``
    remains available globally or per object.

    Args:
        rigid_objects: Live obstacles whose physical shapes define the world.
        env_id: Environment row used for geometry and initial poses.
        representation: Global ``auto`` or forced representation policy.
        overrides: Per-object policies keyed by rigid-object UID.
        dynamic_obstacle_names: Object UIDs whose poses change between plans.
        voxel_size: ESDF voxel edge length in meters.
        voxel_padding: Free-space padding around object-local voxel grids.
        mesh_triangle_threshold: Auto-policy triangle threshold.
        max_voxel_count: Auto-policy upper bound for a dense ESDF grid.
        plane_dims: Workspace-bounded cuboid dimensions used for planes.

    Returns:
        A mixed tensor-backed scene mapping accepted by cuRobo ``Scene.create``.

    Raises:
        ValueError: If configuration or collision geometry is unsupported.
        RuntimeError: If DexSim VisACD decomposition fails.
    """
    rigid_objects = list(rigid_objects)
    if not rigid_objects:
        raise ValueError("rigid_objects must contain at least one RigidObject.")
    overrides = overrides or {}
    supported = {"auto", "voxel", "mesh", "cuboid", "sphere", "capsule"}
    if representation not in supported or any(
        value not in supported for value in overrides.values()
    ):
        raise ValueError(f"representation policies must be one of {sorted(supported)}.")
    if voxel_size <= 0.0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}.")
    if voxel_padding < 0.0:
        raise ValueError(f"voxel_padding must be non-negative, got {voxel_padding}.")
    if mesh_triangle_threshold < 0:
        raise ValueError("mesh_triangle_threshold must be non-negative.")
    if max_voxel_count <= 0:
        raise ValueError("max_voxel_count must be positive.")
    if len(plane_dims) != 3 or any(value <= 0.0 for value in plane_dims):
        raise ValueError("plane_dims must contain three positive dimensions.")

    scene: dict[str, dict[str, dict[str, object]]] = {}
    object_names: set[str] = set()
    for object_idx, obj in enumerate(rigid_objects):
        object_name = getattr(obj, "uid", None) or f"obstacle_{object_idx}"
        if object_name in object_names:
            raise ValueError(
                f"Duplicate obstacle name {object_name!r}; RigidObject uids must be unique."
            )
        object_names.add(object_name)
        shapes = obj.get_collision_shapes(env_id=env_id)
        object_pose = (
            torch.as_tensor(
                obj.get_local_pose(to_matrix=True)[env_id], dtype=torch.float32
            )
            .detach()
            .cpu()
        )
        for shape_idx, shape in enumerate(shapes):
            obstacle_name = (
                object_name if len(shapes) == 1 else f"{object_name}__shape_{shape_idx}"
            )
            shape_pose = object_pose @ shape.local_pose
            policy = overrides.get(object_name, representation)
            if policy == "auto":
                policy = _auto_collision_representation(
                    shape,
                    is_dynamic=object_name in dynamic_obstacle_names,
                    voxel_size=voxel_size,
                    voxel_padding=voxel_padding,
                    mesh_triangle_threshold=mesh_triangle_threshold,
                    max_voxel_count=max_voxel_count,
                    plane_dims=plane_dims,
                )
            _validate_forced_representation(policy, shape)
            if shape.shape_type == RigidBodyShape.PLANE:
                offset = torch.eye(4, dtype=torch.float32)
                offset[2, 3] = -0.5 * plane_dims[2]
                shape_pose = shape_pose @ offset

            fields: dict[str, object]
            if policy == "cuboid":
                if shape.shape_type == RigidBodyShape.PLANE:
                    dims = list(plane_dims)
                else:
                    assert shape.half_extents is not None
                    dims = (2.0 * shape.half_extents).tolist()
                fields = {"pose": _pose_matrix_to_list(shape_pose), "dims": dims}
            elif policy == "sphere":
                assert shape.radius is not None
                fields = {
                    "pose": _pose_matrix_to_list(shape_pose),
                    "radius": shape.radius,
                }
            elif policy == "capsule":
                assert shape.radius is not None and shape.half_height is not None
                fields = {
                    "pose": _pose_matrix_to_list(shape_pose),
                    "radius": shape.radius,
                    "base": [0.0, 0.0, -shape.half_height],
                    "tip": [0.0, 0.0, shape.half_height],
                }
            elif policy == "mesh":
                vertices, triangles = _collision_shape_mesh(shape, plane_dims)
                fields = {
                    "pose": _pose_matrix_to_list(shape_pose),
                    "vertices": vertices.tolist(),
                    "faces": triangles.reshape(-1).tolist(),
                }
            elif policy == "voxel":
                vertices, triangles = _collision_shape_mesh(shape, plane_dims)
                _, fields = _convex_hulls_to_voxel_entry(
                    obstacle_name,
                    vertices,
                    triangles,
                    shape_pose,
                    voxel_size=voxel_size,
                    voxel_padding=voxel_padding,
                )
            else:  # pragma: no cover - policy is validated above
                raise AssertionError(f"Unhandled collision policy {policy!r}.")
            scene.setdefault(policy, {})[obstacle_name] = fields

    unknown_overrides = sorted(set(overrides) - object_names)
    if unknown_overrides:
        raise ValueError(
            f"representation overrides reference unknown RigidObject UIDs: "
            f"{unknown_overrides}."
        )

    if not scene:
        raise ValueError(
            "No collision obstacles could be generated from the given RigidObjects."
        )
    if "voxel" in scene:
        scene["voxel"] = dict(
            sorted(
                scene["voxel"].items(),
                key=lambda item: int(item[1]["feature_tensor"].numel()),
                reverse=True,
            )
        )
    return scene


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


def _get_or_create_dexsim_material(
    env: Any,
    name: str,
    color: list[float],
) -> Any:
    """Return a named DexSim material without accumulating duplicates."""
    material = env.find_material(name)
    if material is None:
        return env.create_color_material(color, name, has_alpha=len(color) == 4)
    material.set_base_color(color)
    return material


def _create_open3d_sphere_mesh(
    centers: torch.Tensor,
    radii: torch.Tensor,
) -> Any:
    """Build one Open3D mesh containing all requested collision spheres."""
    import numpy as np
    import open3d as o3d

    centers = (
        torch.as_tensor(centers, dtype=torch.float32).detach().cpu().reshape(-1, 3)
    )
    radii = torch.as_tensor(radii, dtype=torch.float32).detach().cpu().reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError(
            "Visualization sphere centers and radii must have the same length, got "
            f"{centers.shape[0]} and {radii.shape[0]}."
        )
    if torch.any(radii <= 0.0):
        raise ValueError("Visualization sphere radii must all be positive.")
    if centers.shape[0] == 0:
        raise ValueError("At least one visualization sphere is required.")

    sphere_template = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=8)
    sphere_template.compute_vertex_normals()
    template_vertices = np.asarray(sphere_template.vertices)
    template_triangles = np.asarray(sphere_template.triangles)
    template_normals = np.asarray(sphere_template.vertex_normals)
    centers_np = centers.numpy()
    radii_np = radii.numpy()

    # Vectorized assembly avoids repeated ``combined_mesh += sphere`` reallocations,
    # which become quadratic for a dense obstacle surface.
    sphere_count = centers_np.shape[0]
    vertices_per_sphere = template_vertices.shape[0]
    vertices = (
        template_vertices[None, :, :] * radii_np[:, None, None] + centers_np[:, None, :]
    ).reshape(-1, 3)
    triangle_offsets = (np.arange(sphere_count, dtype=np.int64) * vertices_per_sphere)[
        :, None, None
    ]
    triangles = (template_triangles[None, :, :] + triangle_offsets).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.vertex_normals = o3d.utility.Vector3dVector(
        np.tile(template_normals, (sphere_count, 1))
    )
    return mesh


def _load_dexsim_sphere_mesh(
    env: Any,
    centers: torch.Tensor,
    radii: torch.Tensor,
    material: Any,
) -> Any:
    """Write one combined sphere mesh to ``/tmp`` and load it into DexSim."""
    import os
    import tempfile

    import open3d as o3d

    mesh = _create_open3d_sphere_mesh(centers, radii)
    with tempfile.NamedTemporaryFile(
        prefix="curobo_collision_spheres_",
        suffix=".ply",
        dir="/tmp",
        delete=False,
    ) as temp_file:
        mesh_path = temp_file.name

    actor = None
    try:
        if not o3d.io.write_triangle_mesh(mesh_path, mesh, write_ascii=False):
            raise RuntimeError(
                f"Could not write collision sphere mesh to {mesh_path!r}."
            )
        actor = env.load_actor(mesh_path)
        if actor is None:
            raise RuntimeError(f"DexSim could not load collision mesh {mesh_path!r}.")
        actor.set_material(material)
        return actor
    except Exception:
        if actor is not None:
            env.remove_actor(actor)
        raise
    finally:
        try:
            os.unlink(mesh_path)
        except FileNotFoundError:
            pass


def _remove_dexsim_visualization_actors(env: Any, actors: Sequence[Any]) -> None:
    """Remove every temporary actor, continuing if an individual removal fails."""
    for actor in reversed(actors):
        try:
            env.remove_actor(actor)
        except Exception as exc:  # noqa: BLE001
            logger.log_warning(f"Could not remove a cuRobo visualization actor: {exc}")


def _world_collision_sphere_data(world_scene: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space samples and radii for the collision-world overlay."""
    if isinstance(world_scene, dict):
        voxel_entries = list(world_scene.get("voxel", {}).items())
    else:
        voxel_entries = [
            (voxel.name, voxel) for voxel in (getattr(world_scene, "voxel", None) or [])
        ]
    centers: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    for name, entry in voxel_entries:
        get_value = (
            entry.get if isinstance(entry, dict) else lambda key: getattr(entry, key)
        )
        features = torch.as_tensor(get_value("feature_tensor")).detach().cpu()
        voxel_size = float(get_value("voxel_size"))
        local_points = _voxel_grid_coordinates(tuple(features.shape), voxel_size)
        surface = torch.abs(features.reshape(-1)) <= 0.5 * voxel_size
        if not torch.any(surface):
            logger.log_warning(
                f"Voxel collision entry {name!r} has no samples near its zero level set."
            )
            continue
        pose = torch.as_tensor(get_value("pose"), dtype=torch.float32).detach().cpu()
        rotation = matrix_from_quat(pose[3:7])
        world_points = local_points[surface] @ rotation.T + pose[:3]
        centers.append(world_points)
        radii.append(torch.full((world_points.shape[0],), 0.5 * voxel_size))

    representation_names = ("cuboid", "sphere", "capsule", "mesh")
    for representation in representation_names:
        if isinstance(world_scene, dict):
            entries = list(world_scene.get(representation, {}).items())
        else:
            entries = [
                (entry.name, entry)
                for entry in (getattr(world_scene, representation, None) or [])
            ]
        for _, entry in entries:
            get_value = (
                entry.get
                if isinstance(entry, dict)
                else lambda key: getattr(entry, key)
            )
            pose = torch.as_tensor(get_value("pose"), dtype=torch.float32)
            rotation = matrix_from_quat(pose[3:7])
            if representation == "sphere":
                local_points = torch.zeros((1, 3), dtype=torch.float32)
                sample_radii = torch.tensor([float(get_value("radius"))])
            elif representation == "capsule":
                base = torch.as_tensor(get_value("base"), dtype=torch.float32)
                tip = torch.as_tensor(get_value("tip"), dtype=torch.float32)
                steps = torch.linspace(0.0, 1.0, 9).unsqueeze(-1)
                local_points = base + steps * (tip - base)
                sample_radii = torch.full(
                    (local_points.shape[0],), float(get_value("radius"))
                )
            elif representation == "cuboid":
                dims = torch.as_tensor(get_value("dims"), dtype=torch.float32)
                signs = torch.tensor(
                    [
                        [x, y, z]
                        for x in (-0.5, 0.5)
                        for y in (-0.5, 0.5)
                        for z in (-0.5, 0.5)
                    ],
                    dtype=torch.float32,
                )
                local_points = signs * dims
                sample_radii = torch.full(
                    (local_points.shape[0],), max(0.005, float(dims.amin()) * 0.1)
                )
            else:
                local_points = torch.as_tensor(
                    get_value("vertices"), dtype=torch.float32
                ).reshape(-1, 3)
                if local_points.shape[0] > 10_000:
                    stride = (local_points.shape[0] + 9_999) // 10_000
                    local_points = local_points[::stride]
                sample_radii = torch.full((local_points.shape[0],), 0.005)
            world_points = local_points @ rotation.T + pose[:3]
            centers.append(world_points)
            radii.append(sample_radii)

    if not centers:
        raise ValueError(
            "The cuRobo world scene contains no visible collision surface."
        )
    return torch.cat(centers), torch.cat(radii)


def visualize_curobo_world_collision_model(
    rigid_objects: Sequence[RigidObject],
    world_scene: Any,
    env_id: int = 0,
    *,
    env: Any | None = None,
    material: Any | None = None,
) -> list[Any]:
    """Add a sampled cuRobo collision-world overlay to the DexSim scene.

    The rigid objects are already present in the live DexSim scene, so this
    function only adds an overlay for the collision data consumed by cuRobo.
    Voxel zero-level samples, analytic primitives, and mesh vertices are rendered
    as spheres. All samples are merged into one Open3D mesh, written temporarily
    under ``/tmp``, and imported as one DexSim actor.

    Args:
        rigid_objects: Live simulator obstacles represented by the cache.
        world_scene: Tensor-backed scene mapping or a cuRobo ``Scene`` instance.
        env_id: Simulator environment instance represented by ``world_scene``.
            Retained for API consistency; world-scene poses are already in the
            selected environment's world frame.
        env: DexSim environment that receives the visualization actors. Uses
            the environment of :func:`dexsim.default_world` when omitted.
        material: Optional DexSim material for the collision-surface spheres.

    Returns:
        A one-element list containing the combined DexSim actor. The caller owns
        this actor and must remove it with
        :meth:`dexsim.environment.Env.remove_actor`.
    """
    import dexsim

    # Keep these parameters in the public API because the collision cache is
    # associated with the supplied live objects and simulator environment.
    _ = rigid_objects, env_id
    if env is None:
        env = dexsim.default_world().get_env()
    if material is None:
        material = _get_or_create_dexsim_material(
            env,
            "curobo_world_collision_material",
            [1.0, 0.0, 0.0, 0.45],
        )

    centers, radii = _world_collision_sphere_data(world_scene)
    return [_load_dexsim_sphere_mesh(env, centers, radii, material)]


def visualize_curobo_collision_models(
    robot: Robot,
    robot_yaml_path: str,
    rigid_objects: Sequence[RigidObject] | None = None,
    world_scene: Any | None = None,
    env_id: int = 0,
) -> None:
    """Show robot and world collision models in DexSim until Enter is pressed.

    Robot spheres are loaded from the generated cuRobo YAML and transformed by
    the current simulator link poses. Obstacle samples show the mixed scene data
    passed to cuRobo. Robot and obstacle samples are each merged into one
    temporary DexSim actor so they can use blue and red materials respectively.
    Both actors are removed before the function returns, including when ``input``
    is interrupted.
    """
    import dexsim
    import yaml

    world = dexsim.default_world()
    env = world.get_env()
    robot_material = _get_or_create_dexsim_material(
        env,
        "curobo_robot_collision_material",
        [0.0, 0.0, 1.0, 0.45],
    )
    obstacle_material = _get_or_create_dexsim_material(
        env,
        "curobo_world_collision_material",
        [1.0, 0.0, 0.0, 0.45],
    )

    with open(robot_yaml_path, encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    kinematics = data["robot_cfg"]["kinematics"]
    cached_spheres = kinematics.get("collision_spheres", {})
    sphere_buffer = float(kinematics.get("collision_sphere_buffer", 0.0))

    robot_center_batches: list[torch.Tensor] = []
    robot_radius_batches: list[torch.Tensor] = []
    visualization_actors: list[Any] = []
    try:
        for link_name, link_spheres in cached_spheres.items():
            if not link_spheres:
                continue
            link_pose = (
                torch.as_tensor(
                    robot.get_link_pose(link_name, env_ids=[env_id], to_matrix=True)[0],
                    dtype=torch.float32,
                )
                .detach()
                .cpu()
            )
            centers_local = torch.as_tensor(
                [sphere["center"] for sphere in link_spheres], dtype=torch.float32
            ).reshape(-1, 3)
            centers_world = centers_local @ link_pose[:3, :3].T + link_pose[:3, 3]
            radii = torch.as_tensor(
                [float(sphere["radius"]) + sphere_buffer for sphere in link_spheres],
                dtype=torch.float32,
            )
            robot_center_batches.append(centers_world)
            robot_radius_batches.append(radii)

        sphere_count = 0
        if robot_center_batches:
            robot_centers = torch.cat(robot_center_batches)
            robot_radii = torch.cat(robot_radius_batches)
            visualization_actors.append(
                _load_dexsim_sphere_mesh(
                    env, robot_centers, robot_radii, robot_material
                )
            )
            sphere_count += robot_centers.shape[0]
        if rigid_objects and world_scene is not None:
            world_centers, world_radii = _world_collision_sphere_data(world_scene)
            visualization_actors.append(
                _load_dexsim_sphere_mesh(
                    env, world_centers, world_radii, obstacle_material
                )
            )
            sphere_count += world_centers.shape[0]
        if not visualization_actors:
            raise ValueError("The cuRobo caches contain no collision geometry.")

        input(
            f"Showing {sphere_count} cuRobo collision spheres in "
            "DexSim. Press Enter to remove them and continue..."
        )
    finally:
        _remove_dexsim_visualization_actors(env, visualization_actors)
