#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
robot_surface_pointcloud_torch.py

Purpose:
1. Load a URDF using pytorch_kinematics.
2. Obtain robot link, joint, and FK information.
3. Parse each link's visual mesh.
4. Accept the robot's joint angles as a torch.Tensor.
5. Transform each link mesh to the current pose using FK.
6. Use Open3D RaycastingScene to raycast from multiple external views.
7. Sample only visible outer-surface points from the meshes.
8. Output a point cloud as a torch.Tensor of shape [N, 3].

Dependencies:
    pip install torch pytorch-kinematics open3d trimesh numpy
"""

import os
import re
import math
import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import trimesh
import open3d as o3d
import pytorch_kinematics as pk


# ============================================================
# Utility functions
# ============================================================


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def remove_xml_encoding_declaration(xml_text: str) -> str:
    """
    Fallback helper:
    Some versions of pytorch_kinematics may not accept bytes input
    if the XML declaration contains an encoding attribute. This
    removes the encoding declaration from the XML text.
    """
    return re.sub(
        r'^\s*<\?xml[^?]*encoding=["\'][^"\']+["\'][^?]*\?>',
        "",
        xml_text,
        count=1,
        flags=re.IGNORECASE,
    )


def parse_float_tensor(
    s: Optional[str],
    default: Sequence[float],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Parse xyz/rpy/scale/size fields from URDF and return a torch.Tensor.
    """
    if s is None:
        data = list(default)
    else:
        data = [float(x) for x in s.strip().split()]

    return torch.tensor(data, dtype=dtype, device=device)


def rpy_to_matrix_torch(rpy: torch.Tensor) -> torch.Tensor:
    """
    URDF rpy: roll, pitch, yaw.

    Return a rotation matrix with shape [3, 3].
    """
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]

    cr = torch.cos(roll)
    sr = torch.sin(roll)
    cp = torch.cos(pitch)
    sp = torch.sin(pitch)
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)

    dtype = rpy.dtype
    device = rpy.device

    one = torch.tensor(1.0, dtype=dtype, device=device)
    zero = torch.tensor(0.0, dtype=dtype, device=device)

    rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, cr, -sr]),
            torch.stack([zero, sr, cr]),
        ]
    )

    ry = torch.stack(
        [
            torch.stack([cp, zero, sp]),
            torch.stack([zero, one, zero]),
            torch.stack([-sp, zero, cp]),
        ]
    )

    rz = torch.stack(
        [
            torch.stack([cy, -sy, zero]),
            torch.stack([sy, cy, zero]),
            torch.stack([zero, zero, one]),
        ]
    )

    return rz @ ry @ rx


def make_transform_torch(
    xyz: torch.Tensor,
    rpy: torch.Tensor,
) -> torch.Tensor:
    """
    Construct a homogeneous transform matrix from `xyz` and `rpy`.
    Returns a [4, 4] torch.Tensor.
    """
    dtype = xyz.dtype
    device = xyz.device

    transform = torch.eye(4, dtype=dtype, device=device)
    transform[:3, :3] = rpy_to_matrix_torch(rpy)
    transform[:3, 3] = xyz

    return transform


def parse_origin_torch(
    element: Optional[ET.Element],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Parse a URDF <origin xyz="..." rpy="..."> element and return
    a 4x4 transform as a torch.Tensor.
    """
    if element is None:
        return torch.eye(4, dtype=dtype, device=device)

    xyz = parse_float_tensor(
        element.attrib.get("xyz"),
        [0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
    )

    rpy = parse_float_tensor(
        element.attrib.get("rpy"),
        [0.0, 0.0, 0.0],
        dtype=dtype,
        device=device,
    )

    return make_transform_torch(xyz, rpy)


def torch_transform_to_numpy(transform: torch.Tensor) -> np.ndarray:
    """
    Convert a torch transform matrix to numpy. Some trimesh/open3d
    legacy APIs still require numpy arrays.
    """
    return transform.detach().cpu().numpy().astype(np.float64)


def apply_transform_to_trimesh(
    mesh: trimesh.Trimesh,
    transform: torch.Tensor,
) -> trimesh.Trimesh:
    """
    Copy a mesh and apply a transform given as a torch.Tensor.
    """
    m = mesh.copy()
    m.apply_transform(torch_transform_to_numpy(transform))
    return m


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_unreferenced_vertices()
    o3d_mesh.compute_vertex_normals()

    return o3d_mesh


def merge_trimeshes(meshes: List[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    valid_meshes = []

    for m in meshes:
        if m is None:
            continue
        if len(m.vertices) == 0 or len(m.faces) == 0:
            continue
        valid_meshes.append(m)

    if len(valid_meshes) == 0:
        return None

    return trimesh.util.concatenate(valid_meshes)


def fibonacci_sphere_torch(
    num: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate uniformly distributed directions on the sphere and return
    a torch.Tensor with shape [num, 3].
    """
    dirs = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(num):
        z = 1.0 - 2.0 * (i + 0.5) / num
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden_angle * i

        x = math.cos(theta) * r
        y = math.sin(theta) * r

        dirs.append([x, y, z])

    return torch.tensor(dirs, dtype=dtype, device=device)


def choose_camera_up_torch(direction: torch.Tensor) -> torch.Tensor:
    """
    Choose an 'up' vector for the camera based on the viewing direction.
    """
    dtype = direction.dtype
    device = direction.device

    z_up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    y_up = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device)

    d = direction / torch.clamp(torch.linalg.norm(direction), min=1e-12)

    if torch.abs(torch.dot(d, z_up)) > 0.95:
        return y_up
    return z_up


# ============================================================
# Data structures
# ============================================================


@dataclass
class LinkVisualGeometry:
    link_name: str
    visual_name: str
    mesh: trimesh.Trimesh

    # Store the local transform from link to visual as a torch.Tensor
    link_to_visual: torch.Tensor


# ============================================================
# Main class
# ============================================================


class URDFRobotPointCloudSampler:
    def __init__(
        self,
        urdf_path: str,
        package_dirs: Optional[List[str]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        use_visual: bool = True,
    ):
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)

        self.package_dirs = package_dirs or []
        self.package_dirs = [os.path.abspath(p) for p in self.package_dirs]

        self.device = device
        self.dtype = dtype
        self.use_visual = use_visual

        # --------------------------------------------------------
        # Load URDF
        # --------------------------------------------------------
        with open(self.urdf_path, "rb") as f:
            self.urdf_bytes = f.read()

        self.urdf_text = self.urdf_bytes.decode("utf-8", errors="ignore")

        # --------------------------------------------------------
        # Build the kinematic chain using pytorch_kinematics.
        # Prefer bytes input to avoid lxml encoding declaration errors.
        # --------------------------------------------------------
        try:
            self.chain = pk.build_chain_from_urdf(self.urdf_bytes)
        except Exception:
            clean_text = remove_xml_encoding_declaration(self.urdf_text)
            self.chain = pk.build_chain_from_urdf(clean_text)

        self.chain = self.chain.to(dtype=dtype, device=device)

        self.joint_names = list(self.chain.get_joint_parameter_names())

        try:
            self.pk_link_names = list(self.chain.get_link_names())
        except Exception:
            self.pk_link_names = []

        # --------------------------------------------------------
        # XML parsing
        # --------------------------------------------------------
        self.xml_root = ET.fromstring(self.urdf_bytes)
        self.robot_name = self.xml_root.attrib.get("name", "unnamed_robot")

        self.xml_link_names = self._parse_link_names()
        self.xml_joint_info = self._parse_joint_info()
        self.root_link_name = self._infer_root_link()

        self.link_visuals: Dict[str, List[LinkVisualGeometry]] = (
            self._parse_link_visual_geometries()
        )

    # ------------------------------------------------------------
    # URDF information parsing
    # ------------------------------------------------------------

    def _parse_link_names(self) -> List[str]:
        link_names = []

        for child in self.xml_root:
            if strip_namespace(child.tag) == "link":
                name = child.attrib.get("name")
                if name:
                    link_names.append(name)

        return link_names

    def _parse_joint_info(self) -> List[Dict]:
        joint_info = []

        for child in self.xml_root:
            if strip_namespace(child.tag) != "joint":
                continue

            joint_name = child.attrib.get("name", "")
            joint_type = child.attrib.get("type", "")

            parent_name = None
            child_name = None
            axis = None
            limit = {}

            for elem in child:
                tag = strip_namespace(elem.tag)

                if tag == "parent":
                    parent_name = elem.attrib.get("link")

                elif tag == "child":
                    child_name = elem.attrib.get("link")

                elif tag == "axis":
                    axis = parse_float_tensor(
                        elem.attrib.get("xyz"),
                        [0.0, 0.0, 1.0],
                        dtype=self.dtype,
                        device=self.device,
                    )

                elif tag == "limit":
                    for k, v in elem.attrib.items():
                        try:
                            limit[k] = float(v)
                        except ValueError:
                            limit[k] = v

            joint_info.append(
                {
                    "name": joint_name,
                    "type": joint_type,
                    "parent": parent_name,
                    "child": child_name,
                    "axis": axis,
                    "limit": limit,
                }
            )

        return joint_info

    def _infer_root_link(self) -> Optional[str]:
        all_links = set(self.xml_link_names)
        child_links = set()

        for j in self.xml_joint_info:
            if j["child"] is not None:
                child_links.add(j["child"])

        roots = list(all_links - child_links)

        if len(roots) == 0:
            return None

        return roots[0]

    # ------------------------------------------------------------
    # Mesh path resolution and loading
    # ------------------------------------------------------------

    def _resolve_mesh_path(self, filename: str) -> str:
        if filename.startswith("file://"):
            filename = filename[len("file://") :]

        if os.path.isabs(filename):
            if os.path.exists(filename):
                return filename
            raise FileNotFoundError(f"Mesh file not found: {filename}")

        if filename.startswith("package://"):
            rest = filename[len("package://") :]
            parts = rest.split("/", 1)

            if len(parts) == 1:
                package_name = parts[0]
                relative_path = ""
            else:
                package_name, relative_path = parts

            search_roots = []
            search_roots.extend(self.package_dirs)
            search_roots.append(self.urdf_dir)
            search_roots.append(os.path.dirname(self.urdf_dir))

            candidates = []

            for root in search_roots:
                candidates.append(os.path.join(root, package_name, relative_path))
                candidates.append(os.path.join(root, relative_path))

            for c in candidates:
                if os.path.exists(c):
                    return os.path.abspath(c)

            raise FileNotFoundError(
                "Cannot resolve package mesh path:\n"
                f"  filename: {filename}\n"
                f"  package_name: {package_name}\n"
                f"  relative_path: {relative_path}\n"
                f"  package_dirs: {self.package_dirs}\n"
                f"  urdf_dir: {self.urdf_dir}"
            )

        candidates = [
            os.path.join(self.urdf_dir, filename),
            os.path.join(os.getcwd(), filename),
        ]

        for root in self.package_dirs:
            candidates.append(os.path.join(root, filename))

        for c in candidates:
            if os.path.exists(c):
                return os.path.abspath(c)

        raise FileNotFoundError(f"Cannot resolve mesh path: {filename}")

    def _load_mesh_file(
        self,
        filename: str,
        scale: torch.Tensor,
    ) -> trimesh.Trimesh:
        mesh_path = self._resolve_mesh_path(filename)

        loaded = trimesh.load(mesh_path, force="mesh", process=False)

        if isinstance(loaded, trimesh.Scene):
            meshes = []
            for geom in loaded.geometry.values():
                if isinstance(geom, trimesh.Trimesh):
                    meshes.append(geom)

            if len(meshes) == 0:
                raise RuntimeError(f"No valid geometry in mesh file: {mesh_path}")

            mesh = trimesh.util.concatenate(meshes)

        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded

        else:
            raise RuntimeError(f"Unsupported mesh type from file: {mesh_path}")

        mesh = mesh.copy()

        scale_np = scale.detach().cpu().numpy().astype(np.float64)

        if scale_np.shape[0] == 3:
            mesh.vertices *= scale_np.reshape(1, 3)
        else:
            raise ValueError(f"Invalid mesh scale: {scale_np}")

        return mesh

    def _parse_link_visual_geometries(self) -> Dict[str, List[LinkVisualGeometry]]:
        """
        解析每个 link 下的 visual/collision geometry。

        注意：
        这里只使用 <mesh> 类型的 geometry。
        以下 primitive 不参与点云采样：
            - <box>
            - <cylinder>
            - <sphere>

        也就是说，只有类似下面这种会被加入：
            <geometry>
                <mesh filename="xxx.stl" scale="1 1 1"/>
            </geometry>
        """
        result: Dict[str, List[LinkVisualGeometry]] = {}

        for link_elem in self.xml_root:
            if strip_namespace(link_elem.tag) != "link":
                continue

            link_name = link_elem.attrib.get("name")

            if not link_name:
                continue

            visuals: List[LinkVisualGeometry] = []

            for visual_elem in link_elem:
                visual_tag = strip_namespace(visual_elem.tag)

                if self.use_visual:
                    if visual_tag != "visual":
                        continue
                else:
                    if visual_tag != "collision":
                        continue

                visual_name = visual_elem.attrib.get("name", "")

                origin_elem = None
                geometry_elem = None

                for elem in visual_elem:
                    tag = strip_namespace(elem.tag)

                    if tag == "origin":
                        origin_elem = elem
                    elif tag == "geometry":
                        geometry_elem = elem

                if geometry_elem is None:
                    continue

                link_to_visual = parse_origin_torch(
                    origin_elem,
                    dtype=self.dtype,
                    device=self.device,
                )

                mesh_obj = None

                # ----------------------------------------------------
                # 这里只处理 <mesh>，其余 primitive 全部跳过
                # ----------------------------------------------------
                for geom_child in geometry_elem:
                    geom_tag = strip_namespace(geom_child.tag)

                    if geom_tag != "mesh":
                        # 跳过 box / cylinder / sphere 等 primitive
                        continue

                    filename = geom_child.attrib.get("filename")
                    if filename is None:
                        continue

                    scale = parse_float_tensor(
                        geom_child.attrib.get("scale"),
                        [1.0, 1.0, 1.0],
                        dtype=self.dtype,
                        device=self.device,
                    )

                    mesh_obj = self._load_mesh_file(filename, scale)

                    # 一个 geometry 正常只会有一个 mesh，找到后即可退出
                    break

                if mesh_obj is None:
                    # 当前 visual/collision 不是 mesh，或者 mesh 加载失败，则跳过
                    continue

                visuals.append(
                    LinkVisualGeometry(
                        link_name=link_name,
                        visual_name=visual_name,
                        mesh=mesh_obj,
                        link_to_visual=link_to_visual,
                    )
                )

            result[link_name] = visuals

        return result

    # ------------------------------------------------------------
    # Information printing
    # ------------------------------------------------------------

    def print_robot_info(self):
        print("=" * 80)
        print(f"Robot name: {self.robot_name}")
        print(f"URDF path: {self.urdf_path}")
        print(f"Root link: {self.root_link_name}")
        print("-" * 80)

        print(f"Number of XML links: {len(self.xml_link_names)}")
        print("XML links:")
        for name in self.xml_link_names:
            n_visual = len(self.link_visuals.get(name, []))
            print(f"  - {name}, visuals: {n_visual}")

        print("-" * 80)
        print(
            f"Number of active joints from pytorch_kinematics: {len(self.joint_names)}"
        )
        print("Active joint order:")
        for i, name in enumerate(self.joint_names):
            print(f"  [{i}] {name}")

        print("-" * 80)
        print(f"Number of XML joints: {len(self.xml_joint_info)}")
        print("XML joints:")
        for j in self.xml_joint_info:
            axis = j["axis"]

            if isinstance(axis, torch.Tensor):
                axis_str = axis.detach().cpu().numpy()
            else:
                axis_str = axis

            print(
                f"  - {j['name']}: "
                f"type={j['type']}, "
                f"parent={j['parent']}, "
                f"child={j['child']}, "
                f"axis={axis_str}, "
                f"limit={j['limit']}"
            )

        print("-" * 80)
        print("Kinematic tree from pytorch_kinematics:")
        try:
            self.chain.print_tree()
        except Exception as e:
            print(f"  chain.print_tree() failed: {e}")

        print("=" * 80)

    # ------------------------------------------------------------
    # Joint angle (q) handling
    # ------------------------------------------------------------

    def _normalize_joint_angles(self, q) -> torch.Tensor:
        """
        Supported input formats:
        1. torch.Tensor
        2. list / tuple / np.ndarray
        3. dict with joint names as keys

        Returns a torch.Tensor with shape [1, num_joints].
        """
        n = len(self.joint_names)

        if isinstance(q, dict):
            q_list = []
            for name in self.joint_names:
                if name not in q:
                    raise KeyError(f"Joint '{name}' missing in q dict")
                q_list.append(float(q[name]))

            q_tensor = torch.tensor(
                q_list,
                dtype=self.dtype,
                device=self.device,
            )

        elif isinstance(q, torch.Tensor):
            q_tensor = q.to(dtype=self.dtype, device=self.device)

        else:
            q_tensor = torch.tensor(
                q,
                dtype=self.dtype,
                device=self.device,
            )

        if q_tensor.ndim == 1:
            if q_tensor.numel() != n:
                raise ValueError(
                    f"Expected {n} joint values, got {q_tensor.numel()}.\n"
                    f"Joint order is: {self.joint_names}"
                )

            q_tensor = q_tensor.reshape(1, n)

        elif q_tensor.ndim == 2:
            if q_tensor.shape[0] != 1:
                raise ValueError("This implementation expects batch size 1.")

            if q_tensor.shape[1] != n:
                raise ValueError(
                    f"Expected q shape [1, {n}], got {list(q_tensor.shape)}.\n"
                    f"Joint order is: {self.joint_names}"
                )

        else:
            raise ValueError(f"Invalid q shape: {list(q_tensor.shape)}")

        return q_tensor

    # ------------------------------------------------------------
    # Forward kinematics (FK)
    # ------------------------------------------------------------

    def compute_link_transforms(self, q) -> Dict[str, torch.Tensor]:
        """
        Compute transforms for each link under the current joint angles.

        Returns:
            Dict[str, torch.Tensor]

        Each transform has shape [4, 4].
        """
        q_tensor = self._normalize_joint_angles(q)

        with torch.no_grad():
            try:
                fk_result = self.chain.forward_kinematics(
                    q_tensor,
                    end_only=False,
                )
            except TypeError:
                fk_result = self.chain.forward_kinematics(q_tensor)

        if not isinstance(fk_result, dict):
            raise RuntimeError(
                "forward_kinematics() did not return a dict. "
                "Please check your pytorch_kinematics version."
            )

        transforms: Dict[str, torch.Tensor] = {}

        for link_name, transform in fk_result.items():
            mat = transform.get_matrix()

            if mat.ndim == 3:
                mat = mat[0]
            elif mat.ndim == 2:
                pass
            else:
                raise RuntimeError(
                    f"Unexpected transform matrix shape for link {link_name}: {mat.shape}"
                )

            transforms[link_name] = mat.to(
                dtype=self.dtype,
                device=self.device,
            )

        if self.root_link_name is not None and self.root_link_name not in transforms:
            transforms[self.root_link_name] = torch.eye(
                4,
                dtype=self.dtype,
                device=self.device,
            )

        return transforms

    # ------------------------------------------------------------
    # Meshes transformed to the current pose
    # ------------------------------------------------------------

    def get_transformed_link_meshes(self, q) -> List[trimesh.Trimesh]:
        """
        Transform all link visual meshes to world coordinates for the
        given joint angles.

        Notes:
        - Internal transforms use torch.Tensor.
        - When applying transforms to `trimesh`, tensors are converted
            to numpy arrays.
        """
        link_transforms = self.compute_link_transforms(q)

        world_meshes: List[trimesh.Trimesh] = []

        for link_name, visuals in self.link_visuals.items():
            if len(visuals) == 0:
                continue

            if link_name not in link_transforms:
                print(
                    f"[WARN] Link '{link_name}' has visual mesh but no FK transform. Skipped."
                )
                continue

            T_world_link = link_transforms[link_name]

            for visual in visuals:
                T_world_visual = T_world_link @ visual.link_to_visual
                mesh_world = apply_transform_to_trimesh(
                    visual.mesh,
                    T_world_visual,
                )
                world_meshes.append(mesh_world)

        return world_meshes

    # ------------------------------------------------------------
    # Raycast outer-surface point cloud sampling
    # ------------------------------------------------------------

    def sample_pointcloud_from_joint_angles(
        self,
        q,
        num_points: int = 4096,
        n_views: int = 80,
        width_px: int = 160,
        height_px: int = 120,
        fov_deg: float = 60.0,
        voxel_size: Optional[float] = None,
        random_seed: Optional[int] = 0,
    ) -> torch.Tensor:
        """
        Given joint angles, sample the robot's outer-surface point cloud.

        Args:
            q: Joint angles. Prefer a torch.Tensor. Shape may be
               [num_joints] or [1, num_joints].

            num_points: Maximum number of returned points.

            n_views: Number of external raycast viewpoints.

            width_px, height_px: Virtual camera resolution per view.

            fov_deg: Virtual camera field of view in degrees.

            voxel_size: Voxel size for Open3D downsampling. If None,
                it will be estimated automatically.

            random_seed: Seed for final random sampling.

        Returns:
            torch.Tensor of shape [N, 3] with the same device and dtype
            as the sampler.
        """
        world_meshes = self.get_transformed_link_meshes(q)
        combined = merge_trimeshes(world_meshes)

        if combined is None:
            raise RuntimeError("No valid robot mesh found. Check URDF visual meshes.")

        legacy_mesh = trimesh_to_open3d(combined)

        bbox = legacy_mesh.get_axis_aligned_bounding_box()

        center_np = np.asarray(bbox.get_center(), dtype=np.float64)
        extent_np = np.asarray(bbox.get_extent(), dtype=np.float64)

        center = torch.tensor(
            center_np,
            dtype=self.dtype,
            device=self.device,
        )

        extent = torch.tensor(
            extent_np,
            dtype=self.dtype,
            device=self.device,
        )

        diag = torch.linalg.norm(extent)

        if float(diag.detach().cpu()) < 1e-9:
            raise RuntimeError("Mesh bounding box is too small or invalid.")

        # Open3D RaycastingScene
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(tensor_mesh)

        radius = 0.5 * diag
        half_fov = math.radians(fov_deg) * 0.5

        camera_distance = radius / max(math.sin(half_fov), 1e-6)
        camera_distance = camera_distance * 1.25

        directions = fibonacci_sphere_torch(
            n_views,
            dtype=self.dtype,
            device=self.device,
        )

        all_points_torch = []

        center_o3d = center.detach().cpu().numpy().astype(np.float32)

        for i in range(n_views):
            d = directions[i]
            eye = center + camera_distance * d
            up = choose_camera_up_torch(d)

            eye_o3d = eye.detach().cpu().numpy().astype(np.float32)
            up_o3d = up.detach().cpu().numpy().astype(np.float32)

            rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
                fov_deg=float(fov_deg),
                center=center_o3d,
                eye=eye_o3d,
                up=up_o3d,
                width_px=int(width_px),
                height_px=int(height_px),
            )

            ans = scene.cast_rays(rays)

            t_hit_np = ans["t_hit"].numpy()
            rays_np = rays.numpy()

            mask_np = np.isfinite(t_hit_np)

            if not np.any(mask_np):
                continue

            ray_origins_np = rays_np[..., :3][mask_np]
            ray_dirs_np = rays_np[..., 3:6][mask_np]
            t_vals_np = t_hit_np[mask_np].reshape(-1, 1)

            points_np = ray_origins_np + ray_dirs_np * t_vals_np

            points_torch = torch.tensor(
                points_np,
                dtype=self.dtype,
                device=self.device,
            )

            all_points_torch.append(points_torch)

        if len(all_points_torch) == 0:
            raise RuntimeError(
                "Raycast did not hit the robot mesh. Check mesh scale and URDF."
            )

        points = torch.cat(all_points_torch, dim=0)

        valid = torch.isfinite(points).all(dim=1)
        points = points[valid]

        if points.numel() == 0:
            raise RuntimeError("No valid points after raycasting.")

        # --------------------------------------------------------
        # Open3D voxel downsample
        # Convert to numpy temporarily because legacy Open3D PointCloud
        # APIs require numpy arrays.
        # --------------------------------------------------------
        if voxel_size is None:
            voxel_size = float((diag / 350.0).detach().cpu())

        if voxel_size is not None and voxel_size > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                points.detach().cpu().numpy().astype(np.float64)
            )

            pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

            points_np = np.asarray(pcd.points, dtype=np.float64)

            points = torch.tensor(
                points_np,
                dtype=self.dtype,
                device=self.device,
            )

        # --------------------------------------------------------
        # Randomly sample down to `num_points`
        # --------------------------------------------------------
        if num_points is not None and num_points > 0 and points.shape[0] > num_points:
            if random_seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(random_seed))
                idx = torch.randperm(
                    points.shape[0],
                    device=self.device,
                    generator=generator,
                )[:num_points]
            else:
                idx = torch.randperm(
                    points.shape[0],
                    device=self.device,
                )[:num_points]

            points = points[idx]

        return points


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate robot outer surface point cloud from URDF using pytorch_kinematics and Open3D raycasting."
    )

    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to URDF file.",
    )

    parser.add_argument(
        "--package_dirs",
        type=str,
        nargs="*",
        default=[],
        help="Directories used to resolve package:// mesh paths.",
    )

    parser.add_argument(
        "--q",
        type=float,
        nargs="*",
        default=None,
        help="Joint angles in pytorch_kinematics joint order.",
    )

    parser.add_argument(
        "--num_points",
        type=int,
        default=327680,
        help="Maximum number of output points.",
    )

    parser.add_argument(
        "--n_views",
        type=int,
        default=80,
        help="Number of external raycasting views.",
    )

    parser.add_argument(
        "--width_px",
        type=int,
        default=160,
        help="Virtual camera width.",
    )

    parser.add_argument(
        "--height_px",
        type=int,
        default=120,
        help="Virtual camera height.",
    )

    parser.add_argument(
        "--fov_deg",
        type=float,
        default=60.0,
        help="Virtual camera field of view in degrees.",
    )

    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Voxel size for point cloud downsampling. If omitted, automatically estimated.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device. Recommended cpu because Open3D raycasting is CPU-oriented in this script.",
    )

    parser.add_argument(
        "--no_info",
        action="store_true",
        help="Do not print robot information.",
    )

    args = parser.parse_args()

    sampler = URDFRobotPointCloudSampler(
        urdf_path=args.urdf,
        package_dirs=args.package_dirs,
        device=args.device,
        dtype=torch.float32,
        use_visual=True,
    )

    if not args.no_info:
        sampler.print_robot_info()

    if args.q is None or len(args.q) == 0:
        print("[INFO] No --q provided, using all-zero joint angles.")
        q = torch.zeros(
            len(sampler.joint_names),
            dtype=sampler.dtype,
            device=sampler.device,
        )
    else:
        q = torch.tensor(
            args.q,
            dtype=sampler.dtype,
            device=sampler.device,
        )

    print("[INFO] Joint order:")
    for i, name in enumerate(sampler.joint_names):
        print(f"  q[{i}] -> {name}")

    print(f"[INFO] q tensor shape: {tuple(q.shape)}")
    print(f"[INFO] q tensor device: {q.device}")
    print(f"[INFO] q tensor dtype: {q.dtype}")

    points = sampler.sample_pointcloud_from_joint_angles(
        q=q,
        num_points=args.num_points,
        n_views=args.n_views,
        width_px=args.width_px,
        height_px=args.height_px,
        fov_deg=args.fov_deg,
        voxel_size=args.voxel_size,
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        points.detach().cpu().numpy().astype(np.float64)
    )
    o3d.visualization.draw_geometries([pcd], window_name="Robot Surface Point Cloud")

    print(f"[INFO] Output point cloud type: {type(points)}")
    print(f"[INFO] Output point cloud shape: {tuple(points.shape)}")
    print(f"[INFO] Output point cloud device: {points.device}")
    print(f"[INFO] Output point cloud dtype: {points.dtype}")

    # `points` is the final point cloud as a torch.Tensor
    # shape: [N, 3]
    return points


if __name__ == "__main__":
    main()
