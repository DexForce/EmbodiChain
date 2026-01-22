import open3d as o3d
import numpy as np
from typing import List, Tuple

from dexsim.kit.meshproc import face_uv_to_vert_uv
from dexsim.kit.meshproc.compute_uv import get_mesh_auto_uv
from dexsim.kit.meshproc import (
    simplification_decimation,
    remesh_isotropic_explicit,
)
import dexsim.utility as dexsutil

from embodichain.toolkits.processor.entity import MeshEntity
from embodichain.toolkits.processor.component import TriangleComponent

from .base import MeshProcessor


class ComputeUV(MeshProcessor):
    def __init__(
        self,
        compute_vertex_uvs: bool = True,
        max_triangle_count: int = 10000,
        remesh: bool = False,
    ):
        """Compute UV coordinates.

        Args:
            compute_vertex_uvs (bool, optional): Compute texture uvs or triangle uvs, if True, compute texture uvs. Defaults to True.
            max_triangle_count (int, optional): If the number of faces is larger than this value and there is no uvs, simplification will be applied.
                                            It will cost more time to compute uvs if the number of faces is large .Defaults to 10000.
            remesh (bool, optional): If set to True, remesh will be applied, and the uvs will be re-computed. Defaults to False.
        """
        self.compute_vertex_uvs = compute_vertex_uvs
        self.max_triangle_count = max_triangle_count
        self.remesh = remesh

    def apply(self, meshes: List[MeshEntity]) -> List[MeshEntity]:
        for mesh in meshes:
            tri_comp: TriangleComponent = mesh.get_component(TriangleComponent)
            has_uvs = tri_comp.vertex_uvs.size > 0 or tri_comp.triangle_uvs.size > 0

            mesh_o3d = mesh.get_o3d_mesh(add_scale=False, add_transform=False)
            mesh_o3dt = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
            # if the number of faces is larger than max_triangle_count and there is no uvs, simplification will be applied
            if not has_uvs:
                if tri_comp.triangles.shape[0] > self.max_triangle_count:
                    # simplification
                    is_success, mesh_o3dt = simplification_decimation(
                        mesh_o3dt, sample_triangle_num=self.max_triangle_count
                    )
                    if not is_success:
                        dexsutil.log_warning("failed to do simplification.")
            # remesh need to apply after simplification
            if self.remesh:
                is_success, mesh_o3dt = remesh_isotropic_explicit(
                    mesh_o3dt, is_visual=False
                )
                # has_uvs = False     # need to recompute uvs
            if self.compute_vertex_uvs:
                if tri_comp.vertex_uvs.size == 0 or self.remesh:
                    if tri_comp.triangle_uvs.size > 0:
                        vertex_uvs = face_uv_to_vert_uv(
                            tri_comp.triangles,
                            tri_comp.triangle_uvs,
                            len(tri_comp.vertices),
                        )
                    else:
                        _, vertex_uvs = get_mesh_auto_uv(mesh_o3dt)
                    tri_comp = tri_comp.new(vertex_uvs=vertex_uvs)
                    mesh.add_component(tri_comp)
            else:
                dexsutil.log_error("Not implemented for compute triangle uvs.")
        return meshes


class MeshNormalize(MeshProcessor):
    def __init__(
        self,
        set_origin: str = "center",
        scale: float = 1.0,
        unify_longest_side: bool = False,
    ):
        """Normalize the mesh to a standard size and origin.

        Args:
            set_origin (str, optional): Set the origin location of the mesh to it's center or it's bottom center.
                                Choices=["center", "bottom"]. Defaults to 'center'.
            scale (float, optional): Scale factor for the mesh . Defaults to 1.0.
            unify_longest_side (float, optional): If True, the longest side of the mesh will be scaled to the scale factor.
                                Defaults to False.
        """
        assert set_origin in [
            "center",
            "bottom",
        ], f"Invalid value for set_origin: {set_origin}"
        self.set_origin = set_origin
        self.scale = scale
        self.unify_longest_side = unify_longest_side

    def apply(self, meshes: List[MeshEntity]) -> List[MeshEntity]:
        for mesh in meshes:
            tri_comp: TriangleComponent = mesh.get_component(TriangleComponent)
            vertices = tri_comp.vertices
            # set center of the mesh to the origin
            if self.set_origin == "center":
                center = np.mean(vertices, axis=0)
            elif self.set_origin == "bottom":
                center_xy = np.mean(vertices[:, :2], axis=0)
                center = np.array([center_xy[0], center_xy[1], np.min(vertices[:, 2])])
            else:
                raise ValueError(f"Invalid value for set_origin: {self.set_origin}")
            vertices -= center  # in-place operation

            # scale the mesh
            if self.unify_longest_side:
                max_length = np.max(vertices, axis=0) - np.min(vertices, axis=0)
                scale = self.scale / np.max(max_length)
            else:
                scale = self.scale
            vertices *= scale
        return meshes


class MeshAlign(MeshProcessor):
    def __init__(
        self,
        method: str = "obb",
        symmetry_axis: int = 0,
        is_larger_positive: bool = True,
    ):
        assert method in ["obb", "svd"], f"Invalid value for method: {method}"
        self.method = method
        self.symmetry_axis = symmetry_axis
        self.is_larger_positive = is_larger_positive

    def apply(self, meshes: List[MeshEntity]) -> List[MeshEntity]:
        from dexsim.kit.meshproc import cad_standardlize_svd, cad_standardlize_obb

        for mesh in meshes:
            mesh_o3d = mesh.get_o3d_mesh(add_scale=False, add_transform=False)
            if self.method == "obb":
                is_success, mesh_o3dt = cad_standardlize_obb(
                    mesh_o3d,
                    is_use_mesh_clean=False,
                    is_cad_eliminate_symmetry=True,
                    symmetry_axis=self.symmetry_axis,
                    is_larger_positive=self.is_larger_positive,
                )
            elif self.method == "svd":
                is_success, mesh_o3dt = cad_standardlize_svd(
                    mesh_o3d,
                    is_use_mesh_clean=False,
                    is_cad_eliminate_symmetry=True,
                    symmetry_axis=self.symmetry_axis,
                    is_larger_positive=self.is_larger_positive,
                )
            vertices = mesh_o3dt.vertex.positions.numpy()
            triangles = mesh_o3dt.triangle.indices.numpy()
            tri_comp = mesh.get_component(TriangleComponent)
            tri_comp = tri_comp.new(vertices=vertices, triangles=triangles)
            mesh.add_component(tri_comp)
        return meshes
