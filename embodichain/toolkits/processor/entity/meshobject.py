# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d

from embodichain.toolkits.processor.component import EntityComponent
from embodichain.toolkits.processor.component import (
    OrientedBoundingBox,
    ScaleComponent,
    SpatializationComponenet,
    TriangleComponent,
    AxisAlignedBoundingBox,
    TriangleComponent,
    VisualComponent,
)
from .entity_base import EntityBase


class MeshEntity(EntityBase):
    def __init__(
        self,
        name: str,
        triangle_comp: TriangleComponent = None,
        spatialization_comp: SpatializationComponenet = None,
        visual_comp: VisualComponent = None,
    ) -> None:
        super().__init__(name)
        if triangle_comp is not None:
            self.add_component(triangle_comp)
            if visual_comp is None:
                visual_comp = VisualComponent()
            self.add_component(visual_comp)

        # init with default component
        if spatialization_comp is None:
            spatialization_comp = SpatializationComponenet()
        # add the initial component
        self.add_component(spatialization_comp)

    def is_visible(self) -> bool:
        # if not self.has_component(TriangleComponent):
        #     return False
        visual_comp = self.get_component(VisualComponent)
        if visual_comp:
            visual = visual_comp.is_visual
        else:
            visual = False
        return visual

    def add_component(self, *component: EntityComponent):
        for comp in component:
            if isinstance(comp, TriangleComponent):
                self.triangle_comp = comp
                # remove the old bounding box component
                if self.has_component(AxisAlignedBoundingBox):
                    self.remove_componenet(AxisAlignedBoundingBox)
                if self.has_component(OrientedBoundingBox):
                    self.remove_componenet(OrientedBoundingBox)
                # add default visual component
                if not self.has_component(VisualComponent):
                    self.add_component(VisualComponent())
        super().add_component(*component)

    def get_axis_aligned_bounding_box(self) -> o3d.geometry.AxisAlignedBoundingBox:
        triangle_comp: TriangleComponent = self.get_component(TriangleComponent)
        scale_comp = self.get_component(ScaleComponent)
        vertices = triangle_comp.vertices
        scale = np.array([1, 1, 1])
        if scale_comp is not None:
            scale = scale_comp.scale
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices * scale),
            o3d.utility.Vector3iVector(triangle_comp.triangles),
        )
        spatial_comp: SpatializationComponenet = self.get_component(
            SpatializationComponenet
        )
        o3d_mesh.transform(spatial_comp.get_pose())
        aabbox_o3d = o3d_mesh.get_axis_aligned_bounding_box()
        return aabbox_o3d

    # def get_oriented_bounding_box(self) -> o3d.geometry.OrientedBoundingBox:
    #     if not self.has_component(OrientedBoundingBox):
    #         # self.add_component(OrientedBoundingBox(self.triangle_comp.vertices))
    #         o3d_mesh = o3d.geometry.TriangleMesh(self.triangle_comp.vertices,
    #                                              self.triangle_comp.triangles)
    #         bbox = o3d_mesh.get_oriented_bounding_box()
    #         obb = OrientedBoundingBox(bbox.get_center(), bbox.get_extent(),
    #                                   bbox.get_rotation())
    #         self.add_component(obb)
    #     obbox = self.get_component(OrientedBoundingBox)
    #     return obbox

    def get_o3d_mesh(
        self, add_scale: bool = True, add_transform: bool = True
    ) -> o3d.geometry.TriangleMesh:
        triangle_comp: TriangleComponent = self.get_component(TriangleComponent)
        scale_comp = self.get_component(ScaleComponent)
        vertices = triangle_comp.vertices
        scale = np.array([1, 1, 1])
        if add_scale and scale_comp is not None:
            scale = scale_comp.scale
        o3d_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices * scale),
            o3d.utility.Vector3iVector(triangle_comp.triangles),
        )
        if add_transform:
            spatial_comp: SpatializationComponenet = self.get_component(
                SpatializationComponenet
            )
            o3d_mesh.transform(spatial_comp.get_pose())
        return o3d_mesh

    def save_mesh(self, file_path: str):
        from dexsim.kit.meshproc.mesh_io import save_mesh

        tri_comp: TriangleComponent = self.get_component(TriangleComponent)
        save_mesh(file_path, **tri_comp.save())
