# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
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
import os
import xml.etree.ElementTree as ET
import open3d as o3d
from dexsim.kit.meshproc.convex_decomposition import convex_decomposition_coacd
from dexsim.kit.meshproc.utility import mesh_list_to_file
from embodichain.utils import logger
import trimesh
import numpy as np


def generate_urdf_collision_convexes(
    urdf_path: str,
    output_urdf_name: str,
    max_convex_hull_num: int = 16,
    recompute_inertia: bool = False,
    scale: np.ndarray = None,
):
    decomposer = URDFModifider(
        urdf_path,
        max_convex_hull_num=max_convex_hull_num,
        recompute_inertia=recompute_inertia,
        scale=scale,
    )
    decomposer.decompose_collisions()
    decomposer.save_urdf(output_urdf_name)


class URDFModifider:
    def __init__(self, urdf_path: str, **kwargs):
        self.urdf_path = urdf_path
        self.urdf_dir = os.path.dirname(urdf_path)
        self.urdf = ET.parse(urdf_path)
        self.root = self.urdf.getroot()

        self.max_convex_hull_num = kwargs.get("max_convex_hull_num", 16)
        self.recompute_inertia = kwargs.get("recompute_inertia", False)
        self.scale = kwargs.get("scale", None)

    def decompose_collisions(self):
        link_dict = dict()
        if self.scale is not None:
            self.scale_urdf(self.scale)

        for link in self.root.findall("link"):
            link_name = link.get("name")
            link_dict[link_name] = link

        for link_name, link in link_dict.items():
            self.decomposite_link_collision(link_name, link)
            if self.recompute_inertia:
                self.recompute_link_inertia(link)

    def decomposite_link_collision(
        self,
        link_name: str,
        link: ET.Element,
    ):
        visual = link.find("visual")
        collision = link.find("collision")
        if visual is None and collision is None:
            logger.log_warning(
                f"Link {link_name} has no visual and collision geometry."
            )
            return
        if collision is None:
            geom = visual.find("geometry")
            # use visual geometry mesh
        else:
            # use collision geometry mesh
            geom = collision.find("geometry")

        if geom is None:
            logger.log_warning(f"Link {link_name} has no geometry.")
            return
        mesh_elem = geom.find("mesh")
        if mesh_elem is None:
            logger.log_warning(f"Link {link_name} geometry is not a mesh.")
            return
        mesh_filename = mesh_elem.get("filename")
        mesh_path = os.path.join(self.urdf_dir, mesh_filename)
        mesh_base_name = os.path.basename(mesh_filename).split(".")[0]
        if not os.path.isfile(mesh_path):
            logger.log_warning(f"Mesh file {mesh_path} does not exist.")
            return

        mesh = o3d.t.io.read_triangle_mesh(mesh_path)
        _, convex_meshes = convex_decomposition_coacd(
            mesh, max_convex_hull_num=self.max_convex_hull_num
        )
        convex_mesh_file = f"{mesh_base_name}_auto_convex.obj"
        # create collision mesh dir
        collision_dir = os.path.join(self.urdf_dir, "Collision")
        if not os.path.exists(collision_dir):
            os.makedirs(collision_dir)
        collision_relative_path = os.path.join("Collision", convex_mesh_file)

        mesh_list_to_file(
            save_path=os.path.join(self.urdf_dir, collision_relative_path),
            mesh_list=convex_meshes,
        )

        if collision is None:
            # create collision element and save to urdf xml tree
            collision = ET.SubElement(link, "collision")
            collision_origin = ET.SubElement(collision, "origin")
            visual_origin = visual.find("origin")
            if visual_origin is not None:
                collision_origin.set("xyz", visual_origin.get("xyz", "0 0 0"))
                collision_origin.set("rpy", visual_origin.get("rpy", "0 0 0"))
            else:
                collision_origin.set("xyz", "0 0 0")
                collision_origin.set("rpy", "0 0 0")
            geom = ET.SubElement(collision, "geometry")
            mesh = ET.SubElement(geom, "mesh")
            mesh.set("filename", collision_relative_path)
        else:
            # update collision mesh file path
            geom = collision.find("geometry")
            mesh = geom.find("mesh")
            mesh.set("filename", collision_relative_path)

    def recompute_link_inertia(self, link: ET.Element):
        collision = link.find("collision")
        if collision is None:
            return
        has_mesh = self.has_mesh(collision)
        if not has_mesh:
            return
        geom = collision.find("geometry")
        mesh_elem = geom.find("mesh")
        mesh_filename = mesh_elem.get("filename")
        mesh_path = os.path.join(self.urdf_dir, mesh_filename)
        mass, com, inertia = self.compute_inertia_from_mesh(mesh_path)
        if mass is None or com is None or inertia is None:
            return
        inertial = link.find("inertial")
        if inertial is None:
            inertial = ET.SubElement(link, "inertial")
        mass_elem = inertial.find("mass")
        if mass_elem is None:
            mass_elem = ET.SubElement(inertial, "mass")
        mass_elem.set("value", str(mass))
        origin_elem = inertial.find("origin")
        if origin_elem is None:
            origin_elem = ET.SubElement(inertial, "origin")
        origin_elem.set("xyz", f"{com[0]} {com[1]} {com[2]}")
        origin_elem.set("rpy", "0 0 0")
        inertia_elem = inertial.find("inertia")
        if inertia_elem is None:
            inertia_elem = ET.SubElement(inertial, "inertia")
        inertia_elem.set("ixx", str(inertia[0]))
        inertia_elem.set("iyy", str(inertia[1]))
        inertia_elem.set("izz", str(inertia[2]))
        inertia_elem.set("ixy", str(inertia[3]))
        inertia_elem.set("ixz", str(inertia[4]))
        inertia_elem.set("iyz", str(inertia[5]))

    @staticmethod
    def compute_inertia_from_mesh(mesh_path: str, density: float = 1.0):
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                logger.log_warning(f"Mesh scene {mesh_path} has no geometry.")
                return None, None, None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        if not mesh.is_watertight:
            logger.log_warning(
                f"Mesh {mesh_path} is not watertight. Inertia computation may be inaccurate."
            )
        mesh.density = density

        mass = mesh.mass
        center_of_mass = mesh.center_mass
        inertia = mesh.moment_inertia
        inertia = [
            inertia[0, 0],
            inertia[1, 1],
            inertia[2, 2],
            inertia[0, 1],
            inertia[0, 2],
            inertia[1, 2],
        ]

        return mass, center_of_mass, inertia

    def has_mesh(self, colli_visual: ET.Element | None) -> bool:
        if colli_visual is None:
            return False
        geom = colli_visual.find("geometry")
        if geom is None:
            return False
        mesh = geom.find("mesh")
        if mesh is None:
            return False
        filename = mesh.get("filename", "")
        if not os.path.join(self.urdf_dir, filename):
            return False
        return True

    def scale_urdf(self, scale: list):
        """_summary_

        Args:
            scale (np.ndarray): [scale_x, scale_y, scale_z]
        """
        for link in self.root.findall("link"):
            inertial_elem = link.find("inertial")
            if inertial_elem:
                origin_elem = inertial_elem.find("origin")
                if origin_elem is not None:
                    xyz = origin_elem.get("xyz", "0 0 0").split()
                    xyz = [str(float(x) * s) for x, s in zip(xyz, scale)]
                    origin_elem.set("xyz", " ".join(xyz))
                mass_elem = inertial_elem.find("mass")
                if mass_elem is not None:
                    mass = float(mass_elem.get("value", "1.0"))
                    mass_elem.set("value", str(mass * scale[0] * scale[1] * scale[2]))

            scale_dir = os.path.join(self.urdf_dir, "Scale")
            if not os.path.exists(scale_dir):
                os.makedirs(scale_dir)

            visual_elem = link.find("visual")
            if visual_elem:
                origin_elem = visual_elem.find("origin")
                if origin_elem is not None:
                    xyz = origin_elem.get("xyz", "0 0 0").split()
                    xyz = [str(float(x) * s) for x, s in zip(xyz, scale)]
                    origin_elem.set("xyz", " ".join(xyz))
                geometry_elem = visual_elem.find("geometry")
                if geometry_elem is not None:
                    mesh_elem = geometry_elem.find("mesh")
                    mesh_filename = mesh_elem.get("filename", "")
                    mesh_base_name = os.path.basename(mesh_filename).split(".")[0]
                    mesh_path = os.path.join(self.urdf_dir, mesh_filename)
                    if os.path.isfile(mesh_path):
                        mesh = trimesh.load(mesh_path, process=False)
                        mesh.apply_scale(scale)
                        # TODO: Triemsh output .obj file to the same directory will cause material overwrite issue
                        visual_scale_dir = os.path.join(
                            scale_dir, f"visual_{mesh_base_name}"
                        )
                        if not os.path.exists(visual_scale_dir):
                            os.makedirs(visual_scale_dir)
                        scale_relative_path = os.path.join(
                            "Scale", f"visual_{mesh_base_name}", "scaled.obj"
                        )
                        scaled_mesh_path = os.path.join(
                            self.urdf_dir, scale_relative_path
                        )
                        mesh.export(scaled_mesh_path)
                        mesh_elem.set("filename", scale_relative_path)

            collision_elem = link.find("collision")
            if collision_elem:
                origin_elem = collision_elem.find("origin")
                if origin_elem is not None:
                    xyz = origin_elem.get("xyz", "0 0 0").split()
                    xyz = [str(float(x) * s) for x, s in zip(xyz, scale)]
                    origin_elem.set("xyz", " ".join(xyz))
                geometry_elem = collision_elem.find("geometry")
                if geometry_elem is not None:
                    mesh_elem = geometry_elem.find("mesh")
                    mesh_filename = mesh_elem.get("filename", "")
                    mesh_base_name = os.path.basename(mesh_filename).split(".")[0]
                    mesh_path = os.path.join(self.urdf_dir, mesh_filename)
                    if os.path.isfile(mesh_path):
                        mesh = trimesh.load(mesh_path, process=False)
                        mesh.apply_scale(scale)
                        # TODO: Triemsh output .obj file to the same directory will cause material overwrite issue
                        collision_scale_dir = os.path.join(
                            scale_dir, f"colli_{mesh_base_name}"
                        )
                        if not os.path.exists(collision_scale_dir):
                            os.makedirs(collision_scale_dir)
                        scale_relative_path = os.path.join(
                            "Scale",
                            f"colli_{mesh_base_name}",
                            f"scaled_{mesh_base_name}.obj",
                        )
                        scaled_mesh_path = os.path.join(
                            self.urdf_dir, scale_relative_path
                        )
                        mesh.export(scaled_mesh_path)
                        mesh_elem.set("filename", scale_relative_path)

        for joint in self.root.findall("joint"):
            origin_elem = joint.find("origin")
            if origin_elem is not None:
                xyz = origin_elem.get("xyz", "0 0 0").split()
                xyz = [str(float(x) * s) for x, s in zip(xyz, scale)]
                origin_elem.set("xyz", " ".join(xyz))
            if joint.get("type") == "prismatic":
                limit_elem = joint.find("limit")
                if limit_elem is not None:
                    lower = float(limit_elem.get("lower", "0.0"))
                    upper = float(limit_elem.get("upper", "0.0"))
                    # TODO: can be wrong scale
                    limit_elem.set("lower", str(lower * scale[0]))
                    limit_elem.set("upper", str(upper * scale[0]))

    def save_urdf(self, output_urdf_name: str):
        # save to new urdf file
        output_path = os.path.join(self.urdf_dir, output_urdf_name)
        self.urdf.write(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create and simulate a camera with gizmo in SimulationManager"
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        help="Input urdf file path",
    )
    parser.add_argument(
        "--output_urdf_name",
        type=str,
        default="articulation_acd.urdf",
        help="Output urdf file name, ",
    )
    parser.add_argument(
        "--max_convex_hull_num",
        type=int,
        default=8,
        help="Maximum number of convex hulls for decomposition",
    )
    parser.add_argument(
        "--recompute_inertia",
        default=False,
        action="store_true",
        help="Whether to recompute inertia after convex decomposition",
    )

    parser.add_argument(
        "--scale",
        type=float,
        nargs=3,
        default=None,
        help="Scale the urdf by [scale_x, scale_y, scale_z]",
    )

    args = parser.parse_args()
    generate_urdf_collision_convexes(
        args.urdf_path,
        args.output_urdf_name,
        max_convex_hull_num=args.max_convex_hull_num,
        recompute_inertia=args.recompute_inertia,
        scale=args.scale,
    )
