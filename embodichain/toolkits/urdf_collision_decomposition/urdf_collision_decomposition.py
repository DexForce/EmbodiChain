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


def generate_urdf_convex_decomposition_collision(
    urdf_path: str, output_urdf_name: str, max_convex_hull_num: int = 16
):
    decomposer = URDFCollisionDecomposer(
        urdf_path, max_convex_hull_num=max_convex_hull_num
    )
    decomposer.decompose_collisions(output_urdf_name)


class URDFCollisionDecomposer:
    def __init__(self, urdf_path: str, **kwargs):
        self.urdf_path = urdf_path
        self.urdf_dir = os.path.dirname(urdf_path)
        self.urdf = ET.parse(urdf_path)
        self.root = self.urdf.getroot()

        self.max_convex_hull_num = kwargs.get("max_convex_hull_num", 16)

    def _get_visual_collision_pairs(self):
        visual_collision_pairs = dict()
        for link in self.root.findall("link"):
            link_name = link.get("name")
            visual = link.find("visual")
            collision = link.find("collision")
            visual_collision_pairs[link_name] = (visual, collision, link)

        return visual_collision_pairs

    def generate_link_collision(
            self, link_name: str, visual: ET.Element | None, collision: ET.Element | None, link: ET.Element
        ):
        if collision is None:
            geom = visual.find("geometry")
            # use visual geometry mesh
        elif visual is not None:
            geom = collision.find("geometry")
            # use collision geometry mesh
            pass
        else:
            logger.log_warning(
                f"Link {link_name} has no visual and collision geometry."
            )
            return

        pass

    def decompose_collisions(self, output_urdf_name: str):
        visual_collision_pairs = self._get_visual_collision_pairs()
        for link_name, (visual, collision, link) in visual_collision_pairs.items():
            if collision is None:
                geom = visual.find("geometry")
                # use visual geometry mesh
            elif visual is not None:
                geom = collision.find("geometry")
                # use collision geometry mesh
                pass
            else:
                logger.log_warning(
                    f"Link {link_name} has no visual and collision geometry."
                )
                continue

            if geom is None:
                logger.log_warning(f"Link {link_name} has no geometry.")
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                logger.log_warning(f"Link {link_name} geometry is not a mesh.")
                continue
            mesh_filename = mesh_elem.get("filename")
            mesh_path = os.path.join(self.urdf_dir, mesh_filename)
            mesh_base_name = os.path.basename(mesh_filename).split(".")[0]
            if not os.path.isfile(mesh_path):
                logger.log_warning(f"Mesh file {mesh_path} does not exist.")
                continue

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
                    collision_origin.set(
                        "xyz", visual_origin.get("xyz", "0 0 0")
                    )
                    collision_origin.set(
                        "rpy", visual_origin.get("rpy", "0 0 0")
                    )
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
        help="Output urdf file name, "
    )
    parser.add_argument(
        "--max_convex_hull_num",
        type=int,
        default=8,
        help="Maximum number of convex hulls for decomposition",
    )

    args = parser.parse_args()
    generate_urdf_convex_decomposition_collision(
        args.urdf_path, args.output_urdf_name,  args.max_convex_hull_num
    )
