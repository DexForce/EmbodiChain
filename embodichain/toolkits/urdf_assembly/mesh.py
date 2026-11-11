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

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from embodichain.toolkits.urdf_assembly.logging_utils import (
    URDFAssemblyLogger,
)

__all__ = ["URDFMeshManager"]


class URDFMeshManager:
    r"""Responsible for copying, renaming, and handling dependencies of mesh files."""

    def __init__(self, output_dir: str):
        r"""Initialize the URDFMeshManager with output directory configuration.

        Args:
            output_dir (str): Base directory where mesh files will be organized.
                             Creates subdirectories 'Visual' and 'Collision' for
                             different mesh types.
        """
        self.output_dir = output_dir
        self.logger = URDFAssemblyLogger.get_logger("mesh_manager")

    def ensure_dirs(self):
        r"""Ensure that the output directory contains 'Collision' and 'Visual' subdirectories.
        Creates them if they do not exist.

        Returns:
            tuple: Paths to the 'Collision' and 'Visual' directories.
        """
        collision_dir = os.path.join(self.output_dir, "Collision")
        visual_dir = os.path.join(self.output_dir, "Visual")
        os.makedirs(collision_dir, exist_ok=True)
        os.makedirs(visual_dir, exist_ok=True)
        return collision_dir, visual_dir

    def copy_and_modify_mesh_file(
        self, base_urdf_path: str, mesh_file_name: str, sub_folder: str, comp_name: str
    ):
        r"""Copy a mesh file to the output directory and handle dependencies.

        Args:
            base_urdf_path (str): Path to the base URDF file.
            mesh_file_name (str): Name of the mesh file to copy.
            sub_folder (str): 'Visual' or 'Collision'.
            comp_name (str): Component name, e.g. 'chassis', 'left_arm'.

        Returns:
            str: Relative path to the new mesh file for URDF reference.
        """
        # New mesh path format: output_dir/{sub_folder}/{comp_name}/{original_filename}
        target_dir = os.path.join(self.output_dir, sub_folder, comp_name)
        os.makedirs(target_dir, exist_ok=True)

        # Get URDF directory
        urdf_dir = os.path.dirname(base_urdf_path)

        # Handle different path types
        if os.path.isabs(mesh_file_name):
            # Absolute path
            original_mesh_path = mesh_file_name
        else:
            # Relative path - join with URDF directory and normalize
            original_mesh_path = os.path.join(urdf_dir, mesh_file_name)
            original_mesh_path = os.path.normpath(original_mesh_path)

        # Debug information
        self.logger.debug(f"Processing mesh file:")
        self.logger.debug(f"  URDF path: {base_urdf_path}")
        self.logger.debug(f"  URDF dir: {urdf_dir}")
        self.logger.debug(f"  Mesh reference: {mesh_file_name}")
        self.logger.debug(f"  Resolved path: {original_mesh_path}")

        # Check if file exists
        if not os.path.exists(original_mesh_path):
            # Try some common alternative patterns
            alternatives = []

            # Try removing '../' and looking in same directory as URDF
            if mesh_file_name.startswith("../"):
                alt_path = os.path.join(urdf_dir, mesh_file_name[3:])
                alternatives.append(alt_path)

            # Try looking in parent directory structure
            parent_dir = os.path.dirname(urdf_dir)
            if mesh_file_name.startswith("../"):
                alt_path = os.path.join(parent_dir, mesh_file_name[3:])
                alternatives.append(alt_path)
            else:
                alt_path = os.path.join(parent_dir, mesh_file_name)
                alternatives.append(alt_path)

            # Try looking directly in the mesh filename as basename
            basename = os.path.basename(mesh_file_name)
            alt_path = os.path.join(urdf_dir, basename)
            alternatives.append(alt_path)

            # Check alternatives
            found_alternative = None
            for alt in alternatives:
                alt_normalized = os.path.normpath(alt)
                if os.path.exists(alt_normalized):
                    found_alternative = alt_normalized
                    self.logger.debug(
                        f"Found mesh file at alternative location: {alt_normalized}"
                    )
                    break

            if found_alternative:
                original_mesh_path = found_alternative
            else:
                self.logger.error(f"Mesh file not found: {original_mesh_path}")
                self.logger.debug(f"  Tried alternatives: {alternatives}")
                # Return original path to keep existing URDF reference
                return mesh_file_name

        new_mesh_path = os.path.join(target_dir, os.path.basename(mesh_file_name))

        try:
            shutil.copyfile(original_mesh_path, new_mesh_path)
            self.logger.debug(f"Copied mesh: {original_mesh_path} -> {new_mesh_path}")
        except Exception as e:
            self.logger.error(f"Failed to copy mesh file: {e}", exc_info=True)
            return mesh_file_name

        if mesh_file_name.lower().endswith(".obj"):
            self._copy_obj_materials(Path(original_mesh_path), Path(new_mesh_path))

        # Handle DAE's texture dependency
        if mesh_file_name.lower().endswith(".dae"):
            try:
                dae_tree = ET.parse(original_mesh_path)
                dae_root = dae_tree.getroot()
                ns = {}
                if "}" in dae_root.tag:
                    ns["c"] = dae_root.tag.split("}")[0].strip("{")
                    image_tags = dae_root.findall(".//c:image", ns)
                else:
                    image_tags = dae_root.findall(".//image")
                for image in image_tags:
                    init_from = (
                        image.find("c:init_from", ns) if ns else image.find("init_from")
                    )
                    if init_from is not None and init_from.text:
                        tex_filename = os.path.basename(init_from.text)
                        original_tex_path = os.path.join(
                            os.path.dirname(original_mesh_path), tex_filename
                        )
                        if os.path.exists(original_tex_path):
                            new_tex_path = os.path.join(target_dir, tex_filename)
                            shutil.copyfile(original_tex_path, new_tex_path)
            except Exception as e:
                self.logger.warning(
                    f"Failed to parse DAE texture dependency: {e}", exc_info=True
                )

        return os.path.join(sub_folder, comp_name, os.path.basename(mesh_file_name))

    def _copy_obj_materials(self, source: Path, destination: Path) -> None:
        """Copy the material libraries declared by the source OBJ."""
        with source.open(encoding="utf-8", errors="surrogateescape", newline="") as obj:
            lines = obj.readlines()

        changed = False
        for index, line in enumerate(lines):
            fields = line.split(maxsplit=1)
            if len(fields) != 2 or fields[0] != "mtllib":
                continue
            argument = fields[1].strip()
            if (source.parent / argument).is_file():
                references = [argument]
            else:
                try:
                    references = shlex.split(argument, comments=True)
                except ValueError:
                    self.logger.warning(
                        f"Invalid mtllib declaration in {source}: {line.strip()}"
                    )
                    continue

            relocated = []
            for reference in references:
                material = source.parent / reference
                if not material.is_file():
                    self.logger.warning(f"OBJ material library not found: {material}")
                    relocated.append(reference)
                    continue

                relative = Path(os.path.normpath(reference))
                if relative.is_absolute() or relative.parts[0] == "..":
                    # Keep libraries outside the source OBJ's directory inside
                    # this component's output. Distinct paths can share a basename.
                    key = hashlib.sha256(os.fsencode(material.resolve())).hexdigest()[
                        :16
                    ]
                    relative = Path("materials") / f"{key}.mtl"
                    relocated.append(relative.as_posix())
                else:
                    relocated.append(reference)

                target = destination.parent / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(material, target)

            if relocated != references:
                names = " ".join(
                    f'"{name}"' if " " in name else name for name in relocated
                )
                lines[index] = f"mtllib {names}\n"
                changed = True

        if changed:
            with destination.open(
                "w", encoding="utf-8", errors="surrogateescape", newline=""
            ) as obj:
                obj.writelines(lines)
