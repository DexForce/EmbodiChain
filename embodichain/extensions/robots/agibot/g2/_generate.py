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

"""Reproduce the portable URDF from the pinned ZIP; development-only Xacro tool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
from zipfile import ZipFile

__all__ = []


def _main() -> None:
    import xacro  # Install xacro==2.1.1 only when regenerating the resource.

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name("source.lock.json").read_text())
    if hashlib.sha256(args.archive.read_bytes()).hexdigest() != lock["archive_sha256"]:
        raise ValueError("Source archive differs from the pinned G2 revision")
    with TemporaryDirectory() as temporary:
        with ZipFile(args.archive) as archive:
            archive.extractall(temporary)
        source = Path(temporary) / lock["source_directory"]
        # Resolve ROS package lookups in a disposable copy, without ROS/ament.
        for path in (source / "robots/genie/g2").rglob("*.xacro"):
            path.write_text(
                path.read_text().replace("$(find genie_sim_robot_model)", str(source))
            )
        entry = source / lock["entry_point"]
        entry.write_text(
            entry.read_text().replace(
                'collision_type="coarse"', 'collision_type="fine"'
            )
        )
        document = xacro.process_file(str(entry))
        text = document.toxml().replace(
            "package://genie_sim_robot_model/robots/genie/g2/", ""
        )
        robot = ET.fromstring(text)
        for parent in robot.iter():
            for child in list(parent):
                if child.tag in lock["removed_extensions"]:
                    parent.remove(child)
        for name, repair in lock["inertia_repairs"].items():
            inertia = robot.find(f"link[@name='{name}']/inertial/inertia")
            axes = ("ixx", "iyy", "izz")
            if [float(inertia.get(axis)) for axis in axes] != repair["source_diagonal"]:
                raise ValueError(f"Source inertia for {name} differs from the lock")
            for axis, value in zip(axes, repair["diagonal"]):
                inertia.set(axis, str(value))
        ET.indent(robot, space="  ")
        content = (
            '<?xml version="1.0"?>\n'
            "<!-- Derived from AgiBot Genie G2, Apache-2.0. See LICENSE and source.lock.json. -->\n"
            + ET.tostring(robot, encoding="unicode")
            + "\n"
        ).encode()
        if hashlib.sha256(content).hexdigest() != lock["urdf_sha256"]:
            raise ValueError("Generated G2 URDF differs from the locked resource")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(content)


if __name__ == "__main__":
    _main()
