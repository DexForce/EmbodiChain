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

"""Download pinned YAM assets and reproduce the cable demo's derived robot.

Run with Python dependencies coacd==1.0.14, trimesh, numpy and scipy. Original
MIT assets are fetched from a fixed robosimstudio commit. Generated files are
verified against the robot used to qualify RouteTheCable before installation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

__all__ = ["main"]

_REVISION = "73739abe212f963e8a9f5862b27bca370df091a7"
_REPOSITORY = "https://github.com/robosimstudio/robosimstudio"
_SOURCE_PATH = "assets/embodiments/yam_i2rt"
_BASE_URL = f"https://raw.githubusercontent.com/robosimstudio/robosimstudio/{_REVISION}/{_SOURCE_PATH}"
_COACD_VERSION = "1.0.14"
_THRESHOLD = 0.06
_SEED = 73
_ASSET_NOTICE = """YAM i2rt robot assets
Copyright (c) I2RT Robotics

The YAM arm and linear_4310 gripper models are distributed under the MIT
license; retain the accompanying LICENSE when copying or distributing them.
Upstream: https://github.com/i2rt-robotics/i2rt

The source assembly is from RoboSimStudio, commit
73739abe212f963e8a9f5862b27bca370df091a7, whose NOTICE identifies these models
as I2RT Robotics assets under MIT.
Source: https://github.com/robosimstudio/robosimstudio

EmbodiChain converts the MJCF to URDF, adds named grasp TCP frames and
decomposes the fingertip collision meshes. The original model copyright and
MIT license are retained; conversion details are recorded in PROVENANCE.json.
"""
_SOURCE_FILES = (
    "LICENSE",
    "yam_i2rt_arms.xml",
    "assets/base.stl",
    "assets/link1.stl",
    "assets/link2.stl",
    "assets/link3.stl",
    "assets/link4.stl",
    "assets/link5.stl",
    "assets/gripper.stl",
    "assets/tip_left.stl",
    "assets/tip_right.stl",
)
_EXPECTED_SHA256 = {
    "LICENSE": "7cd0a0224a287ebd6c45703eb6623d264c7810f8958d65de8f99e87ccd9f4891",
    "PROVENANCE.json": "1f47f75c9591b9dedc51181eb891d97fcf0f68cd1480ae2e6800d4610ac32929",
    "assets/base.stl": "7d57efce4cd31db5e461e1bc83362943a7dbe0e8e101cdb94c29de0a72f3ff0f",
    "assets/gripper.stl": "d949c401186197d842ca8f9b23bfcbf24bed8461b421df06093c65e4ead21f5d",
    "assets/link1.stl": "2283bab1300cfdd1a04aaf09503093d7d6d19bf2144abf3429d563b596920903",
    "assets/link2.stl": "4fd4df2916be71391c6c6ea6ffbdb3c270ce67c71c2bb7d42d44736c0be2af98",
    "assets/link3.stl": "5b51658509771514738d0f2271df5f271cd8ae0b6a4bf3787d8463d5ab582593",
    "assets/link4.stl": "f6e980b8064a1396b17e87065de50286abf616f94b55e1ccff92e889218de220",
    "assets/link5.stl": "b32ece2e963cd028d68290a904223826070be52ce8f0988e3ad36bf75be3582f",
    "assets/tip_left.stl": "03d94f53d7b5e9e1871ea6f3925c68d7a2f894b08e0c575fa0b9ed4024684ff7",
    "assets/tip_left_collision_0.obj": "be722957fdb8eaa614c3625689373f83bf966d93250c8727acf0709d65a70941",
    "assets/tip_left_collision_1.obj": "4873ebf9132048f7461dff3dd3b637ff8594faba717de2f8ee063517c411588d",
    "assets/tip_left_collision_10.obj": "03c8c6f9478a1f7e0b0168f28381707f280984243e4b1df0185161f551bfeec4",
    "assets/tip_left_collision_11.obj": "b24c2b17a45d1a673d9ff754245ef9b451c0d76394a9f98fb8e4b69292ea3b32",
    "assets/tip_left_collision_12.obj": "b6e5dd11b5fd7718b3f5bddbd100b24cb9966b2e4ea5e4f45ba35fa9774238e5",
    "assets/tip_left_collision_13.obj": "dc74139cdb72147eda6ea24d665cdf9a072117b69f25ba911ca2d9fe17ae6559",
    "assets/tip_left_collision_14.obj": "9a7de3929f7f94a7bff42eaa8ab1c307108495a867a68865c7ed9fe8617aa75b",
    "assets/tip_left_collision_15.obj": "02d180b7e86e5c05d405b1e686e1d75f99e971336b873a63494220f0133220fd",
    "assets/tip_left_collision_16.obj": "97282f8af188b6734c4bc1735e89be10ee249c3ba486b454df10b79834befb7a",
    "assets/tip_left_collision_17.obj": "100eb36dca3b7012f19d3d3e328636a0e3d3683c3aa8e581fcffe9294d7b5d9f",
    "assets/tip_left_collision_18.obj": "a7754229194f4db7b7222c57e30a9b26f337b398dfec8ba2b2233de059c68eff",
    "assets/tip_left_collision_19.obj": "3595435ab290e6595c08c48c34b9b44a72d61a268141b6e07cc42ee0fd9306a6",
    "assets/tip_left_collision_2.obj": "6645911ab4afa24d375c1cde9cf6b086a42330a4049b3f0295c2b0cfd82c795d",
    "assets/tip_left_collision_20.obj": "6afee55dda23c4864a73f9774952a8a0a292cc76130290ddaf480fe9ea2f42a0",
    "assets/tip_left_collision_21.obj": "6889beacac5136b6a1aae0c284406017918d077faddc737dd76875df65bc200b",
    "assets/tip_left_collision_22.obj": "54a42e94d4fc691846934c37026532479013934621759c66d45efbc1e2c199f7",
    "assets/tip_left_collision_23.obj": "8f3c8d7eba37b3126426691bc1310a7ad8f1bef4bf95d6211844a1a342c112f9",
    "assets/tip_left_collision_3.obj": "2c25f7c192d9c23067057d79c79ac53ef6d122d7067936dc3952b31d3ba371e1",
    "assets/tip_left_collision_4.obj": "26dce0f26216956480b9ff956955780a5f06efe430f5824ffad0730380ba692e",
    "assets/tip_left_collision_5.obj": "58acf4da81482604641685be978bcb39b6128b93a8f6d303bb42493397c06ba2",
    "assets/tip_left_collision_6.obj": "4d5a185ceef7bcafb4316f10918b6191a407a0cc97abcbb152ab8333b09468a3",
    "assets/tip_left_collision_7.obj": "3330109b6b8f025dce4716b2754e2d371cbe2da93be3cbacbb7ee7c4b6b0754f",
    "assets/tip_left_collision_8.obj": "e3f11a7e79ef7a54aff3b55b079162189ce8a62a43c40bc9e6f44a011f365a78",
    "assets/tip_left_collision_9.obj": "79d8d18ebadc20289d32875af9e22f608e173715999e224062a96e931e451823",
    "assets/tip_right.stl": "8f620aae9c15e00e2ea4e5bf5e9a49e7edb2d23128b8955cfacc0da8da45c72f",
    "assets/tip_right_collision_0.obj": "9f721ff8865f1e2c9f2b99cf815574b133595b7473c98f4e25c85c096d73fda1",
    "assets/tip_right_collision_1.obj": "334dc46e059117926cd3f9e42f70d9b3dbf024d4cbe9ca79794fe5e050ae3151",
    "assets/tip_right_collision_10.obj": "e7f2f9f228ca4dc0f004c9f7a0877ee57ebcd53d7b9c8a5dd2dd2a11e4769dd0",
    "assets/tip_right_collision_11.obj": "fb23393715b9ff7478b9faa298b85c21b177d011eac5677e328c786c162542f6",
    "assets/tip_right_collision_12.obj": "5709a388d7de54128ffeeffaa5ee554f42439167b7f30cbc4fcf8695234c046e",
    "assets/tip_right_collision_13.obj": "159d95ed91cec4b1d009ddd44306d7fd07e15c10a1a45ba54c577b29ecba4c31",
    "assets/tip_right_collision_14.obj": "a231b02f5ea416acd34f55d154202dec76b45a8a50ed3e7cb4f6848ceaeed822",
    "assets/tip_right_collision_2.obj": "00c5ff5d494b00e64b3b432164d6be0e9704eae58bf207be67886dd099e607ab",
    "assets/tip_right_collision_3.obj": "bfd94c95d04eefe319ab9c41af0b3753886fa587b2530a571c459aad8534db8d",
    "assets/tip_right_collision_4.obj": "e841efda880497b5da4717baf57edc14de282213c881a7eb2e179cc014feee34",
    "assets/tip_right_collision_5.obj": "8fe376e86226e5fcef11c7df975d44333f51333c8e46db52924f0319dc2f862a",
    "assets/tip_right_collision_6.obj": "6ecd7c8e68b21d545fb629eebb5f82f1fa7beb8c8469488f7d9c835e77562cb0",
    "assets/tip_right_collision_7.obj": "908eccc7be56d9c066d6386a519e22cc4d594d2f5616b416d179b7bc9e99ada7",
    "assets/tip_right_collision_8.obj": "e07df9b12a6aa99da66125a101ec7863f2fb9d4de61095b430719fb6aaabfe67",
    "assets/tip_right_collision_9.obj": "8bbaae45405779bd7733b764378c219eb7e1a24794fcc3f5d1a054a19593b3b5",
    "yam_i2rt.urdf": "ea4582367a942e6d1d4c5d81cc3a69f8c0aefa55bb924726c87cfca7ff6c239b",
    "yam_i2rt_arms.xml": "6146bec6973fb9b450c233b9297c76d0951182aee4883d673573127ea951bf4c",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_file(path: Path, relative: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing regular asset file: {path}")
    actual = _sha256(path)
    if actual != _EXPECTED_SHA256[relative]:
        raise ValueError(
            f"SHA-256 mismatch for {relative}: {actual}; "
            f"expected {_EXPECTED_SHA256[relative]}"
        )


def _verify_bundle(root: Path) -> None:
    for relative in _EXPECTED_SHA256:
        _verify_file(root / relative, relative)
    for mesh in ET.parse(root / "yam_i2rt.urdf").iter("mesh"):
        relative = mesh.attrib["filename"]
        if relative not in _EXPECTED_SHA256:
            raise ValueError(f"Unverified URDF mesh reference: {relative}")


def _fetch_sources(destination: Path, source_dir: Path | None) -> None:
    for relative in _SOURCE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source_dir is not None:
            source = source_dir / relative
            _verify_file(source, relative)
            shutil.copyfile(source, target)
        else:
            request = urllib.request.Request(
                f"{_BASE_URL}/{relative}",
                headers={"User-Agent": "EmbodiChain-YAM-asset-preparation"},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                with target.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
        _verify_file(target, relative)
        print(f"Verified source: {relative}", flush=True)


def _generate_robot(root: Path) -> None:
    import coacd
    import numpy as np
    from scipy.spatial.transform import Rotation
    import trimesh

    version = importlib.metadata.version("coacd")
    if version != _COACD_VERSION:
        raise ValueError(
            f"Install coacd=={_COACD_VERSION} to reproduce the qualified "
            f"collision meshes; found {version}."
        )
    xml = ET.parse(root / "yam_i2rt_arms.xml").getroot()
    urdf = ET.Element("robot", name="yam_i2rt")
    ET.SubElement(urdf, "link", name="base_link")
    meshes = {mesh.attrib["name"]: mesh.attrib["file"] for mesh in xml.find("asset")}
    parts = {}
    for name, expected_count in (("tip_left", 24), ("tip_right", 15)):
        mesh = trimesh.load_mesh(root / "assets" / meshes[name])
        coacd.set_log_level("error")
        hulls = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
            threshold=_THRESHOLD,
            seed=_SEED,
        )
        if len(hulls) != expected_count:
            raise ValueError(
                f"{name}: expected {expected_count} collision hulls, got {len(hulls)}."
            )
        paths = []
        for index, (vertices, faces) in enumerate(hulls):
            path = root / "assets" / f"{name}_collision_{index}.obj"
            trimesh.Trimesh(vertices, faces).export(path)
            relative = f"assets/{path.name}"
            _verify_file(path, relative)
            paths.append(relative)
        parts[name] = paths
        print(f"Generated and verified {name}: {len(paths)} hulls", flush=True)

    def values(text: str | None, default: str) -> np.ndarray:
        return np.array([float(value) for value in (text or default).split()])

    def rotation(text: str | None) -> Rotation:
        return Rotation.from_quat(values(text, "1 0 0 0")[[1, 2, 3, 0]])

    def origin(parent: ET.Element, pos: str | None, quat: str | None = None) -> None:
        ET.SubElement(
            parent,
            "origin",
            xyz=" ".join(map(str, values(pos, "0 0 0"))),
            rpy=" ".join(map(str, rotation(quat).as_euler("xyz"))),
        )

    def body(element: ET.Element, parent: str) -> None:
        name = element.attrib["name"]
        link = ET.SubElement(urdf, "link", name=name)
        inertial = element.find("inertial")
        if inertial is not None:
            entry = ET.SubElement(link, "inertial")
            origin(entry, inertial.get("pos"))
            ET.SubElement(entry, "mass", value=inertial.attrib["mass"])
            matrix = rotation(inertial.get("quat")).as_matrix()
            inertia = (
                matrix
                @ np.diag(values(inertial.get("diaginertia"), "1e-6 1e-6 1e-6"))
                @ matrix.T
            )
            ET.SubElement(
                entry,
                "inertia",
                **{
                    key: str(inertia[row, col])
                    for key, row, col in (
                        ("ixx", 0, 0),
                        ("iyy", 1, 1),
                        ("izz", 2, 2),
                        ("ixy", 0, 1),
                        ("ixz", 0, 2),
                        ("iyz", 1, 2),
                    )
                },
            )
        source_joint = element.find("joint")
        joint = ET.SubElement(
            urdf,
            "joint",
            name=(
                source_joint.attrib["name"]
                if source_joint is not None
                else name + "_fixed"
            ),
            type=(
                ("prismatic" if source_joint.get("type") == "slide" else "revolute")
                if source_joint is not None
                else "fixed"
            ),
        )
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
        origin(joint, element.get("pos"), element.get("quat"))
        if source_joint is not None:
            if np.linalg.norm(values(source_joint.get("pos"), "0 0 0")) > 1e-9:
                raise ValueError("Offset MJCF joint positions are unsupported.")
            ET.SubElement(joint, "axis", xyz=source_joint.get("axis", "0 0 1"))
            lower, upper = values(source_joint.get("range"), "-3.14 3.14")
            ET.SubElement(
                joint,
                "limit",
                lower=str(lower),
                upper=str(upper),
                effort=str(
                    max(abs(values(source_joint.get("actuatorfrcrange"), "-80 80")))
                ),
                velocity="3",
            )
            ET.SubElement(
                joint,
                "dynamics",
                damping=source_joint.get("damping", "0"),
                friction=source_joint.get("frictionloss", "0"),
            )
        for geometry in element.findall("geom"):
            mesh_name = geometry.get("mesh")
            if not mesh_name:
                continue
            for purpose, paths in (
                ("visual", ["assets/" + meshes[mesh_name]]),
                ("collision", parts.get(mesh_name, ["assets/" + meshes[mesh_name]])),
            ):
                if purpose == "collision" and geometry.get("contype") == "0":
                    continue
                for path in paths:
                    entry = ET.SubElement(link, purpose)
                    origin(entry, geometry.get("pos"), geometry.get("quat"))
                    shape = ET.SubElement(entry, "geometry")
                    ET.SubElement(shape, "mesh", filename=path)
        for site in element.findall("site"):
            if site.get("name", "").endswith("grasp_site"):
                tcp = site.attrib["name"].replace("grasp_site", "tcp")
                ET.SubElement(urdf, "link", name=tcp)
                fixed = ET.SubElement(urdf, "joint", name=tcp + "_fixed", type="fixed")
                ET.SubElement(fixed, "parent", link=name)
                ET.SubElement(fixed, "child", link=tcp)
                origin(fixed, site.get("pos"), site.get("quat"))
        for child in element.findall("body"):
            body(child, name)

    for element in xml.find("worldbody").findall("body"):
        body(element, "base_link")
    ET.indent(urdf)
    ET.ElementTree(urdf).write(
        root / "yam_i2rt.urdf", encoding="unicode", xml_declaration=True
    )
    provenance = {
        "repository": _REPOSITORY,
        "commit": _REVISION,
        "source": f"{_SOURCE_PATH}/yam_i2rt_arms.xml",
        "license": "MIT; see LICENSE",
        "conversion": "MJCF local body/joint/geometry frames and rotated diagonal inertias to URDF; named TCP fixed links; fingertip convex decomposition",
        "coacd_version": _COACD_VERSION,
        "coacd_threshold": _THRESHOLD,
        "coacd_seed": _SEED,
        "fingertip_hulls": {name: len(paths) for name, paths in parts.items()},
        "source_sha256": _EXPECTED_SHA256["yam_i2rt_arms.xml"],
    }
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


def _prepare(output: Path, source_dir: Path | None) -> Path:
    if output.exists():
        _verify_bundle(output)
        print(f"Verified existing robot: {output / 'yam_i2rt.urdf'}", flush=True)
        return output / "yam_i2rt.urdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".yam-i2rt-", dir=output.parent) as folder:
        staged = Path(folder) / "YamI2rt"
        staged.mkdir()
        _fetch_sources(staged, source_dir)
        _generate_robot(staged)
        _verify_bundle(staged)
        staged.rename(output)
    print(f"Robot ready: {output / 'yam_i2rt.urdf'}", flush=True)
    return output / "yam_i2rt.urdf"


def _archive(root: Path, destination: Path) -> Path:
    """Package verified assets with stable metadata for the data registry."""
    _verify_bundle(root)
    if destination.is_relative_to(root):
        raise ValueError("The archive must be outside the asset directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".yam-archive-", dir=destination.parent
    ) as folder:
        staged = Path(folder) / destination.name
        with zipfile.ZipFile(
            staged, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as stream:
            # Open3D requires directory entries before their children.
            directories = sorted(
                {
                    str(Path(name).parent) + "/"
                    for name in _EXPECTED_SHA256
                    if Path(name).parent != Path(".")
                }
            )
            for name in directories:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (0o40755 << 16) | 0x10
                stream.writestr(info, b"")
            for name in sorted((*_EXPECTED_SHA256, "NOTICE")):
                data = (
                    _ASSET_NOTICE.encode()
                    if name == "NOTICE"
                    else (root / name).read_bytes()
                )
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                stream.writestr(info, data, compresslevel=9)
        if destination.exists():
            if _sha256(destination) != _sha256(staged):
                raise ValueError(
                    f"Refusing to overwrite a different archive: {destination}"
                )
        else:
            staged.rename(destination)
    print(
        f"Archive ready: {destination}; "
        f"MD5={hashlib.md5(destination.read_bytes()).hexdigest()}; "
        f"SHA-256={_sha256(destination)}",
        flush=True,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    """Prepare verified robot assets from the pinned upstream or a local copy.

    Args:
        argv: CLI arguments, or None to read the process arguments.

    Returns:
        Zero after verifying and installing the complete asset bundle.
    """
    data_root = Path(
        os.environ.get("EMBODICHAIN_DATA_ROOT")
        or Path.home() / ".cache/embodichain_data"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=data_root / "YamI2rt",
        help="Asset directory (default: $EMBODICHAIN_DATA_ROOT/YamI2rt, or ~/.cache/embodichain_data/YamI2rt).",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Also produce a deterministic asset ZIP for publication; outside --output.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Offline directory containing the pinned LICENSE, XML and assets/*.stl.",
    )
    args = parser.parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        _prepare(output, args.source_dir)
        if args.archive is not None:
            _archive(output, args.archive.expanduser().resolve())
    except (
        OSError,
        ValueError,
        ImportError,
        importlib.metadata.PackageNotFoundError,
    ) as error:
        parser.exit(1, f"Asset preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
