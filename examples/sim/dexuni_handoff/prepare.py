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

"""Prepare per-object scene assets for the two trained policies."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from pxr import Usd


def split_rp1(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, geo in [
        ("floor", "floor"),
        ("platform", "platform"),
        ("rp1", "base_link"),
        ("crate", "carry_crate"),
    ]:
        s = Usd.Stage.Open(str(src))
        l = s.Flatten()
        stage = Usd.Stage.Open(l)
        r = stage.GetDefaultPrim()
        for obj in stage.GetPrimAtPath("/RP1Carry/Geometry").GetChildren():
            if obj.GetName() != geo:
                stage.RemovePrim(obj.GetPath())
        for prim in list(r.GetChildren()):
            if prim.GetName() in ("Physics",) and name == "rp1":
                continue
            if prim.GetName() not in ("Geometry", "Materials"):
                stage.RemovePrim(prim.GetPath())
        if name != "rp1":
            r.ClearCustomDataByKey("mujoco")
        if stage.GetPrimAtPath("/PhysicsScene"):
            stage.RemovePrim("/PhysicsScene")
        l.Export(str(dst / (name + ".usdc")))


def triangulate_anymal(src: Path, dst: Path) -> None:
    import trimesh
    from xml.etree import ElementTree as ET

    dst.mkdir(parents=True, exist_ok=True)
    rel = Path("anybotics_anymal_c/urdf/anymal.urdf")
    tree = ET.parse(src / rel)
    for m in tree.iter("mesh"):
        orig = m.attrib["filename"].removeprefix("package://")
        default = src / orig
        new = dst / (Path(orig).stem + ".obj")
        if not new.exists():
            trimesh.load(default, force="mesh", process=False).export(new)
        m.set("filename", new.name)
    tree.write(dst / "anymal.urdf")


def main() -> None:
    from embodichain.extensions.models.policies.partymimic import PartyMimicPolicy
    from embodichain.extensions.models.policies.rp1_carry._reference import (
        make_reference,
    )
    from embodichain.extensions.models.policies.rp1_carry._scene import (
        convert_scene,
        write_mjcf,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--converter-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = args.catalog.resolve(strict=True)
    profiles = json.loads(catalog.read_text())
    files = {k: catalog.parent / v for k, v in profiles["rp1"]["files"].items()}
    policy = PartyMimicPolicy.from_onnx(files["policy"])
    reference = make_reference(
        "step-up", files["robot_xml"], policy, video_step=True, rp_hands=True
    )
    mjcf = files["reference_mjcf"]
    mjcf.parent.mkdir(exist_ok=True, parents=True)
    model = write_mjcf(
        files["robot_xml"],
        mjcf,
        "step-up",
        reference,
        policy,
        payload_kg=reference.payload_kg,
        hand_model="rp",
    )
    stage = mjcf.with_suffix(".usdc")
    convert_scene(
        mjcf, model, stage, args.converter_python.absolute(), policy, reference
    )
    split_rp1(stage, args.output / "rp1")
    triangulate_anymal(
        catalog.parent / profiles["anymal-c"]["files"]["resource_root"],
        args.output / "anymal-c",
    )


if __name__ == "__main__":
    main()
