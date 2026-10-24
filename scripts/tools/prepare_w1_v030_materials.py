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

"""Prepare a local W1 V030 archive with the V021 rendering materials."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import zipfile

import numpy as np
from PIL import Image

__all__ = ["main"]

_SOURCE_MD5 = "74ff7e835160f7309d32ade116172c52"
# Dominant body colors in V021's baselink.png and arml2.png.
_YELLOW = (248, 195, 73)
_BLACK = (46, 46, 46)
# The V021 Collada materials import with shininess 10, metallic 0.
_ROUGHNESS = math.sqrt(2.0 / (10.0 + 2.0))


def _align_materials(raw: bytes) -> bytes:
    magic, version, length = struct.unpack_from("<4sII", raw)
    if (magic, version, length) != (b"glTF", 2, len(raw)):
        raise ValueError("Expected a glTF 2.0 binary asset.")
    json_length, json_type = struct.unpack_from("<I4s", raw, 12)
    binary_length, binary_type = struct.unpack_from("<I4s", raw, 20 + json_length)
    if (json_type, binary_type) != (b"JSON", b"BIN\0"):
        raise ValueError("Expected embedded JSON and BIN chunks.")
    document = json.loads(raw[20 : 20 + json_length])
    binary = bytearray(raw[28 + json_length :])
    if len(binary) != binary_length:
        raise ValueError("Invalid glTF BIN length.")
    palette: dict[tuple[int, int, int], int] = {}

    def add_texture(image: Image.Image, name: str) -> int:
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        payload = stream.getvalue()
        binary.extend(b"\0" * (-len(binary) % 4))
        view = {
            "buffer": 0,
            "byteOffset": len(binary),
            "byteLength": len(payload),
        }
        binary.extend(payload)
        document["bufferViews"].append(view)
        document.setdefault("images", []).append(
            {
                "name": name,
                "mimeType": "image/png",
                "bufferView": len(document["bufferViews"]) - 1,
            }
        )
        document.setdefault("textures", []).append(
            {"source": len(document["images"]) - 1}
        )
        return len(document["textures"]) - 1

    def solid_texture(color: tuple[int, int, int]) -> int:
        if color not in palette:
            palette[color] = add_texture(
                Image.new("RGB", (1, 1), color), f"V021_{color}"
            )
        return palette[color]

    def rebase_texture(
        index: int, source: tuple[int, int, int], target: tuple[int, int, int]
    ) -> int:
        image = document["images"][document["textures"][index]["source"]]
        view = document["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        payload = binary[start : start + view["byteLength"]]
        with Image.open(io.BytesIO(payload)) as original:
            pixels = np.asarray(original.convert("RGB"), dtype=np.float32)
        # Retain the baked lettering, grille shadows and UV layout.
        pixels = np.rint(np.clip(pixels * (np.array(target) / source), 0, 255))
        return add_texture(
            Image.fromarray(pixels.astype(np.uint8)), "V021_" + image["name"]
        )

    for material in document["materials"]:
        pbr = material["pbrMetallicRoughness"]
        name = material["name"]
        texture = pbr.get("baseColorTexture")
        image_name = (
            document["images"][document["textures"][texture["index"]]["source"]].get(
                "name"
            )
            if texture
            else None
        )
        index = None
        if name.startswith("塑胶 (2)"):
            index = solid_texture(_YELLOW)
        elif image_name in ("black", "black_texture") or name == "塑胶.003":
            index = solid_texture(_BLACK)
        elif name == "New_Shell_Baked.001":
            index = rebase_texture(texture["index"], (193, 135, 15), _YELLOW)
            material.pop("normalTexture", None)
        elif name == "New_Grille_Baked.001":
            index = rebase_texture(texture["index"], (6, 6, 6), _BLACK)
            material.pop("normalTexture", None)
        if index is not None:
            # Both DAE and glTF textures carry sRGB bytes. A white factor avoids
            # differing factor color-space conventions between asset importers.
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            pbr["baseColorTexture"] = {"index": index}
        if not material.get("emissiveFactor"):
            pbr["metallicFactor"] = 0.0
            pbr["roughnessFactor"] = _ROUGHNESS

    document["buffers"][0]["byteLength"] = len(binary)
    encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary.extend(b"\0" * (-len(binary) % 4))
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    return b"".join(
        (
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<I4s", len(encoded), b"JSON"),
            encoded,
            struct.pack("<I4s", len(binary), b"BIN\0"),
            binary,
        )
    )


def main() -> None:
    """Write a separate asset archive and checksum manifest without publishing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original V030 w1.zip.")
    parser.add_argument("output", type=Path, help="New local archive path.")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must be separate archives.")
    if hashlib.md5(args.source.read_bytes()).hexdigest() != _SOURCE_MD5:
        parser.error("Source archive does not match the original V030 revision.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    changed = []
    with (
        zipfile.ZipFile(args.source) as source,
        zipfile.ZipFile(args.output, "w") as output,
    ):
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename.startswith(
                "w1/visual_recoloured/"
            ) and info.filename.endswith(".glb"):
                payload = _align_materials(payload)
                changed.append(info.filename)
            output.writestr(info, payload)
    payload = args.output.read_bytes()
    manifest = {
        "source_md5": _SOURCE_MD5,
        "md5": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "reference": "DexforceW1V021 component DAE materials",
        "yellow_srgb": list(_YELLOW),
        "black_srgb": list(_BLACK),
        "roughness": _ROUGHNESS,
        "metallic": 0.0,
        "changed_meshes": changed,
    }
    args.output.with_suffix(".materials.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
