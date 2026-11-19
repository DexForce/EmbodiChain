# SPDX-License-Identifier: MIT
# Decoding adapted from gaussian_renderer 0.2.0, core/super_splat_loader.py.
# Copyright (c) 2025 Yufei Jia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Decode DISCOVERSE's public lab3 compressed PLY into native Inria PLY.

This adapter supports this asset's chunk/vertex schema only. It preserves the
downloaded source and performs no geometric cleanup, rescaling or retouching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

__all__ = ["main"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    with args.source.open("rb") as stream:
        elements: list[tuple[str, int, list[tuple[str, str]]]] = []
        while True:
            line = stream.readline().decode("ascii").strip()
            if line == "end_header":
                break
            if line.startswith("format "):
                assert line == "format binary_little_endian 1.0"
            if line.startswith("element "):
                _, name, count = line.split()
                elements.append((name, int(count), []))
            if line.startswith("property "):
                _, kind, name = line.split()
                elements[-1][2].append((name, {"float": "<f4", "uint": "<u4"}[kind]))
        assert [name for name, _, _ in elements] == ["chunk", "vertex"]
        decoded = {
            name: np.fromfile(stream, dtype=np.dtype(fields), count=count)
            for name, count, fields in elements
        }
        assert not stream.read(1)
    chunks, vertices = decoded["chunk"], decoded["vertex"]
    count = len(vertices)
    chunk_index = np.arange(count) // 256
    assert chunk_index[-1] == len(chunks) - 1

    def unpack_111011(packed: np.ndarray, prefix: str) -> np.ndarray:
        unit = np.stack(
            [
                (packed >> 21) / 2047,
                ((packed >> 11) & 1023) / 1023,
                (packed & 2047) / 2047,
            ],
            axis=-1,
        )
        low = np.stack(
            [chunks[f"min_{prefix}{axis}"][chunk_index] for axis in "xyz"], axis=-1
        )
        high = np.stack(
            [chunks[f"max_{prefix}{axis}"][chunk_index] for axis in "xyz"], axis=-1
        )
        result = low + unit * (high - low)
        assert np.all(result >= low - 1e-6) and np.all(result <= high + 1e-6)
        return result.astype(np.float32)

    xyz = unpack_111011(vertices["packed_position"], "")
    log_scales = unpack_111011(vertices["packed_scale"], "scale_")
    packed = vertices["packed_color"]
    color_unit = np.stack(
        [
            (packed >> 24) / 255,
            ((packed >> 16) & 255) / 255,
            ((packed >> 8) & 255) / 255,
        ],
        axis=-1,
    )
    color_low = np.stack(
        [chunks[f"min_{axis}"][chunk_index] for axis in "rgb"], axis=-1
    )
    color_high = np.stack(
        [chunks[f"max_{axis}"][chunk_index] for axis in "rgb"], axis=-1
    )
    dc = (color_low + color_unit * (color_high - color_low) - 0.5) / 0.28209479177387814
    opacity = np.clip((packed & 255).astype(np.float64) / 255, 1e-7, 1 - 1e-7)
    logits = np.log(opacity / (1 - opacity))
    packed = vertices["packed_rotation"]
    largest = packed >> 30
    small = np.stack(
        [((packed >> shift) & 1023) for shift in [20, 10, 0]], axis=-1
    ).astype(np.float32)
    small = (small / 1023 - 0.5) * np.sqrt(2)
    quaternion = np.zeros((count, 4), dtype=np.float32)
    for component in range(4):
        rows = largest == component
        other = [index for index in range(4) if index != component]
        quaternion[np.ix_(rows, other)] = small[rows]
    quaternion[np.arange(count), largest] = np.sqrt(
        np.clip(1 - np.sum(quaternion * quaternion, axis=-1), 0, 1)
    )
    assert np.max(np.abs(np.linalg.norm(quaternion, axis=-1) - 1)) < 1e-6
    fields = "x y z f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split()
    rows = np.column_stack([xyz, dc, logits, log_scales, quaternion]).astype("<f4")
    assert np.isfinite(rows).all()
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {count}\n"
        + "".join(f"property float {field}\n" for field in fields)
        + "end_header\n"
    )
    with args.destination.open("wb") as stream:
        stream.write(header.encode("ascii"))
        rows.tofile(stream)
    report = dict(
        source=args.source.name,
        destination=args.destination.name,
        vertices=count,
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        destination_sha256=hashlib.sha256(args.destination.read_bytes()).hexdigest(),
        adapter="gaussian_renderer 0.2.0 SuperSplat decoding; alpha endpoint epsilon 1e-7",
        xyz_quantiles=np.quantile(xyz, [0.01, 0.1, 0.5, 0.9, 0.99], axis=0).tolist(),
    )
    args.destination.with_suffix(".conversion.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
