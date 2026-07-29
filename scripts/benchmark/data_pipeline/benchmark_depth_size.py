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

"""Benchmark depth storage size: numeric Parquet vs compressed sidecar video.

Compares the on-disk size of camera depth saved the three ways supported by
``LeRobotRecorder`` (issue #424):

- numeric float32 Parquet (snappy / zstd) -- the PR #422 default; Camera depth
  is float32 metres, and the noisy float32 mantissa is barely compressible.
- numeric uint16-millimetre Parquet (zstd) -- exact raw values, half the bytes.
- compressed ``gray12le``/HEVC sidecar video (lossless and lossy CRF) -- issue
  #424 Path A; depth is quantized to 12-bit codes and encoded as a video.

The depth scene is synthetic but realistic: a smooth background plane, a slowly
moving foreground object, and per-frame sensor noise, so the video codec's
spatial/temporal compression is representative of real depth.

.. note::
   HEVC encoding of long episodes is CPU-bound; the defaults (100 frames) finish
   in a few seconds. Pass ``--frames`` for longer episodes.

Run: python -m scripts.benchmark.data_pipeline.benchmark_depth_size
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embodichain.data_pipeline.depth_video import DepthVideoCfg, DepthVideoWriter


def make_depth_episode(n_frames: int, H: int, W: int, seed: int = 0) -> np.ndarray:
    """Build a realistic float32 depth episode in metres.

    Background is a smooth plane (0.5..2.0 m), a foreground box (0.35 m) moves
    horizontally, and 5 mm Gaussian sensor noise is added per frame.

    Args:
        n_frames: Number of frames.
        H: Frame height.
        W: Frame width.
        seed: RNG seed.

    Returns:
        Array of shape ``(n_frames, H, W)``, dtype ``float32``.
    """
    rng = np.random.default_rng(seed)
    yy, _ = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
    base = (0.5 + 1.5 * yy).astype(np.float32)
    frames = np.empty((n_frames, H, W), dtype=np.float32)
    for t in range(n_frames):
        f = base.copy()
        cx = int(W * 0.5 + 0.15 * W * np.sin(t * 0.05))
        cy = H // 2
        f[cy - H // 6 : cy + H // 6, cx - W // 8 : cx + W // 8] = 0.35
        f += rng.normal(0, 0.005, f.shape).astype(np.float32)
        frames[t] = f
    return frames


def parquet_size(frames: np.ndarray, dtype: str, compression: str) -> int:
    """Write ``frames`` as one Parquet column of fixed-size lists; return bytes."""
    H, W = frames.shape[1], frames.shape[2]
    if dtype == "float32":
        vals = pa.array(frames.ravel().astype(np.float32))
    else:  # uint16 millimetres
        mm = np.rint(frames * 1000.0).clip(0, 65535).astype(np.uint16)
        vals = pa.array(mm.ravel())
    fsl = pa.FixedSizeListArray.from_arrays(vals, H * W)
    table = pa.table({"depth": fsl})
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    pq.write_table(table, path, compression=compression)
    size = Path(path).stat().st_size
    Path(path).unlink()
    return size


def sidecar_size(frames: np.ndarray, lossless: bool, crf: int = 28) -> int:
    """Encode ``frames`` (metres) to a gray12le/HEVC sidecar MP4; return bytes."""
    tmp = Path(tempfile.mkdtemp())
    cfg = DepthVideoCfg(
        enable=True,
        depth_min=0.1,
        depth_max=3.0,
        shift=1.0,
        use_log=True,
        lossless=lossless,
        crf=crf,
        input_unit="m",
        output_unit="m",
    )
    writer = DepthVideoWriter(tmp / "ep.mp4", fps=30, cfg=cfg)
    for f in frames:
        writer.add_frame(f)
    out = writer.close()
    return out.stat().st_size


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def run(configs: Sequence[tuple[int, int, int]]) -> None:
    """Run the benchmark over ``(H, W, n_frames)`` configs and print results."""
    print(
        f"{'config (HxW @30fps)':<22} {'raw f32':<10} {'f32 snappy':<12} "
        f"{'f32 zstd':<12} {'u16 zstd':<12} {'sidecar lossless':<20} {'sidecar CRF28':<14}"
    )
    print("-" * 104)
    for H, W, N in configs:
        frames = make_depth_episode(N, H, W)
        raw = N * H * W * 4  # float32
        p_f32_snappy = parquet_size(frames, "float32", "snappy")
        p_f32_zstd = parquet_size(frames, "float32", "zstd")
        p_u16_zstd = parquet_size(frames, "uint16", "zstd")
        s_lossless = sidecar_size(frames, lossless=True)
        s_crf28 = sidecar_size(frames, lossless=False, crf=28)
        print(
            f"{H}x{W} x{N}f{'':<9} {_human(raw):<10} {_human(p_f32_snappy):<12} "
            f"{_human(p_f32_zstd):<12} {_human(p_u16_zstd):<12} "
            f"{_human(s_lossless):<20} {_human(s_crf28):<14}"
        )

    print()
    print("Compression ratio vs raw float32 (higher = smaller):")
    print(
        f"{'config':<22} {'f32 snappy':<12} {'f32 zstd':<12} {'u16 zstd':<12} {'lossless':<12} {'CRF28':<10}"
    )
    print("-" * 80)
    for H, W, N in configs:
        frames = make_depth_episode(N, H, W)
        raw = N * H * W * 4

        def ratio(sz: int) -> str:
            return f"{raw / sz:.0f}x"

        print(
            f"{H}x{W} x{N}f{'':<9} {ratio(parquet_size(frames, 'float32', 'snappy')):<12} "
            f"{ratio(parquet_size(frames, 'float32', 'zstd')):<12} "
            f"{ratio(parquet_size(frames, 'uint16', 'zstd')):<12} "
            f"{ratio(sidecar_size(frames, True)):<12} "
            f"{ratio(sidecar_size(frames, False, 28)):<10}"
        )


def main() -> None:
    """Parse args and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames", type=int, default=100, help="Frames per config (default 100)."
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="480x640,1280x720",
        help="Comma-separated HxW resolutions (default '480x640,1280x720').",
    )
    args = parser.parse_args()

    configs: list[tuple[int, int, int]] = []
    for res in args.resolutions.split(","):
        H, W = (int(x) for x in res.split("x"))
        configs.append((H, W, args.frames))
    run(configs)


if __name__ == "__main__":
    main()
