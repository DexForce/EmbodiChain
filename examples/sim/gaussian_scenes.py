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

"""Download and render the Garden PLY or DexSim's Library USD acceptance asset.

Run from the repository root with a qualified DexSim build::

    PYTHONPATH=. python examples/sim/gaussian_scenes.py --scene garden
    PYTHONPATH=. python examples/sim/gaussian_scenes.py --scene library-usd

Downloads are SHA256 checked and cached under EMBODICHAIN_DATA_ROOT. The USD
preset requires the corporate network. No capture is bundled or relicensed.
On Linux the native offscreen renderer still requires a working DISPLAY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from urllib.request import urlopen
import zipfile

os.environ.setdefault("EMBODICHAIN_SIM_EXIT_PROCESS", "0")

import numpy as np
from PIL import Image

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT
from embodichain.data.dataset import _dataset_download_lock
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import GaussianSplatCfg, RenderCfg, VisualObjectCfg
from embodichain.lab.sim.sensors import CameraCfg
from embodichain.lab.visualization.cfg import VisualizationCfg

__all__ = ["main"]


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _asset(preset: dict, root: Path) -> Path:
    root = root.expanduser().resolve()
    folder = root / preset["cache_dir"]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / preset["download"]
    with _dataset_download_lock(str(root), preset["cache_dir"]):
        if not archive.exists() or _digest(archive) != preset["sha256"]:
            partial = archive.with_suffix(archive.suffix + ".part")
            try:
                print(f"Downloading {preset['url']}", flush=True)
                with (
                    urlopen(preset["url"], timeout=60) as source,
                    partial.open("wb") as dest,
                ):
                    shutil.copyfileobj(source, dest, length=8 * 1024 * 1024)
                if _digest(partial) != preset["sha256"]:
                    raise ValueError("Asset SHA256 mismatch")
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        if "members" in preset:
            with zipfile.ZipFile(archive) as bundle:
                for name, digest in preset["members"].items():
                    target = folder / name
                    target.resolve().relative_to(folder.resolve())
                    if target.exists() and _digest(target) == digest:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(name) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    if _digest(target) != digest:
                        target.unlink()
                        raise ValueError(f"USD dependency SHA256 mismatch: {name}")
    return folder / preset["entry"]


def _json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    """Download one fixed capture and save a managed Camera RGB/depth observation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["garden", "library-usd"], default="garden")
    parser.add_argument(
        "--data-root", type=Path, default=Path(EMBODICHAIN_DEFAULT_DATA_ROOT)
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/gaussian_scenes"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    presets = json.loads(
        (Path(__file__).parent / "assets/gaussian_scenes.json").read_text()
    )
    preset = presets[args.scene]
    asset = _asset(preset, args.data_root)
    print(f"Verified asset: {asset}", flush=True)
    if args.download_only:
        return
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            num_envs=1,
            startup_summary="off",
            enable_entity_gizmo=False,
            visualization=VisualizationCfg(backend="none"),
            render_cfg=RenderCfg.for_gaussian(renderer="hybrid", spp=1),
        )
    )
    try:
        if args.scene == "library-usd":
            sim.add_usd("library", str(asset))
        else:
            sim.add_visual_object(
                VisualObjectCfg(
                    uid="garden",
                    visuals=GaussianSplatCfg(file_path=str(asset)),
                )
            )
        camera = sim.add_sensor(
            CameraCfg(
                uid="view",
                width=960,
                height=540,
                intrinsics=(700, 700, 480, 270),
                ground_plane_visible=False,
                enable_depth=True,
                extrinsics=CameraCfg.ExtrinsicsCfg(
                    eye=tuple(preset["eye"]),
                    target=tuple(preset["target"]),
                    up=tuple(preset["up"]),
                ),
            )
        )
        sim.prepare()
        for _ in range(4):
            sim.render_camera_group([camera.group_id])
            camera.update(fetch_only=True)
        rgb = camera.get_data()["color"][0, ..., :3].cpu().numpy()
        depth = camera.get_data()["depth"][0].cpu().numpy()
        capture = camera.get_capture_metadata()
        if not capture.get("valid") or not capture["status"]["completed"]:
            raise RuntimeError("No completed Gaussian capture")
        if not capture["output_contract"]["same_frame_identity"]:
            raise RuntimeError("RGB/depth capture identity mismatch")
        if capture["status"]["cameras"][0]["selected_nodes"] != preset["points"]:
            raise RuntimeError("Unexpected Gaussian point count")
        if rgb.std() < 1 or not np.any(np.isfinite(depth) & (depth > 0)):
            raise RuntimeError("Capture has no useful RGB/depth content")
        args.output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output / f"{args.scene}.png")
        np.save(args.output / f"{args.scene}-depth.npy", depth)
        report = dict(
            scene=args.scene,
            asset=str(asset),
            preset=preset,
            rgb_mean=rgb.mean(axis=(0, 1)).tolist(),
            rgb_std=float(rgb.std()),
            finite_depth_fraction=float(np.isfinite(depth).mean()),
            capture=capture,
        )
        # Native contracts represent invalid background as +inf; JSON uses null.
        normalized = json.loads(
            json.dumps(report, default=_json_value), parse_constant=lambda _: None
        )
        text = json.dumps(normalized, indent=2, allow_nan=False)
        (args.output / f"{args.scene}.json").write_text(text + "\n")
        print(f"Saved valid capture to {args.output}", flush=True)
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


if __name__ == "__main__":
    main()
