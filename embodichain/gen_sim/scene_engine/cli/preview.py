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

import argparse
import json
import math
from pathlib import Path
import time
from collections.abc import Sequence
from typing import Any

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import ArticulationCfg, LightCfg, MeshCfg, RigidObjectCfg
from embodichain.lab.visualization import (
    VisualizationCfg,
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def preview_scene_export(
    *,
    output_root: str | Path,
    device: str = "cpu",
    headless: bool = False,
    visualization: VisualizationCfg | None = None,
) -> None:
    """Load ``scene_export/scene_config.json`` and preview its table and assets.

    Args:
        output_root: Scene Engine output root containing ``scene_export/``.
        device: Simulation device, for example ``"cpu"`` or ``"cuda"``.
        headless: Load and validate the scene without an interactive preview.
        visualization: Optional live-visualization configuration.
    """
    resolved_output_root = Path(output_root).expanduser().resolve()
    config_path = resolved_output_root / "scene_export" / "scene_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Scene config not found: {config_path}")

    try:
        scene_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scene config is not valid JSON: {config_path}") from exc
    if not isinstance(scene_config, dict):
        raise ValueError("Scene config must be a JSON object.")
    if scene_config.get("format") != "embodichain.scene-export/v1":
        raise ValueError(
            "Expected an EmbodiChain scene export "
            "(format='embodichain.scene-export/v1')."
        )

    sim = SimulationManager(
        SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=headless,
            physics_dt=1.0 / 100.0,
            sim_device=device,
            visualization=(
                VisualizationCfg() if visualization is None else visualization
            ),
        )
    )
    try:
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        _add_lights(sim)
        _add_objects(
            sim=sim,
            entries=_config_entries(scene_config, "background"),
            config_dir=config_path.parent,
            label="table",
        )
        _add_objects(
            sim=sim,
            entries=_config_entries(scene_config, "rigid_object"),
            config_dir=config_path.parent,
            label="asset",
        )
        _add_articulations(
            sim=sim,
            entries=_config_entries(scene_config, "articulation"),
            config_dir=config_path.parent,
        )

        is_viser = sim.sim_config.visualization.backend == "viser"
        if headless and not is_viser:
            sim.update(step=1)
            print(f"Loaded scene export headlessly: {config_path}")
            return

        if is_viser:
            sim.update(step=1)
            print(f"Previewing in Viser: {config_path}")
        else:
            print(f"Previewing: {config_path}")
            sim.open_window()
        print("Close with Ctrl-C.")
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping preview.")
    finally:
        sim.destroy(exit_process=False)
        SimulationManager.flush_cleanup_queue()


def _config_entries(
    scene_config: dict[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    entries = scene_config.get(field_name, [])
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(
            f"Scene config field {field_name!r} must be a list of objects."
        )
    return entries


def _add_lights(sim: SimulationManager) -> None:
    for index in range(8):
        angle = 2.0 * math.pi * index / 8
        sim.add_light(
            LightCfg(
                uid=f"light_{index + 1}",
                intensity=80.0,
                radius=600,
                init_pos=[5.0 * math.cos(angle), 5.0 * math.sin(angle), 8.0],
            )
        )


def _add_objects(
    *,
    sim: SimulationManager,
    entries: list[dict[str, Any]],
    config_dir: Path,
    label: str,
) -> None:
    """Add exported meshes as static bodies so previewing does not re-simulate them."""
    resolved_config_dir = config_dir.resolve()
    for entry in entries:
        uid = entry.get("uid")
        shape = entry.get("shape")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Scene {label} has no valid uid.")
        if not isinstance(shape, dict) or not isinstance(shape.get("fpath"), str):
            raise ValueError(f"Scene {label} {uid!r} has no shape.fpath.")
        if shape.get("shape_type") != "Mesh":
            raise ValueError(
                f"Scene {label} {uid!r} must use shape_type='Mesh' for preview."
            )

        fpath = Path(shape["fpath"])
        if fpath.is_absolute():
            raise ValueError(
                f"Scene {label} {uid!r} shape.fpath must be a relative path."
            )
        mesh_path = (resolved_config_dir / fpath).resolve()
        if resolved_config_dir not in mesh_path.parents:
            raise ValueError(
                f"Scene {label} {uid!r} shape.fpath must stay within "
                f"{resolved_config_dir}."
            )
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Gym mesh for {uid!r} not found: {mesh_path}")
        init_pos = _vector3(entry.get("init_pos"), field_name=f"{uid}.init_pos")
        init_rot = _vector3(entry.get("init_rot"), field_name=f"{uid}.init_rot")
        body_scale = _vector3(
            entry.get("body_scale", [1.0, 1.0, 1.0]),
            field_name=f"{uid}.body_scale",
        )
        max_convex_hull_num = max(1, int(entry.get("max_convex_hull_num", 32)))

        sim.add_rigid_object(
            RigidObjectCfg(
                uid=uid,
                shape=MeshCfg(fpath=str(mesh_path)),
                # Keep every preview body static: exported poses are already the
                # final gravity-settled poses and should not be simulated again.
                body_type="static",
                init_pos=tuple(init_pos),
                init_rot=tuple(init_rot),
                body_scale=tuple(body_scale),
                max_convex_hull_num=max_convex_hull_num,
                acd_method="vhacd",  # Use vhacd by default.
            )
        )
        print(f"[{label}] {uid}: pos={init_pos} rot={init_rot} scale={body_scale}")


def _add_articulations(
    *,
    sim: SimulationManager,
    entries: list[dict[str, Any]],
    config_dir: Path,
) -> None:
    """Add exported USDC articulations without also loading their GLB proxies."""
    resolved_config_dir = config_dir.resolve()
    for entry in entries:
        uid = entry.get("uid")
        raw_fpath = entry.get("fpath")
        if not isinstance(uid, str) or not uid:
            raise ValueError("Articulation entry has no valid uid.")
        if not isinstance(raw_fpath, str):
            raise ValueError(f"Articulation entry {uid!r} has no fpath.")
        fpath = Path(raw_fpath)
        if fpath.is_absolute() or fpath.suffix.lower() != ".usdc":
            raise ValueError(
                f"Articulation entry {uid!r} fpath must be a relative USDC path."
            )
        usdc_path = (resolved_config_dir / fpath).resolve()
        if resolved_config_dir not in usdc_path.parents:
            raise ValueError(
                f"Articulation entry {uid!r} fpath must stay within "
                f"{resolved_config_dir}."
            )
        if not usdc_path.is_file():
            raise FileNotFoundError(
                f"Articulation USDC for {uid!r} not found: {usdc_path}"
            )
        init_pos = _vector3(entry.get("init_pos"), field_name=f"{uid}.init_pos")
        init_rot = _vector3(entry.get("init_rot"), field_name=f"{uid}.init_rot")
        body_scale = _vector3(
            entry.get("body_scale", [1.0, 1.0, 1.0]),
            field_name=f"{uid}.body_scale",
        )
        if entry.get("fix_base", True) is not True:
            raise ValueError(f"Articulation entry {uid!r} must set fix_base=true.")
        # SimulationManager converts this y-up USDC to z-up with its bottom on XY.
        sim.add_articulation(
            ArticulationCfg(
                uid=uid,
                fpath=str(usdc_path),
                init_pos=tuple(init_pos),
                init_rot=tuple(init_rot),
                body_scale=tuple(body_scale),
                fix_base=True,
                # Generated USDC is not URDF, so it cannot build a PK chain.
                build_pk_chain=False,
            )
        )
        print(f"[articulation] {uid}: pos={init_pos} rot={init_rot} scale={body_scale}")


def _vector3(value: object, *, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Scene config field {field_name!r} must be a length-3 list.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Scene config field {field_name!r} must be numeric.") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="embodichain preview-scene",
        description="Preview a Scene Engine scene export in EmbodiChain simulation.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Scene Engine output root containing scene_export/.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Simulation device, for example cpu or cuda.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Load and validate the exported scene without opening a window.",
    )
    add_viser_args_to_parser(parser)
    args = parser.parse_args(argv)
    preview_scene_export(
        output_root=args.output_root,
        device=args.device,
        headless=args.headless,
        visualization=visualization_cfg_from_args(args),
    )


if __name__ == "__main__":
    main()
