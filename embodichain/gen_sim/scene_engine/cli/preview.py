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
from typing import Any

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import LightCfg, MeshCfg, RigidObjectCfg


def preview_gym_export(
    *,
    output_root: str | Path,
    device: str = "cpu",
    headless: bool = False,
) -> None:
    """Load ``gym_export/gym_config.json`` and preview its table and assets."""
    resolved_output_root = Path(output_root).expanduser().resolve()
    config_path = resolved_output_root / "gym_export" / "gym_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Gym config not found: {config_path}")

    try:
        gym_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gym config is not valid JSON: {config_path}") from exc
    if not isinstance(gym_config, dict):
        raise ValueError("Gym config must be a JSON object.")

    sim = SimulationManager(
        SimulationManagerCfg(
            width=1920,
            height=1080,
            headless=headless,
            physics_dt=1.0 / 100.0,
            sim_device=device,
        )
    )
    try:
        if sim.is_use_gpu_physics:
            sim.init_gpu_physics()
        _add_lights(sim)
        _add_objects(
            sim=sim,
            entries=_config_entries(gym_config, "background"),
            config_dir=config_path.parent,
            label="table",
        )
        _add_objects(
            sim=sim,
            entries=_config_entries(gym_config, "rigid_object"),
            config_dir=config_path.parent,
            label="asset",
        )

        if headless:
            sim.update(step=1)
            print(f"Loaded gym export headlessly: {config_path}")
            return

        print(f"Previewing: {config_path}")
        print("Close with Ctrl-C.")
        sim.open_window()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping preview.")
    finally:
        sim.destroy()


def _config_entries(
    gym_config: dict[str, Any],
    field_name: str,
) -> list[dict[str, Any]]:
    entries = gym_config.get(field_name, [])
    if not isinstance(entries, list) or not all(
        isinstance(entry, dict) for entry in entries
    ):
        raise ValueError(f"Gym config field {field_name!r} must be a list of objects.")
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
    for entry in entries:
        uid = entry.get("uid")
        shape = entry.get("shape")
        if not isinstance(uid, str) or not uid:
            raise ValueError(f"Gym {label} has no valid uid.")
        if not isinstance(shape, dict) or not isinstance(shape.get("fpath"), str):
            raise ValueError(f"Gym {label} {uid!r} has no shape.fpath.")
        if shape.get("shape_type") != "Mesh":
            raise ValueError(
                f"Gym {label} {uid!r} must use shape_type='Mesh' for preview."
            )

        mesh_path = (config_dir / shape["fpath"]).resolve()
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
            )
        )
        print(f"[{label}] {uid}: pos={init_pos} rot={init_rot} scale={body_scale}")


def _vector3(value: object, *, field_name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Gym config field {field_name!r} must be a length-3 list.")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Gym config field {field_name!r} must be numeric.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview a Scene Engine gym export in EmbodiChain simulation."
    )
    parser.add_argument(
        "output_root",
        type=Path,
        help="Scene Engine output root containing gym_export/.",
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
    args = parser.parse_args()
    preview_gym_export(
        output_root=args.output_root,
        device=args.device,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
