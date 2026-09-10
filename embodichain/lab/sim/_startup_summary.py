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

"""Private, read-only startup snapshots and terminal table formatting."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import dexsim
import torch
from prettytable import PrettyTable, TableStyle

from embodichain import __version__
from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

Row = tuple[str, str, str]


def _switch(value: bool) -> str:
    return "ON" if value else "OFF"


def _selection(requested: str, resolved: str | None) -> str:
    if resolved is None or resolved == "auto":
        return f"{requested} -> PENDING"
    return f"{requested} -> {resolved}" if requested != resolved else resolved


def _solver(sim: SimulationManager) -> str:
    if sim.physics.name == "default":
        solver = str(sim.physics.solver_type)
        return "Constraint Dynamics" if solver == "TGS" else solver
    return _selection(sim._requested_solver, sim.physics.solver_type)


def simulation_rows(sim: SimulationManager) -> list[Row]:
    """Read engine configuration without preparing or stepping the scene."""
    cfg = sim.sim_config
    physics = cfg.physics_cfg
    render = cfg.render_cfg
    gpu_name = getattr(sim, "_render_device_name", None) or "name unavailable"
    gpu = f"{gpu_name} · GPU {cfg.gpu_id}"
    rows = [
        (
            "Runtime",
            "Versions",
            f"EmbodiChain {__version__} · DexSim {dexsim.__version__}",
        ),
        ("Runtime", "Compute device", str(sim.device)),
        (
            "Runtime",
            "Parallel environments",
            f"{sim.num_envs} · spacing {cfg.arena_space:g} m",
        ),
        ("Rendering", "Renderer", _selection(sim._requested_renderer, render.renderer)),
        ("Rendering", "Graphics API", "Vulkan"),
        ("Rendering", "Render GPU", gpu),
        ("Rendering", "Native window", "OPEN" if sim.is_window_opened else "CLOSED"),
        (
            "Rendering",
            "Browser viewer",
            cfg.visualization.backend if cfg.visualization.backend != "none" else "OFF",
        ),
        ("Rendering", "Viewer resolution", f"{cfg.width} × {cfg.height}"),
        ("Rendering", "Tone mapping", _switch(render.tone_mapping_enabled)),
        (
            "Physics",
            "Backend",
            "Default" if sim.physics.name == "default" else "Newton",
        ),
        ("Physics", "Solver", _solver(sim)),
        ("Physics", "Collision policy", "isolated"),
        (
            "Physics",
            "Physics timestep",
            f"{physics.physics_dt * 1000:g} ms ({1 / physics.physics_dt:g} Hz)",
        ),
        (
            "Physics",
            "Gravity",
            f"[{', '.join(f'{x:g}' for x in physics.gravity)}] m/s²",
        ),
    ]
    if isinstance(physics, NewtonPhysicsCfg):
        rows.extend(
            [
                (
                    "Physics",
                    "Solver substeps",
                    f"{physics.num_substeps} × {physics.physics_dt * 1000 / physics.num_substeps:g} ms",
                ),
                ("Physics", "CUDA Graph", sim.physics.cuda_graph_status.upper()),
                ("Physics", "Gradients", _switch(physics.requires_grad)),
            ]
        )
    if cfg.startup_summary == "full":
        rows.extend(
            [
                (
                    "Rendering detail",
                    "Samples per frame",
                    f"{render.spp} (configured; renderer-dependent)",
                ),
                ("Rendering detail", "Denoiser", "ON (configured; renderer-dependent)"),
                (
                    "Rendering detail",
                    "Exposure",
                    f"{render.tone_mapping_exposure:g}"
                    + (" (inactive)" if not render.tone_mapping_enabled else ""),
                ),
            ]
        )
        if isinstance(physics, DefaultPhysicsCfg):
            rows.extend(
                [
                    (
                        "Physics detail",
                        "Bounce threshold",
                        f"{physics.bounce_threshold:g} m/s",
                    ),
                    (
                        "Physics detail",
                        "Tolerance scale",
                        f"length {physics.length_tolerance:g} m · speed {physics.speed_tolerance:g} m/s",
                    ),
                ]
            )
            if sim.device.type == "cuda":
                rows.extend(
                    [
                        (
                            "Physics detail",
                            "Contact capacity",
                            str(physics.gpu_memory.max_rigid_contact_count),
                        ),
                        (
                            "Physics detail",
                            "Patch capacity",
                            str(physics.gpu_memory.max_rigid_patch_count),
                        ),
                        (
                            "Physics detail",
                            "Heap capacity",
                            f"{physics.gpu_memory.heap_capacity / 2**20:g} MiB",
                        ),
                    ]
                )
        elif physics.collision_cfg is not None:
            collision = physics.collision_cfg
            rows.extend(
                [
                    (
                        "Physics detail",
                        "Broad phase",
                        str(
                            collision.broad_phase
                            or physics.broad_phase
                            or "backend default"
                        ),
                    ),
                    (
                        "Physics detail",
                        "Collision update",
                        f"every {collision.update_interval or physics.num_substeps} solver substep(s)",
                    ),
                    (
                        "Physics detail",
                        "Contact capacity",
                        str(collision.rigid_contact_max or "scene-derived"),
                    ),
                ]
            )
        if isinstance(physics, NewtonPhysicsCfg):
            solver = physics.solver_cfg
            if isinstance(solver, Mapping):
                rows.extend(
                    ("Solver config", str(key), str(value))
                    for key, value in solver.items()
                )
        rows.extend(
            [
                (
                    "System detail",
                    "Python / PyTorch",
                    f"{sys.version.split()[0]} / {torch.__version__}",
                ),
                (
                    "System detail",
                    "CUDA runtime",
                    str(torch.version.cuda or "unavailable"),
                ),
                (
                    "System detail",
                    "Cache",
                    str(getattr(sim, "_sim_cache_dir", "not initialized")),
                ),
            ]
        )
    return rows


def scene_is_ready(sim: SimulationManager) -> bool:
    """Require successful preparation of the current, unchanged topology."""
    result = sim.spawn_result
    if result is None or result.needs_rebuild:
        return False
    scene = getattr(sim, "_spawn_scene", None)
    if scene is not None and scene.builder.has_pending_changes:
        return False
    return (
        getattr(sim, "_ready_spawn_topology_revision", -1) == result.topology_revision
    )


def scene_rows(sim: SimulationManager) -> list[Row]:
    """Read the current scene snapshot, with counts per replicated environment."""
    if not scene_is_ready(sim):
        return [("Scene", "State", "PENDING (not prepared)")]
    rows = []
    if sim.physics.name == "newton":
        rows.extend(
            [
                ("Physics resolved", "Solver", _solver(sim)),
                (
                    "Physics resolved",
                    "CUDA Graph",
                    sim.physics.cuda_graph_status.upper(),
                ),
            ]
        )
    rows.extend(
        [
            ("Scene / env", "Robots", str(len(sim._robots))),
            ("Scene / env", "Articulations", str(len(sim._articulations))),
            ("Scene / env", "Rigid objects", str(len(sim._rigid_objects))),
            ("Scene / env", "Object groups", str(len(sim._rigid_object_groups))),
            ("Scene / env", "Deformables", str(len(sim._deformable_objects))),
            ("Scene / env", "Sensors", str(len(sim._sensors))),
            (
                "Scene",
                "Default ground",
                "GLOBAL" if sim._default_plane is not None else "NONE",
            ),
            ("Scene", "Native window", "OPEN" if sim.is_window_opened else "CLOSED"),
        ]
    )
    for uid, sensor in sim._sensors.items():
        sensor_cfg = sensor.cfg
        description = type(sensor).__name__
        if hasattr(sensor_cfg, "width"):
            description += f" · {sensor_cfg.width} × {sensor_cfg.height}"
            description += " · " + "/".join(sensor_cfg.get_data_types())
        rows.append(("Sensors / env", uid, description))
    rows.append(("Scene", "State", "READY"))
    return rows


def format_summary(
    title: str,
    rows: Sequence[Row],
    *,
    color: bool | None = None,
    width: int | None = None,
) -> str:
    """Render one bounded-width table; ANSI styling never changes cell layout."""
    if color is None:
        color = "NO_COLOR" not in os.environ and sys.stderr.isatty()
    width = max(64, min(width or shutil.get_terminal_size((100, 24)).columns, 120))
    table = PrettyTable(["Section", "Setting", "Value"])
    table.set_style(TableStyle.SINGLE_BORDER)
    table.align = "l"
    table.title = f"EmbodiChain · {title}"
    table.max_width = {"Section": 16, "Setting": 22, "Value": width - 48}
    previous = None
    for index, (section, key, value) in enumerate(rows):
        table.add_row(
            [section if section != previous else "", key, value],
            divider=index + 1 < len(rows) and rows[index + 1][0] != section,
        )
        previous = section
    lines = table.get_string().splitlines()
    lines[0] = "╭" + lines[0][1:-1] + "╮"
    lines[-1] = "╰" + lines[-1][1:-1] + "╯"
    if color:
        for index, line in enumerate(lines):
            cells = line.split("│")
            if len(cells) == 5:
                if cells[1].strip():
                    cells[1] = f"\033[1;36m{cells[1]}\033[0m"
                value = cells[3]
                code = (
                    "1;33"
                    if "PENDING" in value
                    else (
                        "1;32"
                        if value.strip() in {"READY", "ON", "CAPTURED", "OPEN"}
                        else "1;37"
                    )
                )
                cells[3] = f"\033[{code}m{value}\033[0m"
                lines[index] = "│".join(cells)
            elif "EmbodiChain ·" in line:
                lines[index] = f"\033[1;36m{line}\033[0m"
    return "\n".join(lines)
