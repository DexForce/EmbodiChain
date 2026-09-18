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

"""Read-only functor details for the environment startup summary."""

from __future__ import annotations

import inspect
import os
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import MISSING
from functools import partial

import numpy as np
import torch
from prettytable import PrettyTable, TableStyle

from .managers.cfg import SceneEntityCfg


def _describe(value: object, depth: int = 0) -> str:
    """Bound configuration output without reading tensor data or arbitrary repr."""
    if value is MISSING:
        return "not set"
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return repr(value[:97] + "..." if len(value) > 100 else value)
    if isinstance(value, (torch.Tensor, np.ndarray)):
        device = f", device={value.device}" if isinstance(value, torch.Tensor) else ""
        return f"{type(value).__name__}(shape={tuple(value.shape)}, dtype={value.dtype}{device})"
    if isinstance(value, SceneEntityCfg):
        fields = [f"uid={_describe(value.uid)}"]
        for prefix in ("joint", "body"):
            selected = getattr(value, f"{prefix}_names")
            if selected is not None:
                fields.append(f"{prefix}_names={_describe(selected, depth + 1)}")
            else:
                indices = getattr(value, f"{prefix}_ids")
                if not (isinstance(indices, slice) and indices == slice(None)):
                    fields.append(f"{prefix}_ids={_describe(indices, depth + 1)}")
        return "SceneEntityCfg(" + ", ".join(fields) + ")"
    if isinstance(value, (Mapping, list, tuple)):
        if depth >= 2 or len(value) > 6:
            return f"{type(value).__name__}(len={len(value)})"
        if isinstance(value, Mapping):
            return (
                "{"
                + ", ".join(
                    f"{_describe(k, depth + 1)}: {_describe(v, depth + 1)}"
                    for k, v in value.items()
                )
                + "}"
            )
        contents = ", ".join(_describe(v, depth + 1) for v in value)
        return f"[{contents}]" if isinstance(value, list) else f"({contents})"
    if isinstance(value, slice):
        return f"slice({value.start}, {value.stop}, {value.step})"
    return type(value).__name__


def _callable_name(func: object, full: bool) -> str:
    if isinstance(func, partial):
        return _callable_name(func.func, full)
    if isinstance(func, str):
        path = func.replace(":", ".")
        return path if full else ".".join(path.split(".")[-2:])
    target = func if inspect.isroutine(func) or inspect.isclass(func) else type(func)
    module = target.__module__ or type(func).__module__
    name = target.__qualname__
    if full:
        return f"{module}.{name}"
    if inspect.isroutine(target):
        return f"{module.rsplit('.', 1)[-1]}.{target.__name__}"
    return target.__name__


def _scene_entity_cfgs(value: object) -> Iterator[SceneEntityCfg]:
    """Yield scene entity configurations nested in a functor parameter value."""
    if isinstance(value, SceneEntityCfg):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _scene_entity_cfgs(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _scene_entity_cfgs(nested)


def _scene_entity_label(value: SceneEntityCfg) -> str:
    """Format a resolved scene entity for the compact functor table."""
    selectors: list[str] = []
    for prefix in ("joint", "body"):
        names = getattr(value, f"{prefix}_names")
        if names is not None:
            selectors.append(f"{prefix}_names={_describe(names)}")
        else:
            ids = getattr(value, f"{prefix}_ids")
            if not (isinstance(ids, slice) and ids == slice(None)):
                selectors.append(f"{prefix}_ids={_describe(ids)}")
    suffix = f" ({', '.join(selectors)})" if selectors else ""
    return f"{value.uid}{suffix}"


def _scene_entity_details(params: Mapping[str, object]) -> str | None:
    """Return a compact summary of scene entities referenced by functor params."""
    entities: list[str] = []
    seen: set[int] = set()
    for value in params.values():
        for entity in _scene_entity_cfgs(value):
            if id(entity) in seen:
                continue
            seen.add(id(entity))
            entities.append(_scene_entity_label(entity))
    if not entities:
        return None
    key = "entity" if len(entities) == 1 else "entities"
    return f"{key}=" + ", ".join(entities)


def _functor_cells(
    manager_name: str, manager: object, mode: str, name: str, full: bool
) -> list[str]:
    details = []
    if manager_name == "ActionManager":
        func = manager.get_term(name)
        cfg = func.cfg
        details.append(f"input={func.input_key} · dim={func.action_dim}")
    else:
        cfg = manager.get_functor_cfg(name)
        func = cfg.func
        if manager_name == "EventManager" and mode == "interval":
            details.append(f"every {cfg.interval_step} control steps")
        elif manager_name == "ObservationManager":
            output = getattr(cfg, "name", MISSING)
            if output is not MISSING:
                details.append(f"output={output}")
        elif manager_name == "RewardManager":
            details.append(f"weight={cfg.weight:g}")
        elif manager_name == "DatasetManager":
            setting = "ON" if manager.save_failed_episodes else "OFF"
            details.append(f"save failed episodes={setting}")
    params = getattr(cfg, "params", {})
    if not full:
        entity_details = _scene_entity_details(params)
        if entity_details is not None:
            details.append(entity_details)
    if full:
        if isinstance(func, partial):
            bound = [f"args={_describe(func.args)}"] if func.args else []
            bound.extend(
                f"{k}={_describe(v)}" for k, v in (func.keywords or {}).items()
            )
            if bound:
                details.append("Bound: " + ", ".join(bound))
        details.append(
            "Params:\n"
            + "\n".join(f"{key}={_describe(value)}" for key, value in params.items())
            if params
            else "Params: {}"
        )
    return [name, _callable_name(func, full), mode, "\n".join(details) or "—"]


def format_functor_summary(
    managers: Sequence[tuple[str, object, list[tuple[str, list[str]]]]],
    *,
    full: bool,
    color: bool | None = None,
    width: int | None = None,
) -> str:
    """Render initialized functors in execution groups without invoking them."""
    total = sum(len(names) for _, _, groups in managers for _, names in groups)
    if not total:
        return ""
    if color is None:
        color = "NO_COLOR" not in os.environ and sys.stderr.isatty()
    width = max(64, min(width or shutil.get_terminal_size((100, 24)).columns, 120))
    name_width = min(24, 18 + (width - 64) // 8)
    callable_width = 14 + (width - 64) // 3
    table = PrettyTable(["Name", "Callable", "Mode", "Details"])
    table.set_style(TableStyle.SINGLE_BORDER)
    table.align = "l"
    table.title = f"EmbodiChain · Functor Details · {total} active"
    table.max_width = {
        "Name": name_width,
        "Callable": callable_width,
        "Mode": 8,
        "Details": width - 13 - name_width - callable_width - 8,
    }
    for manager_name, manager, groups in managers:
        count = sum(len(names) for _, names in groups)
        if not count:
            continue
        table.add_row([f"{manager_name} · {count}", "", "", ""], divider=True)
        entries = [(mode, name) for mode, names in groups for name in names]
        for index, (mode, name) in enumerate(entries):
            table.add_row(
                _functor_cells(manager_name, manager, mode, name, full),
                divider=index == len(entries) - 1,
            )
    lines = table.get_string().splitlines()
    lines[0] = "╭" + lines[0][1:-1] + "╮"
    lines[-1] = "╰" + lines[-1][1:-1] + "╯"
    if color:
        for index, line in enumerate(lines):
            cells = line.split("│")
            if len(cells) == 6:
                group = not any(cell.strip() for cell in cells[2:5])
                for column, code in ((1, "1;36" if group else "1;32"), (3, "1;33")):
                    if cells[column].strip():
                        cells[column] = f"\033[{code}m{cells[column]}\033[0m"
                lines[index] = "│".join(cells)
            elif "EmbodiChain ·" in line:
                lines[index] = f"\033[1;36m{line}\033[0m"
    return "\n".join(lines)
