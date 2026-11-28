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

"""Discover a classified DexTrove download and check its single-rigid-body imports."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

from ._objects import _MissingPhysicsVariant, _simready_source

__all__ = ["dextrove_objects"]


def _classified_packages(download_root: Path) -> list[tuple[str, str, Path]]:
    """Validate the complete category view before opening individual USD files."""
    root = download_root.expanduser().resolve(strict=True)
    categories = root if root.name == "categories" else root / "categories"
    if not categories.is_dir():
        raise ValueError(f"Expected the DexTrove categories directory at {categories}")

    seen = {}
    packages = []
    for category in sorted(categories.iterdir()):
        if category.name.startswith("."):
            continue
        if not category.is_dir():
            raise ValueError(f"Expected a DexTrove category directory: {category}")
        for entry in sorted(category.iterdir()):
            if entry.name.startswith("."):
                continue
            try:
                uid = str(UUID(entry.name))
            except ValueError as exc:
                raise ValueError(f"Expected a DexTrove asset UUID at {entry}") from exc
            if uid != entry.name:
                raise ValueError(f"Expected a canonical DexTrove asset UUID at {entry}")
            if uid in seen:
                raise ValueError(
                    f"Duplicate DexTrove UUID {uid} in {seen[uid]} and {category.name}"
                )
            if not entry.is_dir():
                raise ValueError(f"Missing or broken DexTrove package: {entry}")
            wrapper = entry / f"{uid}.usda"
            if not wrapper.is_file():
                raise ValueError(f"Missing DexTrove USD entry point: {wrapper}")
            seen[uid] = category.name
            packages.append((uid, category.name, entry.resolve(strict=True)))
    if not packages:
        raise ValueError(f"No classified DexTrove objects in {categories}")
    return packages


def dextrove_objects(
    download_root: Path, *, physics: str, rigid: dict
) -> tuple[list[dict], list[dict]]:
    """Return executable suite objects and one import decision per downloaded UUID.

    The download's ``categories/`` view defines the population and the category
    of each object. Only the backend variant and the ability to import one rigid
    target with collision geometry are checked here. The physical trials measure
    grasp success.

    Args:
        download_root: Download destination or its ``categories/`` directory.
        physics: Benchmark backend, ``default`` (PhysX) or ``newton`` (DexUni).
        rigid: Shared rigid-object settings for the selected USD sources.

    Returns:
        Accepted suite object definitions and the complete per-object inventory.

    Raises:
        ValueError: The view is malformed, ambiguous or empty, or the backend
            or supplied rigid settings are invalid.
    """
    from embodichain.lab.sim.cfg import RigidObjectCfg
    from embodichain.lab.sim.spawn.usd import rigid_desc_from_usd

    if physics not in {"default", "newton"}:
        raise ValueError(f"Unsupported DexTrove benchmark backend: {physics!r}")
    solver = "dexuni" if physics == "newton" else None
    objects = []
    inventory = []
    for uid, category, package in _classified_packages(download_root):
        row = {"id": uid, "category": category}
        inventory.append(row)
        try:
            source = _simready_source(package, physics)
        except _MissingPhysicsVariant as exc:
            row.update(status="excluded", reason=str(exc))
            continue

        settings = {**rigid, "source": source}
        cfg = RigidObjectCfg.from_dict({**settings, "uid": "target"})
        try:
            desc, _ = rigid_desc_from_usd(cfg, newton_solver_type=solver)
        except (ValueError, NotImplementedError) as exc:
            row.update(status="excluded", reason=str(exc))
            continue
        if not any(shape.enable_collision is not False for shape in desc.collisions):
            row.update(status="excluded", reason="No enabled collision shapes")
            continue
        row["status"] = "ready"
        objects.append({"id": uid, "category": category, "rigid": settings})
    return objects, inventory
