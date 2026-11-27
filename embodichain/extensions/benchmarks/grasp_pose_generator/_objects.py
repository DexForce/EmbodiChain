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

"""Expand local DexTrove SimReady packages into individual benchmark objects."""

from __future__ import annotations

from pathlib import Path

__all__ = ["suite_objects"]

_USD_SUFFIXES = {".usd", ".usda", ".usdc"}


def _suite_asset_root(reference: str | Path, suite_dir: Path) -> Path:
    """Resolve a suite-local directory or a registered Dataset/subdirectory."""
    path = Path(reference).expanduser()
    local = suite_dir / path
    if path.is_absolute() or local.exists():
        return local.resolve()

    from embodichain.data import get_data_path

    return Path(get_data_path(str(path))).resolve()


def _simready_source(package: Path, physics: str) -> dict:
    from pxr import Sdf

    layers = sorted(
        path
        for path in package.iterdir()
        if path.is_file() and path.suffix.lower() in _USD_SUFFIXES
    )
    if len(layers) != 1:
        raise ValueError(
            f"Expected one root USD layer in {package}, found {len(layers)}"
        )
    layer = Sdf.Layer.OpenAsAnonymous(str(layers[0]), metadataOnly=True)
    source = {"file_path": str(layers[0])}
    metadata = layer.customLayerData.get("SimReady_Metadata", {})
    profiles = metadata.get("Variants", {}).get("Physics", {})
    if profiles:
        backend = {"default": "PhysX", "newton": "Newton"}[physics]
        if backend not in profiles:
            raise ValueError(f"No {backend} physics variant in {layers[0]}")
        selections: dict[str, dict[str, str]] = {}
        for name, profile in profiles.items():
            choices = selections.setdefault(profile["prim"], {})
            choices[profile["variantSetName"]] = (
                profile["activateOption"] if name == backend else "Disabled"
            )
        source["variants"] = selections
    return source


def suite_objects(
    suite: dict,
    suite_dir: Path,
    *,
    physics: str = "default",
    representatives: dict[str, str] | None = None,
) -> list[dict]:
    """Resolve explicit objects and unpacked dataset categories for one suite.

    A representatives mapping selects one exact object ID per listed category,
    in mapping order. Each selection must match exactly one source entry or
    unpacked package. Only selected packages have their USD metadata loaded.
    """
    if representatives is not None and (
        not isinstance(representatives, dict)
        or not representatives
        or any(
            not isinstance(name, str) or not name.strip()
            for pair in representatives.items()
            for name in pair
        )
    ):
        raise ValueError(
            "representatives must map category names to nonempty object IDs"
        )

    objects = []
    dataset_root = None
    for entry in suite["objects"]:
        category = entry.get("category", entry.get("family", entry.get("id")))
        if not isinstance(category, str) or not category.strip():
            raise ValueError("Every object needs a nonempty category name")
        if representatives is not None and category not in representatives:
            continue
        wanted = representatives[category] if representatives is not None else None
        if "id" in entry:
            if wanted is None or wanted == entry["id"]:
                objects.append({**entry, "category": category})
            continue
        if Path(category).name != category or category == "..":
            raise ValueError(
                f"Expected a dataset category directory name, got {category!r}"
            )
        if dataset_root is None:
            root = suite.get("dataset_root")
            if not root:
                raise ValueError(f"dataset_root is required for category {category!r}")
            dataset_root = _suite_asset_root(root, suite_dir)
        directory = dataset_root / category
        # Upload layout: <category>/<model>/<root USD>. Older DexTrove
        # extractions place the same files under <source_id>/<model>/simready_usd.
        flat = {
            layer.parent
            for layer in directory.glob("*/*")
            if layer.suffix.lower() in _USD_SUFFIXES and layer.is_file()
        }
        packages = [(package.name, package) for package in sorted(flat)]
        packages.extend(
            (p.parent.name, p)
            for p in sorted(directory.glob("*/*/simready_usd"))
            # A flat package owns its source/ dependencies, including nested USDs.
            if p.parents[1] not in flat
        )
        if not packages and wanted is None:
            raise ValueError(f"No unpacked SimReady packages in {directory}")
        for model, package in packages:
            if wanted is not None and wanted != model:
                continue
            rigid = {
                **entry.get("rigid", {}),
                "source": _simready_source(package, physics),
            }
            objects.append({**entry, "category": category, "id": model, "rigid": rigid})
    if representatives is None:
        return objects
    selected = {}
    for obj in objects:
        category = obj["category"]
        if category in selected:
            raise ValueError(f"Ambiguous representative {category}/{obj['id']}")
        selected[category] = obj
    for category, model in representatives.items():
        if category not in selected:
            raise ValueError(
                f"Representative {category}/{model} was not found in the suite"
            )
    return [selected[category] for category in representatives]
