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

"""CLI tool for pre-downloading EmbodiChain data assets.

Usage::

    # List all available assets
    python -m embodichain.data.download list

    # List assets in a specific category
    python -m embodichain.data.download list --category robot

    # Download a specific asset by name
    python -m embodichain.data.download download --name CobotMagicArm

    # Download all assets in a category
    python -m embodichain.data.download download --category robot

    # Download everything
    python -m embodichain.data.download download --all
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import shutil
import sys

import open3d as o3d

from embodichain.data.constants import EMBODICHAIN_DEFAULT_DATA_ROOT

# Mapping from category name to the module path that defines the asset classes.
CATEGORY_MODULES: dict[str, str] = {
    "demo": "embodichain.data.assets.demo_assets",
    "eef": "embodichain.data.assets.eef_assets",
    "materials": "embodichain.data.assets.materials",
    "obj": "embodichain.data.assets.obj_assets",
    "robot": "embodichain.data.assets.robot_assets",
    "scene": "embodichain.data.assets.scene_assets",
    "w1": "embodichain.data.assets.w1_assets",
}


def _get_asset_classes(module_path: str) -> list[tuple[str, type]]:
    """Return (name, cls) pairs for all DownloadDataset subclasses in *module_path*."""
    module = importlib.import_module(module_path)
    results: list[tuple[str, type]] = []
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, o3d.data.DownloadDataset)
            and obj is not o3d.data.DownloadDataset
            and obj.__module__ == module.__name__
        ):
            results.append((name, obj))
    results.sort(key=lambda x: x[0])
    return results


def get_registry() -> dict[str, list[tuple[str, type]]]:
    """Build ``{category: [(class_name, class), ...]}`` for every category."""
    registry: dict[str, list[tuple[str, type]]] = {}
    for category, module_path in CATEGORY_MODULES.items():
        registry[category] = _get_asset_classes(module_path)
    return registry


def find_asset_class(
    name: str, registry: dict[str, list[tuple[str, type]]]
) -> tuple[str, type] | None:
    """Find an asset class by name (case-insensitive) across all categories."""
    name_lower = name.lower()
    for category, assets in registry.items():
        for cls_name, cls in assets:
            if cls_name.lower() == name_lower:
                return category, cls
    return None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _ensure_extract(data_obj: o3d.data.DownloadDataset, prefix: str) -> None:
    """For non-zip assets, copy the downloaded file into the extract directory.

    ``o3d.data.DownloadDataset`` extracts zip archives automatically but
    leaves single-file downloads (e.g. ``.glb``) only in the download
    directory.  This helper copies them to the extract tree so that
    ``get_data_path`` can find them under ``<data_root>/extract/<prefix>/``.
    """
    extract_dir = os.path.join(EMBODICHAIN_DEFAULT_DATA_ROOT, "extract", prefix)
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        return  # already extracted

    download_dir = os.path.join(EMBODICHAIN_DEFAULT_DATA_ROOT, "download", prefix)
    if not os.path.isdir(download_dir):
        return

    os.makedirs(extract_dir, exist_ok=True)
    for item in os.listdir(download_dir):
        src = os.path.join(download_dir, item)
        dst = os.path.join(extract_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    print(f"  Copied non-zip asset to extract dir: {extract_dir}")


def download_asset(cls_name: str, cls: type) -> None:
    """Instantiate an asset class to trigger download, then ensure extraction."""
    print(f"  Downloading {cls_name} ...")
    try:
        data_obj = cls()
        _ensure_extract(data_obj, cls_name)
        print(f"  ✓ {cls_name} ready")
    except Exception as exc:
        print(f"  ✗ {cls_name} failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> None:
    """List available assets."""
    registry = get_registry()
    categories = [args.category] if args.category else sorted(registry)

    for category in categories:
        assets = registry.get(category)
        if assets is None:
            print(f"Unknown category: {category}", file=sys.stderr)
            print(
                f"Available categories: {', '.join(sorted(registry))}", file=sys.stderr
            )
            sys.exit(1)

        print(f"\n[{category}] ({len(assets)} assets)")
        for cls_name, _ in assets:
            # Show whether it is already downloaded
            extract_dir = os.path.join(
                EMBODICHAIN_DEFAULT_DATA_ROOT, "extract", cls_name
            )
            status = (
                "✓" if os.path.isdir(extract_dir) and os.listdir(extract_dir) else " "
            )
            print(f"  [{status}] {cls_name}")

    print(f"\nData root: {EMBODICHAIN_DEFAULT_DATA_ROOT}")


def cmd_download(args: argparse.Namespace) -> None:
    """Download assets by name, category, or everything."""
    registry = get_registry()

    targets: list[tuple[str, type]] = []

    if args.all:
        for assets in registry.values():
            targets.extend(assets)
    elif args.category:
        assets = registry.get(args.category)
        if assets is None:
            print(f"Unknown category: {args.category}", file=sys.stderr)
            print(
                f"Available categories: {', '.join(sorted(registry))}", file=sys.stderr
            )
            sys.exit(1)
        targets.extend(assets)
    elif args.name:
        result = find_asset_class(args.name, registry)
        if result is None:
            print(f"Asset '{args.name}' not found.", file=sys.stderr)
            print("Use 'list' to see available assets.", file=sys.stderr)
            sys.exit(1)
        _category, cls = result
        targets.append((args.name, cls))
    else:
        print("Specify --name, --category, or --all.", file=sys.stderr)
        sys.exit(1)

    print(f"Data root: {EMBODICHAIN_DEFAULT_DATA_ROOT}")
    print(f"Downloading {len(targets)} asset(s) ...\n")

    for cls_name, cls in targets:
        download_asset(cls_name, cls)

    print(f"\nDone. {len(targets)} asset(s) processed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="embodichain.data.download",
        description="Pre-download EmbodiChain data assets from HuggingFace.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- list ---
    list_parser = subparsers.add_parser("list", help="List available assets.")
    list_parser.add_argument(
        "--category",
        choices=sorted(CATEGORY_MODULES),
        help="Show only assets in this category.",
    )

    # --- download ---
    dl_parser = subparsers.add_parser("download", help="Download assets.")
    dl_group = dl_parser.add_mutually_exclusive_group(required=True)
    dl_group.add_argument("--name", help="Download a single asset by class name.")
    dl_group.add_argument(
        "--category",
        choices=sorted(CATEGORY_MODULES),
        help="Download all assets in a category.",
    )
    dl_group.add_argument(
        "--all", action="store_true", help="Download every registered asset."
    )

    args = parser.parse_args()
    if args.command == "list":
        cmd_list(args)
    elif args.command == "download":
        cmd_download(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
