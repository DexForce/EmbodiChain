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

"""Command-line interface for the SimReady asset pipeline."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def cli_ingest_single(input_dir: str, output_dir: str, category: str) -> None:
    """Ingest one asset directory.

    Args:
        input_dir: Directory containing the source asset.
        output_dir: Root directory for generated assets.
        category: Semantic category assigned to the asset.

    Raises:
        FileNotFoundError: If the input directory does not exist.
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from embodichain.gen_sim.simready_pipeline.io.json_store import JsonStore
    from embodichain.gen_sim.simready_pipeline.parser.base import ParserManager
    from embodichain.gen_sim.simready_pipeline.pipeline.ingest import ingest_one_asset

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    store = JsonStore(output_path)
    manager = ParserManager()

    print(f"Processing Single Asset: {input_path.name} (Category: {category})")

    asset = ingest_one_asset(
        asset_dir=input_path,
        category=category,
        output_root=output_path,
        store=store,
        manager=manager,
    )

    if asset:
        print("Successfully processed")
    else:
        print("No asset returned (might be direct_copy mode)")


def main(argv: Sequence[str] | None = None) -> None:
    """Run the SimReady asset pipeline CLI.

    Args:
        argv: Arguments excluding the command name. Uses ``sys.argv`` when
            omitted.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain simready",
        description="Convert a raw asset directory into a SimReady asset.",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the single asset directory.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory for generated assets.",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Specify the category for this asset (e.g., 'cup', 'chair')",
    )
    args = parser.parse_args(argv)

    cli_ingest_single(args.input_dir, args.output_root, args.category)


if __name__ == "__main__":
    main()


__all__ = ["cli_ingest_single", "main"]
