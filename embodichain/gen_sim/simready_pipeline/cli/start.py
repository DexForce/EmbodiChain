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

import argparse
from pathlib import Path
import os

os.environ["PYOPENGL_PLATFORM"] = "egl"

from embodichain.gen_sim.simready_pipeline.pipeline.ingest import ingest_one_asset
from embodichain.gen_sim.simready_pipeline.io.json_store import JsonStore
from embodichain.gen_sim.simready_pipeline.parser.base import ParserManager


def cli_ingest_single(
    input_dir: str, output_dir: str, category: str
):
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
        print(f"Successfully Processed")
    else:
        print("no asset returned (might be direct_copy mode)")


def main():
    parser = argparse.ArgumentParser(
        description="embodichain.gen_sim.simready_pipeline Asset Ingestion Pipeline"
    )

    parser.add_argument(
        "--input_dir", type=str, help="Path to the single asset directory"
    )
    parser.add_argument("--output_root", type=str, help="Path to the output directory")
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Specify the category for this asset (e.g., 'cup', 'chair')",
    )
    args = parser.parse_args()

    cli_ingest_single(
        args.input_dir, args.output_root, args.category
    )


if __name__ == "__main__":
    main()
