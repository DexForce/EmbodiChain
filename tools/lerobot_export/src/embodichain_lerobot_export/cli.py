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

"""CLI for the standalone export environment."""

from __future__ import annotations

import argparse
import json

from .exporter import export_dataset

__all__ = ["main"]


def main() -> None:
    """Convert a finalized local dataset and print its conversion manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Finalized EmbodiChain LeRobot v3.0 dataset")
    parser.add_argument("destination", help="New dataset directory outside the source")
    args = parser.parse_args()
    try:
        manifest = export_dataset(args.source, args.destination)
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
