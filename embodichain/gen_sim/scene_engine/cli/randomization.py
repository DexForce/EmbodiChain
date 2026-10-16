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

"""Prepare and persist a scene randomization baseline from configuration."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run baseline preparation without starting scene generation or simulation.

    Args:
        argv: Arguments excluding the command name; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain scene-randomization",
        description="Prepare a reusable scene randomization baseline YAML.",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Scene workspace containing scene_export/; the baseline is written there.",
    )
    parser.add_argument("--config", required=True, help="Version-1 YAML configuration.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Recompute an existing baseline cache."
    )
    args = parser.parse_args(argv)

    from embodichain.gen_sim.scene_engine.randomization import (
        load_randomization_config,
        prepare_scene_baseline,
    )

    try:
        cfg = load_randomization_config(args.config, output_root=args.output_root)
        result = prepare_scene_baseline(cfg, rebuild=args.rebuild)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Baseline YAML: {result.path}")
    print(f"Groups: {len(result.data['groups'])}")
    print(f"Cache: {'reused' if result.cache_hit else 'rebuilt'}")


if __name__ == "__main__":
    main()
