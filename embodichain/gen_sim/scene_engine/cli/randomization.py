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

"""Generate and persist grouped scene layouts using the shared scene generator."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> None:
    """Prepare a baseline, generate layouts and save a new run directory.

    Exhausting the attempt budget is a normal result, including when no
    layouts were found. The sampling status reports whether the target was met.

    Args:
        argv: Arguments excluding the command name; defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        prog="embodichain scene-randomization",
        description=(
            "Prepare a reusable baseline, generate grouped XY layouts using "
            "validated YAML parameters, and save each successful layout."
        ),
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

    from embodichain.gen_sim.scene_engine.randomization.cfg import (
        load_randomization_config,
    )
    from embodichain.gen_sim.scene_engine.randomization.generation import (
        randomize_scene,
    )
    from embodichain.gen_sim.scene_engine.randomization.persistence import save_layouts

    try:
        cfg = load_randomization_config(args.config, output_root=args.output_root)
        result = randomize_scene(cfg, rebuild=args.rebuild)
        run_dir = save_layouts(result, output_root=args.output_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Baseline YAML: {result.baseline.path}")
    print(f"Groups: {len(result.baseline.data['groups'])}")
    print(f"Cache: {'reused' if result.baseline.cache_hit else 'rebuilt'}")
    print(f"Sampling: {result.status}")
    print(f"Attempts: {result.attempts}")
    print(f"Layouts: {len(result.candidates)}")
    print(f"Run directory: {run_dir}")
    print(f"Manifest: {run_dir / 'run_result.json'}")
    print(
        "Solver statuses: "
        + ", ".join(
            f"{status}={count}"
            for status, count in sorted(result.solver_status_counts.items())
        )
    )
    if result.status not in ("completed", "budget_exhausted"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
