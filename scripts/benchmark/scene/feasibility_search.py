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

"""Compare proposal strategies under the same full physical acceptance path.

Run: python -m scripts.benchmark.scene.feasibility_search --config PATH --output ROOT
Workspace-only denotes a workspace-conditioned proposal prior; all strategies
retain identical semantic, motion, collision and measured task acceptance gates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import yaml

from embodichain.lab.task_program.integrations.feasibility import run_feasibility
from embodichain.lab.sim.scene.feasibility._values import atomic_json
from embodichain.utils.utility import load_config


def benchmark_search(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    seconds: float = 600.0,
    max_candidates: int | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict:
    """Run comparable joint/uniform/workspace/fixed-installation campaigns.

    Args:
        config_path: Authored production workflow, including optional prior binding.
        output_directory: Separate immutable episode roots for each campaign.
        seconds: Equal total budget for every strategy/seed.
        max_candidates: Optional equal proposal quota override for a smoke run.
        seeds: Explicit repeated-run seeds; defaults to the authored search seed.
    Returns:
        Actual measured yields, coverage and stage costs with raw atlas paths.
        No aggregate superiority claim is inferred from a small smoke campaign.
    """
    source = Path(config_path).resolve()
    root = Path(output_directory).resolve()
    base = load_config(source)
    for key in ("deployment", "program"):
        if key in base["task_program"]:
            base["task_program"][key] = str(
                (source.parent / base["task_program"][key]).resolve()
            )
    seeds = seeds or (base.get("search", {}).get("seed", 0),)
    summaries = []
    for seed in seeds:
        for strategy in ("joint", "uniform", "workspace", "fixed_base"):
            config = copy.deepcopy(base)
            search = config.setdefault("search", {})
            search.update(seed=seed, max_wall_time_s=seconds)
            if max_candidates is not None:
                search["max_candidates"] = max_candidates
            if strategy == "uniform":
                search["global_fraction"] = 1.0
                config["task_program"].pop("workspace_proposal", None)
            elif strategy == "workspace":
                if "workspace_proposal" not in config["task_program"]:
                    summaries.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "status": "unavailable",
                            "reason": "No explicit workspace phase/part/group binding",
                        }
                    )
                    continue
                search["global_fraction"] = 1.0
                config["task_program"]["workspace_proposal"]["uniform_fraction"] = 0.0
            elif strategy == "fixed_base":
                unavailable = False
                for variable in config["request"]["variables"]:
                    if variable["kind"] == "robot":
                        if any(
                            a > 0 or b < 0
                            for a, b in zip(variable["lower"], variable["upper"])
                        ):
                            unavailable = True
                        variable["lower"] = variable["upper"] = [0.0, 0.0, 0.0]
                if unavailable:
                    summaries.append(
                        {
                            "strategy": strategy,
                            "seed": seed,
                            "status": "unavailable",
                            "reason": "Authored installation is outside the allowed comparison domain",
                        }
                    )
                    continue
            folder = root / (strategy + "_" + str(seed))
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "workflow.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=False))
            started = time.perf_counter()
            atlas = run_feasibility(path, folder / "result")
            elapsed = time.perf_counter() - started
            rows = atlas.observations
            summaries.append(
                {
                    "strategy": strategy,
                    "seed": seed,
                    "status": "complete",
                    "elapsed_s": elapsed,
                    "measured_successes": sum(o.evaluation.accepted for o in rows),
                    "committed": sum(o.evaluation.committed for o in rows),
                    "proposals": len(rows),
                    "regions": len(atlas.regions),
                    "confirmed_episodes_per_minute": sum(
                        o.evaluation.committed for o in rows
                    )
                    * 60
                    / elapsed,
                    "calls": dict(atlas.metrics.get("calls", {})),
                    "failure_stages": dict(atlas.metrics.get("failure_stages", {})),
                    "prior": (
                        None
                        if atlas.metrics.get("proposal_source") is None
                        else dict(atlas.metrics["proposal_source"])
                    ),
                    "atlas": str(folder / "result" / "atlas.json"),
                }
            )
            atomic_json(
                root / "comparison.json",
                {
                    "scope": __doc__,
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "threads": torch.get_num_threads(),
                    "seeds": seeds,
                    "seconds_per_job": seconds,
                    "max_candidates": max_candidates,
                    "runs": summaries,
                },
            )
    result = {
        "scope": "proposal strategies with identical full physical gates; small runs are smoke evidence",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "seeds": seeds,
        "seconds_per_job": seconds,
        "max_candidates": max_candidates,
        "runs": summaries,
    }
    atomic_json(root / "comparison.json", result)
    return result


def main() -> None:
    """Run explicitly budgeted physical comparisons sequentially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    if args.seconds <= 0 or (
        args.max_candidates is not None and args.max_candidates < 1
    ):
        parser.error("Budgets must be positive")
    result = benchmark_search(
        args.config,
        args.output,
        seconds=args.seconds,
        max_candidates=args.max_candidates,
        seeds=None if args.seeds is None else tuple(args.seeds),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
