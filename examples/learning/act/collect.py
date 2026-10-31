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

"""Qualify the expert or record physically successful visual demonstrations."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import platform
from pathlib import Path
from typing import Any

from ._environment import close_env, make_env, schema_for_env, task_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qualify", "collect"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a short, unqualified engineering run",
    )
    args = parser.parse_args()
    if args.seed_base is None:
        args.seed_base = 500000 if args.mode == "qualify" else 100000
    if args.episodes < 1 or args.seed_base < 0:
        parser.error("episodes must be positive and seed-base nonnegative")
    identity = task_identity()
    qualification_seeds: set[int] = set()
    if args.mode == "collect" and not args.smoke:
        if args.qualification is None:
            parser.error("collect requires --qualification or --smoke")
        qualification = json.loads(args.qualification.read_text())
        if (
            qualification.get("task_identity") != identity
            or qualification.get("attempts", 0) < 100
            or qualification.get("successes", 0) / qualification["attempts"] < 0.95
        ):
            raise ValueError("Expert qualification must match the task and pass 95/100")
        qualification_seeds = {int(row["seed"]) for row in qualification["records"]}
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from embodichain.lab.gym.envs.demo import execute_demo_episode

    torch.set_num_threads(1)
    env: Any = None
    records: list[dict[str, Any]] = []
    dataset_path = None
    schema = None
    successes = 0
    try:
        env = make_env(
            args.device,
            camera=args.mode == "collect",
            save_path=args.output if args.mode == "collect" else None,
        )
        schema = schema_for_env(env)
        budget = args.episodes if args.mode == "qualify" else args.episodes * 4
        for attempt in range(budget):
            seed = args.seed_base + attempt
            if seed in qualification_seeds:
                raise ValueError(
                    "Collection seed overlaps the expert qualification panel"
                )
            env.reset(seed=seed, options={"save_data": False})
            try:
                result = execute_demo_episode(env, episode_index=attempt)
                success = bool(result.all_success)
                record = {
                    "seed": seed,
                    "success": success,
                    "steps": result.lengths[0],
                    "reason": result.terminal_reason,
                }
            except RuntimeError as error:
                success = False
                record = {"seed": seed, "success": False, "reason": str(error)}
            records.append(record)
            if success:
                successes += 1
                if args.mode == "collect":
                    env.reset(options={"save_data": True})
            print(json.dumps(record), flush=True)
            if args.mode == "collect" and successes >= args.episodes:
                break
        if args.mode == "collect":
            dataset_path = env.unwrapped.dataset_manager.finalize()
    finally:
        close_env(env)
    report = {
        "mode": args.mode,
        "task_identity": identity,
        "seed_base": args.seed_base,
        "attempts": len(records),
        "successes": successes,
        "requested_episodes": args.episodes,
        "dataset": dataset_path,
        "schema": schema,
        "records": records,
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": args.device,
            "gpu": (
                torch.cuda.get_device_name(args.device)
                if args.device.startswith("cuda")
                else None
            ),
            "physics": "default",
            "renderer": "hybrid" if args.mode == "collect" else "no-render",
        },
    }
    for package in ("dexsim", "lerobot"):
        try:
            report["runtime"][package] = version(package)
        except PackageNotFoundError:
            if package == "dexsim":
                import dexsim

                report["runtime"][package] = getattr(dexsim, "__version__", None)
            else:
                report["runtime"][package] = None
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {report_path}", flush=True)
    if args.mode == "collect" and successes < args.episodes:
        raise RuntimeError("Collection exhausted its bounded attempt budget")


if __name__ == "__main__":
    main()
