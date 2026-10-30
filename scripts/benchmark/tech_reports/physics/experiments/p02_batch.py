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

"""Sequential, paired P-02 process launcher; preserves every attempted run.

Run: python -m scripts.benchmark.tech_reports.physics.experiments.p02_batch --help
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "pilot",
            "main",
            "supplement",
            "capture",
            "diagnostic",
            "independent",
            "independent_supplement",
        ],
        required=True,
    )
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--platforms", nargs="+", default=["embodichain", "isaaclab"])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--isolate-initials", action="store_true")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--contact-grouping", action="store_true")
    parser.add_argument("--contact-history", action="store_true")
    parser.add_argument("--no-contact-snap", action="store_true")
    parser.add_argument("--solver", choices=["TGS", "PGS"], default="TGS")
    parser.add_argument("--position-iterations", type=int, default=4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    root = Path(
        os.environ.get("SIM_REPORT_ROOT", Path(__file__).resolve().parents[5])
    ).resolve()
    config = (
        args.config
        or root / "scripts/benchmark/tech_reports/assets/physics/P-02/protocol.json"
    )
    config = config.resolve()
    output_root = (args.output_root or root / "outputs/benchmarks/P-02").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes or [
        s["id"] for s in json.loads(config.read_text())["scenarios"]
    ]
    tag = datetime.now().strftime("%Y%m%d_%H%M%S") + args.tag
    ledger = output_root / f"{tag}_{args.phase}_ledger.jsonl"
    batches = (
        [(i, 1) for i in range(args.episode_start, args.episode_start + args.count)]
        if args.isolate_initials
        else [(args.episode_start, args.count)]
    )
    jobs = [
        (scene, platform, initial, count)
        for scene in scenes
        for initial, count in batches
        for platform in args.platforms
    ]
    for scene, platform, initial, count in jobs:
        output = (
            output_root
            / f"runs/{tag}_{args.phase}_{platform}_{scene}_dt{args.dt:g}_initial{initial:02d}"
        )
        output.mkdir(parents=True)
        source = Path(
            "/home/oem/code/EmbodiChain"
            if platform == "embodichain"
            else "/home/oem/code/IsaacLab"
        )
        python = f"/home/oem/miniconda3/envs/{platform}/bin/python"
        cmd = [
            python,
            "-u",
            "-m",
            f"scripts.benchmark.tech_reports.physics.platforms.{platform}.p02_contact",
            "--config",
            str(config),
            "--scene",
            scene,
            "--output",
            str(output),
            "--source-repo",
            str(source),
            "--spec",
            str(root / "specs/p-02-contact-stability-spec.md"),
            "--dt",
            str(args.dt),
            "--count",
            str(count),
            "--episode-start",
            str(initial),
            "--phase",
            args.phase,
        ]
        if args.duration is not None:
            cmd += ["--duration", str(args.duration)]
        if args.phase == "capture":
            cmd += ["--record"]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root)
        if platform == "embodichain":
            env[
                "PYTHONPATH"
            ] += ":/home/oem/code/EmbodiChain:/home/oem/code/EmbodiChain/extensions:/home/oem/code/EmbodiChain/extensions/embodichain_tasks"
            if args.iterations is not None:
                cmd += ["--iterations", str(args.iterations)]
            if args.contact_grouping:
                cmd += ["--contact-grouping"]
            if args.contact_history:
                cmd += ["--contact-history"]
            if args.no_contact_snap:
                cmd += ["--no-contact-snap"]
        else:
            env["OMNI_KIT_ACCEPT_EULA"] = "YES"
            cmd += [
                "--visualizer",
                "none",
                "--device",
                "cuda:0",
                "--solver",
                args.solver,
                "--position-iterations",
                str(args.position_iterations),
            ]
        entry = dict(
            run_id=output.name, command=cmd, started_at=datetime.now().isoformat()
        )
        print("START", output.name, flush=True)
        with (output / "console.log").open("w") as log:
            result = subprocess.run(
                cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        run_meta = (
            json.loads((output / "run.json").read_text())
            if (output / "run.json").exists()
            else {}
        )
        completed = (
            run_meta.get("status") == "completed"
            and run_meta.get("completed_episodes") == count
        )
        entry.update(
            exit_code=result.returncode,
            run_status=run_meta.get("status", "missing"),
            ended_at=datetime.now().isoformat(),
        )
        with ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print("END", result.returncode, output.name, flush=True)
        if result.returncode or not completed:
            print(
                (output / "console.log").read_text(errors="replace")[-4000:],
                flush=True,
            )
            sys.exit(result.returncode or 1)


if __name__ == "__main__":
    main()
