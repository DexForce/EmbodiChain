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

"""Reproduce both frozen sources, all 156 candidates, offline replay and video."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from embodichain.data_pipeline.real2sim.replay import export_real2sim_replay_manifest
from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    load_real2sim_tabletop_profile,
)
from examples.expansion.trajectory.real2sim.audit import _check_case
from examples.expansion.trajectory.real2sim_carton.prepare import acquire, write_json

__all__: list[str] = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New artifact directory; existing results are never replaced",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional cache for the five checksum-pinned official source files",
    )
    parser.add_argument("--jobs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Explicitly leave the video stage incomplete",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = (args.source_dir or output / "source").resolve()
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[4]
    # Works in the checkout and the installed share/embodichain examples tree.
    paths = [
        repository,
        repository / "extensions",
        repository / "extensions/embodichain_tasks",
    ]
    environment.update(
        EMBODICHAIN_SIM_EXIT_PROCESS="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONPATH=os.pathsep.join(
            [*(str(path) for path in paths), environment.get("PYTHONPATH", "")]
        ),
    )
    stages = []

    def execute(
        name: str, module: str, arguments: list[str], *, matrix: bool = False
    ) -> int:
        with (output / f"{name}.log").open("w") as log:
            status = subprocess.run(
                [sys.executable, "-m", module, *arguments],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        stages.append(dict(stage=name, exit_code=status, log=f"{name}.log"))
        write_json(output / "stages.json", stages)
        if status and not matrix:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        return status

    acquire(source)
    prepared, native, synthetic = (
        output / name for name in ("prepared", "native-reference", "synthetic")
    )
    prefix = "examples.expansion.trajectory.real2sim_carton."
    execute(
        "prepare",
        prefix + "prepare",
        ["prepare", "--source-dir", str(source), "--output", str(prepared)],
    )
    execute(
        "native-capture",
        prefix + "capture",
        [
            "--gym-config",
            str(prepared / "task.native.yaml"),
            "--output-dir",
            str(native),
        ],
    )
    execute(
        "synthetic-input",
        prefix + "prepare_synthetic",
        [
            "--prepared",
            str(prepared),
            "--capture",
            str(native),
            "--output",
            str(synthetic),
        ],
    )
    for origin, task, profile in (
        (
            "measured",
            prepared / "task.measured.yaml",
            prepared / "profile.measured.single.yaml",
        ),
        (
            "synthetic_reference",
            synthetic / "task.yaml",
            synthetic / "profile.single.yaml",
        ),
    ):
        destination = output / f"nominal-{origin}"
        execute(
            f"nominal-{origin}",
            "embodichain",
            [
                "run-task",
                "--gym-config",
                str(task),
                "--real2sim-profile",
                str(profile),
                "--dataset-dir",
                str(destination),
                "--device",
                "cpu",
                "--headless",
                "--renderer",
                "no-render",
                "--disable-sensor",
                "--max_episodes",
                "1",
            ],
        )
        roots = list(destination.glob("*/meta/info.json"))
        if len(roots) != 1:
            raise ValueError("nominal execution must own exactly one recorder")
        root = roots[0].parents[1]
        rows = [
            json.loads(line)
            for line in (root / "meta/embodichain_episodes.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        ]
        if len(rows) != 1:
            raise ValueError("nominal source has no unique confirmed episode")
        audit = _check_case(
            root,
            rows[0],
            load_real2sim_tabletop_profile(profile),
            expected_origin=origin,
        )
        export_real2sim_replay_manifest(
            [root], output / f"nominal-{origin}-attachments.json"
        )
        write_json(output / f"nominal-{origin}-audit.json", audit)
    status = execute(
        "matrix",
        prefix + "matrix",
        [
            "--prepared",
            str(prepared),
            "--synthetic",
            str(synthetic),
            "--output",
            str(output / "matrix"),
            "--jobs",
            str(args.jobs),
        ],
        matrix=True,
    )
    report = output / "matrix/acceptance-report.json"
    if not report.is_file() or not json.loads(report.read_text())["report_complete"]:
        raise RuntimeError("the matrix did not report all 156 candidates")
    execute(
        "replay",
        prefix + "export",
        [
            "--prepared",
            str(prepared),
            "--synthetic",
            str(synthetic),
            "--matrix",
            str(report),
            "--source-dir",
            str(source),
            "--output",
            str(output / "replay"),
        ],
    )
    if not args.no_video:
        execute(
            "video",
            prefix + "video",
            [
                "--prepared",
                str(prepared),
                "--replay",
                str(output / "replay"),
                "--output",
                str(output / "video"),
            ],
        )
    print(f"Replay: {output / 'replay/index.html'}")
    print(
        f"Serve locally: {sys.executable} -m http.server 8000 --bind 127.0.0.1 --directory {output}"
    )
    print(f"Report: {report}")
    if status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
