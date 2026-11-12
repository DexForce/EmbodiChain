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

"""Reproduce the pinned upstream task before comparing EmbodiChain algorithms.

Run with the standard Python library. The upstream checkout owns its frozen
virtual environment, physics, task, algorithm, checkpoints and evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

__all__ = ["main"]

_REVISION = "9816885a3b1fbdb947db71e72c27ed75effcbf9f"
_FAST_TESTS = [
    "tests/test_protocol.py",
    "tests/envs/test_observation_contract.py",
    "tests/test_evaluation_parallel.py",
    "tests/test_evaluation_statistics.py",
    "tests/test_grasp_caches.py",
]
_SIM_TESTS = [
    "tests/envs/test_observation_contract.py::test_mujoco_autoreset_and_simultaneous_termination",
    "tests/test_supermimic_runtime.py::test_mjwarp_capture_is_gpu_resident",
]


def _capture(args: list[str], root: Path) -> str:
    return subprocess.check_output(args, cwd=root, text=True).strip()


def _source(root: Path) -> dict[str, Any]:
    revision = _capture(["git", "rev-parse", "HEAD"], root)
    if revision != _REVISION:
        raise ValueError(f"Expected upstream {_REVISION}; found {revision}")
    if _capture(["git", "status", "--porcelain", "--untracked-files=all"], root):
        raise ValueError(
            "The upstream checkout must have no modified or untracked sources"
        )
    if not (root / ".venv/bin/python").is_file():
        raise FileNotFoundError("Run uv sync --frozen in the upstream checkout first")
    files = _capture(["git", "ls-files"], root).splitlines()
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in files
        if name.startswith(("src/", "tests/")) or name in ("uv.lock", "pyproject.toml")
    }
    return {"revision": revision, "tracked_sha256": hashes}


def main() -> None:
    """Audit, validate, train and evaluate the original fixed baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root, output = args.upstream.resolve(), args.output.resolve()
    source = _source(root)
    output.mkdir(parents=True, exist_ok=False)
    python = str(root / ".venv/bin/python")
    # Isolate generated grasp caches as well as writable XML from other runs.
    # Do not let a user-generated cache silently replace the bundled baseline.
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["SHARPA_RL_UNILAB_ASSET_CACHE"] = str(output / "asset_cache")
    env["PYTHONUNBUFFERED"] = "1"
    overrides = [
        "training.no_play=true",
        "training.logger=no_print",
        f"training.log_dir={output / 'training'}",
    ]
    train = [
        python,
        "-m",
        "sharpa_rl_unilab.cli",
        "--algo",
        "supermimic",
        "--sim",
        "mjwarp",
        *overrides,
    ]
    state: dict[str, Any] = {
        "source": source,
        "started_unix": time.time(),
        "runner_pid": os.getpid(),
        "phase": "audit",
        "reference_only": True,
        "algorithm_comparison_allowed": False,
        "stage": "teacher",
        "sim_backend": "mjwarp",
        "semantic_task_overrides": [],
        "runtime_overrides": overrides,
        "commands": [],
    }

    def save() -> None:
        temporary = output / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "status.json")

    def run(phase: str, command: list[str], log: str) -> None:
        state["phase"] = phase
        state["commands"].append({"phase": phase, "argv": command, "log": log})
        save()
        print(f"[{phase}] {output / log}", flush=True)
        with (output / log).open("w") as stream:
            subprocess.run(
                command,
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    save()
    try:
        run(
            "installed_source",
            [
                python,
                "-c",
                (
                    "from pathlib import Path; import sharpa_rl_unilab as p; "
                    f"assert Path(p.__file__).resolve().parent == Path({str(root / 'src/sharpa_rl_unilab')!r}) "
                    "and p.__file__; print(p.__file__)"
                ),
            ],
            "installed_source.log",
        )
        # Keep the resolved environment, not just a list of intended versions.
        run("packages", ["uv", "pip", "freeze", "--python", python], "packages.txt")
        run("configuration", [*train, "--cfg"], "effective_config.yaml")
        run(
            "contract_tests",
            [python, "-m", "pytest", "-q", *_FAST_TESTS],
            "contracts.log",
        )
        run(
            "simulation_tests",
            [python, "-m", "pytest", "-q", "-m", "slow", *_SIM_TESTS],
            "simulation_contracts.log",
        )
        for backend in ("mujoco", "mjwarp"):
            run(
                f"collection_parity_{backend}",
                [
                    python,
                    str(Path(__file__).with_name("_verify_reference.py").resolve()),
                    "--backend",
                    backend,
                    "--output",
                    str(output / f"parity_{backend}"),
                    *(["--shared-native-step"] if backend == "mjwarp" else []),
                ],
                f"parity_{backend}.log",
            )
        if not args.verify_only:
            run("training", train, "training.log")
            run(
                "evaluation",
                [
                    str(root / ".venv/bin/sharpa-eval"),
                    "--algo",
                    "supermimic",
                    "--sim",
                    "mjwarp",
                    "--checkpoint",
                    str(output / "training/teacher_final.pt"),
                    "--output",
                    str(output / "evaluation.json"),
                ],
                "evaluation.log",
            )
        if _source(root) != source:
            raise RuntimeError("Upstream sources changed during the run")
        state["phase"] = "verified" if args.verify_only else "reference_evaluated"
        state["finished_unix"] = time.time()
        save()
        print(f"[{state['phase']}] {output / 'status.json'}", flush=True)
    except BaseException as exc:
        state["failed_phase"] = state["phase"]
        state["phase"] = "failed"
        state["error"] = repr(exc)
        save()
        raise


if __name__ == "__main__":
    main()
