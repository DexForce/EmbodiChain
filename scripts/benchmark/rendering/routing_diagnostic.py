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

"""Launch a separate untimed routing proof against an original frozen run.

Run this new driver by absolute path; its worker imports the original worktree's
renderer and capture code. ``--prepare-only`` writes the frozen proof inputs and
command without creating a simulator. No original run or source file is changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from typing import Any

if __package__:
    from . import frozen_source as frozen
else:
    import frozen_source as frozen

__all__ = ["main", "prepare_routing_diagnostic"]


def _gpu_index(command: list[str]) -> int:
    try:
        return int(command[command.index("--gpu-id") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError(
            "Original worker command has no unambiguous GPU index"
        ) from error


def prepare_routing_diagnostic(
    source_worktree: Path,
    source_commit: str,
    experiment_root: Path,
    run_id: str,
    output: Path,
    *,
    gpu_id: int | None = None,
    stationary_captures: int = 3,
    settle_captures: int = 2,
    sample_edge: int = 64,
    persist_full: bool = False,
) -> dict[str, Any]:
    """Freeze original inputs and a source-pinned command without GPU imports.

    Args:
        source_worktree: Clean original checkout that produced the timing row.
        source_commit: Explicit original commit, checked against the timed record.
        experiment_root: Original formal artifact root; read only.
        run_id: Exact completed run identifier.
        output: Fresh proof directory outside the original source tree.
        gpu_id: Optional selected physical index; observed UUID must still match.
        stationary_captures: Prespecified stationary noise-control count.
        settle_captures: Additional moved/restored captures; none are discarded.
        sample_edge: Sample grid axis bound, retaining every exposure.
        persist_full: Save full arrays in addition to compact samples.

    Returns:
        Launch config and command. This function never constructs an adapter.
    """
    source_worktree, output = Path(source_worktree).resolve(), Path(output).resolve()
    if output.is_relative_to(source_worktree) or output.is_relative_to(
        Path(experiment_root).resolve()
    ):
        raise ValueError(
            "Routing proof output must be outside the frozen original tree"
        )
    for value, minimum in (
        (stationary_captures, 2),
        (settle_captures, 0),
        (sample_edge, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError("Invalid prespecified routing sample counts")
    original = frozen.load_original_run(experiment_root, run_id)
    source = frozen.verify_source_worktree(original, source_worktree, source_commit)
    output.mkdir(parents=True, exist_ok=False)
    binding, config = frozen.freeze_original_run(original, source, output)
    command = original["row"].get("command")
    if not isinstance(command, list) or not command or not Path(command[0]).is_file():
        raise ValueError("Recorded original interpreter is unavailable")
    gpu_id = _gpu_index(command) if gpu_id is None else gpu_id
    if type(gpu_id) is not int or gpu_id < 0:
        raise ValueError("GPU index must be a nonnegative integer")
    directory = Path(__file__).resolve().parent
    proof_sources = []
    for name in (
        "routing_diagnostic.py",
        "routing_worker.py",
        "routing_proof.py",
        "frozen_source.py",
    ):
        path = directory / name
        copied = output / "proof_source" / name
        copied.parent.mkdir(exist_ok=True)
        shutil.copyfile(path, copied)
        digest = frozen.sha256(path)
        if frozen.sha256(copied) != digest:
            raise ValueError("Follow-up source changed while freezing")
        proof_sources.append(
            {
                "path": str(path),
                "sha256": digest,
                "snapshot": frozen.artifact(output, copied, "routing_producer_source"),
            }
        )
    payload = {
        "schema_version": 1,
        "kind": "original_source_routing_diagnostic",
        "source": source,
        "original_binding": binding,
        "worker_config": config,
        "gpu_id": gpu_id,
        "expected_renderer_controls": frozen.renderer_control_signature(
            original["row"]["renderer"], binding["backend"]
        ),
        "probe_policy": {
            "stationary_captures": stationary_captures,
            "settle_captures": settle_captures,
            "sample_edge": sample_edge,
            "persist_full": persist_full,
        },
        "proof_sources": proof_sources,
        "throughput_claim": False,
    }
    config_path = frozen.write_json(output / "routing_config.json", payload)
    child = [
        str(Path(command[0]).absolute()),
        str(directory / "routing_worker.py"),
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    environment = frozen.original_environment(
        source_worktree,
        Path(command[0]),
        Path(source["isaaclab_root"]) if source.get("isaaclab_root") else None,
    )
    frozen.write_json(
        output / "launch.json",
        {
            "command": child,
            "source_commit": source["commit"],
            "config_sha256": frozen.sha256(config_path),
            "execution_started": False,
        },
    )
    return {
        "config": payload,
        "command": child,
        "environment": environment,
        "output": output,
    }


def main() -> int:
    """Prepare or execute one separate routing proof without modifying timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-worktree", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--stationary-captures", type=int, default=3)
    parser.add_argument("--settle-captures", type=int, default=2)
    parser.add_argument("--sample-edge", type=int, default=64)
    parser.add_argument("--full-samples", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=900)
    args = parser.parse_args()
    prepared = prepare_routing_diagnostic(
        args.source_worktree,
        args.source_commit,
        args.experiment_root,
        args.run_id,
        args.output,
        gpu_id=args.gpu_id,
        stationary_captures=args.stationary_captures,
        settle_captures=args.settle_captures,
        sample_edge=args.sample_edge,
        persist_full=args.full_samples,
    )
    if args.prepare_only:
        return 0
    output = prepared["output"]
    completed = None
    error = None
    try:
        with (output / "worker.log").open("w") as stream:
            completed = subprocess.run(
                prepared["command"],
                cwd=args.source_worktree,
                env=prepared["environment"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout_s,
            )
    except subprocess.TimeoutExpired as exception:
        error = f"worker_timeout: {exception}"
    receipt = {
        "returncode": completed.returncode if completed is not None else None,
        "error": error,
        "throughput_claim": False,
    }
    frozen.write_json(output / "process_exit.json", receipt)
    result_path = output / "result.json"
    if not result_path.exists():
        return 1
    result = frozen.read_json(result_path)
    return (
        0
        if completed is not None
        and completed.returncode == 0
        and result.get("status") == "completed"
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
