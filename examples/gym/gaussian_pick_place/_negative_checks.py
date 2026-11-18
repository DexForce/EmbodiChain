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

"""Exercise offline rejection through the real CLI on isolated recording copies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
import yaml

from _prepare import HERE, digest, write_json

__all__ = ["main"]

CASES = {
    "empty_episode_inventory": "Episode file inventory",
    "missing_video_and_digest": "Episode file inventory",
    "nan_action": "non-finite actions",
    "infinite_action": "non-finite actions",
    "changed_configuration": "Configuration changed",
    "empty_configuration_inventory": "configuration inventory",
    "nan_target_with_updated_digest": "Invalid placement target",
}


def _copy_file(source: str, destination: str) -> str:
    # Untouched media can share storage. Every mutation below replaces its inode.
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def _replace_yaml(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False))
    temporary.replace(path)


def _mutate(directory: Path, case: str) -> None:
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text())
    episode = directory / run["episode"]
    marker_path = episode / "episode.json"
    marker = json.loads(marker_path.read_text())
    if case == "empty_episode_inventory":
        marker["files"] = {}
        write_json(marker_path, marker)
    elif case == "missing_video_and_digest":
        (episode / "cam_wrist.mp4").unlink()
        del marker["files"]["cam_wrist.mp4"]
        write_json(marker_path, marker)
    elif case in {"nan_action", "infinite_action"}:
        path = episode / "trajectory.pt"
        trajectory = torch.load(path, map_location="cpu", weights_only=False)
        trajectory["actions"][0, 0, 0] = (
            float("nan") if case == "nan_action" else float("inf")
        )
        temporary = path.with_suffix(".pt.pending")
        torch.save(trajectory, temporary)
        temporary.replace(path)
        # A self-consistent file hash must not hide an invalid action value.
        marker["files"]["trajectory.pt"] = digest(path)
        write_json(marker_path, marker)
    elif case == "changed_configuration":
        path = directory / "resolved" / f"env.{run['scene']}.yaml"
        config = yaml.safe_load(path.read_text())
        config["render_cfg"]["spp"] += 1
        _replace_yaml(path, config)
    elif case == "empty_configuration_inventory":
        run["configuration_hashes"] = {}
        write_json(run_path, run)
    elif case == "nan_target_with_updated_digest":
        relative = f"task_program/program.{run['scene']}.yaml"
        path = directory / "resolved" / relative
        config = yaml.safe_load(path.read_text())
        config["targets"]["rest_pose"]["position"][0] = float("nan")
        _replace_yaml(path, config)
        run["configuration_hashes"][relative] = digest(path)
        write_json(run_path, run)
    else:
        raise ValueError(f"Unknown negative case: {case}")


def _fingerprint(directory: Path) -> dict[str, str]:
    run = json.loads((directory / "run.json").read_text())
    episode = directory / run["episode"]
    marker = json.loads((episode / "episode.json").read_text())
    paths = [directory / "run.json", episode / "episode.json"]
    paths.extend(episode / name for name in marker["files"])
    paths.extend(directory / "resolved" / name for name in run["configuration_hashes"])
    return {path.relative_to(directory).as_posix(): digest(path) for path in paths}


def main() -> None:
    """Require a valid original and explicit CLI rejection of each damaged copy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, default=HERE / "_verify.py")
    args = parser.parse_args()
    source = args.run.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(source):
        parser.error("Acceptance output must be outside the original recording")
    output.mkdir(parents=True, exist_ok=False)
    before = _fingerprint(source)
    verifier = args.verifier.resolve(strict=True)
    report = {
        "source_run": str(source),
        "verifier": str(verifier),
        "verifier_sha256": digest(verifier),
        "status": "running",
        "cases": [],
    }
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for case in ("baseline", *CASES, "original_after_checks"):
        directory = source
        if case in CASES:
            directory = output / case
            shutil.copytree(source, directory, copy_function=_copy_file)
            _mutate(directory, case)
        result = subprocess.run(
            [sys.executable, "-B", str(verifier), str(directory)],
            env=env,
            capture_output=True,
            text=True,
        )
        (output / f"{case}.log").write_text(result.stdout + result.stderr)
        expected = CASES.get(case)
        passed = (
            result.returncode == 0
            if expected is None
            else result.returncode != 0 and expected in result.stderr
        )
        report["cases"].append(
            {"case": case, "passed": passed, "returncode": result.returncode}
        )
        write_json(output / "acceptance.json", report)
        print(f"{'PASS' if passed else 'FAIL'} {case}: exit {result.returncode}")
        if case == "baseline" and not passed:
            raise RuntimeError(
                "The original recording must pass before negative checks"
            )
    report["original_unchanged"] = before == _fingerprint(source)
    passed = report["original_unchanged"] and all(c["passed"] for c in report["cases"])
    report["status"] = "PASS" if passed else "FAIL"
    write_json(output / "acceptance.json", report)
    if not passed:
        raise RuntimeError(
            f"Offline rejection failed; inspect {output / 'acceptance.json'}"
        )


if __name__ == "__main__":
    main()
