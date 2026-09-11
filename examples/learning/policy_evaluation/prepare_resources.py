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

"""Prepare the pinned public policy and assets for the ANYmal-C example."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from pathlib import Path

__all__ = ["main", "prepare_resources"]

UPSTREAM_URL = "https://github.com/newton-physics/newton-assets.git"
UPSTREAM_REVISION = "261cd1f429619d8ef4f546bd788ab9dea906b5e1"
MODEL_RELATIVE_PATH = Path("anybotics_anymal_c/rl_policies/mjw_anymal.pt")
POLICY_CONFIG_RELATIVE_PATH = Path("anybotics_anymal_c/rl_policies/anymal.yaml")
POLICY_LICENSE_RELATIVE_PATH = Path("anybotics_anymal_c/rl_policies/LICENSE")
ROBOT_LICENSE_RELATIVE_PATH = Path("anybotics_anymal_c/LICENSE")
ROBOT_RELATIVE_PATH = Path("anybotics_anymal_c/urdf/anymal.urdf")
MESH_RELATIVE_PATH = Path("anybotics_anymal_c/meshes/base.dae")
SHA256 = {
    MODEL_RELATIVE_PATH: "00765c1c07e497be3825672b05f9cefff9238f2df72fb0bcb5ac9541155b945f",
    POLICY_CONFIG_RELATIVE_PATH: "b5a463ac418c7f40ebe494c7bcf0d8031f021db70a0625dbcc28a718de8ee817",
    POLICY_LICENSE_RELATIVE_PATH: "59899c6091b540582ed617e8eeaac4919dc985ccfc35459ee9752b699be5205b",
    ROBOT_LICENSE_RELATIVE_PATH: "cef384faae108293b03b5e16a00bc3db8212d44575f69df6296438a3f901700b",
    ROBOT_RELATIVE_PATH: "d6bd20292cdd4873ffdeeb6f8ca3f96c4a0096565d78d8b6204f6edf0d19fb83",
    MESH_RELATIVE_PATH: "785bea9b33831f8c741fc0ca070162e73cbf560ea9b03c53abf8978be877fc48",
}


def prepare_resources(output: Path) -> tuple[Path, Path]:
    """Fetch and verify the pinned upstream files.

    Args:
        output: Cache directory that will contain the sparse Git checkout.

    Returns:
        The local checkpoint and resource-root paths.
    """
    output = output.expanduser().resolve()
    checkout = output / "upstream"
    _status("Preparing the ANYmal-C command policy and robot assets")
    _prepare_checkout(
        checkout,
        UPSTREAM_URL,
        UPSTREAM_REVISION,
        (
            f"/{MODEL_RELATIVE_PATH}",
            f"/{POLICY_CONFIG_RELATIVE_PATH}",
            f"/{POLICY_LICENSE_RELATIVE_PATH}",
            f"/{ROBOT_LICENSE_RELATIVE_PATH}",
            "/anybotics_anymal_c/urdf/**",
            "/anybotics_anymal_c/meshes/**",
        ),
    )

    checkpoint = checkout / MODEL_RELATIVE_PATH
    for relative, digest in SHA256.items():
        _verify_sha256(checkout / relative, digest)
    _status("Resource verification completed")
    return checkpoint, checkout


def main() -> None:
    """Prepare resources and print the paths used by the evaluation command."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / ".cache/embodichain/examples/anymal_c_velocity",
        help="Directory used for the pinned upstream checkout",
    )
    args = parser.parse_args()
    checkpoint, resource_root = prepare_resources(args.output)
    print(f"Checkpoint: {checkpoint}")
    print(f"Resource root: {resource_root}")


def _git(
    checkout: Path,
    *args: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(checkout), *args]
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=capture_output,
            env=environment,
            timeout=600,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Git command did not finish within 10 minutes: {' '.join(command)}"
        ) from error


def _git_output(checkout: Path, *args: str) -> str | None:
    try:
        return _git(checkout, *args, capture_output=True).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _prepare_checkout(
    checkout: Path,
    url: str,
    revision: str,
    includes: tuple[str, ...],
) -> None:
    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(
            f"Resource path exists but is not a Git checkout: {checkout}"
        )

    if checkout.exists():
        remote_url = _git_output(checkout, "remote", "get-url", "origin")
        if remote_url != url:
            raise RuntimeError(
                f"Resource checkout uses an unexpected remote: {remote_url}"
            )
        if _git_output(checkout, "rev-parse", "HEAD") == revision:
            tracked_changes = _git_output(
                checkout, "status", "--porcelain", "--untracked-files=no"
            )
            if tracked_changes == "":
                _status(f"Using cached revision {revision[:8]} from {checkout}")
                return
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        checkout.mkdir()
        _git(checkout, "init", "--quiet")
        _git(checkout, "remote", "add", "origin", url)

    _git(checkout, "sparse-checkout", "init", "--no-cone")
    _git(checkout, "sparse-checkout", "set", *includes)

    _status(f"Fetching revision {revision[:8]} from {url}")
    _git(
        checkout,
        "fetch",
        "--progress",
        "--filter=blob:none",
        "--depth",
        "1",
        "origin",
        revision,
    )

    _status(f"Checking out required files in {checkout}")
    _git(
        checkout,
        "-c",
        "advice.detachedHead=false",
        "checkout",
        "--progress",
        "--force",
        "--detach",
        "FETCH_HEAD",
    )
    actual_revision = _git_output(checkout, "rev-parse", "HEAD")
    if actual_revision != revision:
        raise RuntimeError(
            f"Checkout revision mismatch: expected {revision}, got {actual_revision}"
        )


def _status(message: str) -> None:
    print(f"[resources] {message}", flush=True)


def _verify_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
