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
"""Create a checksum-guarded, isolated MuJoCo Warp EPA horizon overlay.

Run: python scripts/patches/apply_mujoco_warp_epa_horizon.py --output /tmp/mjw-epa128
The installed dependency is read only. No simulator, Torch or Warp is imported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

__all__: list[str] = []

_VERSION = "3.12.0"
_PATCH_NAME = "mujoco_warp_epa_horizon.patch"
_SOURCE_SHA256 = {
    "_src/types.py": "a49625a8994db0d3b52dd34f98adac7f70882d473108c4e4793e4830e81ea71c",
    "_src/collision_convex.py": "9a030d1756b0c0cc6bc9df818011b0c44859bc45420cd37254df7087773bc85c",
    "_src/collision_flex.py": "40dbd903315a6005c169af4d1899fc61a1e01b2a5362737ca16de2e99c8ed151",
    "_src/collision_gjk.py": "fe2741a492f4e108a776aeb64c67dacf518845b16087209a6b9edda6367affdb",
}


def _sha256(path: Path) -> str:
    """Hash source bytes without importing the dependency."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_overlay(output: Path, source_package: Path | None = None) -> Path:
    """Validate the exact dependency, copy it, and apply the capacity patch.

    Args:
        output: New overlay root, which must not already exist.
        source_package: Optional explicit ``mujoco_warp`` package directory.

    Returns:
        Absolute overlay root to prepend to ``PYTHONPATH`` in a fresh process.
    """
    distribution = importlib.metadata.distribution("mujoco-warp")
    if distribution.version != _VERSION:
        raise RuntimeError(
            f"Expected mujoco-warp {_VERSION}; found {distribution.version}."
        )
    source = (
        Path(distribution.locate_file("mujoco_warp"))
        if source_package is None
        else source_package
    ).resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Overlay output must be new: {output}")
    if source.name != "mujoco_warp" or not (source / "__init__.py").is_file():
        raise ValueError(f"Expected a mujoco_warp package directory: {source}")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Overlay and source package must be separate directories.")
    source_hashes = {}
    for relative, expected in _SOURCE_SHA256.items():
        actual = _sha256(source / relative)
        if actual != expected:
            raise RuntimeError(
                f"Source SHA-256 mismatch for {relative}: expected {expected}, got {actual}."
            )
        source_hashes[relative] = actual
    original = (source / "_src/types.py").read_bytes()
    before, after = b"MJ_MAX_EPAHORIZON = 24\n", b"MJ_MAX_EPAHORIZON = 128\n"
    if original.count(before) != 1:
        raise RuntimeError("Expected exactly one unpatched EPA horizon constant.")
    patch = Path(__file__).with_name(_PATCH_NAME).resolve()
    if shutil.which("patch") is None:
        raise RuntimeError(
            "The standard patch command is required to apply this overlay."
        )
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(
            source,
            output / "mujoco_warp",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        command = [
            "patch",
            "--batch",
            "--forward",
            "-p1",
            "-d",
            str(output),
            "-i",
            str(patch),
        ]
        subprocess.run(
            [*command, "--dry-run"], check=True, capture_output=True, text=True
        )
        subprocess.run(command, check=True, capture_output=True, text=True)
        target = output / "mujoco_warp/_src/types.py"
        if target.read_bytes() != original.replace(before, after):
            raise RuntimeError(
                "Patched source differs from the single intended constant change."
            )
        for relative, expected in source_hashes.items():
            if _sha256(source / relative) != expected:
                raise RuntimeError(
                    "Source package changed while the overlay was prepared."
                )
            if (
                relative != "_src/types.py"
                and _sha256(output / "mujoco_warp" / relative) != expected
            ):
                raise RuntimeError(f"Unmodified algorithm source changed: {relative}")
        provenance = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "distribution": "mujoco-warp",
            "version": distribution.version,
            "source_package": str(source),
            "overlay_root": str(output),
            "source_sha256": source_hashes,
            "patched_types_sha256": _sha256(target),
            "patch_sha256": _sha256(patch),
            "epa_horizon_before": 24,
            "epa_horizon_after": 128,
            "additional_convex_horizon_bytes_per_naccdmax": (128 - 24) * 4,
            "qualification": "Capacity allocation patch only; contact/overflow and task physics require separate native qualification.",
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-package", type=Path)
    args = parser.parse_args()
    overlay = apply_overlay(args.output, args.source_package)
    print(f"Overlay ready: {overlay}")
    print(
        "Prepend this directory to PYTHONPATH for a fresh process; inspect provenance.json."
    )
