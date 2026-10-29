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
"""Simulator-free paths and pilot-only guard for the migrated DexUni scripts.

These adapters have NOT passed the revised experiment protocol. An explicit
pilot acknowledgement enables development runs, never formal qualification.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO_ROOT",
    "TECH_ROOT",
    "EXPERIMENTS",
    "PLATFORMS",
    "ASSETS",
    "OUTPUTS",
    "input_dir",
    "output_dir",
    "franka_usd",
    "frozen_robot_urdf",
    "require_pilot",
]

REPO_ROOT = Path(__file__).resolve().parents[6]
TECH_ROOT = REPO_ROOT / "scripts/benchmark/tech_reports"
EXPERIMENTS = TECH_ROOT / "physics/experiments/dexuni"
PLATFORMS = TECH_ROOT / "physics/platforms/dexuni"
ASSETS = TECH_ROOT / "assets/physics/dexuni"
OUTPUTS = TECH_ROOT.parents[2] / "outputs/benchmarks/tech_reports/dexuni"
_TASKS = {"s01c", "s01t", "s02", "s03"}


def input_dir(task: str) -> Path:
    """Return a local input snapshot, without downloading or creating files.

    Args:
        task: Migrated benchmark ID (s01c, s01t or s02).

    Returns:
        Task input directory under DEXUNI_INPUT_ROOT or the local pilot cache.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown migrated task: {task}")
    root = (
        Path(
            os.environ.get(
                "DEXUNI_INPUT_ROOT",
                "~/.cache/embodichain/report-assets/local/dexuni/pilot_v1",
            )
        )
        .expanduser()
        .resolve()
    )
    return root / task


def output_dir(task: str, override: str | Path | None = None) -> Path:
    """Resolve writable development output without overwriting source/archive.

    Args:
        task: Migrated benchmark ID.
        override: Optional explicit output root; relative paths use the cwd.

    Returns:
        Output directory. Creation remains the caller's responsibility.
    """
    if task not in _TASKS:
        raise ValueError(f"Unknown migrated task: {task}")
    path = (
        Path(override).expanduser()
        if override is not None
        else OUTPUTS / "development" / task
    ).resolve()
    protected = (
        TECH_ROOT / "assets",
        TECH_ROOT / "physics",
        TECH_ROOT / "reporting",
        TECH_ROOT / "core",
        TECH_ROOT / "loading",
        TECH_ROOT / "rendering",
        TECH_ROOT / "demonstrations",
        TECH_ROOT / "expert_generation",
        REPO_ROOT / "outputs/benchmarks/dexuni/pilot_v1",
        REPO_ROOT / "dexuni_benchmark",
    )
    if path == TECH_ROOT:
        raise ValueError(f"Output must not overwrite source or pilot archive: {path}")
    if any(path == root or root in path.parents for root in protected):
        raise ValueError(f"Output must not overwrite source or pilot archive: {path}")
    return path


def franka_usd() -> Path:
    """Return the explicitly selected USD or the migrated local pilot asset."""
    default = input_dir("s01c").parent / (
        "generated/fr3_franka_hand/fr3_franka_hand.dexsim/"
        "fr3_franka_hand.dexsim.usda"
    )
    return (
        Path(os.environ.get("DEXUNI_FRANKA_USD", str(default))).expanduser().resolve()
    )


def require_pilot() -> None:
    """Reject execution unless the caller acknowledges unqualified pilot code.

    DEXUNI_ALLOW_PILOT=1 does not attest any gate in PLAN.md. There is currently
    no formal-run switch; implement and validate those gates before adding one.
    """
    if os.environ.get("DEXUNI_ALLOW_PILOT") != "1":
        raise SystemExit(
            "Unqualified pilot adapter: control/timing/denominator/video gates "
            "are pending. Read physics/experiments/dexuni/PLAN.md. "
            "For development only, explicitly set DEXUNI_ALLOW_PILOT=1."
        )


def frozen_robot_urdf(inputs: Path, configured: str) -> str:
    """Prefer the relocated frozen Franka asset beside a task input snapshot.

    Args:
        inputs: The task input directory (s01c, s01t or s02).
        configured: Historical robot path retained in the frozen configuration.

    Returns:
        Restored robot path if present; otherwise the caller's original path.
    """
    local = inputs.parent / "franka_emika_panda" / Path(configured).name
    return str(local.resolve()) if local.is_file() else configured
