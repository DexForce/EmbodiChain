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
"""Prepare an isolated, audited 20-DOF derivative of the pinned learner.

The original checkout is read-only. Dimension changes live in a run-owned
dependency snapshot; native physics never instantiates its Sharpa environment.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

__all__: list[str] = []


def _activate(upstream: Path, output: Path) -> Path:
    if any(n.startswith("sharpa_rl_unilab") for n in sys.modules):
        raise RuntimeError("Activate the Wuji dependency before importing the learner")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=upstream, text=True
    ).strip()
    if revision != "9816885a3b1fbdb947db71e72c27ed75effcbf9f":
        raise ValueError("Expected the pinned learner revision")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=upstream,
        text=True,
    ).strip():
        raise ValueError("Pinned learner checkout must remain clean")
    package = output / "src/sharpa_rl_unilab"
    patches = {
        "tasks/sharpa_inhand/protocol.py": [
            ("ACTION_DIM = NUM_HAND_JOINTS", "ACTION_DIM = 20"),
            (
                'CONTRACT_VERSION = "sharpa-hora-v2"',
                'CONTRACT_VERSION = "wuji-palm-up-hora20-v2"',
            ),
        ],
        "algos/supermimic/networks.py": [
            (" + 22", " + 20"),
            (" * 22", " * 20"),
            ("(batch, 22)", "(batch, 20)"),
        ],
        "algos/supermimic/learner.py": [
            ("            55,", "            51,"),
            (":147", ":135"),
            ("147:", "135:"),
        ],
        "algos/supermimic/replay.py": [("22", "20"), ("55", "51")],
        "training/supermimic_mdp.py": [
            ("CONTEXT_DIM = 55", "CONTEXT_DIM = 51"),
            ("CONTEXT_SPLITS = (22, 22,", "CONTEXT_SPLITS = (20, 20,"),
        ],
    }
    receipt = {
        "base_revision": revision,
        "action_dim": 20,
        "frame_dim": 45,
        "actor_dim": 135,
        "critic_dim": 162,
        "state_dim": 83,
        "context_dim": 51,
        "patches": {},
    }
    if not package.exists():
        package.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            upstream / "src/sharpa_rl_unilab",
            package,
            ignore=shutil.ignore_patterns("assets", "__pycache__", "*.pyc"),
        )
        (package / "assets").mkdir()
        shutil.copy2(
            upstream / "src/sharpa_rl_unilab/assets/__init__.py",
            package / "assets/__init__.py",
        )
        shutil.copy2(upstream / "LICENSE", output / "UPSTREAM_LICENSE")
        for name, substitutions in patches.items():
            original = (package / name).read_text()
            modified = original
            counts = []
            for before, after in substitutions:
                count = modified.count(before)
                if not count:
                    raise ValueError(
                        f"Expected dimension site missing: {name}: {before}"
                    )
                counts.append({"before": before, "after": after, "count": count})
                modified = modified.replace(before, after)
            (package / name).write_text(modified)
            receipt["patches"][name] = {
                "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "adapted_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                "substitutions": counts,
            }
        (output / "adaptation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    else:
        saved = json.loads((output / "adaptation.json").read_text())
        for name, item in saved["patches"].items():
            if (
                hashlib.sha256((package / name).read_bytes()).hexdigest()
                != item["adapted_sha256"]
            ):
                raise ValueError("Adapted learner source changed")
    sys.path.insert(0, str(package.parent.resolve()))
    sys.path.append(str(Path(__file__).resolve().parent.parent / "sharpa_inhand"))
    return package.parent
