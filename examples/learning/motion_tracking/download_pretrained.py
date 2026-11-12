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

"""Download the G1 motion-tracking policy and world-model inference bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from huggingface_hub import HfApi, snapshot_download

_REPOSITORY = "DexForceAI/embodichain_model"
_BUNDLE = "neural_dynamics/motion_tracking/g1/v1"
_FILES = (
    "weights.pt",
    "manifest.json",
    "evaluation.json",
    "README.md",
    "NOTICE",
    "robot-LICENSE.txt",
    "configs/env.yaml",
    "configs/train.yaml",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub commit, tag or branch; resolved once to an immutable revision.",
    )
    args = parser.parse_args()
    revision = args.revision
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        revision = (
            HfApi(endpoint="https://huggingface.co", token=False)
            .model_info(_REPOSITORY, revision=revision)
            .sha
        )
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("The requested model revision did not resolve to a commit.")
    destination = args.output.expanduser().resolve() / revision
    snapshot_download(
        repo_id=_REPOSITORY,
        revision=revision,
        allow_patterns=[f"{_BUNDLE}/{name}" for name in _FILES],
        local_dir=destination,
        endpoint="https://huggingface.co",
        token=False,
    )
    bundle = destination / _BUNDLE
    missing = [name for name in _FILES if not (bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete model bundle: {missing}")
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format") != "embodichain-motion-inference-v1":
        raise ValueError("Unsupported motion-tracking inference bundle format.")
    source = {
        "repo_id": _REPOSITORY,
        "revision": revision,
        "bundle": _BUNDLE,
        "checkpoint": str(bundle / "weights.pt"),
    }
    (args.output.expanduser() / "download.json").write_text(
        json.dumps(source, indent=2) + "\n"
    )
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
