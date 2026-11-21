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

"""Render one trusted ACT checkpoint's PushCube rollout to an MP4 file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ._environment import close_env, make_env
from .evaluate import _load_policy, _rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=300000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    env = None
    try:
        model, update = _load_policy(
            args.checkpoint, args.device, panel="overfit", seeds=[args.seed]
        )
        env = make_env(args.device, camera=True)
        report = _rollouts(
            env,
            model,
            mode="policy",
            seeds=[args.seed],
            device=args.device,
            video_dir=args.output,
        )
        report["checkpoint"], report["update"] = str(args.checkpoint.resolve()), update
        (args.output / "rollout.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Rollout saved in {args.output}", flush=True)
    finally:
        close_env(env)


if __name__ == "__main__":
    main()
