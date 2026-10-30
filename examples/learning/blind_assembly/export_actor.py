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
"""Export a native temporal checkpoint's actor without simulator dependencies.

The artifact contains normalization and deterministic action means, excluding
critic and contact decoder. Exporting is not a task-performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from embodichain.learning.rl.policies import build_policy
from embodichain.utils.utility import load_config


def main() -> None:
    """Restore an explicit native checkpoint and verify its traced actor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config", type=Path, required=True, help="Saved run/configs/train.yaml."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    attributes = checkpoint["metadata"]["policy_attributes"]
    config = load_config(str(args.config))["policy"]
    if config["name"] != "proprioceptive_actor_critic":
        raise ValueError(
            "This example exports the native temporal assembly policy only."
        )
    policy = build_policy(
        config,
        attributes["obs_dim"],
        attributes["action_dim"],
        torch.device("cpu"),
        critic_obs_space=attributes["critic_obs_dim"],
    )
    policy.load_state_dict(checkpoint["policy"])
    actor = policy.export_actor().cpu().eval()
    torch.manual_seed(0)
    sample = torch.randn(3, attributes["obs_dim"])
    with torch.inference_mode():
        reference = actor(sample)
        traced = torch.jit.trace(actor, torch.zeros(1, attributes["obs_dim"]))
        actual = traced(sample)
        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(args.output))
    manifest = {
        "qualification": "export_verified_task_performance_unqualified",
        "num_updates": checkpoint["num_updates"],
        "global_step": checkpoint["global_step"],
        "actor_path": str(args.output),
        "actor_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "checkpoint_path": str(args.checkpoint),
        "config_path": str(args.config),
        "input_shape": ["batch", attributes["obs_dim"]],
        "output_shape": ["batch", attributes["action_dim"]],
        "action_semantics": "normalized joint-position increments; apply task action scales and limits",
        "max_export_error": float((reference - actual).abs().max()),
        "assembly_success_rate": None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
