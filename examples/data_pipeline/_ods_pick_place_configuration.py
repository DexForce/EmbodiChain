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

"""Bind the generic two-rate ODS plan to the fixed PickPlace learner."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from embodichain.data_pipeline.configuration import (
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
)


def load_configuration(path: Path) -> dict[str, Any]:
    """Validate a complete measurement/profile input and preserve its identity."""
    raw = path.read_bytes()
    inputs = json.loads(raw)
    required = {
        "accepted_episodes_per_second",
        "training_chunks_per_second",
        "profile",
        "runtime",
        "binding",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ValueError(f"Auto-config input must contain exactly {sorted(required)}")
    profile = ODSReuseProfileCfg(**inputs["profile"])
    runtime = ODSRuntimeCfg(**inputs["runtime"])
    result = configure_ods(
        inputs["accepted_episodes_per_second"],
        inputs["training_chunks_per_second"],
        profile=profile,
        runtime=runtime,
    )
    return {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input": inputs,
        "configuration": asdict(result),
    }


def apply_configuration(args: argparse.Namespace) -> None:
    """Resolve pipeline settings before creating a producer or model."""
    report = load_configuration(args.auto_config_input)
    expected = {
        "task": "PickPlace-Franka-v1",
        "policy": args.policy,
        "prediction_horizon": args.prediction_horizon,
        "action_horizon": args.action_horizon,
        "pose_ik": args.pose_ik,
    }
    if report["input"]["binding"] != expected:
        raise ValueError(f"Auto-config binding must match {expected}")
    if args.mode != "online" or args.policy != "chunk-pose" or args.stage != "showcase":
        raise ValueError(
            "Auto-config currently requires live online chunk-pose showcase"
        )
    if args.offline_data or args.export_data or args.smoke:
        raise ValueError(
            "Auto-config cannot combine with offline/export data or smoke overrides"
        )
    if args.usage_schedule != "fixed":
        raise ValueError(
            "Auto-config owns the schedule; do not combine with anchor-budget pacing"
        )
    plan = report["configuration"]
    for name in ("batch_size", "producer_envs", "buffer_size"):
        supplied = getattr(args, name)
        if supplied is not None and supplied != plan[name]:
            raise ValueError(
                f"Explicit {name} conflicts with the measured auto-configuration"
            )
    args.batch_size = plan["batch_size"]
    args.producer_envs = plan["producer_envs"]
    args.buffer_size = plan["buffer_size"]
    args.usage_statistics = True
    # Keep capacity measurement free of evaluator contention and always save
    # the final trained policy even when quotas end before eval_every.
    args.defer_evaluation = True
    args.auto_configuration = report
    if args.max_generated_episodes is None:
        args.max_generated_episodes = args.buffer_size * args.rounds
    # This is a finite-data ceiling, not a convergence recommendation. Source
    # retirement and incomplete final batches can reduce actual updates.
    quota_updates = (
        args.max_generated_episodes * plan["episode_sample_budget"] // args.batch_size
    )
    requested_updates = args.rounds * args.updates_per_round
    report["training_budget"] = {
        "accepted_episode_cap": args.max_generated_episodes,
        "assigned_chunk_ceiling": args.max_generated_episodes
        * plan["episode_sample_budget"],
        "quota_update_ceiling": quota_updates,
        "requested_update_cap": requested_updates,
        "maximum_additional_updates": min(quota_updates, requested_updates),
        "scope": "New-run upper bound; retirement and partial batches reduce it; no learning-quality guarantee",
    }
