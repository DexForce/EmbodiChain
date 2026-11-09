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

"""Normalize the online recipe and its saved pre-configuration checkpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_recipe(config: dict[str, Any]) -> dict[str, Any]:
    """Return trainer/policy/algorithm blocks and task-owned workflow settings."""
    result = deepcopy(config)
    if "trainer" in result:
        if set(result) != {"trainer", "policy", "algorithm", "workflow"}:
            raise ValueError(
                "Motion recipes require trainer, policy, algorithm and workflow."
            )
        return result
    workflow = result
    collection = workflow.pop("gym_config")
    trainer = {
        "learning_env": {
            "name": "MotionTrackingModelRL",
            "cfg": {"horizon": workflow.pop("horizon")},
        },
        "num_envs": workflow.pop("pi_batch"),
        "rollout_mode": "complete",
        "deterministic_actions": True,
    }
    actor_hidden = workflow.pop("actor_hidden")
    policy = {
        "name": "motion_tracking",
        "cfg": {
            "latent_dim": workflow.pop("latent"),
            "encoder_hidden": actor_hidden,
            "decoder_hidden": list(actor_hidden),
            "critic_hidden": workflow.pop("critic_hidden"),
            "amp_bfloat16": workflow["amp"],
        },
    }
    algorithm = {
        "name": "apg_actor_critic",
        "cfg": {
            "gamma": workflow.pop("gamma"),
            "gae_lambda": workflow.pop("lambda"),
            "max_grad_norm": 1.0,
            "optimizer": {
                "learning_rate": workflow["lr"],
                "kwargs": {"betas": [0.9, 0.95], "fused": True},
            },
            "critic_optimizer": {
                "learning_rate": workflow["lr"],
                "kwargs": {"fused": True},
            },
            "critic_epochs": workflow.pop("critic_epochs"),
            "critic_minibatches": workflow.pop("critic_minibatches"),
            "actor_grad_value_clip": 0.1,
            "update_critic_normalization": True,
        },
    }
    workflow["collection_env"] = collection
    workflow["evaluation_env"] = collection
    return dict(trainer=trainer, policy=policy, algorithm=algorithm, workflow=workflow)


def validate_recipe(config: dict[str, Any]) -> None:
    """Check the G1/SuperMimic composition before allocating its environment."""
    for path, block, expected in (
        (
            "trainer.learning_env",
            config["trainer"].get("learning_env"),
            "MotionTrackingModelRL",
        ),
        ("policy", config["policy"], "motion_tracking"),
        ("algorithm", config["algorithm"], "apg_actor_critic"),
    ):
        name = block.get("name") if isinstance(block, dict) else None
        if name != expected:
            raise ValueError(
                f"G1 motion training requires {path}.name={expected!r}; got {name!r}."
            )


def recipe_identity(config: dict[str, Any]) -> dict[str, Any]:
    """Select learning settings that must match before restoring any live state."""
    result = normalize_recipe(config)
    workflow = result["workflow"]
    for key in (
        "iterations",
        "eval_every",
        "evaluate_final",
        "save_every",
        "log_every",
        "compile_mode",
        "episode_context_in_info",
        "wm_checkpoint",
        "collection_env",
        "evaluation_env",
    ):
        workflow.pop(key, None)
    result["trainer"]["learning_env"].get("cfg", {}).pop("compile_mode", None)
    # The policy's resolved dimensions and numerical settings have their own identity.
    result.pop("policy")
    return result
