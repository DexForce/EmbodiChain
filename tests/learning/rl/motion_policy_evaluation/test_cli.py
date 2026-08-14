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

from __future__ import annotations

import importlib

import pytest
import torch

from embodichain.learning.rl.motion_policy_evaluation.cli import (
    _resolve_input,
    _validate_native_options,
    parse_args,
)
from embodichain.learning.rl.motion_policy_evaluation.manifest import (
    write_run_manifest,
)


def test_run_uses_manifest_profile_and_latest_checkpoint(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    write_run_manifest(
        run,
        train_config=train,
        latest_checkpoint=checkpoint,
        motion_profile="example-motion",
    )

    resolved = _resolve_input(parse_args((str(run),)))

    assert resolved.profile == "example-motion"
    assert resolved.checkpoint == checkpoint
    assert resolved.selected_checkpoint == "latest"


def test_original_task_overrides_manifest_profile(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    write_run_manifest(
        run,
        train_config=train,
        latest_checkpoint=checkpoint,
        motion_profile="example-motion",
    )

    resolved = _resolve_input(parse_args((str(run), "--original-task")))

    assert resolved.profile is None
    assert resolved.configs["train"] == run / "configs" / "train.yaml"


def test_explicit_configs_override_manifest_configs(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    gym = tmp_path / "gym.yaml"
    gym.write_text("id: Original\n", encoding="utf-8")
    write_run_manifest(
        run,
        train_config=train,
        gym_config=gym,
        latest_checkpoint=checkpoint,
    )
    replacement_train = tmp_path / "replacement-train.yaml"
    replacement_train.write_text("trainer: {}\n", encoding="utf-8")
    replacement_gym = tmp_path / "replacement-gym.yaml"
    replacement_gym.write_text("id: Replacement\n", encoding="utf-8")

    resolved = _resolve_input(
        parse_args(
            (
                str(run),
                "--config",
                str(replacement_train),
                "--gym-config",
                str(replacement_gym),
            )
        )
    )

    assert resolved.configs["train"] == replacement_train
    assert resolved.configs["gym"] == replacement_gym


def test_viewer_defaults_to_hybrid_without_a_time_limit():
    args = parse_args(
        (
            "--profile",
            "example-motion",
            "--checkpoint",
            "policy.pt",
            "--viewer",
        )
    )

    assert args.renderer == "hybrid"
    assert args.physics_backend is None
    assert args.control_steps is None
    assert args.duration is None


def test_run_without_profile_selects_native_task_evaluation(tmp_path, monkeypatch):
    cli_module = importlib.import_module(
        "embodichain.learning.rl.motion_policy_evaluation.cli"
    )
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "policy.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"model_state_dict": {}}, checkpoint)
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    write_run_manifest(
        run,
        train_config=train,
        latest_checkpoint=checkpoint,
    )
    expected = tmp_path / "evaluation.json"
    received = {}

    def fake_native(args, resolved):
        received["args"] = args
        received["resolved"] = resolved
        return expected

    monkeypatch.setattr(cli_module, "discover_task_packages", lambda: None)
    monkeypatch.setattr(cli_module, "execute_init_hooks", lambda: None)
    monkeypatch.setattr(cli_module, "_run_native_task", fake_native)

    report = cli_module.run(parse_args((str(run), "--viewer")))

    assert report == expected
    assert received["resolved"].profile is None
    assert received["resolved"].checkpoint == checkpoint
    assert received["args"].viewer is True


def test_explicit_native_checkpoint_requires_training_config(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")

    args = parse_args(("--checkpoint", str(checkpoint)))

    with pytest.raises(
        ValueError,
        match="--config is required for a native EmbodiChain checkpoint",
    ):
        _resolve_input(args)


@pytest.mark.parametrize(
    "arguments, option",
    [
        (("--command", "0.5"), "--command"),
        (("--physics-backend", "default"), "--physics-backend"),
        (("--scene-config", "classic"), "--scene-config"),
        (("--cache-dir", "cache"), "--cache-dir"),
        (("--offline",), "--offline"),
        (("--resource-root", "resources"), "--resource-root"),
    ],
)
def test_native_task_rejects_profile_only_options(arguments, option):
    args = parse_args(
        (
            "--checkpoint",
            "policy.pt",
            "--config",
            "train.yaml",
            *arguments,
        )
    )

    with pytest.raises(ValueError, match=option):
        _validate_native_options(args)
