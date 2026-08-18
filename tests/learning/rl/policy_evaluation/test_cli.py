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

from embodichain.learning.rl.policy_evaluation.cli import (
    _resolve_input,
    _validate_native_options,
    parse_args,
)
from embodichain.learning.rl.policy_evaluation.manifest import write_run_manifest


def _run(tmp_path):
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
    )
    return run, checkpoint


def test_run_defaults_to_latest_checkpoint(tmp_path):
    run, checkpoint = _run(tmp_path)

    resolved = _resolve_input(parse_args((str(run),)))

    assert resolved.checkpoint == checkpoint
    assert resolved.requested_checkpoint == "latest"
    assert resolved.selected_checkpoint == "latest"


@pytest.mark.parametrize(
    "arguments, handler",
    [
        ((), "_run_native_headless"),
        (("--viewer",), "_run_native_viewer"),
        (("--profile", "example"), "_run_profile"),
    ],
)
def test_cli_routes_one_command_to_the_selected_evaluation(
    tmp_path,
    monkeypatch,
    arguments,
    handler,
):
    module = importlib.import_module("embodichain.learning.rl.policy_evaluation.cli")
    run, _checkpoint = _run(tmp_path)
    expected = tmp_path / "evaluation.json"
    calls = []

    monkeypatch.setattr(module, "discover_task_packages", lambda: None)
    monkeypatch.setattr(module, "execute_init_hooks", lambda: None)
    for name in ("_run_native_headless", "_run_native_viewer", "_run_profile"):
        monkeypatch.setattr(
            module,
            name,
            lambda args, resolved, name=name: calls.append(name) or expected,
        )

    report = module.run(parse_args((str(run), *arguments)))

    assert report == expected
    assert calls == [handler]


def test_explicit_checkpoint_requires_training_config(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="--config is required"):
        _resolve_input(parse_args(("--checkpoint", str(checkpoint))))


def test_native_options_keep_profile_and_viewer_inputs_explicit():
    profile_args = parse_args(
        (
            "--checkpoint",
            "policy.pt",
            "--config",
            "train.yaml",
            "--command",
            "0.5",
        )
    )
    viewer_args = parse_args(
        (
            "--checkpoint",
            "policy.pt",
            "--config",
            "train.yaml",
            "--control-steps",
            "10",
        )
    )

    with pytest.raises(ValueError, match="--command requires --profile"):
        _validate_native_options(profile_args)
    with pytest.raises(ValueError, match="--control-steps.*require --viewer"):
        _validate_native_options(viewer_args)
