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
import json
from types import SimpleNamespace

import pytest
import torch

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


def test_named_run_reports_the_host_owned_policy_outcomes(tmp_path, monkeypatch):
    from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv
    from embodichain.lab.gym.envs.episode_context import EpisodeContext
    from embodichain.lab.gym.envs.managers import (
        ConditionCfg,
        RLTaskCfg,
        TerminationManager,
    )
    from embodichain.lab.gym.envs.managers.conditions import evaluation_condition
    from embodichain.learning.rl.policy_evaluation import cli

    host = object.__new__(EmbodiedEnv)
    host._num_envs = 3
    host.sim = SimpleNamespace(device=torch.device("cpu"))
    host.cfg = SimpleNamespace(seed=79, task_program=None)
    host.episode_context = EpisodeContext(3)
    host._task_initial_states = [{}, {}, {}]
    host._recording_provenance = {}
    host._elapsed_steps = torch.ones(3, dtype=torch.long)
    host.positions = torch.tensor([[0.0, 0, 0], [0, 0, 2], [0, 0, 0]])
    manager = TerminationManager(
        RLTaskCfg(
            success={
                "low": ConditionCfg(
                    func=evaluation_condition,
                    params={
                        "sources": {
                            "pose": {"attribute": "positions", "kind": "position"}
                        },
                        "rules": {
                            "low": {
                                "kind": "below_height",
                                "bindings": {"position": "pose"},
                                "params": {"height": 1.0},
                            }
                        },
                        "root": "low",
                    },
                )
            }
        ),
        host,
    )
    host.termination_manager = manager
    observation = torch.zeros(3, 1)

    def reset(*, seed):
        assert seed == host.cfg.seed
        manager.reset()
        host.episode_context._begin_episode([0, 1, 2])
        host.episode_context._finish_initialization([0, 1, 2])
        host._demo_episode_metadata = [
            host._new_demo_episode_metadata(i) for i in range(3)
        ]
        return observation, {}

    def step(action):
        manager.compute()
        info = {"success": manager.task_state()[0], "task_outcome": manager.snapshot()}
        terminated, truncated = manager.terminated, manager.truncated
        rows = terminated.nonzero().flatten()
        assert rows.tolist() == [0, 2]
        host._seal_step_outcomes(rows, terminated, truncated, info)
        manager.reset(rows)
        assert manager.snapshot()["steps"].tolist() == [0, 1, 0]
        return observation, torch.ones(3), terminated, truncated, info

    class Policy(torch.nn.Module):
        def get_action(self, inputs, deterministic=False):
            assert deterministic
            return {"action": torch.zeros(3, 1)}

    run, checkpoint = _run(tmp_path / "shared-rules-experiment")
    args = parse_args((str(run), "--episodes", "1"))
    resolved = _resolve_input(args)
    runtime = SimpleNamespace(
        env=SimpleNamespace(num_envs=3, reset=reset, step=step),
        policy=Policy(),
        env_id="SharedRules-v1",
        close=lambda: None,
    )
    native = cli.NativeRuntime(
        runtime, host.device, host.device, 79, "hybrid", False, {}
    )
    monkeypatch.setattr(cli, "_build_native_runtime", lambda *args, **kwargs: native)
    report_path = cli._run_native_headless(args, resolved)
    report = json.loads(report_path.read_text())
    assert report["inputs"]["run"] == str(run)
    assert report["inputs"]["checkpoint"]["path"] == str(checkpoint)
    (outcome,) = report["result"]["episode_outcomes"]
    assert outcome == host.get_demo_episode_metadata(0)["episode_outcome"]
    assert outcome["identity"]["seed"] == 79
    assert outcome["acceptance"]["accepted"] is True
    assert outcome["physical"]["conditions"]["conditions"]["success.low"][
        "evaluation_identity"
    ]
    assert (
        host.get_demo_episode_metadata(2)["episode_outcome"]["identity"]["env_id"] == 2
    )
    repeated = json.loads(cli._run_native_headless(args, resolved).read_text())
    (second_outcome,) = repeated["result"]["episode_outcomes"]
    assert second_outcome["physical"] == outcome["physical"]
    assert (
        second_outcome["identity"]["episode_uuid"]
        != outcome["identity"]["episode_uuid"]
    )


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


@pytest.mark.parametrize("renderer", ("hybrid", "fast-rt", "rt"))
def test_cli_accepts_dexsim_renderer_names(renderer):
    args = parse_args(("--renderer", renderer))

    assert args.renderer == renderer


@pytest.mark.parametrize("keymap", ("wasd", "arrows"))
def test_cli_accepts_dexsim_viewer_keymaps(keymap):
    args = parse_args(("--keymap", keymap))

    assert args.keymap == keymap


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

    with pytest.raises(ValueError, match="--command.*require --viewer"):
        _validate_native_options(profile_args)
    with pytest.raises(ValueError, match="--control-steps.*require --viewer"):
        _validate_native_options(viewer_args)


@pytest.mark.parametrize(
    "arguments",
    [
        ("some-run",),
        ("--checkpoint", "best"),
        ("--profile", "example"),
        ("--config", "train.yaml"),
        ("--gym-config", "env.yaml"),
    ],
)
def test_pretrained_conflicts_are_rejected_before_download(arguments, monkeypatch):
    from embodichain.data.assets import policy_assets

    def fail(*args, **kwargs):
        pytest.fail("conflicting input must not download")

    monkeypatch.setattr(policy_assets, "download_pretrained_policy", fail)
    with pytest.raises(ValueError, match="--pretrained cannot be combined"):
        _resolve_input(parse_args(("--pretrained", "example", *arguments)))


def test_revision_requires_pretrained():
    with pytest.raises(ValueError, match="--revision requires --pretrained"):
        _resolve_input(parse_args(("--revision", "main")))


@pytest.mark.parametrize("option", ("--offline", "--physics-backend"))
def test_pretrained_keeps_profile_only_options_explicit(option):
    args = ["--pretrained", "example", option]
    if option == "--physics-backend":
        args.append("newton")
    with pytest.raises(ValueError, match="requires --profile"):
        _resolve_input(parse_args(args))


@pytest.mark.parametrize("viewer", (False, True))
def test_pretrained_uses_native_runtime_and_records_source(
    tmp_path, monkeypatch, viewer
):
    from embodichain.data.assets import policy_assets

    module = importlib.import_module("embodichain.learning.rl.policy_evaluation.cli")
    run_path, checkpoint = _run(tmp_path)
    source = {
        "repo_id": "DexForceAI/embodichain_model",
        "revision": "a" * 40,
        "model_id": "example",
    }
    downloads = []

    def download(model_id, **kwargs):
        downloads.append((model_id, kwargs))
        return run_path, source

    monkeypatch.setattr(policy_assets, "download_pretrained_policy", download)
    monkeypatch.setattr(module, "discover_task_packages", lambda: None)
    monkeypatch.setattr(module, "execute_init_hooks", lambda: None)
    captured = []

    def evaluate(args, resolved):
        captured.append(resolved)
        return tmp_path / "report.json"

    handler = "_run_native_viewer" if viewer else "_run_native_headless"
    monkeypatch.setattr(module, handler, evaluate)
    arguments = [
        "--pretrained",
        "example",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--revision",
        "release",
    ]
    if viewer:
        arguments.append("--viewer")
    module.run(parse_args(arguments))
    resolved = captured[0]
    assert resolved.checkpoint == checkpoint
    assert module._checkpoint_inputs(resolved)["pretrained"] == source
    assert downloads == [
        ("example", {"revision": "release", "cache_dir": str(tmp_path / "cache")})
    ]
    assert (
        module._output_parent(None, resolved)
        == module.Path.cwd() / "outputs/policy_eval/example"
    )
    assert module._output_parent("chosen-output", resolved) == module.Path(
        "chosen-output"
    )


def test_pretrained_incomplete_bundle_fails_before_runtime(tmp_path, monkeypatch):
    from embodichain.data.assets import policy_assets

    run_path, checkpoint = _run(tmp_path)
    checkpoint.unlink()
    monkeypatch.setattr(
        policy_assets, "download_pretrained_policy", lambda *a, **kw: (run_path, {})
    )
    with pytest.raises(FileNotFoundError, match="Run manifest file does not exist"):
        _resolve_input(parse_args(("--pretrained", "example")))
