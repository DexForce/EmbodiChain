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

"""Lazy dynamics CLI routing, actual offline workflow and packaged configs."""

from __future__ import annotations

import json
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml

from examples.learning.nerd.__main__ import main


def test_help_and_dispatcher_are_simulation_free():
    code = "from examples.learning.nerd.__main__ import main; import sys\ntry: main(['--help'])\nexcept SystemExit as exc: assert exc.code == 0\nassert not any(k == 'dexsim' or k.startswith('embodichain.lab.sim') for k in sys.modules)"
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_generate_routing_and_path_resolution(tmp_path, monkeypatch):
    host = ModuleType("examples.learning.nerd._support._generation")
    host.generate_dataset = Mock(return_value={"status": "ok"})
    monkeypatch.setitem(sys.modules, host.__name__, host)
    config = tmp_path / "generate.yaml"
    config.write_text("robot: cartpole\noutput_dir: data\ndevice: cpu\nresume: false\n")
    main(["generate", "--config", str(config), "--seed", "7", "--resume"])
    received = host.generate_dataset.call_args.args[0]
    assert received["output_dir"] == str(tmp_path / "data")
    assert received["seed"] == 7 and received["resume"] is True


def test_default_evaluation_writes_under_current_directory(tmp_path, monkeypatch):
    host = ModuleType("examples.learning.nerd._support._workflow")
    host.run_eval = Mock(return_value={"status": "ok"})
    monkeypatch.setitem(sys.modules, host.__name__, host)
    monkeypatch.chdir(tmp_path)
    resources = tmp_path / "resources"
    prefix = Path("neural_dynamics/nerd/cartpole/v1")
    for kind in ("datasets", "models"):
        (resources / kind / prefix).mkdir(parents=True)
    main(["eval", "--resource-root", str(resources), "--device", "cpu"])
    received = host.run_eval.call_args.args[0]
    assert received["output"] == str(tmp_path / "outputs/nerd/cartpole/offline.json")
    assert received["dataset"] == str(resources / "datasets" / prefix)


def test_invalid_paths_and_conversion_requires_explicit_trust(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("dataset: missing\n")
    with pytest.raises(SystemExit) as error:
        main(["train", "--config", str(config)])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(
            [
                "convert-checkpoint",
                "--checkpoint",
                "model.pt",
                "--output",
                "export",
                "--manifest",
                "manifest.json",
            ]
        )
    assert error.value.code == 2


def test_conversion_manifest_and_trust_reach_converter(tmp_path, monkeypatch):
    import embodichain_models.dynamics.nerd as nerd

    checkpoint = tmp_path / "old.pt"
    checkpoint.write_bytes(b"unused mock fixture")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"codec": {"robot": "cartpole"}, "dt": 1 / 60}))
    convert = Mock()
    monkeypatch.setattr(nerd, "convert_legacy_checkpoint", convert)
    main(
        [
            "convert-checkpoint",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "export"),
            "--manifest",
            str(manifest),
            "--trust-pickle",
        ]
    )
    assert convert.call_args.kwargs == {"trust_pickle": True}
    assert convert.call_args.args[2]["codec"]["robot"] == "cartpole"


@pytest.mark.parametrize("trainer_kind", ["legacy", "sl"])
def test_real_offline_cli_train_export_eval(tmp_path, trainer_kind):
    from embodichain_models.dynamics.nerd.data import DynamicsShardWriter

    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for split in ("train", "validation", "test"):
            q = np.zeros((6, 2), np.float32)
            q[:, 0] = np.arange(6) * 0.01
            qd = np.zeros_like(q)
            qd[:, 0] = 0.6
            writer.add_trajectory(
                split,
                {"q": q, "qd": qd},
                {"joint_f": np.zeros((5, 2), np.float32)},
                split=split,
            )
    training = {
        "dataset": "data",
        "output_dir": "train",
        "model": {
            "input_dims": {"states_embedding": 4, "joint_f": 2},
            "output_dim": 4,
            "n_embd": 12,
            "n_layer": 1,
            "n_head": 3,
            "block_size": 3,
            "history_length": 3,
            "head_hidden_dims": [12],
        },
        "train": {
            "total_steps": 2,
            "batch_size": 2,
            "device": "cpu",
            "validation_interval": 1,
        },
    }
    if trainer_kind == "sl":
        training = {
            "factory": "embodichain_models.dynamics.nerd.sl_recipe:build_supervised_trainer",
            "model": training["model"],
            "data": {"path": str(tmp_path / "data")},
            "trainer": {
                "total_updates": 2,
                "batch_size": 2,
                "device": "cpu",
                "validation_interval": 1,
                "checkpoint_interval": 0,
                "output_dir": str(tmp_path / "train"),
            },
        }
    train_cfg = tmp_path / "train.json"
    train_cfg.write_text(json.dumps(training))
    evaluation = {
        "mode": "offline",
        "dataset": "data",
        "model": "export",
        "output": "reports/evaluation/result.json",
        "device": "cpu",
    }
    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text(yaml.safe_dump(evaluation))
    code = """import sys
from examples.learning.nerd.__main__ import main
main(['train','--config',sys.argv[1]])
main(['export','--checkpoint',sys.argv[2],'--output',sys.argv[3]])
main(['eval','--config',sys.argv[4]])
assert not any(k == 'dexsim' or k.startswith('embodichain.lab.sim') for k in sys.modules)
"""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(train_cfg),
            str(
                tmp_path / "train" / ("last.pt" if trainer_kind == "sl" else "best.pt")
            ),
            str(tmp_path / "export"),
            str(eval_cfg),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "reports/evaluation/result.json").read_text())
    assert report["split"] == "test" and report["windows"] == 3
    if trainer_kind == "sl":
        from embodichain_models.dynamics.nerd import load_model

        model, _ = load_model(tmp_path / "export")
        assert list(model.cfg.input_dims) == ["states_embedding", "joint_f"]
        state = torch.load(tmp_path / "train/last.pt", weights_only=False)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, state["learner"]["model"][name], rtol=0, atol=0
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("model_snapshot", [False, True])
def test_training_checkpoint_cli_export_preserves_dtype_statistics_and_predictions(
    tmp_path, dtype, model_snapshot
):
    from embodichain_models.dynamics.nerd import NeRDModel, NeRDModelCfg, load_model

    torch.set_num_threads(2)
    model = (
        NeRDModel(
            NeRDModelCfg(
                input_dims={"states_embedding": 4, "joint_f": 2},
                output_dim=4,
                n_embd=12,
                n_head=3,
                n_layer=1,
                block_size=3,
                history_length=3,
                head_hidden_dims=(12,),
            )
        )
        .to(dtype=dtype)
        .eval()
    )
    # Values beyond FP32 precision expose a downcast even if tensors were later
    # converted back to FP64. Frozen normalizers must retain their precision too.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1e-10)
    model.set_normalization(
        {
            key: (
                torch.full((width,), 1 / 7, dtype=dtype),
                torch.full((width,), 1 / 3, dtype=dtype),
            )
            for key, width in model.cfg.input_dims.items()
        },
        torch.full((4,), 1 / 11, dtype=dtype),
        torch.full((4,), 1 / 13, dtype=dtype),
    )
    checkpoint = tmp_path / "training.pt"
    state = {
        "format": "embodichain.nerd.training.v1",
        "model_cfg": model.cfg.to_dict(),
        "model": model.state_dict(),
        "train_cfg": {"numerical_profile": "default"},
        "manifest": {"codec": {"robot": "cartpole"}, "dt": 1 / 60},
        "global_step": 7,
        "dataset_fingerprint": "fixed-training-corpus",
    }
    if model_snapshot:
        state.pop("train_cfg")
        state.update(
            resumable=False, numerical_profile="default", attention_backend="default"
        )
    torch.save(state, checkpoint)
    destination = tmp_path / "export"
    main(["export", "--checkpoint", str(checkpoint), "--output", str(destination)])
    loaded, metadata = load_model(destination)
    assert metadata["training"]["global_step"] == 7
    assert all(value.dtype == dtype for value in loaded.state_dict().values())
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], expected, atol=0, rtol=0)
    inputs = {
        name: torch.randn(2, 3, width, dtype=dtype)
        for name, width in model.cfg.input_dims.items()
    }
    torch.testing.assert_close(model(inputs), loaded(inputs), atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize_input", [False, True])
@pytest.mark.parametrize("normalize_output", [False, True])
def test_fine_tune_resume_export_preserves_dtype_updates_and_input_order(
    tmp_path, dtype, normalize_input, normalize_output
):
    from embodichain_models.dynamics.nerd.data import DynamicsShardWriter
    from examples.learning.nerd._support._workflow import run_train
    from embodichain_models.dynamics.nerd import (
        NeRDModel,
        NeRDModelCfg,
        export_model,
        load_model,
    )

    torch.set_num_threads(2)
    metadata = {"codec": {"robot": "cartpole"}, "dt": 1 / 60}
    with DynamicsShardWriter(tmp_path / "data", metadata) as writer:
        for split in ("train", "validation", "test"):
            q = np.zeros((6, 2), np.float32)
            q[:, 0] = np.arange(6) * 0.01
            qd = np.zeros_like(q)
            qd[:, 0] = 0.6
            writer.add_trajectory(
                split,
                {"q": q, "qd": qd},
                {"joint_f": np.zeros((5, 2), np.float32)},
                split=split,
            )
    model = NeRDModel(
        NeRDModelCfg(
            input_dims={"states_embedding": 4, "joint_f": 2},
            output_dim=4,
            n_embd=12,
            n_head=3,
            n_layer=1,
            block_size=3,
            history_length=3,
            head_hidden_dims=(12,),
            normalize_input=normalize_input,
            normalize_output=normalize_output,
        )
    ).to(dtype=dtype)
    model.input_rms["states_embedding"].mean.fill_(1 / 7)
    export_model(tmp_path / "pretrained", model, metadata)
    # Use the same JSON-shaped config as the CLI, including list-valued fields.
    config = json.loads(
        json.dumps(
            {
                "dataset": str(tmp_path / "data"),
                "output_dir": str(tmp_path / "interrupted"),
                "fine_tune": str(tmp_path / "pretrained"),
                "model": model.cfg.to_dict(),
                "train": {
                    "total_steps": 2,
                    "batch_size": 2,
                    "device": "cpu",
                    "validation_interval": 1,
                    "checkpoint_interval": 1,
                },
                "stop_after": 1,
            }
        )
    )

    def train_cli(value, label):
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(value))
        main(["train", "--config", str(path)])

    def evaluate_cli(model_path, label):
        path = tmp_path / f"{label}.json"
        output = tmp_path / f"{label}-result.json"
        path.write_text(
            json.dumps(
                {
                    "mode": "offline",
                    "dataset": str(tmp_path / "data"),
                    "model": str(model_path),
                    "device": "cpu",
                    "output": str(output),
                }
            )
        )
        main(["eval", "--config", str(path)])
        assert json.loads(output.read_text())["windows"] == 3

    evaluate_cli(tmp_path / "pretrained", "pretrained-eval")
    train_cli(config, "fine-tune")
    checkpoint = tmp_path / "interrupted/last.pt"
    saved = torch.load(checkpoint, weights_only=True)
    assert all(value.dtype == dtype for value in saved["model"].values())
    resumed_config = deepcopy(config)
    resumed_config.pop("fine_tune")
    resumed_config.update(
        resume=str(checkpoint), output_dir=str(tmp_path / "resumed"), stop_after=2
    )
    reordered = deepcopy(resumed_config)
    reordered["output_dir"] = str(tmp_path / "reordered")
    reordered["model"]["input_dims"] = dict(
        reversed(list(reordered["model"]["input_dims"].items()))
    )
    with pytest.raises(ValueError, match="input_dims order"):
        run_train(reordered)
    assert list((tmp_path / "reordered").iterdir()) == []
    train_cli(resumed_config, "resume")
    uninterrupted = deepcopy(config)
    uninterrupted.update(output_dir=str(tmp_path / "full"), stop_after=2)
    train_cli(uninterrupted, "uninterrupted")
    expected = torch.load(tmp_path / "full/last.pt", weights_only=True)
    actual = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    assert actual["global_step"] == 2
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, atol=0, rtol=0)
    assert actual["best_step"] == expected["best_step"]
    main(
        [
            "export",
            "--checkpoint",
            str(tmp_path / "resumed/best.pt"),
            "--output",
            str(tmp_path / "export"),
        ]
    )
    loaded, _ = load_model(tmp_path / "export")
    assert all(value.dtype == dtype for value in loaded.state_dict().values())
    assert loaded.cfg.normalize_input is normalize_input
    assert loaded.cfg.normalize_output is normalize_output
    for name, value in actual["best_model"].items():
        torch.testing.assert_close(loaded.state_dict()[name], value, atol=0, rtol=0)
    inputs = {
        name: torch.randn(2, 3, width) for name, width in model.cfg.input_dims.items()
    }
    prediction = loaded(inputs)
    assert prediction.dtype == dtype
    assert torch.isfinite(prediction).all()
    evaluate_cli(tmp_path / "export", "resumed-eval")


def test_packaged_workflow_configs_decode_through_cli(tmp_path, monkeypatch):
    """Exercise the actual YAML loader/path resolver and model/train constructors."""
    import examples.learning.nerd._support._workflow as workflow
    from examples.learning.nerd._support._trainer import DynamicsTrainCfg
    from embodichain_models.dynamics.nerd import NeRDModelCfg

    host = ModuleType("examples.learning.nerd._support._generation")
    host.generate_dataset = Mock(return_value={"status": "not_run"})
    monkeypatch.setitem(sys.modules, host.__name__, host)

    def train(config):
        model = NeRDModelCfg(**config["model"])
        model.validate()
        cfg = DynamicsTrainCfg(**config["train"])
        cfg.validate()
        assert "id" not in config
        if cfg.total_steps > 100:
            assert cfg.total_steps == 999 * 5000
            assert cfg.lr_update_interval == 5000 and cfg.lr_offset == 1
        return {"status": "not_run"}

    monkeypatch.setattr(workflow, "run_train", train)
    monkeypatch.setattr(workflow, "run_eval", lambda config: {"status": "not_run"})
    repo = Path(__file__).resolve().parents[4]
    (tmp_path / "data").mkdir()
    (tmp_path / "export").mkdir()
    files = []
    expected_configs = {
        "dynamics.generate.smoke.yaml",
        "dynamics.generate.full.yaml",
        "dynamics.generate.warp-reference.yaml",
        "dynamics.train.smoke.yaml",
        "dynamics.train.full.yaml",
        "dynamics.eval.smoke.yaml",
        "dynamics.eval.full.yaml",
        "dynamics.eval.pretrained.yaml",
    }
    for robot in ("cartpole", "ant"):
        robot_configs = list(
            (repo / "examples" / "learning" / "nerd" / "configs" / robot).glob(
                "dynamics.*.yaml"
            )
        )
        assert {path.name for path in robot_configs} == expected_configs
        files.extend(robot_configs)
    for source in files:
        config = yaml.safe_load(source.read_text())
        # Temporary destinations let path validation run without generating data.
        if "dataset" in config:
            config["dataset"] = "data"
        if isinstance(config.get("model"), str):
            config["model"] = "export"
        if "output_dir" in config:
            config["output_dir"] = "out"
        if "output" in config:
            config["output"] = "report.json"
        copied = tmp_path / source.name
        copied.write_text(yaml.safe_dump(config, sort_keys=False))
        operation = source.name.split(".")[1]
        main([operation, "--config", str(copied)])
