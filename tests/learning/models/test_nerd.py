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
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from embodichain.compute.dynamics import append_history
from embodichain.learning.models.nerd import (
    NeRDModel,
    NeRDModelCfg,
    export_model,
    load_model,
    convert_legacy_checkpoint,
)


def tiny() -> NeRDModel:
    return NeRDModel(
        NeRDModelCfg(
            input_dims={"states_embedding": 4, "joint_f": 2},
            output_dim=4,
            n_embd=12,
            n_head=3,
            n_layer=2,
            block_size=10,
            history_length=10,
        )
    )


def test_normalizers_and_parameters_follow_dtype() -> None:
    model = tiny().double()
    assert all(v.dtype == torch.double for v in model.state_dict().values())
    model.set_normalization(
        {
            k: (torch.ones(d), torch.full((d,), 2.0))
            for k, d in model.cfg.input_dims.items()
        },
        torch.ones(4),
        torch.full((4,), 3.0),
    )
    value = torch.randn(2, 3, 4, dtype=torch.double)
    torch.testing.assert_close(
        model.input_rms["states_embedding"](value), (value - 1) / (2 + 1e-5) ** 0.5
    )


def test_variable_prefix_matches_individual_unpadded_calls() -> None:
    model = tiny().eval()
    inputs = {k: torch.randn(3, 10, d) for k, d in model.cfg.input_dims.items()}
    copies = {k: v.clone() for k, v in inputs.items()}
    lengths = torch.tensor([1, 9, 10])
    outputs = model.evaluate(inputs, lengths)
    for row, length in enumerate(lengths):
        expected = model.evaluate(
            {k: v[row : row + 1, :length] for k, v in inputs.items()}
        )
        torch.testing.assert_close(
            outputs[row : row + 1], expected, atol=1e-6, rtol=1e-5
        )
    for k in inputs:
        torch.testing.assert_close(inputs[k], copies[k])


def test_multistep_history_control_and_parameter_gradients() -> None:
    model = tiny().double().eval()
    history = {
        k: torch.zeros(1, 10, d, dtype=torch.double)
        for k, d in model.cfg.input_dims.items()
    }
    lengths = torch.zeros(1, dtype=torch.long)
    initial = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
    state = initial
    controls = [
        torch.randn(1, 2, dtype=torch.double, requires_grad=True) for _ in range(12)
    ]
    for control in controls:
        history, lengths = append_history(
            history, lengths, {"states_embedding": state, "joint_f": control}
        )
        state = state + model.evaluate(history, lengths).squeeze(1)
    state.square().mean().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert all(c.grad is not None and c.grad.abs().sum() > 0 for c in controls)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_export_roundtrip_and_checksum(tmp_path: Path) -> None:
    model = tiny().double().eval()
    inputs = {
        k: torch.randn(2, 3, d, dtype=torch.double)
        for k, d in model.cfg.input_dims.items()
    }
    export_model(tmp_path, model, {"codec": {"robot": "cartpole"}, "dt": 1 / 60})
    loaded, manifest = load_model(tmp_path)
    assert manifest["codec"]["robot"] == "cartpole"
    torch.testing.assert_close(model(inputs), loaded(inputs), atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        export_model(tmp_path, model, {})
    with (tmp_path / "weights.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_legacy_pickle_requires_explicit_trust(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="trust_pickle"):
        convert_legacy_checkpoint(tmp_path / "missing", tmp_path, {})


def test_attention_backend_preserves_initial_weights_and_rng() -> None:
    torch.manual_seed(19)
    default = tiny()
    expected_rng = torch.get_rng_state().clone()
    cfg = default.cfg.copy()
    cfg.attention_backend = "math"
    torch.manual_seed(19)
    math_model = NeRDModel(cfg)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert all(
        torch.equal(value, math_model.state_dict()[key])
        for key, value in default.state_dict().items()
    )


def test_math_attention_export_load_and_provider_prediction_restore_flags(
    tmp_path: Path, monkeypatch
) -> None:
    model = tiny().eval()
    model.cfg.attention_backend = "math"
    export_model(
        tmp_path,
        model,
        {
            "codec": {"robot": "cartpole"},
            "dt": 1 / 60,
            "numerical_profile": "reference",
        },
    )
    loaded, manifest = load_model(tmp_path)
    assert loaded.cfg.attention_backend == manifest["attention_backend"] == "math"
    assert manifest["numerical_profile"] == "reference"
    original = torch.nn.functional.scaled_dot_product_attention
    flags = []
    before = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )

    def record(*args, **kwargs):
        flags.append(
            (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    inputs = {
        key: torch.randn(2, 3, width) for key, width in loaded.cfg.input_dims.items()
    }
    assert torch.equal(loaded.evaluate(inputs), model.evaluate(inputs))
    assert flags and all(row == (False, True, False) for row in flags)
    assert before == (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


def test_legacy_portable_export_without_execution_fields_remains_default(
    tmp_path: Path,
) -> None:
    model = tiny().eval()
    export_model(tmp_path, model, {"codec": {"robot": "cartpole"}, "dt": 1 / 60})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest.pop("attention_backend")
    manifest.pop("numerical_profile")
    manifest["model_config"].pop("attention_backend")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    loaded, _ = load_model(tmp_path)
    assert loaded.cfg.attention_backend == "default"
    assert all(
        torch.equal(value, loaded.state_dict()[key])
        for key, value in model.state_dict().items()
    )


@pytest.mark.parametrize(
    "mutation", ["attention_backend", "numerical_profile", "reference_requires_math"]
)
def test_export_rejects_conflicting_execution_metadata(
    tmp_path: Path, mutation: str
) -> None:
    model = tiny()
    model.cfg.attention_backend = "math"
    export_model(tmp_path, model, {"numerical_profile": "reference"})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if mutation == "attention_backend":
        manifest["attention_backend"] = "default"
    elif mutation == "numerical_profile":
        manifest["numerical_profile"] = "unknown"
    else:
        manifest["attention_backend"] = "default"
        manifest["model_config"]["attention_backend"] = "default"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="attention_backend|numerical_profile"):
        load_model(tmp_path)


@pytest.mark.parametrize("robot", ["Cartpole", "Ant"])
def test_explicit_official_checkpoint_forward_gradient_and_adam(
    robot: str, tmp_path: Path
) -> None:
    # Reference code is only loaded when the operator explicitly opts into a local
    # trusted checkout. Run it in a subprocess to avoid polluting production imports.
    reference = os.environ.get("NERD_REFERENCE_ROOT")
    if reference is None:
        pytest.skip("Set NERD_REFERENCE_ROOT to the pinned trusted upstream checkout")
    script = Path(__file__).with_name("_nerd_reference_check.py")
    subprocess.run(
        [sys.executable, str(script), reference, robot, str(tmp_path)],
        check=True,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (
                        os.environ.get("NERD_REFERENCE_DEPS"),
                        str(Path(__file__).resolve().parents[3]),
                    ),
                )
            ),
        },
    )
