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

"""NMG model execution and gradient contracts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from embodichain.extensions.models.policy import nmg as nmg_module
from embodichain.extensions.models.policy.nmg import NeuralMotionGenerator

pytestmark = pytest.mark.no_sim


def test_onnx_runtime_respects_host_thread_budget(monkeypatch, tmp_path):
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="obs", type="tensor(float)", shape=[None, 32])
        ],
        get_outputs=lambda: [
            SimpleNamespace(name="action", type="tensor(float)", shape=[None, 7])
        ],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={}),
        get_providers=lambda: ["CPUExecutionProvider"],
    )
    received = {}

    def create_session(path, *, sess_options, providers):
        received["options"] = sess_options
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=create_session,
        ),
    )
    messages = []
    monkeypatch.setattr(nmg_module.logger, "log_info", messages.append)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 3)
    model = tmp_path / "policy.onnx"
    model.touch()
    NeuralMotionGenerator.from_onnx(model)
    assert messages == ["NMG ONNX execution providers: CPUExecutionProvider"]
    assert received["options"].intra_op_num_threads == 3
    assert received["options"].inter_op_num_threads == 1


@pytest.mark.parametrize(
    "providers,use_tf32,device,expected_providers,expected_options",
    [
        (None, None, None, None, None),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            None,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            None,
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"use_tf32": "0"}, {}],
        ),
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            True,
            None,
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            [{}, {"use_tf32": "1"}],
        ),
        (
            ["CPUExecutionProvider"],
            False,
            None,
            ["CPUExecutionProvider"],
            [{}],
        ),
        (
            None,
            False,
            None,
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"use_tf32": "0"}, {}],
        ),
        (
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            False,
            torch.device("cuda:0"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "0", "use_tf32": "0"}, {}],
        ),
        (
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            False,
            torch.device("cuda:3"),
            ["CPUExecutionProvider", "CUDAExecutionProvider"],
            [{}, {"device_id": "3", "use_tf32": "0"}],
        ),
        (
            None,
            False,
            torch.device("cuda:2"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "2", "use_tf32": "0"}, {}],
        ),
        (
            None,
            None,
            torch.device("cuda:4"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "4"}, {}],
        ),
        (
            ["CPUExecutionProvider"],
            False,
            torch.device("cuda:2"),
            ["CPUExecutionProvider"],
            [{}],
        ),
        (
            None,
            False,
            torch.device("cuda"),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            [{"device_id": "2", "use_tf32": "0"}, {}],
        ),
        (None, None, torch.device("cpu"), None, None),
    ],
)
def test_onnx_precision_is_explicit_and_provider_scoped(
    monkeypatch,
    tmp_path,
    providers,
    use_tf32,
    device,
    expected_providers,
    expected_options,
):
    """Freeze CUDA arithmetic without changing CPU options or historical defaults."""
    session = SimpleNamespace(
        get_inputs=lambda: [
            SimpleNamespace(name="obs", type="tensor(float)", shape=[None, 32])
        ],
        get_outputs=lambda: [
            SimpleNamespace(name="action", type="tensor(float)", shape=[None, 7])
        ],
        get_modelmeta=lambda: SimpleNamespace(custom_metadata_map={}),
        get_providers=lambda: expected_providers or ["CPUExecutionProvider"],
    )
    received = {}

    def create_session(path, **kwargs):
        received.update(kwargs)
        return session

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
            InferenceSession=create_session,
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    model = tmp_path / "policy.onnx"
    model.touch()
    NeuralMotionGenerator.from_onnx(
        model, providers=providers, cuda_use_tf32=use_tf32, device=device or "cpu"
    )
    assert received["providers"] == expected_providers
    assert received.get("provider_options") == expected_options
    if expected_options is None:
        assert "provider_options" not in received


@pytest.mark.parametrize("invalid", [0, 1, "false", 1.0])
def test_onnx_precision_rejects_non_boolean_values(tmp_path, invalid):
    with pytest.raises(TypeError, match="onnx_cuda_use_tf32 must be a bool or None"):
        NeuralMotionGenerator.from_onnx(tmp_path / "policy.onnx", cuda_use_tf32=invalid)


def _linear_policy(tmp_path: Path, batch: int | str) -> tuple[Path, torch.Tensor]:
    onnx = pytest.importorskip("onnx")
    weight = torch.tensor([[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]])
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("MatMul", ["obs", "weight"], ["action"])],
        "nmg-linear-policy",
        [onnx.helper.make_tensor_value_info("obs", onnx.TensorProto.FLOAT, [batch, 3])],
        [
            onnx.helper.make_tensor_value_info(
                "action", onnx.TensorProto.FLOAT, [batch, 2]
            )
        ],
        [onnx.numpy_helper.from_array(weight.numpy(), name="weight")],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=10
    )
    onnx.helper.set_model_props(model, {"nmg.observation_layout": "linear-test"})
    path = tmp_path / "linear.onnx"
    onnx.save(model, path)
    return path, weight


@pytest.mark.parametrize("enable_grad", [False, True], ids=["ort", "torch"])
def test_dynamic_batch_and_analytic_input_vjp(tmp_path, monkeypatch, enable_grad):
    path, weight = _linear_policy(tmp_path, "batch")
    if enable_grad:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        pytest.importorskip("onnx2pytorch")
    else:
        pytest.importorskip("onnxruntime")
        monkeypatch.setitem(sys.modules, "onnx", None)
        monkeypatch.setitem(sys.modules, "onnx2pytorch", None)
    policy = NeuralMotionGenerator.from_onnx(path, enable_grad=enable_grad)
    assert (policy.obs_dim, policy.output_dim, policy.fixed_batch_size) == (3, 2, None)
    assert policy.observation_metadata == {"nmg.observation_layout": "linear-test"}

    source = torch.arange(9, dtype=torch.float32).reshape(3, 3).requires_grad_()
    scale = torch.tensor([2.0, -1.0, 0.5])
    obs = source * scale + 1.0
    batch_one = policy(obs[:1])
    batch_three = policy(obs)
    torch.testing.assert_close(batch_one, obs[:1].detach() @ weight)
    torch.testing.assert_close(batch_three, obs.detach() @ weight)
    if enable_grad:
        seed = torch.tensor([[-1.0, 0.5], [2.0, 1.0], [0.25, -3.0]])
        batch_three.backward(seed)
        torch.testing.assert_close(source.grad, (seed @ weight.T) * scale)
        assert all(not parameter.requires_grad for parameter in policy.parameters())
        assert all(parameter.grad is None for parameter in policy.parameters())
        with torch.no_grad():
            assert not policy(obs).requires_grad
    else:
        assert not batch_one.requires_grad
        assert batch_three.grad_fn is None


@pytest.mark.parametrize("enable_grad", [False, True], ids=["ort", "torch"])
def test_validates_observation_signature(tmp_path, enable_grad):
    path, weight = _linear_policy(tmp_path, 2)
    pytest.importorskip("onnx2pytorch" if enable_grad else "onnxruntime")
    policy = NeuralMotionGenerator.from_onnx(path, enable_grad=enable_grad)
    assert policy.fixed_batch_size == 2
    obs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    torch.testing.assert_close(policy(obs), obs @ weight)
    with pytest.raises(ValueError, match=r"shape \[batch, 3\]"):
        policy(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="fixed batch size 2"):
        policy(obs[:1])
    with pytest.raises(TypeError, match="float32"):
        policy(obs.double())


@pytest.mark.parametrize(
    ("enable_grad", "package", "extra"),
    [(False, "onnxruntime", "policy-deploy"), (True, "onnx2pytorch", "nmg")],
)
def test_missing_backend_names_its_optional_extra(
    tmp_path, monkeypatch, enable_grad, package, extra
):
    path = tmp_path / "local.onnx"
    path.touch()
    monkeypatch.setitem(sys.modules, package, None)
    with pytest.raises(ImportError, match=extra):
        NeuralMotionGenerator.from_onnx(path, enable_grad=enable_grad)


def test_requires_a_local_onnx_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        NeuralMotionGenerator.from_onnx(tmp_path / "missing.onnx")


def test_torch_backend_rejects_ort_provider_options(tmp_path):
    with pytest.raises(ValueError, match="provider options require enable_grad=False"):
        NeuralMotionGenerator.from_onnx(
            tmp_path / "policy.onnx",
            enable_grad=True,
            providers=["CPUExecutionProvider"],
        )


_RELEASE_MODELS = {
    "franka": (
        7,
        417,
        "2361954fda7b6f0d21445ebfb5099d31bdd67199bdadcecba71e0835f2157ad5",
    ),
    "piper": (
        6,
        391,
        "223a02d7f1c3db1ea96352404f623ebfde7e1b00f3bc3d7a186a634dfea5ecf0",
    ),
}


def _release_observation(batch: int, joints: int) -> torch.Tensor:
    slots = 12
    qpos = torch.linspace(-0.65, 0.92, joints) + 0.04 * torch.arange(batch)[:, None]
    ee_pose = torch.tensor([[0.48, -0.04, 0.32, 0.0, 0.0, 0.0, 1.0]]).expand(batch, -1)
    positions = torch.zeros(batch, slots, 3)
    positions[:, :2] = torch.tensor([[0.51, -0.08, 0.41], [0.47, 0.10, 0.36]])
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(batch, slots)
    valid = torch.zeros(batch, slots)
    valid[:, :2] = 1.0
    pos = valid.clone()
    rot = valid.clone()
    joint = torch.zeros_like(valid)
    if batch == 3:
        # Joint, orientation and full pose targets with different active slots.
        pos[0, 0] = rot[0, 0] = 0
        joint[0, 0] = 1
        pos[1, 1] = 0
        valid[2, 1] = pos[2, 1] = rot[2, 1] = 0
    active = torch.nn.functional.one_hot(torch.tensor([0, 1, 0])[:batch], slots).float()
    return torch.cat(
        (
            qpos,
            ee_pose,
            (positions * pos[..., None]).flatten(1),
            identity_quat,
            ((qpos[:, None] + 0.12) * joint[..., None]).flatten(1),
            active,
            valid,
            pos,
            rot,
            joint,
            torch.zeros_like(qpos),
            ((positions - ee_pose[:, None, :3]) * pos[..., None]).flatten(1),
            identity_quat,
            (torch.full_like(qpos[:, None], 0.12) * joint[..., None]).flatten(1),
        ),
        dim=1,
    )


@pytest.fixture
def _full_precision_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("robot", _RELEASE_MODELS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_release_policy_vjp_matches_onnx_central_difference(
    robot, device, _full_precision_matmul
):
    """Use the pinned v1 files from a local directory for numeric qualification."""
    directory = os.environ.get("EMBODICHAIN_NMG_MODEL_DIR")
    if not directory:
        pytest.skip("Set EMBODICHAIN_NMG_MODEL_DIR to the pinned v1 model directory.")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for this NMG gradient check.")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx2pytorch")
    joints, obs_dim, sha256 = _RELEASE_MODELS[robot]
    path = Path(directory) / f"{robot}.onnx"
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == sha256
    inference = NeuralMotionGenerator.from_onnx(
        path, providers=["CPUExecutionProvider"]
    )
    policy = NeuralMotionGenerator.from_onnx(path, enable_grad=True, device=device)
    cpu_policy = (
        NeuralMotionGenerator.from_onnx(path, enable_grad=True)
        if device == "cuda"
        else None
    )
    assert policy.observation_metadata == inference.observation_metadata
    assert (policy.obs_dim, policy.output_dim, policy.fixed_batch_size) == (
        obs_dim,
        joints,
        None,
    )
    for batch in (1, 3):
        encoded = _release_observation(batch, joints)
        obs = encoded.to(device).requires_grad_(True)
        seed = torch.linspace(-0.7, 0.9, batch * joints).reshape(batch, joints)
        action = policy(obs)
        (grad,) = torch.autograd.grad(action, obs, grad_outputs=seed.to(device))
        assert action.device == grad.device == obs.device
        assert torch.isfinite(grad).all() and grad.abs().max() > 0
        torch.testing.assert_close(
            action.detach().cpu(), inference(encoded), rtol=1e-4, atol=2e-6
        )

        direction = torch.zeros_like(encoded)
        direction[:, :joints] = torch.linspace(-0.3, 0.4, joints)
        step = 1e-3
        delta = inference(encoded + step * direction) - inference(
            encoded - step * direction
        )
        expected = (delta * seed).sum() / (2 * step)
        torch.testing.assert_close(
            (grad.cpu() * direction).sum(), expected, rtol=1e-2, atol=1.5e-3
        )

        if cpu_policy is not None:
            cpu_obs = encoded.detach().requires_grad_(True)
            (cpu_grad,) = torch.autograd.grad(
                cpu_policy(cpu_obs), cpu_obs, grad_outputs=seed
            )
            torch.testing.assert_close(grad.cpu(), cpu_grad, rtol=5e-4, atol=3e-5)
