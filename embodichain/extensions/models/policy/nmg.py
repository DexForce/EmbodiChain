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

"""Single-step execution of exported Neural Motion Generator policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from embodichain.utils import logger

from ._nmg_exported import _load_exported_actor

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

__all__ = ["NeuralMotionGenerator"]


class NeuralMotionGenerator(nn.Module):
    """Evaluate batched observations with a deployed NMG policy.

    An ONNX export uses ONNX Runtime for detached actions. An exported PyTorch
    program contains the actor graph and frozen weights and follows the caller's
    autograd context for first-order gradients of actions with respect to
    continuous observations. Keep the active waypoint and masks fixed
    when differentiating.

    Attributes:
        obs_dim: Width of the exported observation.
        output_dim: Number of output joint actions.
        fixed_batch_size: Required batch size, or ``None`` for dynamic batches.
        observation_metadata: Observation contract carried by the model artifact.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        enable_grad: bool = False,
        device: str | torch.device = "cpu",
        providers: Sequence[str] | None = None,
        cuda_use_tf32: bool | None = None,
        sha256: str | None = None,
    ) -> None:
        super().__init__()
        if type(cuda_use_tf32) is not bool and cuda_use_tf32 is not None:
            raise TypeError("onnx_cuda_use_tf32 must be a bool or None.")
        if enable_grad and (providers is not None or cuda_use_tf32 is not None):
            raise ValueError("ONNX Runtime provider options require enable_grad=False.")

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"NMG policy not found: {path}")
        valid_suffixes = {".pt2", ".pt"} if enable_grad else {".onnx"}
        if path.suffix.lower() not in valid_suffixes:
            message = (
                "Input gradients require an exported PyTorch actor (.pt2). "
                "Call NeuralMotionGenerator.from_checkpoint()."
                if enable_grad
                else f"NeuralMotionGenerator only accepts standalone .onnx policies, got {path}."
            )
            raise ValueError(message)

        device = torch.device(device)
        self._session: InferenceSession | None = None
        self._ort_cuda_device: torch.device | None = None
        self._torch_model: nn.Module | None = None
        if enable_grad:
            self._load_torch(path, device, sha256)
        else:
            self._load_ort(path, device, providers, cuda_use_tf32)
        self.eval()

    @classmethod
    def from_onnx(
        cls,
        path: str | Path,
        *,
        enable_grad: bool = False,
        device: str | torch.device = "cpu",
        providers: Sequence[str] | None = None,
        cuda_use_tf32: bool | None = None,
    ) -> NeuralMotionGenerator:
        """Load a local ONNX policy for ONNX Runtime inference.

        Args:
            path: Path to the standalone policy export.
            enable_grad: Deprecated selector. Observation gradients use :meth:`from_checkpoint`.
            device: CPU or CUDA device for ONNX Runtime.
                Reload the model to bind an ORT session to a different device.
            providers: Optional ORT execution providers in priority order.
            cuda_use_tf32: Optional TF32 setting for the ORT CUDA provider.

        Returns:
            A policy ready for inference.

        Raises:
            FileNotFoundError: If the local model file is missing.
            ImportError: If ONNX Runtime is unavailable.
            TypeError: If ``cuda_use_tf32`` is neither a bool nor ``None``.
            ValueError: If the file suffix, model signature or backend options
                are incompatible.
        """
        if enable_grad:
            raise ValueError(
                "Input gradients require an exported PyTorch actor (.pt2). "
                "Call NeuralMotionGenerator.from_checkpoint()."
            )
        return cls(
            path,
            device=device,
            providers=providers,
            cuda_use_tf32=cuda_use_tf32,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        sha256: str | None = None,
    ) -> NeuralMotionGenerator:
        """Load a graph-bearing NMG checkpoint for differentiable evaluation.

        Args:
            path: Path to a local PT2 archive with a ``.pt2`` or ``.pt`` suffix.
            device: Device on which to evaluate the policy.
            sha256: Trusted digest for an unpublished policy. The published
                v1 artifacts are recognized by their built-in SHA256.

        Returns:
            A policy with frozen weights and input gradients controlled by
            the caller's PyTorch autograd context.

        Raises:
            ValueError: If the digest, archive format, or observation contract
                is incompatible.
        """
        return cls(path, enable_grad=True, device=device, sha256=sha256)

    def _load_ort(
        self,
        path: Path,
        device: torch.device,
        providers: Sequence[str] | None,
        cuda_use_tf32: bool | None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "NMG inference requires ONNX Runtime. Install it with "
                "pip install 'embodichain[policy-deploy]'."
            ) from exc

        options = ort.SessionOptions()
        # Use the host's thread budget when running alongside other workers.
        options.intra_op_num_threads = torch.get_num_threads()
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cuda_options = {}
        if device.type == "cuda":
            # ORT ordinals are process-local, including under CUDA_VISIBLE_DEVICES.
            cuda_options["device_id"] = str(
                torch.cuda.current_device() if device.index is None else device.index
            )
        if cuda_use_tf32 is not None:
            cuda_options["use_tf32"] = "1" if cuda_use_tf32 else "0"
        provider_kwargs = {}
        if cuda_options:
            if providers is None:
                # Explicit CUDA options select the CUDA/CPU path.
                available = ort.get_available_providers()
                providers = [
                    name
                    for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if name in available
                ]
            provider_kwargs["provider_options"] = [
                dict(cuda_options) if name == "CUDAExecutionProvider" else {}
                for name in providers
            ]
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=providers, **provider_kwargs
        )
        inputs, outputs = session.get_inputs(), session.get_outputs()
        self._set_signature(
            [
                value.shape if value.type == "tensor(float)" else None
                for value in inputs
            ],
            [
                value.shape if value.type == "tensor(float)" else None
                for value in outputs
            ],
            session.get_modelmeta().custom_metadata_map,
        )
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._session = session
        session_providers = session.get_providers()
        if session_providers and session_providers[0] == "CUDAExecutionProvider":
            cuda_options = session.get_provider_options()["CUDAExecutionProvider"]
            self._ort_cuda_device = torch.device(
                "cuda", int(cuda_options.get("device_id", "0"))
            )
        logger.log_info("NMG ONNX execution providers: " + ", ".join(session_providers))

    def _load_torch(self, path: Path, device: torch.device, sha256: str | None) -> None:
        model, inputs, outputs, metadata = _load_exported_actor(
            path, device=device, sha256=sha256
        )
        self._set_signature([inputs], [outputs], metadata)
        self._torch_model = model

    def _set_signature(
        self,
        inputs: Sequence[Sequence[int | str | None] | None],
        outputs: Sequence[Sequence[int | str | None] | None],
        metadata: Mapping[str, str],
    ) -> None:
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "NMG ONNX policy must expose exactly one input and one output; "
                f"got {len(inputs)} inputs and {len(outputs)} outputs."
            )
        input_shape, output_shape = inputs[0], outputs[0]
        if (
            input_shape is None
            or len(input_shape) != 2
            or not isinstance(input_shape[1], int)
            or input_shape[1] < 1
        ):
            raise ValueError(
                f"Expected a float32 ONNX input [batch, obs], got {input_shape}."
            )
        if (
            output_shape is None
            or len(output_shape) != 2
            or not isinstance(output_shape[1], int)
            or output_shape[1] < 1
        ):
            raise ValueError(
                f"Expected a float32 ONNX output [batch, joints], got {output_shape}."
            )

        self.obs_dim: int = input_shape[1]
        self.output_dim: int = output_shape[1]
        self.fixed_batch_size: int | None = (
            input_shape[0] if isinstance(input_shape[0], int) else None
        )
        self.observation_metadata: dict[str, str] = dict(metadata)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate one policy step.

        Args:
            obs: Float32 NMG observations of shape ``(B, obs_dim)`` with the
                export's feature layout.

        Returns:
            Raw actions of shape ``(B, output_dim)`` on the observation device.

        Raises:
            ValueError: If the observation shape or batch size is incompatible.
            TypeError: If observations are supplied in a different dtype.
        """
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected NMG observations with shape [batch, {self.obs_dim}], "
                f"got {tuple(obs.shape)}."
            )
        if self.fixed_batch_size is not None and obs.shape[0] != self.fixed_batch_size:
            raise ValueError(
                f"ONNX policy has fixed batch size {self.fixed_batch_size}, "
                f"but received {obs.shape[0]}."
            )
        if obs.dtype != torch.float32:
            raise TypeError(f"Expected float32 NMG observations, got {obs.dtype}.")

        if self._torch_model is not None:
            with torch.autocast(obs.device.type, enabled=False):
                return self._torch_model(obs)

        assert self._session is not None
        if obs.device == self._ort_cuda_device and obs.shape[0] > 0:
            return self._forward_ort_cuda(obs)

        obs_np = np.ascontiguousarray(obs.detach().cpu().numpy())
        action = self._session.run([self._output_name], {self._input_name: obs_np})[0]
        return torch.as_tensor(action, dtype=torch.float32, device=obs.device)

    def _forward_ort_cuda(self, obs: torch.Tensor) -> torch.Tensor:
        assert self._session is not None
        obs = obs.detach().resolve_neg().contiguous()
        action = torch.empty(
            (obs.shape[0], self.output_dim), dtype=obs.dtype, device=obs.device
        )
        device_id = obs.get_device()
        binding = self._session.io_binding()
        # ORT can read on a different CUDA stream, starting at bind_input.
        # Wait for the Torch producer. The synchronous ORT run finishes writing
        # the action before this method returns it to the caller.
        torch.cuda.current_stream(obs.device).synchronize()
        binding.bind_input(
            self._input_name,
            "cuda",
            device_id,
            np.float32,
            tuple(obs.shape),
            obs.data_ptr(),
        )
        binding.bind_output(
            self._output_name,
            "cuda",
            device_id,
            np.float32,
            tuple(action.shape),
            action.data_ptr(),
        )
        self._session.run_with_iobinding(binding)
        return action
