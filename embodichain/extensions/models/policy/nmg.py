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

if TYPE_CHECKING:
    from onnxruntime import InferenceSession

__all__ = ["NeuralMotionGenerator"]


class _DeviceRange(nn.Module):
    def forward(
        self, start: torch.Tensor, limit: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return torch.arange(start, limit, delta, dtype=start.dtype, device=start.device)


class NeuralMotionGenerator(nn.Module):
    """Run a standalone NMG ONNX policy on batched observations.

    The model includes observation normalization and returns unscaled actions.
    The default ONNX Runtime backend returns detached actions. With
    ``enable_grad=True``, the ONNX graph becomes a frozen PyTorch model whose
    first-order input gradients follow the caller's autograd context. Keep
    discrete masks and the active waypoint fixed when differentiating.

    Attributes:
        obs_dim: Width of the exported observation.
        output_dim: Number of output joint actions.
        fixed_batch_size: Required batch size, or ``None`` for dynamic batches.
        observation_metadata: Metadata carried by the ONNX export.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        enable_grad: bool = False,
        device: str | torch.device = "cpu",
        providers: Sequence[str] | None = None,
        cuda_use_tf32: bool | None = None,
    ) -> None:
        super().__init__()
        if type(cuda_use_tf32) is not bool and cuda_use_tf32 is not None:
            raise TypeError("onnx_cuda_use_tf32 must be a bool or None.")
        if enable_grad and (providers is not None or cuda_use_tf32 is not None):
            raise ValueError("ONNX Runtime provider options require enable_grad=False.")

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {path}")
        if path.suffix.lower() != ".onnx":
            raise ValueError(
                f"NeuralMotionGenerator only accepts standalone .onnx policies, got {path}."
            )

        device = torch.device(device)
        self._session: InferenceSession | None = None
        self._torch_model: nn.Module | None = None
        if enable_grad:
            self._load_torch(path, device)
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
        """Load the selected backend from a local ONNX file.

        Args:
            path: Path to the standalone policy export.
            enable_grad: Enable first-order observation gradients with PyTorch.
            device: Initial PyTorch model device or CUDA ordinal for ORT.
                Reload the model to bind an ORT session to a different device.
            providers: Optional ORT execution providers in priority order.
            cuda_use_tf32: Optional TF32 setting for the ORT CUDA provider.

        Returns:
            A policy ready to evaluate with the selected backend.

        Raises:
            FileNotFoundError: If the local model file is missing.
            ImportError: If the selected backend's optional dependencies are absent.
            TypeError: If ``cuda_use_tf32`` is neither a bool nor ``None``.
            ValueError: If the file suffix, model signature or backend options
                are incompatible.
        """
        return cls(
            path,
            enable_grad=enable_grad,
            device=device,
            providers=providers,
            cuda_use_tf32=cuda_use_tf32,
        )

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
        logger.log_info(
            "NMG ONNX execution providers: " + ", ".join(session.get_providers())
        )

    def _load_torch(self, path: Path, device: torch.device) -> None:
        try:
            import onnx
            from onnx2pytorch import ConvertModel
            from onnx2pytorch.operations import Range
        except ImportError as exc:
            raise ImportError(
                "NMG input gradients require ONNX and onnx2pytorch. Install them with "
                "pip install 'embodichain[nmg-grad]'."
            ) from exc

        def float_shape(value: onnx.ValueInfoProto) -> list[int | None] | None:
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT or not tensor.HasField(
                "shape"
            ):
                return None
            return [
                dim.dim_value if dim.HasField("dim_value") else None
                for dim in tensor.shape.dim
            ]

        model = onnx.load(path, load_external_data=False)
        initializers = {value.name for value in model.graph.initializer}
        self._set_signature(
            [
                float_shape(value)
                for value in model.graph.input
                if value.name not in initializers
            ],
            [float_shape(value) for value in model.graph.output],
            {item.key: item.value for item in model.metadata_props},
        )
        converted = ConvertModel(model)
        # onnx2pytorch 0.6.0 creates Range results on CPU by default.
        for name, child in list(converted.named_children()):
            if isinstance(child, Range):
                converted.set_submodule(name, _DeviceRange())
        self._torch_model = converted.requires_grad_(False).to(device)

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
        obs_np = np.ascontiguousarray(obs.detach().cpu().numpy())
        action = self._session.run([self._output_name], {self._input_name: obs_np})[0]
        return torch.as_tensor(action, dtype=torch.float32, device=obs.device)
