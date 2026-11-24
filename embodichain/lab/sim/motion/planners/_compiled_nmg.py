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

"""Fused ``torch.compile`` CUDA-Graph closed-loop rollout backend for NeuralPlanner.

Lowers standalone ONNX actor graphs directly into static-shape PyTorch modules
and fuses observation construction, policy evaluation, joint-limit clipping,
serial-chain forward kinematics, and arrival checks into one CUDA Graph per
batch size, without importing external training repositories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn
import torch.nn.functional as F

from embodichain.utils.math import matrix_from_quat

_ONNX_DTYPE: dict[int, torch.dtype] = {
    1: torch.float32,
    6: torch.int32,
    7: torch.int64,
    9: torch.bool,
    11: torch.float64,
}


def _quat_from_rotmat_xyzw(rot_mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` into unit ``xyzw`` quaternions."""
    r00, r01, r02 = rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 0, 2]
    r10, r11, r12 = rot_mat[..., 1, 0], rot_mat[..., 1, 1], rot_mat[..., 1, 2]
    r20, r21, r22 = rot_mat[..., 2, 0], rot_mat[..., 2, 1], rot_mat[..., 2, 2]
    tr = r00 + r11 + r22
    s0 = torch.sqrt((1.0 + tr).clamp(min=1e-12)) * 2.0
    q0 = torch.stack(
        [(r21 - r12) / s0, (r02 - r20) / s0, (r10 - r01) / s0, 0.25 * s0], dim=-1
    )
    s1 = torch.sqrt((1.0 + r00 - r11 - r22).clamp(min=1e-12)) * 2.0
    q1 = torch.stack(
        [0.25 * s1, (r01 + r10) / s1, (r02 + r20) / s1, (r21 - r12) / s1], dim=-1
    )
    s2 = torch.sqrt((1.0 + r11 - r00 - r22).clamp(min=1e-12)) * 2.0
    q2 = torch.stack(
        [(r01 + r10) / s2, 0.25 * s2, (r12 + r21) / s2, (r02 - r20) / s2], dim=-1
    )
    s3 = torch.sqrt((1.0 + r22 - r00 - r11).clamp(min=1e-12)) * 2.0
    q3 = torch.stack(
        [(r02 + r20) / s3, (r12 + r21) / s3, 0.25 * s3, (r10 - r01) / s3], dim=-1
    )
    c0 = (tr > 0.0).unsqueeze(-1)
    c1 = ((r00 >= r11) & (r00 >= r22)).unsqueeze(-1)
    c2 = (r11 >= r22).unsqueeze(-1)
    q = torch.where(c0, q0, torch.where(c1, q1, torch.where(c2, q2, q3)))
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _quat_err_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Geodesic SO(3) angle error in radians between unit ``xyzw`` quaternions."""
    dot = (q1 * q2).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _quat_inverse_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Conjugate inverse of unit ``xyzw`` quaternions."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``xyzw`` quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _canonicalize_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Canonicalize quaternion sign with first non-negligible coordinate positive."""
    eps = 1.0e-7
    w, x, y, z = q[..., 3], q[..., 0], q[..., 1], q[..., 2]
    sign = torch.where(
        w.abs() > eps,
        torch.where(w >= 0.0, 1.0, -1.0),
        torch.where(
            x.abs() > eps,
            torch.where(x >= 0.0, 1.0, -1.0),
            torch.where(
                y.abs() > eps,
                torch.where(y >= 0.0, 1.0, -1.0),
                torch.where(z >= 0.0, 1.0, -1.0),
            ),
        ),
    )
    return q * sign.unsqueeze(-1)


def _rewrite_bool_scatter(nodes: list[onnx.NodeProto]) -> list[onnx.NodeProto]:
    """Rewrite ``torch.onnx``'s 8-node ``NonZero..ScatterND`` mask put into ``Where``."""
    out: list[onnx.NodeProto] = []
    i = 0
    while i < len(nodes):
        if (
            i + 7 < len(nodes)
            and [n.op_type for n in nodes[i : i + 8]]
            == [
                "NonZero",
                "Transpose",
                "Reshape",
                "Cast",
                "ReduceSum",
                "Expand",
                "Reshape",
                "ScatterND",
            ]
            and nodes[i + 7].input[1] == nodes[i + 1].output[0]
            and nodes[i + 7].input[2] == nodes[i + 6].output[0]
        ):
            out.append(
                onnx.helper.make_node(
                    "Where",
                    [nodes[i].input[0], nodes[i + 5].input[0], nodes[i + 7].input[0]],
                    [nodes[i + 7].output[0]],
                )
            )
            i += 8
        else:
            out.append(nodes[i])
            i += 1
    return out


class StaticBatchOnnxModule(nn.Module):
    """Standalone ONNX-to-PyTorch static-batch module compatible with CUDA Graphs."""

    def __init__(self, onnx_path: str, batch_size: int, device: torch.device) -> None:
        """Fold static batch shapes and register ONNX initializers as buffers."""
        super().__init__()
        model = onnx.load(onnx_path)
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("StaticBatchOnnxModule expects single-input/output ONNX.")
        nodes = _rewrite_bool_scatter(list(model.graph.node))
        if any(n.op_type in ("NonZero", "ScatterND", "If", "Loop") for n in nodes):
            raise ValueError("Dynamic control or indexing ops remain in ONNX graph.")

        cpu_consts: dict[str, np.ndarray] = {}
        self._buf_map: dict[str, str] = {}
        for idx, init in enumerate(model.graph.initializer):
            arr = numpy_helper.to_array(init).copy()
            if arr.size <= 32 and arr.dtype.kind in ("i", "b"):
                cpu_consts[init.name] = arr
            bname = f"_c_{idx}"
            self.register_buffer(bname, torch.from_numpy(arr).to(device=device))
            self._buf_map[init.name] = bname

        in_info = model.graph.input[0]
        in_dims = [
            (
                batch_size
                if (d.dim_param or d.dim_value <= 0) and j == 0
                else int(d.dim_value)
            )
            for j, d in enumerate(in_info.type.tensor_type.shape.dim)
        ]
        shapes: dict[str, tuple[int, ...]] = {in_info.name: tuple(in_dims)}
        for init in model.graph.initializer:
            shapes[init.name] = tuple(int(x) for x in init.dims)

        self._ops: list[tuple[str, list[str], str, dict[str, Any]]] = []
        for idx, n in enumerate(nodes):
            op = n.op_type
            ins = list(n.input)
            out0 = n.output[0]
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}

            if op == "Shape":
                sh = shapes[ins[0]]
                s = int(attrs.get("start", 0))
                e = int(attrs.get("end", len(sh)))
                arr = np.asarray(sh[s:e], dtype=np.int64)
                cpu_consts[out0] = arr
                shapes[out0] = arr.shape
                bname = f"_s_{idx}"
                self.register_buffer(bname, torch.from_numpy(arr).to(device=device))
                self._buf_map[out0] = bname
                continue

            nonempty_ins = [x for x in ins if x]
            if all(x in cpu_consts for x in nonempty_ins) and op in (
                "Squeeze",
                "Unsqueeze",
                "Concat",
                "Add",
                "Sub",
                "Mul",
                "Div",
                "Cast",
                "Range",
                "Reshape",
                "Expand",
                "Gather",
                "Slice",
            ):
                vals = [cpu_consts[x] for x in nonempty_ins]
                arr = None
                if op == "Squeeze":
                    arr = (
                        np.squeeze(vals[0])
                        if len(vals) == 1
                        else np.squeeze(
                            vals[0],
                            axis=tuple(int(a) for a in np.atleast_1d(vals[1])),
                        )
                    )
                elif op == "Unsqueeze":
                    arr = vals[0]
                    for ax in sorted(int(a) for a in np.atleast_1d(vals[1])):
                        arr = np.expand_dims(arr, axis=ax)
                elif op == "Concat":
                    arr = np.concatenate(vals, axis=int(attrs["axis"]))
                elif op == "Add":
                    arr = vals[0] + vals[1]
                elif op == "Sub":
                    arr = vals[0] - vals[1]
                elif op == "Mul":
                    arr = vals[0] * vals[1]
                elif op == "Div":
                    arr = (
                        vals[0] // vals[1]
                        if vals[0].dtype.kind == "i"
                        else vals[0] / vals[1]
                    )
                elif op == "Cast":
                    dt_map = {7: np.int64, 6: np.int32, 9: np.bool_, 1: np.float32}
                    arr = vals[0].astype(dt_map[int(attrs["to"])])
                elif op == "Range":
                    arr = np.arange(
                        int(vals[0]), int(vals[1]), int(vals[2]), dtype=np.int64
                    )
                elif op == "Reshape":
                    arr = vals[0].reshape([int(x) for x in vals[1]])
                elif op == "Expand":
                    arr = np.broadcast_to(vals[0], [int(x) for x in vals[1]]).copy()
                elif op == "Gather":
                    arr = np.take(vals[0], vals[1], axis=int(attrs.get("axis", 0)))
                elif op == "Slice":
                    starts = [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                    ends = [int(x) for x in np.atleast_1d(cpu_consts[ins[2]])]
                    axes = (
                        [int(x) for x in np.atleast_1d(cpu_consts[ins[3]])]
                        if len(ins) > 3 and ins[3]
                        else list(range(len(starts)))
                    )
                    steps = (
                        [int(x) for x in np.atleast_1d(cpu_consts[ins[4]])]
                        if len(ins) > 4 and ins[4]
                        else [1] * len(starts)
                    )
                    arr = vals[0].copy()
                    for st, en, ax, sp in zip(starts, ends, axes, steps):
                        sl = [slice(None)] * arr.ndim
                        sl[ax if ax >= 0 else arr.ndim + ax] = slice(st, en, sp)
                        arr = arr[tuple(sl)]
                if arr is not None:
                    arr = np.asarray(arr)
                    if arr.size <= 4096:
                        cpu_consts[out0] = arr
                    shapes[out0] = arr.shape
                    bname = f"_k_{idx}"
                    self.register_buffer(
                        bname, torch.from_numpy(arr.copy()).to(device=device)
                    )
                    self._buf_map[out0] = bname
                    continue

            sh0 = shapes[ins[0]]
            if op in ("Sub", "Add", "Mul", "Div", "Min", "Equal", "Greater", "And"):
                shapes[out0] = tuple(np.broadcast_shapes(sh0, shapes[ins[1]]))
            elif op == "Where":
                shapes[out0] = tuple(
                    np.broadcast_shapes(sh0, shapes[ins[1]], shapes[ins[2]])
                )
            elif op in (
                "Not",
                "Cast",
                "Clip",
                "Tanh",
                "Erf",
                "Relu",
                "Sigmoid",
                "Softmax",
                "LayerNormalization",
            ):
                shapes[out0] = sh0
            elif op == "Slice":
                starts = [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                ends = [int(x) for x in np.atleast_1d(cpu_consts[ins[2]])]
                axes = (
                    [int(x) for x in np.atleast_1d(cpu_consts[ins[3]])]
                    if len(ins) > 3 and ins[3]
                    else list(range(len(starts)))
                )
                steps = (
                    [int(x) for x in np.atleast_1d(cpu_consts[ins[4]])]
                    if len(ins) > 4 and ins[4]
                    else [1] * len(starts)
                )
                new_sh = list(sh0)
                slice_specs = []
                for st, en, ax, sp in zip(starts, ends, axes, steps):
                    ax = ax if ax >= 0 else len(new_sh) + ax
                    new_sh[ax] = len(range(*slice(st, en, sp).indices(new_sh[ax])))
                    slice_specs.append((ax, st, en, sp))
                shapes[out0] = tuple(new_sh)
                attrs["_slice_specs"] = slice_specs
            elif op == "Reshape":
                raw = [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                allowzero = int(attrs.get("allowzero", 0))
                target = [
                    sh0[j] if (d == 0 and not allowzero) else d
                    for j, d in enumerate(raw)
                ]
                if -1 in target:
                    prod = 1
                    for d in target:
                        if d != -1:
                            prod *= d
                    tot = int(np.prod(sh0))
                    target = [tot // prod if d == -1 else d for d in target]
                shapes[out0] = tuple(target)
                attrs["_shape"] = tuple(target)
            elif op == "Expand":
                raw = [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                raw = [1] * max(0, len(sh0) - len(raw)) + raw
                sh_pad = [1] * max(0, len(raw) - len(sh0)) + list(sh0)
                target = [s if r in (1, -1) else r for s, r in zip(sh_pad, raw)]
                shapes[out0] = tuple(target)
                attrs["_shape"] = tuple(target)
            elif op == "Unsqueeze":
                axes = [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                rk = len(sh0) + len(axes)
                axes_norm = sorted(ax if ax >= 0 else rk + ax for ax in axes)
                new_sh = list(sh0)
                for ax in axes_norm:
                    new_sh.insert(ax, 1)
                shapes[out0] = tuple(new_sh)
                attrs["_axes"] = axes_norm
            elif op == "Squeeze":
                axes = (
                    [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                    if len(ins) > 1 and ins[1]
                    else [j for j, d in enumerate(sh0) if d == 1]
                )
                axes_norm = sorted(
                    (ax if ax >= 0 else len(sh0) + ax for ax in axes), reverse=True
                )
                new_sh = list(sh0)
                for ax in axes_norm:
                    new_sh.pop(ax)
                shapes[out0] = tuple(new_sh)
                attrs["_axes"] = axes_norm
            elif op == "Concat":
                ax = int(attrs["axis"])
                ax = ax if ax >= 0 else len(sh0) + ax
                new_sh = list(sh0)
                new_sh[ax] = sum(shapes[x][ax] for x in ins)
                shapes[out0] = tuple(new_sh)
                attrs["_axis"] = ax
            elif op == "Transpose":
                perm = [int(x) for x in attrs["perm"]]
                shapes[out0] = tuple(sh0[p] for p in perm)
                attrs["_perm"] = tuple(perm)
            elif op == "ArgMax":
                ax = int(attrs.get("axis", 0))
                kd = bool(int(attrs.get("keepdims", 1)))
                ax = ax if ax >= 0 else len(sh0) + ax
                new_sh = list(sh0)
                if kd:
                    new_sh[ax] = 1
                else:
                    new_sh.pop(ax)
                shapes[out0] = tuple(new_sh)
                attrs["_axis"] = ax
                attrs["_keepdims"] = kd
            elif op == "ReduceSum":
                axes = (
                    [int(x) for x in np.atleast_1d(cpu_consts[ins[1]])]
                    if len(ins) > 1 and ins[1]
                    else list(range(len(sh0)))
                )
                kd = bool(int(attrs.get("keepdims", 1)))
                axes_norm = sorted(
                    (ax if ax >= 0 else len(sh0) + ax for ax in axes), reverse=True
                )
                new_sh = list(sh0)
                for ax in axes_norm:
                    if kd:
                        new_sh[ax] = 1
                    else:
                        new_sh.pop(ax)
                shapes[out0] = tuple(new_sh)
                attrs["_axes"] = tuple(axes_norm)
                attrs["_keepdims"] = kd
            elif op == "Gemm":
                tb = int(attrs.get("transB", 0))
                shapes[out0] = (sh0[0], shapes[ins[1]][0] if tb else shapes[ins[1]][1])
            elif op == "MatMul":
                s1 = shapes[ins[1]]
                shapes[out0] = tuple(np.broadcast_shapes(sh0[:-2], s1[:-2])) + (
                    sh0[-2],
                    s1[-1],
                )
            elif op == "Gather":
                ax = int(attrs.get("axis", 0))
                ax = ax if ax >= 0 else len(sh0) + ax
                shapes[out0] = sh0[:ax] + shapes[ins[1]] + sh0[ax + 1 :]
                attrs["_axis"] = ax
                if ins[1] in cpu_consts and cpu_consts[ins[1]].ndim == 0:
                    attrs["_scalar_idx"] = int(cpu_consts[ins[1]])
            elif op == "GatherElements":
                ax = int(attrs.get("axis", 0))
                attrs["_axis"] = ax if ax >= 0 else len(sh0) + ax
                shapes[out0] = shapes[ins[1]]
            elif op == "GatherND":
                idx_sh = shapes[ins[1]]
                shapes[out0] = idx_sh[:-1] + sh0[idx_sh[-1] :]
            else:
                raise ValueError(f"Unsupported ONNX op for static compilation: {op}")
            self._ops.append((op, ins, out0, attrs))

        self._in_name = in_info.name
        self._out_name = model.graph.output[0].name

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the lowered ONNX policy graph on ``obs``."""
        env: dict[str, torch.Tensor] = {self._in_name: obs}
        for orig, bname in self._buf_map.items():
            env[orig] = getattr(self, bname)
        for op, ins, out0, attrs in self._ops:
            if op == "Add":
                env[out0] = env[ins[0]] + env[ins[1]]
            elif op == "Sub":
                env[out0] = env[ins[0]] - env[ins[1]]
            elif op == "Mul":
                env[out0] = env[ins[0]] * env[ins[1]]
            elif op == "Div":
                a, b = env[ins[0]], env[ins[1]]
                env[out0] = (
                    torch.div(a, b, rounding_mode="trunc")
                    if not a.is_floating_point()
                    else a / b
                )
            elif op == "Min":
                env[out0] = torch.minimum(env[ins[0]], env[ins[1]])
            elif op == "Clip":
                lo = env[ins[1]] if len(ins) > 1 and ins[1] else None
                hi = env[ins[2]] if len(ins) > 2 and ins[2] else None
                env[out0] = torch.clamp(env[ins[0]], min=lo, max=hi)
            elif op == "Where":
                env[out0] = torch.where(env[ins[0]], env[ins[1]], env[ins[2]])
            elif op == "Equal":
                env[out0] = torch.eq(env[ins[0]], env[ins[1]])
            elif op == "Greater":
                env[out0] = torch.gt(env[ins[0]], env[ins[1]])
            elif op == "And":
                env[out0] = env[ins[0]] & env[ins[1]]
            elif op == "Not":
                env[out0] = ~env[ins[0]]
            elif op == "Cast":
                env[out0] = env[ins[0]].to(_ONNX_DTYPE[int(attrs["to"])])
            elif op == "Slice":
                x = env[ins[0]]
                for ax, st, en, sp in attrs["_slice_specs"]:
                    sl = [slice(None)] * x.ndim
                    sl[ax] = slice(st, en, sp)
                    x = x[tuple(sl)]
                env[out0] = x
            elif op == "Reshape":
                env[out0] = env[ins[0]].reshape(attrs["_shape"])
            elif op == "Expand":
                env[out0] = env[ins[0]].expand(attrs["_shape"])
            elif op == "Unsqueeze":
                x = env[ins[0]]
                for ax in attrs["_axes"]:
                    x = x.unsqueeze(ax)
                env[out0] = x
            elif op == "Squeeze":
                x = env[ins[0]]
                for ax in attrs["_axes"]:
                    x = x.squeeze(ax)
                env[out0] = x
            elif op == "Concat":
                env[out0] = torch.cat([env[k] for k in ins], dim=attrs["_axis"])
            elif op == "Transpose":
                env[out0] = env[ins[0]].permute(attrs["_perm"])
            elif op == "ArgMax":
                env[out0] = torch.argmax(
                    env[ins[0]], dim=attrs["_axis"], keepdim=attrs["_keepdims"]
                )
            elif op == "ReduceSum":
                env[out0] = torch.sum(
                    env[ins[0]], dim=attrs["_axes"], keepdim=attrs["_keepdims"]
                )
            elif op == "Gemm":
                a, b = env[ins[0]], env[ins[1]]
                c = env[ins[2]] if len(ins) > 2 and ins[2] else None
                if int(attrs.get("transA", 0)):
                    a = a.transpose(-2, -1)
                if (
                    int(attrs.get("transB", 1)) == 1
                    and c is not None
                    and float(attrs.get("alpha", 1.0)) == 1.0
                    and float(attrs.get("beta", 1.0)) == 1.0
                ):
                    env[out0] = F.linear(a, b, c)
                else:
                    if int(attrs.get("transB", 0)):
                        b = b.transpose(-2, -1)
                    y = float(attrs.get("alpha", 1.0)) * torch.matmul(a, b)
                    if c is not None:
                        y = y + float(attrs.get("beta", 1.0)) * c
                    env[out0] = y
            elif op == "MatMul":
                env[out0] = torch.matmul(env[ins[0]], env[ins[1]])
            elif op == "Gather":
                x = env[ins[0]]
                ax = attrs["_axis"]
                sl = [slice(None)] * x.ndim
                sl[ax] = (
                    attrs["_scalar_idx"]
                    if "_scalar_idx" in attrs
                    else env[ins[1]].long()
                )
                env[out0] = x[tuple(sl)]
            elif op == "GatherElements":
                env[out0] = torch.gather(
                    env[ins[0]], attrs["_axis"], env[ins[1]].long()
                )
            elif op == "GatherND":
                x = env[ins[0]]
                idx_t = env[ins[1]].long()
                env[out0] = x[tuple(idx_t[..., j] for j in range(idx_t.shape[-1]))]
            elif op == "Softmax":
                env[out0] = torch.softmax(env[ins[0]], dim=int(attrs.get("axis", -1)))
            elif op == "LayerNormalization":
                eps = float(attrs.get("epsilon", 1e-5))
                w = env[ins[1]]
                b = env[ins[2]] if len(ins) > 2 and ins[2] else None
                env[out0] = F.layer_norm(env[ins[0]], w.shape, w, b, eps)
            elif op == "Erf":
                env[out0] = torch.erf(env[ins[0]])
            elif op == "Relu":
                env[out0] = F.relu(env[ins[0]])
            elif op == "Sigmoid":
                env[out0] = torch.sigmoid(env[ins[0]])
            elif op == "Tanh":
                env[out0] = torch.tanh(env[ins[0]])
        return env[self._out_name]


class FastSerialFK(nn.Module):
    """Zero-graph-break GPU serial-chain FK matching ``pytorch_kinematics``."""

    def __init__(
        self, serial_chain: Any, tcp_matrix: torch.Tensor | None, device: torch.device
    ) -> None:
        """Extract static link offsets and revolute axes from ``serial_chain``."""
        super().__init__()
        self.jtypes: list[str] = []
        frames = serial_chain._serial_frames
        for idx, frame in enumerate(frames):
            off = (
                frame.joint.offset.get_matrix()[0]
                .detach()
                .to(device=device, dtype=torch.float32)
                if frame.joint.offset is not None
                else torch.eye(4, device=device, dtype=torch.float32)
            )
            self.jtypes.append(frame.joint.joint_type)
            self.register_buffer(f"T_{idx}", off)
            if frame.joint.axis is not None:
                ax = frame.joint.axis.detach().to(device=device, dtype=torch.float32)
                self.register_buffer(f"ax_{idx}", ax / ax.norm().clamp(min=1e-8))
        self.n = len(frames)
        if tcp_matrix is not None:
            self.register_buffer(
                "tcp", tcp_matrix.detach().to(device=device, dtype=torch.float32)
            )
            self.has_tcp = True
        else:
            self.has_tcp = False

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Compute batched ``(B, 4, 4)`` end-effector transforms for joints ``q``."""
        b = q.shape[0]
        t_mat = (
            torch.eye(4, device=q.device, dtype=q.dtype).unsqueeze(0).expand(b, 4, 4)
        )
        j = 0
        for idx in range(self.n):
            off = getattr(self, f"T_{idx}").unsqueeze(0)
            t_mat = torch.matmul(t_mat, off)
            if self.jtypes[idx] == "revolute":
                ang = q[:, j]
                j += 1
                ax = getattr(self, f"ax_{idx}")
                c = torch.cos(ang)
                s = torch.sin(ang)
                omc = 1.0 - c
                ux, uy, uz = ax[0], ax[1], ax[2]
                r00 = c + ux * ux * omc
                r01 = ux * uy * omc - uz * s
                r02 = ux * uz * omc + uy * s
                r10 = uy * ux * omc + uz * s
                r11 = c + uy * uy * omc
                r12 = uy * uz * omc - ux * s
                r20 = uz * ux * omc - uy * s
                r21 = uz * uy * omc + ux * s
                r22 = c + uz * uz * omc
                z = torch.zeros_like(c)
                o = torch.ones_like(c)
                rot = torch.stack(
                    [
                        torch.stack([r00, r01, r02, z], dim=-1),
                        torch.stack([r10, r11, r12, z], dim=-1),
                        torch.stack([r20, r21, r22, z], dim=-1),
                        torch.stack([z, z, z, o], dim=-1),
                    ],
                    dim=-2,
                )
                t_mat = torch.matmul(t_mat, rot)
        if self.has_tcp:
            t_mat = torch.matmul(t_mat, self.tcp.unsqueeze(0))
        return t_mat


class CompiledStepModule(nn.Module):
    """Single closed-loop control step: obs -> actor -> clip -> FK -> arrival."""

    def __init__(
        self,
        actor: nn.Module,
        fast_fk: FastSerialFK,
        *,
        num_waypoints: int,
        action_dim: int,
        action_scale: float,
        use_relative_obs: bool,
        canonicalize_quat_obs: bool,
        joint_periodic_features: bool,
        pos_eps: float,
        rot_eps: float,
        joint_eps: float,
        tcp_origin: torch.Tensor | None,
    ) -> None:
        """Configure static rollout buffers and arrival tolerances."""
        super().__init__()
        self.actor = actor
        self.fast_fk = fast_fk
        self.num_waypoints = num_waypoints
        self.action_dim = action_dim
        self.action_scale = float(action_scale)
        self.use_relative_obs = use_relative_obs
        self.canonicalize_quat_obs = canonicalize_quat_obs
        self.joint_periodic_features = joint_periodic_features
        self.pos_eps = float(pos_eps)
        self.rot_eps = float(rot_eps)
        self.joint_eps = float(joint_eps)
        if tcp_origin is not None:
            self.register_buffer("tcp_origin", tcp_origin.detach().float())
            self.has_tcp_origin = True
        else:
            self.has_tcp_origin = False
        self.register_buffer(
            "ident_quat", torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        )
        self.register_buffer("slot_ids", torch.arange(num_waypoints, dtype=torch.long))

    def forward(
        self,
        qpos: torch.Tensor,
        ee_pose: torch.Tensor,
        last_action: torch.Tensor,
        active_idx: torch.Tensor,
        converged: torch.Tensor,
        path_lengths: torch.Tensor,
        arrival_indices: torch.Tensor,
        pos_block: torch.Tensor,
        quat_block: torch.Tensor,
        joint_block: torch.Tensor,
        valid_mask: torch.Tensor,
        pos_mask: torch.Tensor,
        rot_mask: torch.Tensor,
        joint_mask: torch.Tensor,
        episode_k: torch.Tensor,
        step_limits: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        base_pose: torch.Tensor,
        policy_left: torch.Tensor,
        policy_right: torch.Tensor,
        step_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Execute one fused closed-loop step across a static batch."""
        b = qpos.shape[0]
        arm_q = qpos[:, : self.action_dim]
        active_clamped = torch.clamp(active_idx, max=self.num_waypoints - 1)
        active_onehot = (self.slot_ids.unsqueeze(0) == active_clamped.unsqueeze(1)).to(
            qpos.dtype
        )

        parts = [
            arm_q,
            ee_pose,
            pos_block.reshape(b, self.num_waypoints * 3),
            quat_block.reshape(b, self.num_waypoints * 4),
            joint_block.reshape(b, self.num_waypoints * self.action_dim),
            active_onehot,
            valid_mask,
            pos_mask,
            rot_mask,
            joint_mask,
            last_action,
        ]
        if self.use_relative_obs:
            inv_eef = _quat_inverse_xyzw(ee_pose[:, 3:7])
            rel_pos = (pos_block - ee_pose[:, None, :3]) * pos_mask.unsqueeze(-1)
            rel_quat = _quat_mul_xyzw(
                quat_block, inv_eef[:, None, :].expand_as(quat_block)
            )
            if self.canonicalize_quat_obs:
                rel_quat = _canonicalize_quat_xyzw(rel_quat)
            rel_quat = torch.where(
                rot_mask.unsqueeze(-1) > 0.5,
                rel_quat,
                self.ident_quat.view(1, 1, 4),
            )
            joint_err = (joint_block - arm_q[:, None, :]) * joint_mask.unsqueeze(-1)
            parts.extend(
                [
                    rel_pos.reshape(b, self.num_waypoints * 3),
                    rel_quat.reshape(b, self.num_waypoints * 4),
                    joint_err.reshape(b, self.num_waypoints * self.action_dim),
                ]
            )
        if self.joint_periodic_features:
            parts.extend([arm_q.sin(), arm_q.cos()])
        obs = torch.cat(parts, dim=-1)

        raw_action = self.actor(obs).clamp(-1.0, 1.0)
        active = (~converged) & (step_idx < step_limits)
        action = torch.where(
            active.unsqueeze(-1), raw_action, torch.zeros_like(raw_action)
        )
        next_arm_q = torch.clamp(arm_q + action * self.action_scale, lower, upper)
        next_qpos = (
            torch.cat([next_arm_q, qpos[:, self.action_dim :]], dim=-1)
            if qpos.shape[1] > self.action_dim
            else next_arm_q
        )
        next_last_action = torch.where(active.unsqueeze(-1), action, last_action)

        local_fk = self.fast_fk(next_arm_q)
        world_pose = torch.bmm(base_pose, local_fk)
        pol_mat = torch.bmm(torch.bmm(policy_left, world_pose), policy_right)
        pol_pos = pol_mat[:, :3, 3]
        pol_quat = _quat_from_rotmat_xyzw(pol_mat[:, :3, :3])
        if self.canonicalize_quat_obs:
            pol_quat = _canonicalize_quat_xyzw(pol_quat)
        next_ee_pose = torch.cat([pol_pos, pol_quat], dim=-1)

        next_path_lengths = path_lengths + active.to(path_lengths.dtype)

        w = active_onehot.unsqueeze(1)
        active_pos = torch.bmm(w, pos_block).squeeze(1)
        active_quat = torch.bmm(w, quat_block).squeeze(1)
        active_joint = torch.bmm(w, joint_block).squeeze(1)
        active_pos_mask = (active_onehot * pos_mask).sum(dim=-1) > 0.5
        active_rot_mask = (active_onehot * rot_mask).sum(dim=-1) > 0.5
        active_joint_mask = (active_onehot * joint_mask).sum(dim=-1) > 0.5

        pos_err = next_ee_pose[:, :3] - active_pos
        if self.has_tcp_origin:
            pos_err = (
                pos_err
                + (
                    matrix_from_quat(next_ee_pose[:, 3:7])
                    - matrix_from_quat(active_quat)
                )
                @ self.tcp_origin
            )
        pos_dist = pos_err.norm(dim=-1)
        rot_dist = _quat_err_angle(next_ee_pose[:, 3:7], active_quat)
        joint_dist = torch.amax(torch.abs(next_arm_q - active_joint), dim=-1)

        pos_ok = torch.where(
            active_pos_mask, pos_dist < self.pos_eps, torch.ones_like(active_pos_mask)
        )
        rot_ok = torch.where(
            active_rot_mask, rot_dist < self.rot_eps, torch.ones_like(active_rot_mask)
        )
        joint_ok = torch.where(
            active_joint_mask,
            joint_dist <= self.joint_eps,
            torch.ones_like(active_joint_mask),
        )
        reached = active & pos_ok & rot_ok & joint_ok

        slot_mask = self.slot_ids.unsqueeze(0) == active_clamped.unsqueeze(1)
        next_arrival_indices = torch.where(
            slot_mask & reached.unsqueeze(1),
            (step_idx + 1).to(arrival_indices.dtype).expand_as(arrival_indices),
            arrival_indices,
        )
        next_active_idx = torch.where(reached, active_idx + 1, active_idx)
        next_converged = converged | (next_active_idx >= episode_k)
        done_all = (next_converged | ((step_idx + 1) >= step_limits)).all()
        return (
            next_qpos,
            world_pose,
            next_ee_pose,
            next_last_action,
            next_active_idx,
            next_converged,
            next_path_lengths,
            next_arrival_indices,
            done_all,
        )


_ONNX_ACTOR_CACHE: dict[tuple[str, int, str], nn.Module] = {}


def load_compiled_onnx_actor(
    onnx_path: str,
    batch_size: int,
    device: torch.device,
) -> nn.Module:
    """Load and cache a static-batch PyTorch module directly from ``onnx_path``."""
    key = (str(Path(onnx_path).resolve()), int(batch_size), str(device))
    if key not in _ONNX_ACTOR_CACHE:
        _ONNX_ACTOR_CACHE[key] = StaticBatchOnnxModule(
            onnx_path, batch_size, device
        ).eval()
    return _ONNX_ACTOR_CACHE[key]
