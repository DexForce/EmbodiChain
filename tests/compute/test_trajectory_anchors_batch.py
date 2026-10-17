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

"""Ragged batch, backend parity and exact-event contracts for anchored resampling."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.compute.trajectory import (
    prepare_anchored_batch,
    resample_with_anchors,
    resample_with_anchors_batched,
)

_EXECUTIONS = [
    ("cpu", "torch"),
    ("cpu", "warp"),
    pytest.param("cuda:0", "torch", marks=pytest.mark.gpu),
    pytest.param("cuda:0", "warp", marks=pytest.mark.gpu),
]


def _inputs(dtype=torch.float64, device="cpu"):
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    times = torch.full((3, 6), float("nan"), dtype=torch.float64)
    q = torch.full((3, 6, 4), float("nan"), dtype=dtype)
    clocks = [
        [0.0, 0.011, 0.021, 0.041, 0.070, 0.097],
        [0.0, 0.025, 0.047, 0.083],
        [0.0, 0.09],
    ]
    for b, clock in enumerate(clocks):
        t = torch.tensor(clock, dtype=torch.float64)
        times[b, : len(t)] = t
        q[b, : len(t)] = torch.stack(
            (torch.sin(t * 20) + b, t * t, -t, t * 0 + 3), dim=1
        ).to(dtype)
    q[0, 2, 3] = -0.0
    return (
        q.to(device),
        times,
        torch.tensor([[4, 2, 2, -999], [2, 1, -999, -999], [-999] * 4]),
        dict(
            source_lengths=torch.tensor([6, 4, 2]),
            anchor_counts=torch.tensor([3, 2, 0]),
        ),
    )


def _assert_reference(result, plan, q, times, anchors, args):
    host = result.detach().cpu().numpy()
    source = q.detach().cpu().numpy()
    for b, row in enumerate(plan.rows):
        n = args["source_lengths"][b].item()
        k = args["anchor_counts"][b].item()
        expected, mapping = resample_with_anchors(
            source[b, :n], times[b, :n].numpy(), 0.01, anchors[b, :k].tolist()
        )
        m = len(expected)
        tolerance = 2e-7 if q.dtype == torch.float32 else 2e-14
        np.testing.assert_allclose(
            host[b, :m], expected, atol=tolerance, rtol=tolerance
        )
        for field in mapping.__dataclass_fields__:
            np.testing.assert_array_equal(getattr(row, field), getattr(mapping, field))
        # Check the representation too: copying anchors must preserve signed zero.
        bits = np.int32 if q.dtype == torch.float32 else np.int64
        np.testing.assert_array_equal(
            host[b, row.anchor_execution_indices].view(bits),
            source[b, row.anchor_source_indices].view(bits),
        )
        np.testing.assert_array_equal(
            host[b, m:], np.broadcast_to(source[b, n - 1], host[b, m:].shape)
        )
        assert plan.valid_counts[b] == m
        assert plan.valid_mask[b].sum() == m
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("device,backend", _EXECUTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ragged_batch_matches_single_paths_and_preserves_exact_anchors(
    device, backend, dtype
):
    q, times, anchors, args = _inputs(dtype, device)
    original = q.clone()
    result, plan = resample_with_anchors_batched(
        q, times, 0.01, anchors, backend=backend, **args
    )
    _assert_reference(result, plan, q, times, anchors, args)
    assert result.device == q.device and result.dtype == dtype
    assert plan.valid_counts.device == q.device
    torch.testing.assert_close(q, original, rtol=0, atol=0, equal_nan=True)
    # Returned metadata is owned, and mutations cannot change a reusable plan.
    counts = plan.valid_counts
    counts.zero_()
    assert (plan.valid_counts >= 2).all()


@pytest.mark.parametrize("device,backend", _EXECUTIONS)
def test_reuses_mapping_for_changed_values_and_noncontiguous_paths(device, backend):
    q, times, anchors, args = _inputs(device=device)
    plan = prepare_anchored_batch(times, 0.01, anchors, device=device, **args)
    storage = torch.stack((q, q), dim=-1)
    changed = storage[..., 0]
    assert not changed.is_contiguous()
    changed.add_(0.125)
    result = plan.apply(changed, backend=backend)
    _assert_reference(result, plan, changed, times, anchors, args)
    fresh, _ = resample_with_anchors_batched(
        changed, times, 0.01, anchors, backend=backend, **args
    )
    torch.testing.assert_close(result, fresh, rtol=0, atol=0)


@pytest.mark.parametrize("device,backend", _EXECUTIONS)
def test_random_batches_match_independent_scalar_interpolation(device, backend):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    rng = np.random.default_rng(192)
    batch, width, dimensions = 13, 41, 7
    lengths = rng.integers(10, width + 1, size=batch)
    times = np.full((batch, width), np.nan)
    values = np.full((batch, width, dimensions), np.nan)
    anchors = np.full((batch, 4), -1)
    for b, n in enumerate(lengths):
        times[b, :n] = np.r_[0.0, np.cumsum(rng.uniform(0.015, 0.09, n - 1))]
        values[b, :n] = rng.normal(size=(n, dimensions))
        anchors[b] = rng.choice(n, size=4, replace=False)
    q = torch.tensor(values, device=device)
    result, plan = resample_with_anchors_batched(
        q,
        torch.tensor(times),
        0.01,
        torch.tensor(anchors),
        source_lengths=torch.tensor(lengths),
        anchor_counts=torch.full((batch,), 4),
        backend=backend,
    )
    for b, row in enumerate(plan.rows):
        n, m = lengths[b], len(row.execution_times)
        expected = np.stack(
            [
                np.interp(row.source_query_times, times[b, :n], values[b, :n, d])
                for d in range(dimensions)
            ],
            axis=1,
        )
        np.testing.assert_allclose(
            result[b, :m].cpu().numpy(), expected, atol=2e-14, rtol=2e-14
        )


def test_known_event_tick_mapping_and_stationary_trajectory():
    q = torch.tensor(
        [[[2.0], [5.0], [8.0]], [[-1.0], [-1.0], [-1.0]]], dtype=torch.float64
    )
    times = torch.tensor([[0.0, 0.021, 0.070], [0.0, 0.03, 0.04]], dtype=torch.float64)
    out, plan = resample_with_anchors_batched(
        q,
        times,
        0.01,
        torch.tensor([[1], [1]]),
        source_lengths=torch.tensor([3, 3]),
        anchor_counts=torch.tensor([1, 1]),
    )
    np.testing.assert_array_equal(plan.rows[0].anchor_execution_indices, [0, 3, 7])
    np.testing.assert_allclose(
        out[0, :, 0], [2, 3, 4, 5, 5.75, 6.5, 7.25, 8], atol=1e-14
    )
    assert torch.equal(out[1], torch.full_like(out[1], -1))
    assert plan.valid_counts.tolist() == [8, 5]


@pytest.mark.parametrize(
    "bad_field", ["lengths", "counts", "anchors", "clock", "finite"]
)
def test_invalid_valid_prefixes_fail_with_no_implicit_row_dropping(bad_field):
    q, times, anchors, args = _inputs()
    if bad_field == "lengths":
        args["source_lengths"][1] = 7
    elif bad_field == "counts":
        args["anchor_counts"][1] = 5
    elif bad_field == "anchors":
        anchors[1, 0] = 4
    elif bad_field == "clock":
        times[1, 1] = times[1, 0]
    else:
        q[1, 1] = float("nan")
    with pytest.raises(ValueError):
        resample_with_anchors_batched(q, times, 0.01, anchors, **args)


def test_dense_events_report_the_failing_row():
    q = torch.zeros((2, 4, 2), dtype=torch.float64)
    times = torch.tensor(
        [[0.0, 0.011, 0.021, 0.04], [0.0, 0.011, 0.012, 0.04]], dtype=torch.float64
    )
    with pytest.raises(ValueError, match="Batch row 1:.*anchors cannot fit"):
        resample_with_anchors_batched(
            q,
            times,
            0.01,
            torch.tensor([[1, 2], [1, 2]]),
            source_lengths=torch.tensor([4, 4]),
            anchor_counts=torch.tensor([2, 2]),
        )


def test_total_padding_and_output_element_bounds_are_checked():
    q, times, anchors, args = _inputs()
    with pytest.raises(ValueError, match="max_samples"):
        resample_with_anchors_batched(q, times, 0.01, anchors, max_samples=20, **args)
    plan = prepare_anchored_batch(times, 0.01, anchors, **args)
    with pytest.raises(ValueError, match="max_output_elements"):
        plan.apply(q, max_output_elements=10)


@pytest.mark.parametrize(
    "field",
    [
        "source_lengths",
        "anchor_counts",
        "anchor_indices",
        "positions",
        "source_times",
        "shape",
    ],
)
def test_invalid_shapes_and_dtypes_are_rejected(field):
    q, times, anchors, args = _inputs()
    if field in args:
        args[field] = args[field].float()
    elif field == "anchor_indices":
        anchors = anchors.float()
    elif field == "positions":
        q = q.to(torch.float16)
    elif field == "source_times":
        times = times.long()
    else:
        q = q[:2]
    with pytest.raises(ValueError):
        resample_with_anchors_batched(q, times, 0.01, anchors, **args)


def test_zero_explicit_anchors_preserve_source_endpoints():
    q, times, _, args = _inputs()
    out, plan = resample_with_anchors_batched(
        q,
        times,
        0.01,
        torch.empty((3, 0), dtype=torch.long),
        source_lengths=args["source_lengths"],
        anchor_counts=torch.zeros(3, dtype=torch.long),
    )
    assert all(len(row.anchor_source_indices) == 2 for row in plan.rows)
    assert torch.equal(out[:, 0], q[:, 0])


def test_explicit_backend_errors_propagate_and_torch_autograd_remains_available(
    monkeypatch,
):
    from embodichain.compute.trajectory._warp import anchored

    q, times, anchors, args = _inputs()
    plan = prepare_anchored_batch(times, 0.01, anchors, **args)

    def fail(*args):
        raise RuntimeError("test launch failed")

    monkeypatch.setattr(anchored, "evaluate", fail)
    with pytest.raises(RuntimeError, match="test launch failed"):
        plan.apply(q, backend="warp")
    with pytest.raises(ValueError, match="backend"):
        plan.apply(q, backend="unknown")
    q.requires_grad_()
    with pytest.raises(ValueError, match="autograd"):
        plan.apply(q, backend="warp")
    plan.apply(q).sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    for b, n in enumerate(args["source_lengths"]):
        assert torch.count_nonzero(q.grad[b, n:]) == 0


@pytest.mark.gpu
def test_warp_follows_nondefault_torch_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    import warp as wp

    wp.init()
    q, times, anchors, args = _inputs(device="cuda:0")
    plan = prepare_anchored_batch(times, 0.01, anchors, device=q.device, **args)
    # Warm compilation separately, then order writes, launch and consumption on
    # one non-default Torch stream without relying on default-stream ordering.
    plan.apply(q, backend="warp")
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        changed = q + 0.25
        result = plan.apply(changed, backend="warp")
        consumed = result + 0.5
    stream.synchronize()
    torch.testing.assert_close(
        consumed, plan.apply(q + 0.25) + 0.5, rtol=1e-14, atol=1e-14
    )


def test_batch_row_permutation_does_not_change_individual_results():
    q, times, anchors, args = _inputs()
    out, _ = resample_with_anchors_batched(q, times, 0.01, anchors, **args)
    order = torch.tensor([2, 0, 1])
    reordered, plan = resample_with_anchors_batched(
        q[order],
        times[order],
        0.01,
        anchors[order],
        source_lengths=args["source_lengths"][order],
        anchor_counts=args["anchor_counts"][order],
    )
    torch.testing.assert_close(reordered, out[order], rtol=0, atol=0)
    assert plan.source_lengths.tolist() == [2, 6, 4]


@pytest.mark.gpu
def test_cuda_metadata_preparation_and_plan_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    q, times, anchors, args = _inputs(device="cuda:0")
    plan = prepare_anchored_batch(
        times.cuda(),
        0.01,
        anchors.cuda(),
        **{key: value.cuda() for key, value in args.items()},
    )
    assert plan.device == q.device
    _assert_reference(plan.apply(q, backend="warp"), plan, q, times, anchors, args)
    with pytest.raises(ValueError, match="plan device"):
        plan.apply(q.cpu())
