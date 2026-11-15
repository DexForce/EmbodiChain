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

"""Host replay transfers preserve samples, dtypes and result ownership."""

from __future__ import annotations

import pytest
import torch

from embodichain.learning.rl.experimental._host_replay import (
    _host_replay_state_dict,
    _HostReplayAdapter,
    _load_host_replay_state_dict,
)


class _Storage:
    device = torch.device("cpu")

    def __init__(self):
        self.features = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)[..., ::2]
        self.mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        self.added = None

    def add(self, transition):
        assert all(value.device.type == "cpu" for value in transition.values())
        self.added = {key: value.clone() for key, value in transition.items()}

    def sample_windows(self, batch, window):
        assert (batch, window) == (2, 3)
        return {"features": self.features, "mask": self.mask}

    def sample_starts(self, batch, history):
        assert (batch, history) == (2, 3)
        return {"features": self.features[:, 0]}, {"features": self.features}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_host_transfer_preserves_data_and_owns_results(device) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    storage = _Storage()
    adapter = _HostReplayAdapter(storage, device)
    source = (
        torch.arange(8, device=device, dtype=torch.float32)
        .reshape(2, 4)
        .requires_grad_()
    )
    done = torch.tensor([True, False], device=device)
    adapter.add({"x": source, "done": done})
    with torch.no_grad():
        source.add_(100)
    torch.testing.assert_close(
        storage.added["x"], torch.arange(8, dtype=torch.float32).reshape(2, 4)
    )
    assert not storage.added["x"].requires_grad
    assert storage.added["done"].dtype == torch.bool
    windows = adapter.sample_windows(2, 3)
    start, history = adapter.sample_starts(2, 3)
    expected = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8)[..., ::2].clone()
    storage.features.zero_()
    storage.mask.zero_()
    torch.testing.assert_close(windows["features"].cpu(), expected)
    torch.testing.assert_close(start["features"].cpu(), expected[:, 0])
    torch.testing.assert_close(history["features"].cpu(), expected)
    torch.testing.assert_close(
        windows["mask"].cpu(), torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    )
    for result in [windows, start, history]:
        assert all(
            value.device.type == device
            and value.dtype == torch.float32
            and not value.requires_grad
            for value in result.values()
        )


def test_host_transfer_rejects_device_resident_storage() -> None:
    storage = _Storage()
    storage.device = torch.device("cuda")
    with pytest.raises(ValueError, match="CPU"):
        _HostReplayAdapter(storage, "cpu")


class _Ring:
    device = torch.device("cpu")
    steps = 4
    num_envs = 2

    def __init__(self, ptr: int, size: int) -> None:
        self.ptr, self.size = ptr, size
        self.data = {
            "s": torch.arange(24, dtype=torch.float32).reshape(4, 2, 3),
            "ep_step": torch.arange(8, dtype=torch.int32).reshape(4, 2),
            "done": torch.tensor([[False, True]] * 4),
        }


@pytest.mark.parametrize("ptr,size,order", [(3, 3, [0, 1, 2]), (1, 4, [1, 2, 3, 0])])
def test_host_replay_checkpoint_preserves_ring_history_and_owns_restore(
    tmp_path, ptr, size, order
) -> None:
    source = _Ring(ptr, size)
    path = tmp_path / "replay.pt"
    torch.save(_host_replay_state_dict(source), path)
    state = torch.load(path, weights_only=True)
    target = _Ring(0, 0)
    for tensor in target.data.values():
        tensor.zero_()
    _load_host_replay_state_dict(target, state)
    assert (target.ptr, target.size) == (ptr, size)
    for key, tensor in source.data.items():
        torch.testing.assert_close(target.data[key], tensor)
        assert target.data[key].data_ptr() != state["data"][key].data_ptr()
    oldest = (target.ptr - target.size) % target.steps
    indices = (oldest + torch.arange(size)) % target.steps
    torch.testing.assert_close(
        target.data["ep_step"][indices], source.data["ep_step"][order]
    )
    state["data"]["s"].zero_()
    torch.testing.assert_close(target.data["s"], source.data["s"])


@pytest.mark.parametrize("mismatch", ["capacity", "cursor", "dtype", "shape", "fields"])
def test_host_replay_checkpoint_rejects_mismatch_before_mutation(mismatch) -> None:
    source = _Ring(1, 4)
    state = _host_replay_state_dict(source)
    if mismatch == "capacity":
        state["steps"] += 1
    elif mismatch == "cursor":
        state["ptr"] = 5
    elif mismatch == "dtype":
        state["data"]["done"] = state["data"]["done"].float()
    elif mismatch == "shape":
        state["data"]["done"] = state["data"]["done"][:1]
    else:
        del state["data"]["done"]
    target = _Ring(0, 0)
    for tensor in target.data.values():
        tensor.zero_()
    with pytest.raises(ValueError):
        _load_host_replay_state_dict(target, state)
    assert (target.ptr, target.size) == (0, 0)
    assert all(not tensor.any() for tensor in target.data.values())
