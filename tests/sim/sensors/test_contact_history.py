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

import pytest
import torch

from embodichain.lab.sim.sensors.contact_history import ContactHistory


def sample(pairs, valid=True):
    ids = torch.tensor(pairs)
    shape = ids.shape[:2]
    normal = torch.zeros((*shape, 3))
    normal[..., 2] = 1.0
    return dict(
        user_ids=ids,
        is_valid=torch.full(shape, valid),
        normal=normal,
        friction=torch.zeros_like(normal),
        impulse=torch.ones(shape),
    )


@pytest.mark.parametrize("pair", ((10, 11), (11, 10), (10, 99)))
def test_only_selected_counterpart_counts_as_support(pair):
    history = ContactHistory(torch.tensor([[10]]), counterpart_ids=torch.tensor([[0]]))
    history.update(sample([[pair]]), 0.1)
    assert not history.found.any()
    assert not history.force.any()


def test_unknown_counterpart_requires_explicit_opt_in():
    for allow in (False, True):
        history = ContactHistory(
            torch.tensor([[10]]),
            counterpart_ids=torch.tensor([[0]]),
            include_unknown_counterpart=allow,
        )
        history.update(sample([[[10, -1]]]), 0.1)
        assert history.found.item() == allow


def test_early_substep_contact_survives_later_empty_sample():
    history = ContactHistory(torch.tensor([[10]]))
    history.begin_control_step()
    history.update(sample([[[0, 10]]]), 0.1)
    history.update(sample([[[0, 10]]], valid=False), 0.1)
    assert not history.contact.item()
    assert history.found.item()
    assert history.first_contact.item()
    assert history.peak_force[0, 0, 2] == 10.0
    assert history.force[0, 0, 2] == 0.0


def test_landing_retains_completed_air_time_until_next_landing():
    history = ContactHistory(torch.tensor([[10]]))
    for _ in range(3):
        history.begin_control_step()
        history.update(sample([[[0, 10]]], valid=False), 0.1)
    history.begin_control_step()
    history.update(sample([[[0, 10]]]), 0.1)
    assert history.current_air_time.item() == 0.0
    assert history.last_air_time.item() == pytest.approx(0.4)
    assert history.first_contact.item()
    history.update(sample([[[0, 10]]]), 0.1)
    assert history.last_air_time.item() == pytest.approx(0.4)
    history.begin_control_step()
    assert not history.first_contact.item()


def test_selective_reset_preserves_every_untouched_history_buffer():
    history = ContactHistory(torch.tensor([[10], [20]]))
    history.update(sample([[[0, 10]], [[0, 20]]], valid=False), 0.1)
    history.update(sample([[[0, 10]], [[0, 20]]]), 0.1)
    fields = (
        "contact",
        "found",
        "first_contact",
        "force",
        "peak_force",
        "current_air_time",
        "last_air_time",
        "contact_count",
    )
    before = {name: getattr(history, name)[1].clone() for name in fields}
    history.reset([0])
    for name in fields:
        assert not getattr(history, name)[0].any()
        assert torch.equal(getattr(history, name)[1], before[name])


def test_actor_sides_preserve_signed_impulse_and_invalid_nan_is_ignored():
    history = ContactHistory(torch.tensor([[10]]))
    data = sample([[[10, 0], [0, 10], [0, 10]]])
    data["impulse"][0] = torch.tensor([1.0, 2.0, float("nan")])
    data["is_valid"][0, 2] = False
    history.update(data, 0.1)
    assert torch.equal(history.force, torch.tensor([[[0.0, 0.0, 10.0]]]))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
)
def test_sparse_rows_use_device_count_and_environment_qualified_actor_ids(device):
    from dexsim.scene import ContactBuffer

    history = ContactHistory(
        torch.tensor([[10, 11], [10, 11]], device=device),
        counterpart_ids=torch.tensor([[0], [0]], device=device),
    )
    buffer = ContactBuffer.allocate(6, device)
    if hasattr(buffer, "quantity_valid"):
        buffer.quantity_valid.fill_(True)
    buffer.data.fill_(float("nan"))
    buffer.data[:4] = torch.tensor(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 100, 0],
        ],
        device=device,
    )
    buffer.env_ids[:4] = torch.tensor([0, 1, 0, 0], device=device)
    buffer.actor_ids[:4] = torch.tensor(
        [[0, 10], [11, 0], [0, 10], [10, 11]], device=device
    )
    buffer.device_counts[0] = 4
    assert buffer.count == 0  # Host count remains unread, as with fetch_async().
    history._update_from_query(buffer, 0.1)
    torch.testing.assert_close(
        history.force[:, :, 2], torch.tensor([[30.0, 0.0], [0.0, -30.0]], device=device)
    )
    assert history.contact.tolist() == [[True, False], [False, True]]
    assert history.contact_count.tolist() == [1.0, 1.0]


@pytest.mark.parametrize("threshold,found", [(0.0, True), (1.0, False)])
def test_geometry_only_rows_do_not_create_force_or_force_threshold_events(
    threshold, found
):
    history = ContactHistory(torch.tensor([[10]]), force_threshold=threshold)
    data = sample([[[0, 10]]])
    data["impulse_valid"] = torch.zeros((1, 1), dtype=torch.bool)
    data["friction_valid"] = torch.zeros((1, 1), dtype=torch.bool)
    data["impulse"].fill_(float("nan"))
    data["friction"].fill_(float("nan"))
    history.update(data, 0.1)
    assert not history.force.any()
    assert history.found.item() == found


@pytest.mark.parametrize(
    "pair,expected", [((10, 0), (5.0, 0.0, -20.0)), ((0, 10), (-5.0, 0.0, 20.0))]
)
def test_force_preserves_friction_impulse_on_actor_zero(
    pair: tuple[int, int], expected: tuple[float, float, float]
) -> None:
    history = ContactHistory(torch.tensor([[10]]))
    data = sample([[pair]])
    data["impulse"][0, 0] = 2.0
    data["friction"][0, 0] = torch.tensor([0.5, 0.0, 0.0])
    history.update(data, 0.1)
    torch.testing.assert_close(history.force[0, 0], torch.tensor(expected))
