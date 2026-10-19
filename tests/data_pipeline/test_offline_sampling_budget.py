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

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from tensordict import TensorDict
from embodichain.data_pipeline.datasets import (
    TensorDataSource,
    SequenceSpecCfg,
)

from embodichain.data_pipeline.engine.data import OnlineDataBudgetExhaustedError


def _corpus() -> TensorDataSource:
    valid = torch.tensor([[True] * 4, [True, True, False, False], [False] * 4])
    ids = torch.zeros(3, 4, 3, dtype=torch.int64)
    ids[..., 0] = 12
    ids[..., 1] = torch.arange(3)[:, None]
    ids[..., 2] = torch.arange(4)
    return TensorDataSource(
        TensorDict(
            {
                "obs": torch.arange(24, dtype=torch.float32).reshape(3, 4, 2),
                "actions": torch.arange(12, dtype=torch.float32).reshape(3, 4, 1),
                "valid": valid,
                "frame_ids": ids,
                "generations": torch.tensor([[1] * 4, [2] * 4, [3] * 4]),
            },
            batch_size=valid.shape,
        ),
        schema={},
        source_id="quota-test",
    )


def _sample(
    corpus: TensorDataSource, count: int, generator: torch.Generator | None = None
) -> dict[str, torch.Tensor]:
    return corpus.sample_context(count, SequenceSpecCfg(2, 3), generator=generator)


def test_context_draws_spend_each_source_allowance_once() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(3, 5)
    for _ in range(3):
        _sample(corpus, 1)
    audit = corpus.get_sampling_budget_snapshot()
    assert audit["total_delivered_chunks"] == audit["remaining_chunks"] == 3
    assert len(audit["rows"]) == 2
    assert all(0 <= row["delivered"] <= 3 for row in audit["rows"])


def test_depletion_counts_duplicate_batch_rows_and_leaves_no_partial_spend() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(3, 3)
    _sample(corpus, 5, torch.Generator().manual_seed(7))
    before = corpus.get_sampling_budget_snapshot()
    assert sorted(row["delivered"] for row in before["rows"]) == [2, 3]
    with pytest.raises(OnlineDataBudgetExhaustedError):
        _sample(corpus, 2)
    assert corpus.get_sampling_budget_snapshot() == before
    _sample(corpus, 1)
    assert corpus.get_sampling_budget_snapshot()["remaining_chunks"] == 0


def test_copy_failure_never_debits(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(3, 3)
    before = corpus.get_sampling_budget_snapshot()

    def fail(*args: object) -> None:
        raise MemoryError("failed copy")

    monkeypatch.setattr(corpus, "_select", fail)
    with pytest.raises(MemoryError):
        _sample(corpus, 2)
    assert corpus.get_sampling_budget_snapshot() == before


def test_post_copy_failure_does_not_debit(monkeypatch: pytest.MonkeyPatch) -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(3, 3)
    select = corpus._select

    def invalid_shape(
        rows: torch.Tensor, times: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        result = select(rows, times)
        result["obs"] = torch.empty(0)
        return result

    monkeypatch.setattr(corpus, "_select", invalid_shape)
    with pytest.raises(RuntimeError, match="batch dimension mismatch"):
        _sample(corpus, 2)
    assert corpus.get_sampling_budget_snapshot()["total_delivered_chunks"] == 0


def test_generous_budget_preserves_unlimited_rng_and_copied_results() -> None:
    baseline, bounded = _corpus(), _corpus()
    bounded.configure_sampling_budget(1000, 1000)
    rng_a, rng_b = torch.Generator().manual_seed(31), torch.Generator().manual_seed(31)
    for _ in range(3):
        first, second = _sample(baseline, 7, rng_a), _sample(bounded, 7, rng_b)
        assert first.keys() == second.keys()
        assert all(torch.equal(first[key], second[key]) for key in first.keys())
        assert torch.equal(rng_a.get_state(), rng_b.get_state())
        second["obs"].fill_(-100)
        assert not (bounded.data["obs"] == -100).any()


@pytest.mark.parametrize(
    "budget,limit", [(0, 1), (1, 0), (3, 2), (True, 2), (2, 2.0), (2**63, 2**63)]
)
def test_invalid_budget_rejected_without_enabling(
    budget: object, limit: object
) -> None:
    corpus = _corpus()
    with pytest.raises(ValueError):
        corpus.configure_sampling_budget(budget, limit)
    assert corpus.get_sampling_budget_snapshot()["enabled"] is False


def test_budget_cannot_be_reset_or_enabled_after_unlimited_sampling() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(2, 3)
    _sample(corpus, 1)
    with pytest.raises(RuntimeError, match="once"):
        corpus.configure_sampling_budget(3, 3)
    other = _corpus()
    _sample(other, 1)
    with pytest.raises(RuntimeError, match="before sampling"):
        other.configure_sampling_budget(2, 2)


@pytest.mark.parametrize("duplicate", [False, True])
def test_identity_validation_prevents_budget_multiplication(duplicate: bool) -> None:
    corpus = _corpus()
    if duplicate:
        corpus.data["frame_ids"][1, :, 1] = 0
    else:
        corpus.data["frame_ids"][0, 1, 1] = 7
    with pytest.raises(ValueError, match="identity"):
        corpus.configure_sampling_budget(2, 2)
    assert corpus.get_sampling_budget_snapshot()["enabled"] is False


def test_identity_mutation_after_assignment_rejected() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(2, 2)
    corpus.data["frame_ids"][0, :, 1] = 8
    with pytest.raises(RuntimeError, match="identity changed"):
        corpus.sample_context(1, SequenceSpecCfg(2, 3, generations=(1,)))
    assert corpus.get_sampling_budget_snapshot()["total_delivered_chunks"] == 0


def test_generation_filter_does_not_borrow_other_source_budget() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(2, 2)
    corpus.sample_context(2, SequenceSpecCfg(2, 3, generations=(1,)))
    with pytest.raises(OnlineDataBudgetExhaustedError):
        corpus.sample_context(1, SequenceSpecCfg(2, 3, generations=(1,)))
    assert corpus.get_sampling_budget_snapshot()["remaining_chunks"] == 2


def test_budgeted_sampler_rejects_other_thread_and_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(2, 2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_sample, corpus, 1)
        with pytest.raises(RuntimeError, match="owner-process/thread"):
            future.result()
    monkeypatch.setattr(
        "embodichain.data_pipeline.datasets.sources.os.getpid", lambda: -1
    )
    with pytest.raises(RuntimeError, match="owner-process/thread"):
        _sample(corpus, 1)
    with pytest.raises(RuntimeError, match="owner-process/thread"):
        corpus.get_sampling_budget_snapshot()


def test_cpu_sampling_survives_non_cpu_default_device() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(3, 3)
    with torch.device("meta"):
        batch = _sample(corpus, 2, torch.Generator(device="cpu").manual_seed(2))
    assert all(t.device.type == "cpu" for t in batch.values())


def test_snapshot_is_detached_and_counts_large_limits_without_expansion() -> None:
    corpus = _corpus()
    corpus.configure_sampling_budget(2**62, 2**63 - 1)
    _sample(corpus, 256)
    audit = corpus.get_sampling_budget_snapshot()
    assert audit["total_delivered_chunks"] == 256
    audit["rows"][0]["delivered"] = -1
    assert corpus.get_sampling_budget_snapshot()["rows"][0]["delivered"] >= 0
