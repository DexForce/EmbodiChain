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
"""CPU fake-state checks for pre-reset native physics diagnostics."""

from __future__ import annotations

from pathlib import Path
import enum
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.benchmark.assembly.peg_reachability import _PhysicsHealth


@pytest.mark.parametrize(
    "flag,capacity,convergence,unknown",
    [(1, True, False, False), (4, False, True, False), (16, False, False, True)],
)
def test_composite_enum_aliases_do_not_change_overflow_classification(
    monkeypatch: pytest.MonkeyPatch,
    flag: int,
    capacity: bool,
    convergence: bool,
    unknown: bool,
) -> None:
    from mujoco_warp._src import types

    class Flags(enum.IntFlag):
        NONE = 0
        NEFC = 1
        BROADPHASE = 2
        ITERATIONS = 4
        LS_ITERATIONS = 8
        ALL = 15

    monkeypatch.setattr(types, "OverflowType", Flags)
    env, data = _environment()
    data.overflow.values[1] = flag
    health = _PhysicsHealth(env, scope="control", strict=True)
    if capacity or unknown:
        with pytest.raises(RuntimeError, match="capacity/collision loss"):
            env.sim.update(0.001, step=20)
    else:
        env.sim.update(0.001, step=20)
    assert health.values["capacity_overflow_seen"] is capacity
    assert health.values["convergence_budget_exhausted_seen"] is convergence
    assert health.values["unclassified_overflow_seen"] is unknown
    assert "ALL" not in health.values["overflow_flags"]


@pytest.mark.parametrize(
    "warp_version,mujoco_version,horizon,passed",
    [
        ("3.12.0", "3.12.0", 128, True),
        ("3.15.0", "3.15.0", 128, True),
        ("3.15.0", "3.12.0", 128, False),
        ("3.15.0", "3.15.0", 24, False),
    ],
)
def test_epa_preflight_requires_prepared_and_matching_reference(
    monkeypatch, warp_version, mujoco_version, horizon, passed
) -> None:
    from mujoco_warp._src import collision_convex, types
    from scripts.benchmark.assembly import train_peg

    monkeypatch.setattr(
        train_peg,
        "version",
        lambda package: {"mujoco-warp": warp_version, "mujoco": mujoco_version}[
            package
        ],
    )
    monkeypatch.setattr(types, "MJ_MAX_EPAHORIZON", horizon)
    monkeypatch.setattr(collision_convex, "MJ_MAX_EPAHORIZON", horizon)
    assert train_peg._epa_preflight()["passed"] is passed


class _Array:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def numpy(self) -> np.ndarray:
        return self.values


class _Solver:
    def __init__(self, data: SimpleNamespace) -> None:
        self.mjw_data = data


def _environment() -> tuple[SimpleNamespace, SimpleNamespace]:
    """Build readable native-like buffers without any simulation runtime."""
    data = SimpleNamespace(
        overflow=_Array(np.zeros(3, dtype=np.int32)),
        qpos=_Array(np.zeros((3, 4), dtype=np.float32)),
        qvel=_Array(np.zeros((3, 3), dtype=np.float32)),
        qacc=_Array(np.zeros((3, 3), dtype=np.float32)),
    )
    solver = _Solver(data)
    part = SimpleNamespace(
        _entities=[
            SimpleNamespace(
                _physics_binding=SimpleNamespace(
                    _runtime=SimpleNamespace(solver=solver)
                )
            )
        ]
    )
    env = SimpleNamespace(
        physics_dt=0.001,
        num_envs=3,
        sim=SimpleNamespace(
            get_rigid_object=lambda uid: part, update=lambda *args, **kwargs: None
        ),
    )
    return env, data


@pytest.mark.parametrize(
    "field,value", [("qpos", np.nan), ("qvel", np.inf), ("qacc", -np.inf)]
)
def test_strict_health_rejects_nonfinite_native_state_before_reset(
    field: str, value: float
) -> None:
    env, data = _environment()
    getattr(data, field).values[1, 0] = value
    original = getattr(data, field).values.copy()
    health = _PhysicsHealth(
        env, scope="control", strict=True, include_world_flags=False
    )
    with pytest.raises(
        RuntimeError, match="nonfinite qpos/qvel/qacc before auto-reset"
    ):
        env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["nonfinite_seen"] is True
    assert state["all_samples_finite"] is False
    assert state["first_nonfinite"]["field"] == field
    assert state["first_nonfinite"]["first_bad_world_ids"] == [1]
    assert state["field_summaries"][field]["nonfinite_world_observations"] == 1
    assert state["field_summaries"][field]["nonfinite_value_observations"] == 1
    assert health.values["control_boundaries_observed"] == 1
    np.testing.assert_equal(getattr(data, field).values, original)


def test_strict_health_requires_all_native_finite_state_fields() -> None:
    env, data = _environment()
    del data.qacc
    with pytest.raises(RuntimeError, match="finite-state fields.*qacc"):
        _PhysicsHealth(env, scope="control", strict=True)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_nonstrict_health_records_missing_and_bad_state_without_raising(
    nonfinite: bool,
) -> None:
    env, data = _environment()
    del data.qacc
    if nonfinite:
        data.qvel.values[2, 0] = np.inf
    health = _PhysicsHealth(env, scope="control")
    env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["available"] is False
    assert state["missing_fields"] == ["qacc"]
    assert state["complete_observations"] == 0
    assert state["nonfinite_seen"] is nonfinite
    assert state["all_samples_finite"] is (False if nonfinite else None)
    assert health.values["overflow_seen"] is False


def test_strict_health_counts_complete_finite_control_samples() -> None:
    env, _ = _environment()
    health = _PhysicsHealth(env, scope="control", strict=True)
    for _ in range(2):
        env.sim.update(0.001, step=20)
    state = health.values["native_state"]
    assert state["available"] is True
    assert state["complete_observations"] == 2
    assert state["all_samples_finite"] is True
    assert state["first_nonfinite"] is None
    assert [item["observations"] for item in state["field_summaries"].values()] == [
        2,
        2,
        2,
    ]


def _failure_dump_trainer() -> tuple[SimpleNamespace, Any]:
    """Create a scalar PPO-loss fixture with no environment or simulator."""
    import torch
    from tensordict import TensorDict

    class Policy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(100.0))

        def evaluate_actions(self, batch: Any) -> Any:
            return TensorDict(
                {
                    "sample_log_prob": self.weight.expand(batch.batch_size[0]),
                    "value": torch.zeros(batch.batch_size[0]),
                    "entropy": torch.zeros(batch.batch_size[0]),
                },
                batch_size=batch.batch_size,
            )

    class Algorithm:
        def __init__(self, policy: Any) -> None:
            self.policy = policy
            self.cfg = SimpleNamespace(
                clip_coef=0.2, normalize_advantage_per_mini_batch=False
            )
            self.optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)

        def _clip_gradients(self) -> None:
            try:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 1.0, error_if_nonfinite=True
                )
            except RuntimeError as error:
                raise FloatingPointError("native gradient guard") from error

    policy = Policy()
    trainer = SimpleNamespace(
        policy=policy,
        algorithm=Algorithm(policy),
        global_step=128,
        num_updates=3,
        device=torch.device("cpu"),
    )
    batch = TensorDict(
        {
            "obs": torch.zeros(4, 2),
            "sample_log_prob": torch.zeros(4),
            "advantage": torch.ones(4),
        },
        batch_size=[4],
    )
    return trainer, batch


def test_failure_dump_preserves_gradient_failure_and_replay_inputs(
    tmp_path: Path,
) -> None:
    import torch
    from scripts.benchmark.assembly.train_peg import _PPOFailureDump

    trainer, batch = _failure_dump_trainer()
    output = tmp_path / "failure.pt"
    hook = _PPOFailureDump(trainer, output, {"algorithm": {"name": "ppo"}})
    evaluated = trainer.policy.evaluate_actions(batch)
    ratio = (evaluated["sample_log_prob"] - batch["sample_log_prob"]).exp()
    loss = -torch.minimum(ratio, ratio.clamp(0.8, 1.2)).mean()
    assert torch.isfinite(loss)
    loss.backward()
    with pytest.raises(FloatingPointError, match="native gradient guard"):
        trainer.algorithm._clip_gradients()
    captured = torch.load(output, weights_only=True)
    assert captured["bad_gradient_names"] == ["weight"]
    assert captured["all_gradient_elements_finite"] is False
    assert captured["policy"]["weight"] == 100.0
    assert captured["batch_size"] == [4]
    assert torch.isinf(captured["loss_inputs"]["ratio_reconstructed_on_cpu"]).all()
    assert torch.isnan(trainer.policy.weight.grad)
    assert trainer.algorithm.optimizer.state == {}
    batch["obs"].fill_(42)
    assert torch.count_nonzero(captured["batch"]["obs"]) == 0
    hook.restore()


def test_failure_dump_normal_path_keeps_only_current_device_minibatch(
    tmp_path: Path,
) -> None:
    import torch
    from scripts.benchmark.assembly.train_peg import _PPOFailureDump

    trainer, batch = _failure_dump_trainer()
    with torch.no_grad():
        trainer.policy.weight.zero_()
    output = tmp_path / "failure.pt"
    hook = _PPOFailureDump(trainer, output, {})
    evaluated = trainer.policy.evaluate_actions(batch)
    assert evaluated["sample_log_prob"].requires_grad
    assert not hook.last_evaluated["sample_log_prob"].requires_grad
    assert hook.last_batch is batch
    evaluated["sample_log_prob"].mean().backward()
    trainer.algorithm._clip_gradients()
    assert not output.exists()
    hook.restore()


def test_failure_dump_write_error_does_not_replace_native_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.benchmark.assembly.train_peg import _PPOFailureDump

    trainer, _ = _failure_dump_trainer()
    hook = _PPOFailureDump(trainer, tmp_path / "failure.pt", {})

    def reject() -> None:
        raise FloatingPointError("original gradient failure")

    def broken_dump(error: FloatingPointError) -> None:
        raise OSError("diagnostic disk unavailable")

    hook._clip = reject
    monkeypatch.setattr(hook, "dump", broken_dump)
    with pytest.raises(FloatingPointError, match="original gradient failure") as caught:
        trainer.algorithm._clip_gradients()
    assert any("diagnostic disk unavailable" in note for note in caught.value.__notes__)
    hook.restore()
