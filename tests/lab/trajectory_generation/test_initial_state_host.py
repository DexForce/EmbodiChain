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

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.sim.motion.expansion import (
    SceneCase,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.trajectory_generation.initial_state import (
    FixedSceneHost,
    InitialStateProfile,
)


def _passed(name: str = "task_initial") -> ValidationResult:
    return ValidationResult((ValidationCheck(name, "passed"),))


class _Adapter:
    def __init__(self) -> None:
        self.position = torch.tensor([[1.0], [2.0]])
        self.robot = object()
        self.sim = SimpleNamespace(num_envs=2, update=self._settle)
        self.settle_calls = 0
        self.restore_calls = 0
        self.settle_delta = 0.0
        self.read_failed = False

    def _settle(self, dt: float, steps: int) -> None:
        self.settle_calls += steps
        self.position += self.settle_delta

    def signature(self) -> str:
        return "two-row-robot"

    def capture(self) -> torch.Tensor:
        if self.read_failed:
            raise ValueError("backend cannot capture")
        return self.position.clone()

    def restore(self, value: torch.Tensor) -> None:
        self.restore_calls += 1
        self.position.copy_(value)

    def verify(self, value: torch.Tensor) -> ValidationResult:
        return ValidationResult(
            (
                ValidationCheck(
                    "initial_state",
                    "passed" if torch.equal(self.position, value) else "failed",
                ),
            )
        )


def _cases() -> tuple[SceneCase, ...]:
    return tuple(
        SceneCase(f"case-{row}", f"initial-{row}", f"scene-{row}", "pick", "robot")
        for row in range(2)
    )


def _profile(**kwargs: object) -> InitialStateProfile:
    values = dict(
        profile_id="fixed_cube",
        prepare=lambda: None,
        signature=lambda: "fixed-properties",
        verify=lambda cases: _passed(),
        settling_steps=1,
    )
    values.update(kwargs)
    return InitialStateProfile(**values)


def test_repeated_full_batch_restoration_preserves_case_rows_and_invalidates_tokens() -> (
    None
):
    adapter = _Adapter()
    with FixedSceneHost(adapter, _profile()) as host:
        previous = host.acquire_case(_cases())
        for _ in range(3):
            adapter.position += 10
            current = host.restore_initial()
            assert torch.equal(adapter.position, torch.tensor([[1.0], [2.0]]))
            assert host.verify_initial(current).accepted
            assert current.cases == _cases()
            assert current.epoch > previous.epoch
            with pytest.raises(RuntimeError, match="obsolete"):
                host.assert_current(previous)
            previous = current
        assert adapter.restore_calls == 3
        assert adapter.settle_calls == 4
    with pytest.raises(RuntimeError, match="own"):
        host.assert_current(previous)
    assert adapter.sim._trajectory_generation_owner is None


def test_acquire_captures_after_preparation_and_settling() -> None:
    adapter = _Adapter()
    adapter.settle_delta = 0.5
    observed = []
    profile = _profile(
        prepare=lambda: adapter.position.fill_(3),
        verify=lambda cases: (observed.append(adapter.position.clone()) or _passed()),
    )
    with FixedSceneHost(adapter, profile) as host:
        binding = host.acquire_case(_cases())
        assert torch.equal(observed[0], torch.full((2, 1), 3.5))
        adapter.settle_delta = 0
        adapter.position.fill_(9)
        binding = host.restore_initial()
        assert torch.equal(adapter.position, torch.full((2, 1), 3.5))
        host.assert_current(binding)


def test_fixed_condition_change_rejects_before_prepare_or_restore() -> None:
    adapter = _Adapter()
    state = {"signature": "initial"}
    calls = []
    with FixedSceneHost(
        adapter,
        _profile(
            signature=lambda: state["signature"],
            prepare=lambda: calls.append("prepare"),
        ),
    ) as host:
        old = host.acquire_case(_cases())
        state["signature"] = "changed-material"
        with pytest.raises(RuntimeError, match="conditions changed"):
            host.restore_initial()
        assert calls == ["prepare"]
        assert adapter.restore_calls == 0
        with pytest.raises(RuntimeError, match="obsolete"):
            host.assert_current(old)


def test_failed_verification_does_not_publish_binding_and_can_retry_saved_state() -> (
    None
):
    adapter = _Adapter()
    with FixedSceneHost(adapter, _profile()) as host:
        old = host.acquire_case(_cases())
        adapter.settle_delta = 1
        with pytest.raises(RuntimeError, match="verification failed"):
            host.restore_initial()
        with pytest.raises(RuntimeError, match="obsolete"):
            host.assert_current(old)
        adapter.settle_delta = 0
        assert host.verify_initial(host.restore_initial()).accepted


def test_failed_capture_does_not_create_a_restorable_initial_state() -> None:
    adapter = _Adapter()
    adapter.read_failed = True
    with FixedSceneHost(adapter, _profile()) as host:
        with pytest.raises(ValueError, match="cannot capture"):
            host.acquire_case(_cases())
        with pytest.raises(RuntimeError, match="acquire_case"):
            host.restore_initial()
        adapter.read_failed = False
        host.assert_current(host.acquire_case(_cases()))


def test_second_host_cannot_take_batch_and_closed_host_can_be_replaced() -> None:
    adapter = _Adapter()
    first = FixedSceneHost(adapter, _profile())
    with pytest.raises(RuntimeError, match="already has"):
        FixedSceneHost(adapter, _profile())
    first.close()
    first.close()
    with FixedSceneHost(adapter, _profile()) as second:
        second.assert_current(second.acquire_case(_cases()))


def test_partial_batch_is_rejected_without_running_preparation() -> None:
    adapter = _Adapter()
    calls = []
    with FixedSceneHost(adapter, _profile(prepare=lambda: calls.append(True))) as host:
        with pytest.raises(ValueError, match="every simulator row"):
            host.acquire_case(_cases()[:1])
        assert not calls
        host.acquire_case(_cases())
        with pytest.raises(RuntimeError, match="already acquired"):
            host.acquire_case(_cases())


@pytest.mark.parametrize("status", ["failed", "unavailable", "not_run"])
def test_profile_must_run_and_pass_all_required_checks(status: str) -> None:
    adapter = _Adapter()
    profile = _profile(
        verify=lambda cases: ValidationResult(
            (ValidationCheck("scene_signature", status, "cannot certify"),)
        )
    )
    with FixedSceneHost(adapter, profile) as host:
        with pytest.raises(RuntimeError, match="scene_signature"):
            host.acquire_case(_cases())


@pytest.mark.parametrize(
    "override",
    [
        {"physics_dt": 0},
        {"physics_dt": float("nan")},
        {"settling_steps": -1},
        {"settling_steps": True},
        {"prepare": None},
        {"profile_id": ""},
        {"allowed_interval_events": ("push", "push")},
    ],
)
def test_invalid_preparation_profile_is_rejected(override: dict) -> None:
    with pytest.raises(ValueError):
        _profile(**override)


def test_gym_unknown_interval_event_is_rejected_before_acquiring_lease() -> None:
    adapter = _Adapter()
    env = SimpleNamespace(
        sim=adapter.sim,
        robot=adapter.robot,
        num_envs=2,
        physics_dt=0.01,
        event_manager=SimpleNamespace(active_functors={"interval": ["random_push"]}),
    )
    with pytest.raises(ValueError, match="uncertified interval"):
        FixedSceneHost(adapter, _profile(), env=env)
    assert not hasattr(adapter.sim, "_trajectory_generation_owner")


def test_normal_sim_reset_is_blocked_until_host_releases_batch() -> None:
    from embodichain.lab.sim.sim_manager import SimulationManager as SimManager

    adapter = _Adapter()
    with FixedSceneHost(adapter, _profile()):
        with pytest.raises(RuntimeError, match="generation host owns"):
            SimManager.reset_objects_state(adapter.sim)


def test_topology_change_invalidates_prepared_execution() -> None:
    adapter = _Adapter()
    with FixedSceneHost(adapter, _profile()) as host:
        binding = host.acquire_case(_cases())
        adapter.signature = lambda: "new-robot-control-layout"
        with pytest.raises(RuntimeError, match="topology"):
            host.assert_current(binding)
