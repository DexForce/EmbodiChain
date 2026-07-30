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

import json
import types

import pytest
import torch

from embodichain.lab.gym.envs.managers.cfg import EventCfg, FunctorCfg
from embodichain.lab.gym.envs.managers.event_manager import EventManager
from embodichain.lab.gym.envs.managers.manager_base import ManagerBase
from embodichain.lab.gym.utils.profiler import EnvProfiler, EnvProfilerCfg


def _step(prof: EnvProfiler) -> None:
    """Mimic the section nesting used by BaseEnv.step / get_obs."""
    with prof.section("step", is_root=True):
        with prof.section("sim_update"):
            pass
        with prof.section("get_obs"):
            with prof.section("proprio"):
                pass
            with prof.section("sensor"):
                with prof.section("render_camera_group"):
                    pass
                with prof.section("sensor_fetch"):
                    pass


class TestEnvProfilerDisabled:
    def test_noop_when_cfg_none(self):
        prof = EnvProfiler(None, torch.device("cpu"))
        assert not prof.enabled
        _step(prof)
        assert prof._stats == {}
        assert prof.report() == {}


class TestEnvProfilerRecording:
    def test_records_hierarchical_names(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        _step(prof)

        assert "step" in prof._stats
        assert "step.sim_update" in prof._stats
        assert "step.get_obs" in prof._stats
        assert "step.get_obs.proprio" in prof._stats
        assert "step.get_obs.sensor" in prof._stats
        assert "step.get_obs.sensor.render_camera_group" in prof._stats
        assert "step.get_obs.sensor.sensor_fetch" in prof._stats

    def test_counts_increment_per_call(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        _step(prof)
        _step(prof)

        assert prof._stats["step"].n == 2
        assert prof._stats["step.get_obs"].n == 2
        assert prof._stats["step.get_obs.proprio"].n == 2

    def test_parent_total_includes_children(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        _step(prof)

        step_total = prof._stats["step"].total_s
        get_obs_total = prof._stats["step.get_obs"].total_s
        assert step_total >= get_obs_total

    def test_outside_root_is_skipped(self):
        # Sections entered without a step/reset root (e.g. init-time get_obs)
        # must not pollute the report.
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        with prof.section("proprio"):
            pass

        assert prof._stats == {}


class TestEnvProfilerWarmup:
    def test_warmup_discards_first_roots(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=2), torch.device("cpu")
        )
        _step(prof)  # warmup
        _step(prof)  # warmup
        _step(prof)  # first recorded

        assert prof._stats["step"].n == 1
        assert prof._stats["step.get_obs"].n == 1


class TestEnvProfilerAutoReset:
    def test_nested_reset_does_not_open_duplicate_root(self):
        # reset() called during step's auto-reset must attribute its children to
        # the outer step root, not create a duplicate top-level "reset" entry.
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        with prof.section("step", is_root=True):
            with prof.section("auto_reset"):
                with prof.section("reset", is_root=True):
                    with prof.section("event_reset"):
                        pass
                    with prof.section("obs_reset"):
                        pass

        assert "step" in prof._stats
        assert "step.auto_reset" in prof._stats
        assert "step.auto_reset.event_reset" in prof._stats
        assert "step.auto_reset.obs_reset" in prof._stats
        # No duplicate top-level reset root from the nested call.
        assert "reset" not in prof._stats
        assert "reset.event_reset" not in prof._stats


class TestEnvProfilerReport:
    def test_report_returns_sections(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        _step(prof)

        data = prof.report()
        assert "sections" in data
        assert "step.sim_update" in data["sections"]
        assert data["sections"]["step.sim_update"]["calls"] == 1
        assert data["sections"]["step.sim_update"]["mean_ms"] >= 0.0

    def test_report_json_dump(self, tmp_path):
        out = tmp_path / "report.json"
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0, output_path=str(out)),
            torch.device("cpu"),
        )
        _step(prof)
        prof.report()

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "sections" in data
        assert "step" in data["sections"]

    def test_report_no_samples_does_not_crash(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=10), torch.device("cpu")
        )
        _step(prof)  # consumed by warmup, nothing recorded
        data = prof.report()
        assert data["sections"] == {}


def _make_manager(profiler) -> ManagerBase:
    """Minimal concrete ManagerBase whose env exposes a profiler (or None)."""

    class _FakeManager(ManagerBase):
        @property
        def active_functors(self):
            return []

        def _prepare_functors(self):
            pass

    env = types.SimpleNamespace(_profiler=profiler)
    return _FakeManager(cfg=None, env=env)


class TestManagerCallFunctor:
    """Per-functor timing via ManagerBase._call_functor."""

    def test_records_section_and_calls_func(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        m = _make_manager(prof)
        calls = {"n": 0}

        def my_func(env, env_ids, scale=1.0):
            calls["n"] += 1
            return env_ids

        cfg = FunctorCfg(func=my_func, params={"scale": 2.0})
        with prof.section("step", is_root=True):
            with prof.section("event_interval"):
                result = m._call_functor("my_func", cfg, None, 42)

        assert calls["n"] == 1
        assert result == 42
        # per-functor section nests under the active parent
        assert "step.event_interval.my_func" in prof._stats
        assert prof._stats["step.event_interval.my_func"].n == 1

    def test_noop_when_no_profiler(self):
        m = _make_manager(profiler=None)
        calls = {"n": 0}

        def my_func(env, env_ids):
            calls["n"] += 1
            return env_ids

        cfg = FunctorCfg(func=my_func, params={})
        result = m._call_functor("my_func", cfg, None, 7)

        assert calls["n"] == 1
        assert result == 7

    def test_disabled_profiler_still_calls_func(self):
        prof = EnvProfiler(None, torch.device("cpu"))  # disabled
        m = _make_manager(prof)
        calls = {"n": 0}

        def my_func(env, env_ids):
            calls["n"] += 1
            return env_ids

        cfg = FunctorCfg(func=my_func, params={})
        result = m._call_functor("my_func", cfg, None, 3)

        assert calls["n"] == 1
        assert result == 3
        assert prof._stats == {}


class TestEventFunctorIntervalProfiling:
    """An interval event functor must be timed per-firing, not per-step.

    Functors with ``interval_step > 1`` fire only every N steps. The profiler
    must record a sample only on those firings, so ``calls`` reflects the firing
    count (not the step count) and ``mean`` is the per-firing execution time
    (not diluted by the non-firing steps).
    """

    def test_interval_functor_recorded_per_firing(self):
        prof = EnvProfiler(
            EnvProfilerCfg(enable_time=True, warmup_steps=0), torch.device("cpu")
        )
        env = types.SimpleNamespace(
            _profiler=prof, num_envs=4, device=torch.device("cpu"), sim=None
        )
        fired = {"n": 0}

        def my_event(env, env_ids, scale=1.0):
            fired["n"] += 1

        em = EventManager(
            {
                "my_event": EventCfg(
                    func=my_event,
                    mode="interval",
                    interval_step=3,
                    is_global=True,
                    params={"scale": 2.0},
                )
            },
            env,
        )

        # 10 steps; is_global interval_step=3 fires when count % 3 == 0.
        # count increments to 1..10, so fires at 3, 6, 9 -> 3 firings.
        for _ in range(10):
            with prof.section("step", is_root=True):
                with prof.section("update_sim_state"):
                    with prof.section("event_interval"):
                        em.apply("interval", None)

        assert fired["n"] == 3  # functor actually fired 3 times
        key = "step.update_sim_state.event_interval.my_event"
        assert key in prof._stats
        s = prof._stats[key]
        # recorded 3 firings, NOT 10 steps -> mean is per-firing, not diluted
        assert s.n == 3
        assert s.total_s > 0
        # parent event_interval runs every step -> 10 samples, distinct from the
        # functor's 3 firings
        assert prof._stats["step.update_sim_state.event_interval"].n == 10
