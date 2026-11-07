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

"""Recorded contacts use the real raw owner and the existing substep timeline."""

from __future__ import annotations

from types import SimpleNamespace as NS
import pytest
import torch
from embodichain.lab.gym.envs.episode_context import EpisodeContext
from embodichain.lab.sim.sensors.contact.timeline import ContactTimelineCfg
from embodichain.lab.task_program.integrations.simulation.recorded_contacts import (
    SimulationRecordedContactCfg,
    SimulationRecordedContactCollector,
)
from ._observation_fixture import observation_host


def contact_case():
    cfg = SimulationRecordedContactCfg(
        sensor_uid="contacts", retention=ContactTimelineCfg(step_dt=0.04)
    )
    host = observation_host(capacity=3, recorded_cfg=cfg)
    collector = SimulationRecordedContactCollector(
        host.env, host.bridge, host.registration, cfg, observations=host.manager
    )
    return host, collector


def test_actual_contact_collector_registers_first_and_preserves_raw_window_clocks():
    host, collector = contact_case()
    assert len(host.hooks) == 1 and host.fetches == [0]
    host.advance()
    host.advance()
    for body in host.bodies.values():
        body.get_local_pose = lambda **kw: pytest.fail(
            "Historical capture cannot acquire current poses"
        )
    captured = collector.capture("cargo", "tray", start_frame=0, end_frame=2)
    assert host.fetches == [4]
    assert [s["step"] for s in captured["timeline"]["environments"][0]["samples"]] == [
        1,
        2,
    ]
    field = captured["pose_frame"]["batches"]["rigid.pose_window:cargo"]
    assert field["value_dtype"] == "torch.float64" and torch.tensor(
        field["values"]
    ).shape == (1, 3, 4, 4)
    raw = captured["recorded_acquisition"]["cargo"]
    assert raw["observation_revisions"] == [[0, 1, 2]]
    assert raw["timestamps"][0] == pytest.approx([0.1, 0.14, 0.18])
    assert raw["start_frames"] == [0] and raw["end_frames"] == [2]
    assert raw["clock_domain"] == "simulation_time"
    assert raw["valid"] == [[True, True, True]]
    host.manager.read_history_window = lambda *a, **kw: pytest.fail(
        "Contact-only capture cannot copy poses"
    )
    only = collector.capture_timeline("cargo", "tray", boundary_frame=2)
    assert only["timeline"] == captured["timeline"] and "pose_frame" not in only
    assert host.fetches == [4]
    collector.close()
    collector.close()
    assert not host.hooks
    assert host.manager.read_history_step_counts(episode_ids=(0,)) == (2,)


@pytest.mark.parametrize("damage", ["invalid_finite", "timestamp", "slot"])
def test_recorded_contact_missing_raw_facts_cannot_become_available_poses(damage):
    host, collector = contact_case()
    host.advance(
        invalid="cargo" if damage == "invalid_finite" else None,
        timestamp=damage != "timestamp",
    )
    host.advance()
    if damage == "slot":
        host.storage["raw_revision"][0, 1] = -1
    with pytest.raises((ValueError, RuntimeError), match="unavailable|recorded"):
        collector.capture("cargo", "tray", start_frame=0, end_frame=2)
    assert host.fetches == [4]
    collector.close()


@pytest.mark.parametrize("damage", ["episode_owner", "sensor", "native", "reset"])
def test_recorded_contact_rejects_replaced_owner_and_generation(damage):
    host, collector = contact_case()
    host.advance()
    if damage == "episode_owner":
        owner = EpisodeContext(1)
        owner._begin_episode([0])
        owner._finish_initialization([0])
        host.env.episode_context = owner
    elif damage == "sensor":
        host.env.sim.get_sensor = lambda uid: NS()
    elif damage == "native":
        host.bodies["cargo"] = NS(uid="cargo", body_type="dynamic")
    else:
        collector.timeline.reset([0])
    with pytest.raises((ValueError, RuntimeError), match="owner|reset|episode"):
        collector.capture_timeline("cargo", "tray", boundary_frame=1)
    collector.close()


def test_recorded_contact_requires_ordinary_suite_and_source_query_budget():
    cfg = SimulationRecordedContactCfg(
        sensor_uid="contacts", retention=ContactTimelineCfg(step_dt=0.04)
    )
    host = observation_host(recorded_cfg=cfg)
    host.env.sensors = {}
    with pytest.raises(ValueError, match="ordinary substep suite"):
        SimulationRecordedContactCollector(
            host.env, host.bridge, host.registration, cfg, observations=host.manager
        )
    assert not host.hooks
    host.env.sensors = {"contacts": host.sensor}
    host.sensor.cfg.max_contacts_per_env = 512
    with pytest.raises(ValueError, match="source sensor scope"):
        SimulationRecordedContactCollector(
            host.env, host.bridge, host.registration, cfg, observations=host.manager
        )
    assert not host.hooks
