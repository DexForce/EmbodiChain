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

"""Cargo adaptation reads original raw prefixes with native owner lifetime checks."""

from __future__ import annotations

from types import SimpleNamespace as NS
import pytest
import torch
import trimesh
from embodichain.lab.task_program.integrations.simulation.cargo_observation import (
    SimulationCargoEnvelopeCfg,
    SimulationCargoEnvelopeCollector,
)
from _observation_fixture import observation_host


def configure_meshes(host, tmp_path):
    sources = []
    for name in host.bodies:
        path = tmp_path / (name + ".glb")
        trimesh.creation.box(
            extents=(1.0, 0.2, 0.8) if name == "tray" else (0.06, 0.06, 0.06)
        ).export(path)
        data = {
            "uid": name,
            "shape": {"fpath": str(path)},
            "body_scale": [1.0, 1.0, 1.0],
        }
        sources.append(NS(uid=name, to_dict=lambda data=data: data))
    host.env.cfg = NS(rigid_object=sources)


def cargo_case(tmp_path):
    host = observation_host()
    configure_meshes(host, tmp_path)
    cfg = SimulationCargoEnvelopeCfg(carrier_uids=("tray",), excluded_uids=("tool",))
    collector = SimulationCargoEnvelopeCollector(
        host.env, host.bridge, cfg, observations=host.manager
    )
    return host, collector


def test_cargo_collector_reads_complete_original_prefix_and_terminal_facts(tmp_path):
    host, collector = cargo_case(tmp_path)
    assert host.reads == [] and host.fetches == [0]
    for step in range(1, 131):
        host.bodies["cargo"].pose[0, 0, 3] = 1.0 if step == 64 else 0.0
        host.advance()
    result = collector.capture_report(invocation_id="physical_facts/final:0")
    assert len(result["contents"]) == 1
    observed = result["contents"][0]
    assert observed["valid_mask"] == [True] and observed["observed_mask"] == [False]
    assert observed["first_failed_frames"] == [64]
    assert result["episode_ids"] == [0] and result["timestamp"] == 5.2
    assert result["clock_domain"] == "execution_step_clock"
    acquisition = result["recorded_acquisition"]["cargo"]
    assert acquisition["start_frames"] == [0] and acquisition["end_frames"] == [129]
    assert acquisition["observation_revisions"][0] == list(range(130))
    assert acquisition["timestamps"][0][0] == 0.1
    assert acquisition["timestamps"][0][-1] == pytest.approx(5.26)
    assert host.fetches == [0]
    collector.close()
    with pytest.raises(RuntimeError, match="closed"):
        collector.capture(invocation_id="physical_facts/final:0")


@pytest.mark.parametrize("damage", ["invalid_finite", "timestamp", "slot"])
def test_cargo_missing_raw_history_preserves_unavailability_reason(tmp_path, damage):
    host, collector = cargo_case(tmp_path)
    for step in range(1, 4):
        host.advance(
            invalid="cargo" if damage == "invalid_finite" and step == 1 else None,
            timestamp=not (damage == "timestamp" and step == 1),
        )
    if damage == "slot":
        host.storage["raw_revision"][0, 1] = -1
    result = collector.capture_report(invocation_id="physical_facts/final:0")
    value = result["contents"][0]
    assert value["valid_mask"] == [False] and value["observed_mask"] == [None]
    assert value["acquisition_errors"][0]
    if damage == "invalid_finite":
        assert "raw source unavailable" in value["acquisition_errors"][0]


def test_cargo_clock_change_during_raw_read_cannot_publish_facts(tmp_path):
    host, collector = cargo_case(tmp_path)
    host.advance()
    original = host.manager.read_history_window

    def changed(*a, **kw):
        value = original(*a, **kw)
        host.bridge.clock.advance_after_env_step()
        return value

    host.manager.read_history_window = changed
    with pytest.raises(ValueError, match="clock advanced"):
        collector.capture(invocation_id="physical_facts/final:0")


@pytest.mark.parametrize("damage", ["replacement", "uid"])
def test_cargo_native_replacement_cannot_substitute_current_pose(tmp_path, damage):
    host, collector = cargo_case(tmp_path)
    host.advance()
    host.bodies["cargo"] = NS(get_local_pose=lambda **kw: torch.eye(4)[None])
    with pytest.raises(ValueError, match="native owner"):
        collector.capture(invocation_id="physical_facts/final:0")
