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

"""The standard factory assembles actual declared collectors and owns cleanup."""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace as NS
import pytest
from embodichain.lab.sim.sensors.contact.timeline import ContactTimelineCfg
from embodichain.lab.task_program.integrations.simulation.environment import (
    SimulationTaskProgramFactory,
)
from embodichain.lab.task_program.integrations.simulation.cargo_observation import (
    SimulationCargoEnvelopeCfg,
)
from embodichain.lab.task_program.integrations.simulation.recorded_contacts import (
    SimulationRecordedContactCfg,
)
from embodichain.lab.task_program.integrations.simulation.observations import (
    SimulationStateObservationCollector,
)
from ._observation_fixture import observation_host
from .test_cargo_observation import configure_meshes


def factory(host):
    return SimulationTaskProgramFactory(
        host.env.sim,
        host.robot,
        host.registration,
        step_dt=0.04,
        motion_generator_factory=host.motion,
    )


def declared_host(tmp_path):
    host = observation_host(
        recorded_cfg=SimulationRecordedContactCfg(
            sensor_uid="contacts", retention=ContactTimelineCfg(step_dt=0.04)
        ),
        cargo_cfg=SimulationCargoEnvelopeCfg(
            carrier_uids=("tray",), excluded_uids=("tool",)
        ),
    )
    configure_meshes(host, tmp_path)
    return host


def test_standard_factory_unselected_observation_services_add_no_hook_or_capture():
    host = observation_host()
    for body in host.bodies.values():
        body.get_local_pose = lambda **kw: pytest.fail(
            "Unselected services cannot acquire native poses"
        )
    owner = factory(host)
    services = owner.create_adapter().create_observation_services(
        host.env, host.bridge, observations=host.manager
    )
    assert type(services.state) is SimulationStateObservationCollector
    assert hasattr(services.state, "capture_held_poses")
    assert (
        services.controls == {} and services.cargo is services.recorded_contacts is None
    )
    assert host.fetches == [0] and not host.hooks
    services.close()
    assert host.manager.read_history_step_counts(episode_ids=(0,)) == (0,)
    assert host.bridge.post_policy_port.observation_clock is host.bridge.clock


def test_actual_declared_services_and_factory_teardown_close_only_owned_hooks(tmp_path):
    host = declared_host(tmp_path)
    owner = factory(host)
    services = owner.create_observation_services(
        host.env, host.bridge, observations=host.manager
    )
    assert services.cargo is not None and services.recorded_contacts is not None
    assert len(host.hooks) == 1 and host.fetches == [0]
    host.advance()
    host.advance()
    report = services.recorded_contacts.capture(
        "cargo", "tray", start_frame=0, end_frame=2
    )
    assert report["recorded_acquisition"]["cargo"]["observation_revisions"] == [
        [0, 1, 2]
    ]
    assert services.cargo.capture_report(invocation_id="physical_facts/final:0")[
        "contents"
    ][0]["valid_mask"] == [True]
    owner.close_runtime_resources()
    assert services.closed and not host.hooks and host.fetches == [4]
    assert host.manager.read_history_step_counts(episode_ids=(0,)) == (2,)
    assert host.bridge.post_policy_port.observation_clock is host.bridge.clock


def test_failed_cargo_assembly_releases_already_registered_contact_hook(tmp_path):
    host = declared_host(tmp_path)
    owner = factory(host)
    host.env.cfg = NS(
        rigid_object=[
            NS(uid=uid, to_dict=lambda uid=uid: {"uid": uid}) for uid in host.bodies
        ]
    )
    with pytest.raises(ValueError, match="mesh geometry"):
        owner.create_observation_services(
            host.env, host.bridge, observations=host.manager
        )
    assert not host.hooks and host.fetches == [0]
    assert host.manager.has_history


def test_closed_services_are_not_retained_for_later_factory_episodes(tmp_path):
    host = declared_host(tmp_path)
    owner = factory(host)
    for _ in range(3):
        services = owner.create_observation_services(
            host.env, host.bridge, observations=host.manager
        )
        reference = weakref.ref(services)
        services.close()
        del services
        gc.collect()
        assert reference() is None and not host.hooks
