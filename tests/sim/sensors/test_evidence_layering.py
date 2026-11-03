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

"""Sensor and Atomic evidence run without importing Task Program implementations."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_shared_grip_can_be_observed_without_task_program():
    source = """
import sys
from types import SimpleNamespace
import torch
from embodichain.lab.sim.evidence import PhysicalEvidenceRequest, PhysicalEvidenceBatch
from embodichain.lab.sim.sensors.contact_sampling import ContactEvidenceSampler
from embodichain.lab.sim.sensors.contact_timeline import ContactTimeline
from embodichain.lab.sim.atomic_actions.grip_evidence import GripContactObserver
from embodichain.lab.sim.atomic_actions.evidence import PhysicalEvidenceBatch as ExistingBatch
assert ExistingBatch is PhysicalEvidenceBatch
sensor = SimpleNamespace(
    num_instances=1, dropped_contacts=0,
    contact_capabilities=SimpleNamespace(geometry=True, impulse=True),
    update=lambda: None,
    get_data=lambda: {
        "is_valid": torch.tensor([[True, True]]),
        "user_ids": torch.tensor([[[1, 2], [1, 3]]]),
        "normal": torch.tensor([[[1., 0., 0.], [-1., 0., 0.]]]),
        "position": torch.zeros(1, 2, 3), "impulse": torch.ones(1, 2),
    },
)
observer = GripContactObserver(ContactEvidenceSampler(sensor),
    {"object": torch.tensor([[1]])}, {"hand": torch.tensor([[2, 3]])}, sensor_uid="contacts")
result = observer.observe(PhysicalEvidenceRequest(1., 0, torch.tensor([0])),
    object_id="object", control_part="hand", expected_contact=True)
assert result.values.tolist() == [True]
assert result.valid.tolist() == [True]
assert result.acquisition_errors == (None,)
assert not any(name.startswith("embodichain.lab.task_program") for name in sys.modules)
assert not any(name.startswith("embodichain.gen_sim.task_engine") for name in sys.modules)
"""
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=root,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
