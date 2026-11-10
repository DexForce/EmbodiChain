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
from embodichain.lab.sim.sensors.contact.sampling import ContactEvidenceSampler
from embodichain.lab.sim.sensors.contact.timeline import ContactTimeline
from embodichain.lab.sim.atomic_skills.grip_evidence import GripContactObserver, DriverOpeningBinding
from embodichain.lab.sim.sensors.support_observation import NativeSupportObservation, NativeSupportResult, SupportObservationRegistry
from embodichain.lab.sim.atomic_skills.evidence import PhysicalEvidenceBatch as ExistingBatch
assert ExistingBatch is PhysicalEvidenceBatch
assert DriverOpeningBinding.__module__ == "embodichain.lab.sim.atomic_skills.grip_evidence"
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

robot = SimpleNamespace(joint_names=["drive"], get_joint_ids=lambda name: [0], get_qpos=lambda name: torch.zeros(1, 1))
opening = DriverOpeningBinding("hand", ("drive",), (0,), (0.,), .05).bind(robot)
assert opening(0)["open"] is True
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


def test_shared_evidence_owners_do_not_import_semantic_or_task_engines():
    import ast
    from importlib.util import resolve_name

    root = Path(__file__).resolve().parents[3]
    failures = []
    for area in (
        "embodichain/lab/sim/sensors",
        "embodichain/lab/sim/atomic_skills",
        "embodichain/compute",
    ):
        for path in (root / area).rglob("*.py"):
            package = ".".join(path.relative_to(root).parent.parts)
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if node.level:
                        module = resolve_name("." * node.level + module, package)
                    names = [
                        module,
                        *(module + "." + alias.name for alias in node.names),
                    ]
                else:
                    continue
                forbidden = (
                    "embodichain.lab.task_program",
                    "embodichain.gen_sim.task_engine",
                )
                if area.endswith("sensors"):
                    forbidden += ("embodichain.lab.sim.atomic_skills",)
                if any(
                    name == prefix or name.startswith(prefix + ".")
                    for name in names
                    for prefix in forbidden
                ):
                    failures.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not failures, "Upward evidence dependencies: " + ", ".join(failures)
