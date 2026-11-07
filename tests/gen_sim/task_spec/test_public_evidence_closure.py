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

"""Cold-import the complete normative evidence surface used by host consumers."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import embodichain

from .test_condition_group_evidence import _declaration
from .test_control_evidence import _case


def test_complete_evidence_surface_is_available_without_physical_or_numeric_imports():
    root = Path(embodichain.__file__).resolve().parents[1]
    normalizations = (
        (
            "condition_group_evidence",
            "ConditionEvidenceGroupBinding",
            _declaration(recorded=False).to_dict(),
        ),
        ("control_evidence", "ControlEvidenceBinding", _case()[1].to_dict()),
    )
    script = f"""
import importlib, importlib.abc, pkgutil, sys
sys.path.insert(0, {str(root)!r})
class PureOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(('embodichain.lab', 'embodichain.gen_sim.task_engine',
                            'dexsim', 'curobo', 'newton', 'numpy', 'torch')):
            raise AssertionError(name)
sys.meta_path.insert(0, PureOnly())
import embodichain.gen_sim.task_spec as task_spec
for info in pkgutil.iter_modules(task_spec.__path__, task_spec.__name__ + '.'):
    module = importlib.import_module(info.name)
    for name in module.__all__:
        assert hasattr(module, name), (info.name, name)
from embodichain.gen_sim.task_spec.manipulation_evidence import (
    HoldAcceptanceCriteria, ManipulationEvidenceBinding, evaluate_manipulation_evidence)
from embodichain.gen_sim.task_spec.evidence_report import evaluate_physical_evidence_report
assert all(callable(value) for value in (HoldAcceptanceCriteria, ManipulationEvidenceBinding,
    evaluate_manipulation_evidence, evaluate_physical_evidence_report))
for module_name, class_name, payload in {normalizations!r}:
    module = importlib.import_module('embodichain.gen_sim.task_spec.' + module_name)
    assert getattr(module, class_name).from_dict(payload).to_dict() == payload
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
