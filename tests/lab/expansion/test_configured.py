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

"""Native file jobs fail before execution for unsupported schemas and scopes."""

from __future__ import annotations
from pathlib import Path
import pytest
from embodichain.lab.expansion import ExpansionEngine, load_expansion_job


def test_task_program_declarations_are_pure_and_protect_contact_phases(tmp_path):
    import json
    import os
    import subprocess
    import sys
    import yaml
    from embodichain.lab.expansion import load_task_program_expansion

    path = Path(__file__).parent / "fixtures/task_program_job.yaml"
    code = "from embodichain.lab.expansion import load_task_program_expansion; import json,sys; value=load_task_program_expansion(sys.argv[1]); assert not any(k=='dexsim' or k.startswith('dexsim.') for k in sys.modules); print(json.dumps(value))"
    process = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )
    value = json.loads(process.stdout)
    assert (value["engine"]["workload_count"], value["engine"]["batch_size"]) == (2, 1)
    assert value["trajectory"]["operators"] == ["object_pose", "joint_residual"]
    value["trajectory"]["free_phases"]["pick_up"] = ["approach", "lift"]
    invalid = tmp_path / "protected.yaml"
    invalid.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="protected free phase"):
        load_task_program_expansion(invalid)


@pytest.mark.parametrize("name", ["default", "newton", "rgbd", "motion"])
def test_example_preflight_resolves_owned_backend_and_budget(name, tmp_path):
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine"
        / f"job.{name}.yaml"
    )
    cfg = load_expansion_job(path)
    cfg.output_dir = str(tmp_path / "untouched")
    engine = ExpansionEngine.from_job(cfg)
    report = engine.preflight()
    assert report["physics_backend"] == ("newton" if name == "newton" else "default")
    assert report["pending_numeric_bytes"] == cfg.batch_size * cfg.episode_byte_budget
    assert not Path(cfg.output_dir).exists()
    assert not report["capabilities"]["single_body_damping"]


def test_native_media_requires_complete_decision_contract():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.default.yaml"
    )
    cfg = load_expansion_job(path)
    cfg.render_profiles = ("blue",)
    with pytest.raises(ValueError, match="observation_independent"):
        ExpansionEngine.from_job(cfg)


def test_resource_changes_preserve_semantics_but_record_new_execution(tmp_path):
    from embodichain.lab.expansion.configured import preflight_expansion_job

    path = (
        Path(__file__).resolve().parents[3]
        / "examples/expansion/engine/job.default.yaml"
    )
    cfg = load_expansion_job(path)
    a = preflight_expansion_job(cfg)
    cfg.batch_size = 32
    cfg.max_attempts = 64
    cfg.target_episodes = 32
    cfg.episode_byte_budget *= 2 // 2
    cfg.output_dir = str(tmp_path)
    b = preflight_expansion_job(cfg)
    assert a["semantic_id"] == b["semantic_id"]
    assert a["execution_id"] != b["execution_id"]
    cfg.mass_scale = {"cube": (2.0, 3.0)}
    assert preflight_expansion_job(cfg)["semantic_id"] != a["semantic_id"]
