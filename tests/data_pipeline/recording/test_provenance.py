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
import torch

from embodichain.data_pipeline.recording import (
    build_recording_provenance,
    stable_config_hash,
)


def test_config_fingerprint_is_order_independent_and_sensitive_to_values() -> None:
    first = {"robot": {"gain": 1.0}, "seed": 12}
    reordered = {"seed": 12, "robot": {"gain": 1.0}}

    assert stable_config_hash(first) == stable_config_hash(reordered)
    assert stable_config_hash(first) != stable_config_hash(dict(first, seed=13))


def test_provenance_records_effective_backend_timing_and_program() -> None:
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            task_program={"program_id": "pick"}, sim_steps_per_control=4
        ),
        step_dt=0.04,
        physics_dt=0.01,
        control_frequency=25,
        sim=SimpleNamespace(physics_backend="newton"),
    )

    result = build_recording_provenance(env)

    assert result["program_hash"] == stable_config_hash(env.cfg.task_program)
    assert result["config_hash"] == stable_config_hash(env.cfg)
    assert result["provenance"]["physics"] == "newton"
    assert result["provenance"]["step_dt"] == 0.04
    assert result["provenance"]["sim_steps_per_control"] == 4


def test_tensor_configuration_fingerprint_uses_values() -> None:
    first = {"pose": torch.tensor([1.0, 2.0]), "device": torch.device("cpu")}
    second = {"pose": torch.tensor([1.0, 3.0]), "device": torch.device("cpu")}

    assert stable_config_hash(first) != stable_config_hash(second)
