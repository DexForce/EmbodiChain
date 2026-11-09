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

"""Expected source acceptance failures stay local; runtime failures stop the host."""

from __future__ import annotations

from types import SimpleNamespace
import json

import pytest
import torch

from embodichain.lab.expansion._gym_episode import GymEpisodeAssembler
from embodichain.lab.expansion.contracts import (
    CandidateIdentity,
    ValidationCheck,
    ValidationResult,
)
from embodichain.lab.task_program.integrations.simulation.real2sim_collection import (
    Real2SimAcceptanceRejected,
)


@pytest.mark.parametrize("failed_row", [0, 1])
def test_source_rejection_preserves_successful_peer(tmp_path, failed_row):
    ctx, assignments, collector, services, result = _assembly_case(tmp_path, failed_row)
    episodes = GymEpisodeAssembler(ctx).assemble(
        assignments, collector, services, result, {}
    )
    assert episodes[failed_row] is None
    peer = episodes[1 - failed_row]
    assert peer is not None and peer.validation.accepted
    assert len(peer.actions) == 1 and len(peer.timestamps) == 2
    record = json.loads((tmp_path / str(failed_row) / "attempt.json").read_text())
    assert (
        record["status"] == "rejected"
        and record["reason"] == "source_acceptance_failed"
    )
    assert any(
        c["check_id"] == "task" and c["status"] == "failed"
        for c in record["validation"]
    )


@pytest.mark.parametrize(
    "error", [RuntimeError("native error"), ValueError("damaged source")]
)
def test_unexpected_qualification_errors_remain_fatal(tmp_path, error):
    ctx, assignments, collector, services, result = _assembly_case(tmp_path, 0)

    def fail(**kwargs):
        raise error

    services[0].qualify_episode = fail
    with pytest.raises(type(error), match=str(error)):
        GymEpisodeAssembler(ctx).assemble(assignments, collector, services, result, {})


def _assembly_case(root, failed_row):
    valid = ValidationResult((ValidationCheck("source", "passed"),))
    spec = SimpleNamespace(
        metadata=lambda **kw: {"joint_names": ["joint"]}, joint_command_mode="position"
    )
    env = SimpleNamespace(
        active_joint_ids=[0],
        step_dt=0.04,
        expert_action_spec=spec,
        sim=SimpleNamespace(physics_backend="default"),
    )
    recipe = SimpleNamespace(recipe_id="recipe", to_dict=lambda: {})
    ctx = SimpleNamespace(
        cfg=SimpleNamespace(visual_profile=None),
        report={"runtime_versions": {}, "input_files": {}},
        env=env,
        names=("joint",),
        limits=torch.tensor([[-1.0, 1.0]]),
        timings={},
        source_artifacts={},
        profile=SimpleNamespace(object_uid="box"),
        program=SimpleNamespace(program_id="program", instruction="task"),
        collision=None,
        physics_before={},
        journal=SimpleNamespace(job_instance_id="job"),
        requests={
            i: SimpleNamespace(sequence=i, recipe=recipe, attempt_id=0) for i in (0, 1)
        },
        compatibility=SimpleNamespace(key="compatible"),
        preparation_checks={0: valid, 1: valid},
    )
    assignments = {}
    records = {}
    services = {}
    for i in (0, 1):
        identity = CandidateIdentity(
            "case",
            "initial",
            "candidate" + str(i),
            "geometry" + str(i),
            "source",
            "v1",
            "template",
        )
        assignments[i] = (
            identity,
            "episode" + str(i),
            "commit" + str(i),
            root / str(i),
            {"recipe": {}, "case": {}, "initial_state": []},
        )
        records[i] = {
            "qpos": [torch.tensor([0.0]), torch.tensor([0.1])],
            "qvel": [torch.zeros(1), torch.zeros(1)],
            "qf": [torch.zeros(1), torch.zeros(1)],
            "actions": [torch.tensor([0.1])],
            "objects": {"box": [torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])] * 2},
            "scene": {},
        }

        def qualify(index=i, **kwargs):
            if index == failed_row:
                raise Real2SimAcceptanceRejected(
                    ValidationResult(
                        (ValidationCheck("task", "failed", "placement drift"),)
                    )
                )
            return SimpleNamespace(
                metadata={},
                validation=ValidationResult((ValidationCheck("prototype", "passed"),)),
                source_episode_uuids=("source-episode",),
            )

        services[i] = SimpleNamespace(qualify_episode=qualify)
    result = SimpleNamespace(
        env_ids=(0, 1),
        segments=((), ()),
        lengths=(1, 1),
        reasons=("completed", "completed"),
        accepted=(True, True),
        expansion_records=(),
    )
    return (
        ctx,
        assignments,
        SimpleNamespace(records=records, spools={}),
        services,
        result,
    )
