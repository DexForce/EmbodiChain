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

"""Canonical lifecycle ownership and opt-in trajectory coverage compatibility."""

from __future__ import annotations
import os
import subprocess
import sys
import pytest
import torch
from embodichain.lab.expansion import ExpansionSession, ExpansionSessionCfg
from embodichain.lab.expansion.contracts import (
    SceneCase,
    EpisodeCandidate,
    ExpertEpisode,
    ValidationResult,
    ValidationCheck,
    CommitReceipt,
)
from embodichain.lab.sim.motion.expansion.cfg import (
    TrajectoryExpansionCfg,
    TrajectoryExpansionJobCfg,
)


def test_canonical_session_import_needs_no_motion_or_simulation_package():
    script = """
import sys
from embodichain.lab.expansion import ExpansionSession, ExpansionSessionCfg
session=ExpansionSession(ExpansionSessionCfg())
assert session.snapshot()['counts']['committed']==0
assert not any(name.startswith('embodichain.lab.sim') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


def test_old_and_new_session_paths_export_the_identical_class():
    from embodichain.lab.sim.motion.expansion import ExpansionSession as facade
    from embodichain.lab.sim.motion.expansion.session import ExpansionSession as module

    assert facade is module is ExpansionSession
    assert ExpansionSession.__module__ == "embodichain.lab.expansion.session"


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_policy_keeps_legacy_trajectory_dedup_and_generic_episode_credit(
    legacy,
):
    cfg = {"collection": {"target_committed_episodes": 2}}
    session = ExpansionSession(
        TrajectoryExpansionJobCfg.from_mapping(cfg)
        if legacy
        else ExpansionSessionCfg.from_mapping(cfg)
    )
    case = SceneCase("case", "initial", "scene", "task", "robot")
    session.register_case(case, torch.tensor([[-1.0, 1.0]]), joint_names=("joint",))
    valid = ValidationResult(
        (
            ValidationCheck("path_collision", "passed"),
            ValidationCheck("task_success", "passed"),
        )
    )
    accepted = []
    for index in range(2):
        identity, _ = session.propose(
            "case",
            "initial",
            source_id="source",
            source_revision="v1",
            template_id="recipe" + str(index),
            operator_id="episode",
        )
        session.add_prepared_episode(EpisodeCandidate(identity, "{}"), valid)
        session.take_ready(
            "case",
            "initial",
            episode_byte_budget=1024 * 1024,
            candidate_id=identity.candidate_id,
        )
        session.mark_rollout_started(identity)
        episode_id, commit_id = session.episode_ids(identity)
        episode = ExpertEpisode(
            identity,
            {"joint_positions": torch.tensor([[0.0], [0.1]])},
            torch.tensor([[0.1]]),
            torch.tensor([0.0, 0.1]),
            "qpos",
            valid,
            episode_id,
            commit_id,
        )
        result = session.accept_episode(episode)
        accepted.append(result)
        if result:
            receipt = CommitReceipt(
                episode_id,
                identity.candidate_id,
                identity.attempt_id,
                "frozen",
                commit_id,
                "case",
            )
            assert session.apply_receipt(receipt)
            assert not session.apply_receipt(receipt)
    assert accepted == ([True, False] if legacy else [True, True])
    assert session.snapshot()["counts"]["committed"] == (1 if legacy else 2)
    assert session.snapshot()["coverage"]["case"] == (1 if legacy else 2)


@pytest.mark.parametrize(
    "cfg_type", [TrajectoryExpansionCfg, TrajectoryExpansionJobCfg]
)
def test_both_legacy_config_types_keep_validation_and_seeded_identity(cfg_type):
    payload = {"augmentation": {"seed": 17}}
    cfg = cfg_type.from_mapping(payload)
    session = ExpansionSession(cfg)
    case = SceneCase("case", "initial", "scene", "task", "robot")
    session.register_case(case, torch.tensor([[-1.0, 1.0]]), joint_names=("joint",))
    identity, _ = session.propose(
        "case",
        "initial",
        source_id="source",
        source_revision="v1",
        template_id="recipe",
        operator_id="episode",
    )
    replay = ExpansionSession(cfg_type.from_mapping(payload))
    replay.register_case(case, torch.tensor([[-1.0, 1.0]]), joint_names=("joint",))
    assert (
        replay.propose(
            "case",
            "initial",
            source_id="source",
            source_revision="v1",
            template_id="recipe",
            operator_id="episode",
        )[0]
        == identity
    )
    cfg.execution.ready_high_watermark = "invalid"
    with pytest.raises(ValueError):
        ExpansionSession(cfg)
