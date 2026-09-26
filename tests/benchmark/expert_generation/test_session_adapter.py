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

"""The G-03 adapter exercises the existing #670 GenerationSession contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _session_and_case():
    from embodichain.lab.sim.motion.expansion.cfg import TrajectoryGenerationJobCfg
    from embodichain.lab.sim.motion.expansion.contracts import (
        SceneCase,
        ValidationCheck,
        ValidationResult,
    )
    from embodichain.lab.sim.motion.expansion.session import GenerationSession

    session = GenerationSession(TrajectoryGenerationJobCfg())
    scene = SceneCase("scene-0", "initial-0", "scene-v1", "lift", "robot")
    session.register_case(
        scene,
        torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        joint_names=("joint_a", "joint_b"),
    )
    return (
        session,
        scene,
        ValidationResult(
            (
                ValidationCheck("path_collision", "passed"),
                ValidationCheck("task_success", "passed"),
            )
        ),
    )


def test_session_adapter_confirms_a_real_generation_session_receipt() -> None:
    from embodichain.lab.sim.motion.expansion.contracts import (
        CandidateTrajectoryBatch,
        CommitReceipt,
        ExpertEpisode,
    )
    from scripts.benchmark.expert_generation.contracts import GenerationCase
    from scripts.benchmark.expert_generation.session_adapter import (
        GenerationSessionExecutor,
    )

    session, scene, validation = _session_and_case()
    case = GenerationCase(
        experiment_id="g03-session",
        case_id="case-0",
        source_kind="atomic_action",
        source_id="fixture-action",
        source_revision="fixture:v1",
        scene_case_id=scene.scene_case_id,
        initial_state_id=scene.initial_state_id,
        seed=0,
    )

    @dataclass
    class Host:
        session: object
        validation: object

        def propose(self, current, attempt_id):
            identity, _ = self.session.propose(
                scene.scene_case_id,
                scene.initial_state_id,
                source_id="handwritten_qpos",
                source_revision="v1",
                template_id="reference-0",
                operator_id="joint_residual",
            )
            batch = CandidateTrajectoryBatch(
                torch.zeros(1, 2, 2),
                torch.tensor([[0.0, 1.0]]),
                torch.tensor([2], dtype=torch.int64),
                (identity,),
                ("joint_a", "joint_b"),
            )
            self.session.add_planned(batch, self.validation)
            assert self.session.take_ready(scene.scene_case_id, scene.initial_state_id)
            return identity

        def rollout(self, current, identity):
            episode_id, commit_id = self.session.episode_ids(identity)
            return ExpertEpisode(
                identity=identity,
                observations={
                    "joint_positions": torch.tensor([[0.0, 0.0], [0.5, 0.0]])
                },
                actions=torch.tensor([[0.5, 0.0]]),
                timestamps=torch.tensor([0.0, 1.0]),
                action_representation="qpos",
                validation=self.validation,
                episode_id=episode_id,
                commit_id=commit_id,
            )

        def persist(self, current, episode):
            return CommitReceipt(
                episode_id=episode.episode_id,
                candidate_id=episode.identity.candidate_id,
                attempt_id=episode.identity.attempt_id,
                storage_id="memory:episode",
                commit_id=episode.commit_id,
                scene_case_id=episode.identity.scene_case_id,
            )

    attempt = GenerationSessionExecutor(session, Host(session, validation))(case, 0)
    assert attempt.execution_status == "completed"
    assert attempt.validation_status == "passed"
    assert attempt.task_status == "passed"
    assert attempt.receipt is not None and attempt.receipt.confirmed
    assert session.snapshot()["counts"]["committed"] == 1


def test_session_adapter_rejects_measured_validation_before_persistence() -> None:
    from embodichain.lab.sim.motion.expansion.contracts import (
        CandidateTrajectoryBatch,
        ExpertEpisode,
        ValidationCheck,
        ValidationResult,
    )
    from scripts.benchmark.expert_generation.contracts import GenerationCase
    from scripts.benchmark.expert_generation.session_adapter import (
        GenerationSessionExecutor,
    )

    session, scene, _ = _session_and_case()
    failed = ValidationResult((ValidationCheck("task_success", "failed"),))
    case = GenerationCase(
        experiment_id="g03-session-failed",
        case_id="case-0",
        source_kind="handwritten",
        source_id="fixture",
        source_revision="fixture:v1",
        scene_case_id=scene.scene_case_id,
        initial_state_id=scene.initial_state_id,
        seed=0,
    )

    class Host:
        def propose(self, current, attempt_id):
            identity, _ = session.propose(
                scene.scene_case_id,
                scene.initial_state_id,
                source_id="handwritten_qpos",
                source_revision="v1",
                template_id="reference-0",
                operator_id="joint_residual",
            )
            session.add_planned(
                CandidateTrajectoryBatch(
                    torch.zeros(1, 2, 2),
                    torch.tensor([[0.0, 1.0]]),
                    torch.tensor([2], dtype=torch.int64),
                    (identity,),
                    ("joint_a", "joint_b"),
                ),
                ValidationResult((ValidationCheck("path_collision", "passed"),)),
            )
            session.take_ready(scene.scene_case_id, scene.initial_state_id)
            return identity

        def rollout(self, current, identity):
            episode_id, commit_id = session.episode_ids(identity)
            return ExpertEpisode(
                identity,
                {"joint_positions": torch.tensor([[0.0, 0.0], [0.5, 0.0]])},
                torch.tensor([[0.5, 0.0]]),
                torch.tensor([0.0, 1.0]),
                "qpos",
                failed,
                episode_id,
                commit_id,
            )

        def persist(self, current, episode):
            raise AssertionError("failed validation must not reach persistence")

    attempt = GenerationSessionExecutor(session, Host())(case, 0)
    assert attempt.validation_status == "failed"
    assert attempt.receipt is None
    assert session.snapshot()["counts"]["committed"] == 0
