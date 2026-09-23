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

import torch

from embodichain.lab.sim.motion.expansion import (
    CandidateCoordinator,
    CommitReceipt,
    ExpertEpisode,
    GenerationSession,
    SceneCase,
    SingleSlotRunner,
    SourceContext,
    TemplateSourceAdapter,
    TrajectoryGenerationJobCfg,
    TrajectoryPhase,
    TrajectoryTemplate,
    ValidationCheck,
    ValidationResult,
)


class _Restorer:
    def __init__(self) -> None:
        self.items = []

    def restore(self, item):
        self.items.append(item.spec.identity.candidate_id)
        return {"epoch": len(self.items)}


class _Executor:
    def validate_plan(self, item):
        return ValidationResult((ValidationCheck("path_collision", "passed"),))

    def execute(self, item, prepared, *, episode_id, commit_id):
        assert prepared["epoch"] == 1
        positions = item.template.positions
        return ExpertEpisode(
            identity=item.spec.identity,
            observations={"joint_positions": positions},
            actions=positions[:-1],
            timestamps=item.template.dt.cumsum(0),
            action_representation="qpos",
            validation=ValidationResult(
                (
                    ValidationCheck("path_collision", "passed"),
                    ValidationCheck("task_success", "passed"),
                )
            ),
            episode_id=episode_id,
            commit_id=commit_id,
            phases=item.template.phases,
        )


class _Sink:
    def submit(self, episode, *, submission_id):
        return CommitReceipt(
            episode_id=episode.episode_id,
            candidate_id=episode.identity.candidate_id,
            attempt_id=episode.identity.attempt_id,
            storage_id="memory://episode",
            commit_id=episode.commit_id,
            scene_case_id=episode.identity.scene_case_id,
            submission_id=submission_id,
        )


def test_single_slot_runner_completes_measured_receipt_lifecycle():
    cfg = TrajectoryGenerationJobCfg.from_mapping(
        {
            "augmentation": {
                "factors": {
                    "spatial": {
                        "enabled": True,
                        "method": ["joint_residual"],
                        "joint_offset_scale": 0.1,
                    }
                },
                "coverage": {"joint_dedup_normalized_tol": 0.001},
            },
            "collection": {"target_committed_episodes": 1},
        }
    )
    case = SceneCase("case", "initial", "signature", "task", "robot")
    limits = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]])
    session = GenerationSession(cfg)
    session.register_case(case, limits, joint_names=("arm", "tool"))
    context = SourceContext("handwritten", "test", "unit", case, 0.1)
    template = TrajectoryTemplate(
        source_id="source",
        source_revision="source",
        template_id="template",
        joint_names=("arm", "tool"),
        positions=torch.tensor([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        dt=torch.tensor([0.0, 0.1, 0.1]),
        phases=(TrajectoryPhase("free", 0, 3, "free", ("joint_residual",)),),
        allowed_operators=("joint_residual",),
        controlled_joint_indices=(0, 1),
    )
    coordinator = CandidateCoordinator(
        cfg,
        session=session,
        source_adapter=TemplateSourceAdapter(),
        source_context=context,
        joint_limits=limits,
    )
    coordinator.enqueue_source(template, count=1)
    restorer, executor, sink = _Restorer(), _Executor(), _Sink()

    outcome = SingleSlotRunner(
        coordinator,
        restorer,
        executor,
        sink,
        episode_byte_budget=1024 * 1024,
    ).run_next()

    assert outcome.status == "committed"
    assert outcome.receipt is not None and outcome.receipt.confirmed
    assert len(restorer.items) == 1
    assert session.snapshot()["counts"]["committed"] == 1
