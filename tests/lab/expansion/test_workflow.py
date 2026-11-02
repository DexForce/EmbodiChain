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

"""The configured pipeline uses the shared scheduler and independent artifact stages."""

from __future__ import annotations
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
import torch
from test_journal import recipe
from embodichain.lab.expansion.configured import load_expansion_job
from embodichain.lab.expansion.workflow import run_expansion_job
from embodichain.lab.expansion.engine import PreparedEpisodeWorkItem
from embodichain.lab.expansion.contracts import (
    EpisodeCandidate,
    SceneCase,
    ExpertEpisode,
    ValidationResult,
    ValidationCheck,
)
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.data_pipeline.recording.media import _atomic_json


class Host:
    fail_once = False
    all_success = False
    opened = 0
    closed = 0

    def __init__(self, cfg, report, root, journal, artifacts):
        type(self).opened += 1
        self.capacity = cfg.batch_size
        self.report = report
        self.journal = journal
        self.appearance = None
        self.env = SimpleNamespace(_profiler=SimpleNamespace(report=lambda: {}))
        self.timings = {"render_variants": 0.0, "export": 0.0}

    def recipe(self, ordinal):
        return recipe(ordinal)

    def prepare_batch(self, requests, session):
        items = []
        self.requests = {}
        for row, request in enumerate(requests):
            case = SceneCase(
                "case" + str(request.recipe.context.ordinal),
                "initial",
                "scene",
                "task",
                "robot",
            )
            session.register_case(
                case, torch.tensor([[-2.0, 2.0]]), joint_names=("joint",)
            )
            identity, _ = session.propose(
                case.scene_case_id,
                "initial",
                source_id="source",
                source_revision="v1",
                template_id=request.recipe.recipe_id,
                operator_id="episode",
                attempt_id=request.attempt_id,
            )
            episode_id, commit_id = session.episode_ids(identity)
            self.journal.update(
                request, status="prepared", commit_id=commit_id, episode_id=episode_id
            )
            item = PreparedEpisodeWorkItem(
                EpisodeCandidate(identity, "{}"), request.recipe.compatibility.key, row
            )
            items.append(item)
            self.requests[row] = request
        return items

    def preflight(self, items):
        pass

    def validate_plan(self, item):
        return ValidationResult((ValidationCheck("preparation", "passed"),))

    def execute_batch(self, assignments, *, on_started):
        result = []
        for item in assignments:
            request = self.requests[item.env_id]
            on_started(item.item.spec.identity)
            self.journal.update(request, status="running", steps=2)
            if type(self).fail_once:
                type(self).fail_once = False
                raise RuntimeError("interrupted native call")
            if not type(self).all_success and request.recipe.context.ordinal % 2:
                self.journal.update(request, status="rejected", reason="task_failed")
                result.append(None)
                continue
            result.append(
                ExpertEpisode(
                    item.item.spec.identity,
                    {
                        "joint_positions": torch.tensor([[0.0], [0.1], [0.2]]),
                        "joint_velocities": torch.zeros(3, 1),
                        "joint_efforts": torch.zeros(3, 1),
                    },
                    torch.tensor([[0.1], [0.2]]),
                    torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64),
                    "qpos",
                    ValidationResult(
                        (
                            ValidationCheck("path_collision", "passed"),
                            ValidationCheck("task_success", "passed"),
                        )
                    ),
                    item.episode_id,
                    item.commit_id,
                    metadata={
                        "attempt_sequence": request.sequence,
                        "recipe_id": request.recipe.recipe_id,
                        "expert_action_spec": {"joint_names": ["joint"]},
                    },
                )
            )
        return result

    def on_acceptance(self, assignment, episode, accepted):
        self.journal.update(
            self.requests[assignment.env_id],
            status="accepted" if accepted else "rejected",
            measured_accepted=episode.validation.accepted,
        )

    def on_deferred(self, item):
        self.journal.update(self.requests[item.env_id], status="interrupted")

    def close(self):
        type(self).closed += 1


def config(tmp_path):
    root = Path(__file__).resolve().parents[3]
    cfg = load_expansion_job(root / "examples/expansion/engine/job.default.yaml")
    cfg.batch_size = 2
    cfg.target_episodes = 2
    cfg.max_attempts = 2
    cfg.output_dir = str(tmp_path / "job")
    return cfg


def patch_host(monkeypatch):
    import embodichain.lab.expansion.host as module

    Host.opened = Host.closed = 0
    Host.fail_once = False
    Host.all_success = False
    monkeypatch.setattr(module, "GymExpansionHost", Host)


def test_partial_collection_still_exports_and_export_only_resume_opens_no_world(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    cfg = config(tmp_path)
    result = run_expansion_job(cfg)
    assert result["committed"] == 1 and not result["collection_complete"]
    from embodichain.lab.expansion import export as exporter

    observed = []

    def export(env, root, *, verify_only=False):
        assert env is None
        receipts = ArtifactEpisodeSink(root).inspect()
        observed.append(tuple(r.episode_id for r in receipts))
        value = {
            "dataset_path": "frozen",
            "episode_ids": list(observed[-1]),
            "episodes": 1,
        }
        _atomic_json(Path(root) / "lerobot-export.json", value)
        return value

    monkeypatch.setattr(exporter, "export_expansion_lerobot", export)
    cfg.export_lerobot = True
    resumed = run_expansion_job(cfg, resume=True)
    assert resumed["committed"] == 1 and not resumed["complete"]
    assert resumed["stages"]["export"]["complete"] and len(observed[0]) == 1
    assert Host.opened == Host.closed == 1


def test_interrupted_execution_reuses_recipe_with_new_attempt_and_capacity(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    Host.fail_once = True
    Host.all_success = True
    cfg = config(tmp_path)
    with pytest.raises(RuntimeError, match="interrupted native"):
        run_expansion_job(cfg)
    root = Path(cfg.output_dir)
    originals = [
        json.loads(p.read_text())
        for p in sorted((root / "attempts").glob("*/attempt.json"))
    ]
    cfg.batch_size = 8
    cfg.max_attempts = 4
    result = run_expansion_job(cfg, resume=True)
    assert result["complete"] and result["committed"] == 2
    records = [
        json.loads(p.read_text())
        for p in sorted((root / "attempts").glob("*/attempt.json"))
    ]
    assert [r["recipe_id"] for r in records[2:]] == [r["recipe_id"] for r in originals]
    assert [r["attempt_id"] for r in records[2:]] == [1, 1]
    assert Host.opened == Host.closed == 2


def test_renderer_failure_does_not_block_export_of_confirmed_parents(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    Host.all_success = True
    cfg = config(tmp_path)
    cfg.capture_media = True
    cfg.export_lerobot = True
    cfg.visual_profile_file = str(
        Path(__file__).resolve().parents[3] / "examples/expansion/engine/visual.yaml"
    )
    cfg.render_profiles = ("blue",)
    cfg.observation_independent = True
    import embodichain.lab.expansion.visual as visual
    import embodichain.lab.expansion.export as exporter

    def broken(*args, **kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(visual, "render_observation_variants", broken)

    def export(env, root, *, verify_only=False):
        assert env is None
        value = {
            "dataset_path": "frozen",
            "episode_ids": [r.episode_id for r in ArtifactEpisodeSink(root).inspect()],
        }
        _atomic_json(Path(root) / "lerobot-export.json", value)
        return value

    monkeypatch.setattr(exporter, "export_expansion_lerobot", export)
    result = run_expansion_job(cfg)
    assert result["committed"] == 2 and result["collection_complete"]
    assert (
        not result["complete"]
        and result["stages"]["render"]["error"] == "render failed"
    )
    assert result["stages"]["export"]["complete"]


def test_inspection_counts_interrupted_running_attempt_and_new_render_obligations(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    Host.all_success = True
    cfg = config(tmp_path)
    cfg.capture_media = True
    cfg.observation_independent = True
    cfg.visual_profile_file = str(
        Path(__file__).resolve().parents[3] / "examples/expansion/engine/visual.yaml"
    )
    cfg.render_profiles = ("blue",)
    import embodichain.lab.expansion.stages as stages
    import embodichain.lab.expansion.workflow as workflow

    # Stop after physical receipts, leaving the genuine pre-collection empty plan.
    monkeypatch.setattr(stages, "run_postprocessing", lambda *args, **kwargs: None)
    result = run_expansion_job(cfg)
    assert result["committed"] == 2
    assert result["stages"]["render"]["pending"] == 2
    assert not result["stages"]["render"]["complete"]
    root = Path(cfg.output_dir)
    path = root / "attempts" / "00000002" / "attempt.json"
    _atomic_json(
        path,
        {
            "sequence": 2,
            "ordinal": 2,
            "recipe_id": "interrupted",
            "status": "interrupted",
            "steps": 0,
            "executed_steps_lower_bound": 1,
        },
    )
    assert workflow.inspect_expansion_job(root)["rollout_attempted"] == 3


def test_render_host_startup_failure_keeps_offline_export_available(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    Host.all_success = True
    cfg = config(tmp_path)
    cfg.capture_media = True
    cfg.observation_independent = True
    run_expansion_job(cfg)
    import embodichain.lab.expansion.host as host_module
    import embodichain.lab.expansion.export as exporter

    def unavailable(*args, **kwargs):
        raise RuntimeError("renderer startup unavailable")

    monkeypatch.setattr(host_module, "GymExpansionHost", unavailable)
    cfg.visual_profile_file = str(
        Path(__file__).resolve().parents[3] / "examples/expansion/engine/visual.yaml"
    )
    cfg.render_profiles = ("blue",)
    cfg.export_lerobot = True

    def export(env, root, *, verify_only=False):
        assert env is None
        value = {
            "dataset_path": "frozen",
            "episode_ids": [r.episode_id for r in ArtifactEpisodeSink(root).inspect()],
        }
        _atomic_json(Path(root) / "lerobot-export.json", value)
        return value

    monkeypatch.setattr(exporter, "export_expansion_lerobot", export)
    result = run_expansion_job(cfg, resume=True)
    assert result["stages"]["export"]["complete"]
    assert result["stages"]["render"]["error"] == "renderer startup unavailable"
    assert not result["complete"]


def test_cli_export_recovery_does_not_resolve_source_or_open_world(
    tmp_path, monkeypatch, capsys
):
    patch_host(monkeypatch)
    Host.all_success = True
    cfg = config(tmp_path)
    run_expansion_job(cfg)
    import embodichain.lab.expansion.workflow as workflow
    import embodichain.lab.expansion.export as exporter
    from embodichain.cli.expansion import main

    def forbidden(*args, **kwargs):
        raise AssertionError("live source resolution is forbidden during frozen export")

    monkeypatch.setattr(workflow, "resolve_expansion_job", forbidden)

    def export(env, root, *, verify_only=False):
        assert env is None
        value = {
            "dataset_path": "frozen",
            "episode_ids": [r.episode_id for r in ArtifactEpisodeSink(root).inspect()],
        }
        _atomic_json(Path(root) / "lerobot-export.json", value)
        return value

    monkeypatch.setattr(exporter, "export_expansion_lerobot", export)
    assert main(["resume", "--output", cfg.output_dir, "--stage", "export"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["stage_complete"] and result["operation"] == "export"
    assert Host.opened == 1


def test_verified_target_supersedes_prior_collection_startup_failure(
    tmp_path, monkeypatch
):
    patch_host(monkeypatch)
    cfg = config(tmp_path)
    assert run_expansion_job(cfg)["committed"] == 1
    import embodichain.lab.expansion.host as module

    def unavailable(*args, **kwargs):
        raise RuntimeError("host unavailable")

    monkeypatch.setattr(module, "GymExpansionHost", unavailable)
    cfg.max_attempts = 4
    failed = run_expansion_job(cfg, resume=True)
    assert failed["stages"]["collection"]["error"] == "host unavailable"
    cfg.target_episodes = 1
    recovered = run_expansion_job(cfg, resume=True)
    assert recovered["committed"] == 1 and recovered["complete"]
    assert recovered["stages"]["collection"] == {"complete": True}
