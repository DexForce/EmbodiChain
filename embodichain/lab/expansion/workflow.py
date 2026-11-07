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

"""Assemble native ports around the shared engine and independent production stages."""

from __future__ import annotations
from pathlib import Path
import fcntl
import hashlib
import json
import resource
import numpy as np
import torch
from embodichain.data_pipeline.recording.media import _atomic_json, _digest, _mkdir
from .session import ExpansionSession
from .session_cfg import ExpansionSessionCfg
from .configured import ExpansionRunCfg, preflight_expansion_job, resolve_expansion_job
from .engine import ExpansionEngine, ExpansionJobCfg
from .journal import ExpansionJournal
from .persistence import ArtifactEpisodeSink
from ._gym_utils import _images

__all__ = ["run_expansion_job", "inspect_expansion_job", "resume_expansion_export"]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


class _RecordedSink:
    def __init__(self, sink, journal):
        self.sink, self.journal = sink, journal

    def _attempt(self, episode):
        return (
            self.journal.root
            / "attempts"
            / f"{episode.metadata['attempt_sequence']:08d}"
        )

    def stage(self, episode):
        path = self.sink.stage(episode)
        self.journal.update(
            self._attempt(episode), status="pending_write", artifact_id=_digest(path)
        )
        return path

    def submit(self, episode, *, submission_id):
        receipt = self.sink.submit(episode, submission_id=submission_id)
        self.journal.update(
            self._attempt(episode),
            status="committed",
            storage_id=str(Path(receipt.storage_id).relative_to(self.journal.root)),
        )
        return receipt


class _RecipeSource:
    def __init__(self, cfg, host, journal):
        self.cfg, self.host, self.journal = cfg, host, journal

    def next_batch(self, capacity):
        if self.journal.time_usage()[0] >= self.cfg.max_wall_time_s:
            return ()
        for field, checksum in self.host.report["input_files"].items():
            if _digest(Path(getattr(self.cfg, field))) != checksum:
                raise ValueError(f"job input {field} changed after preflight")
        return self.journal.next_batch(
            self.host.recipe,
            capacity=capacity,
            target=self.cfg.target_episodes,
            max_attempts=self.cfg.max_attempts,
            retry_interrupted=self.cfg.retry_interrupted,
        )

    def after_batch(self, snapshot):
        self.journal.checkpoint(force=True)
        _atomic_json(
            self.journal.root / "progress.json",
            {
                "counts": dict(snapshot["counts"]),
                "engine_seconds": dict(snapshot.get("timings", {})),
                "host_seconds": dict(self.host.timings),
                "wall_time_lower_bound_s": self.journal.time_usage()[0],
            },
        )


def inspect_expansion_job(directory: str | Path) -> dict[str, object]:
    """Audit physical commits and independently report collection/render/export progress."""
    root = Path(directory).resolve()
    job = json.loads((root / "job.json").read_text())
    cfg = (
        json.loads((root / "execution.json").read_text())["config"]
        if (root / "execution.json").exists()
        else job["config"]
    )
    receipts = ArtifactEpisodeSink(root).inspect()
    records = ExpansionJournal(root).attempts()
    outcomes = {
        status: sum(r["status"] == status for r in records)
        for status in sorted({r["status"] for r in records})
    }
    result = {
        "job_id": job["job_id"],
        "semantic_id": job.get("semantic_id", job["job_id"]),
        "committed": len(receipts),
        "attempts": len(records),
        "target_episodes": cfg["target_episodes"],
        "outcomes": outcomes,
        "rollout_attempted": sum(
            max(r.get("steps", 0), r.get("executed_steps_lower_bound", 0)) > 0
            or r["status"] == "running"
            for r in records
        ),
        "measured_successes": sum(
            bool(r.get("measured_accepted", r["status"] == "committed"))
            for r in records
        ),
    }
    result["collection_complete"] = len(receipts) >= cfg["target_episodes"]
    from .visual import inspect_observation_variants

    variants = inspect_observation_variants(root)
    result["observation_variants"] = len(variants)
    plan = (
        json.loads((root / "postprocessing.json").read_text())
        if (root / "postprocessing.json").exists()
        else None
    )
    requested = set(cfg.get("render_profiles", ()))
    known = {variant["variant_id"] for variant in variants}
    profiles = {} if plan is None else plan.get("profiles", {})
    if (root / "execution.json").exists():
        profiles = (
            json.loads((root / "execution.json").read_text()).get(
                "render_profile_snapshots"
            )
            or profiles
        )
    from .visual import _variant_key

    render_pending = []
    for parent in receipts:
        for name in requested:
            if name in profiles:
                key = _variant_key(parent.commit_id, name, profiles[name]["revision"])
                if key not in known:
                    render_pending.append(key)
            elif not any(
                v["parent_commit_id"] == parent.commit_id and v["profile_id"] == name
                for v in variants
            ):
                render_pending.append((parent.commit_id, name))
    result["stages"] = {
        "collection": {"complete": result["collection_complete"]},
        "render": {"complete": not render_pending, "pending": len(render_pending)},
        "export": {
            "complete": not cfg.get("export_lerobot", False),
            "pending": int(bool(receipts) and cfg.get("export_lerobot", False)),
        },
    }
    if cfg.get("export_lerobot") and (root / "lerobot-export.json").exists():
        from .export import export_expansion_lerobot

        try:
            exported = export_expansion_lerobot(None, root, verify_only=True)
            result["lerobot"] = exported
            result["stages"]["export"] = {"complete": True, "pending": 0}
        except ValueError as error:
            result["stages"]["export"]["reason"] = str(error)
    for path in (root / "stages").glob("*/*.json"):
        stage = json.loads(path.read_text())
        requested = (
            bool(cfg.get("render_profiles"))
            if stage["stage"] == "render"
            else bool(cfg.get("export_lerobot")) if stage["stage"] == "export" else True
        )
        satisfied_collection = (
            stage["stage"] == "collection" and result["collection_complete"]
        )
        if requested and stage.get("state") == "failed" and not satisfied_collection:
            result["stages"].setdefault(stage["stage"], {}).update(
                error=stage.get("error"), complete=False
            )
    seconds, uncertain = ExpansionJournal(root).time_usage()
    result["wall_time_lower_bound_s"], result["wall_time_uncertain"] = (
        seconds,
        uncertain,
    )
    from .reporting import build_expansion_report

    coverage = build_expansion_report(root, receipts=receipts, variants=variants)
    result["coverage"] = {
        "geometry_buckets": coverage["geometry"]["buckets"],
        "physical_conditions": coverage["physics"]["condition_fingerprints"],
        "visual_variants": coverage["visual"]["variants"],
        "source_groups": len(coverage["sources"]["ancestor_groups"]),
    }
    result["complete"] = all(stage["complete"] for stage in result["stages"].values())
    return result


def _migrate_legacy(root, cfg, report):
    path = root / "job.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())
    if previous.get("schema_version") == 2:
        return
    old_cfg = resolve_expansion_job(ExpansionRunCfg(**previous["config"]))
    old_report = preflight_expansion_job(old_cfg)
    if previous.get("resolved_deployment") != old_report["resolved_deployment"]:
        raise ValueError(
            "legacy resolved deployment changed or is missing; frozen export remains available"
        )
    if (
        old_report["semantic_id"] != report["semantic_id"]
        or old_report["input_files"] != previous["input_files"]
    ):
        raise ValueError(
            "legacy job inputs changed; cannot migrate its recorded evidence"
        )
    _atomic_json(root / "legacy-job.json", previous)
    for item in (root / "attempts").glob("*/attempt.json"):
        record = json.loads(item.read_text())
        _atomic_json(root / "legacy_attempts" / (item.parent.name + ".json"), record)
        record.update(
            sequence=int(item.parent.name),
            recipe_id=record.get("identity", {}).get(
                "template_id", str(record["ordinal"])
            ),
            attempt_id=record.get("identity", {}).get("attempt_id", 0),
        )
        if record["status"] in {"prepared", "running", "interrupted"}:
            record.update(
                status="abandoned", reason="legacy attempt lacks a typed replay recipe"
            )
        _atomic_json(item, record)
    observed = []
    if (root / "runtime.json").exists():
        observed.append(
            json.loads((root / "runtime.json").read_text())
            .get("seconds", {})
            .get("elapsed", 0.0)
        )
    if (root / "progress.json").exists():
        observed.append(
            json.loads((root / "progress.json").read_text()).get("elapsed_seconds", 0.0)
        )
    _atomic_json(
        root / "runs" / "legacy" / "status.json",
        {
            "state": "closed",
            "elapsed_seconds": max(observed, default=0.0),
            "time_uncertain": True,
        },
    )
    _atomic_json(
        path,
        {
            **report,
            "schema_version": 2,
            "config": cfg.to_dict(),
            "legacy_job_id": previous["job_id"],
        },
    )


def run_expansion_job(
    cfg: ExpansionRunCfg, *, resume: bool = False
) -> dict[str, object]:
    """Assemble one native host, shared scheduler and durable stage coordinator.

    Resource budgets may change on resume; physical recipe semantics remain pinned.
    Accepted parents are postprocessed even when collection exhausts its budget.
    """
    cfg = resolve_expansion_job(cfg)
    report = preflight_expansion_job(cfg)
    root = Path(cfg.output_dir).resolve()
    _mkdir(root)
    with (root / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume:
            _migrate_legacy(root, cfg, report)
        journal = ExpansionJournal(root)
        journal.open_run(report, cfg.to_dict(), resume=resume)
        sink = ArtifactEpisodeSink(root)
        host = None
        engine = None
        try:
            journal.reconcile(sink)
            completed = len(sink.inspect())
            prior = journal.attempts()
            from .stages import plan_postprocessing, render_pending, run_postprocessing

            plan = plan_postprocessing(cfg, root, sink.inspect())
            status = inspect_expansion_job(root)
            if not status["complete"]:
                source_artifacts = {
                    field: {
                        "path": str(Path(getattr(cfg, field)).resolve()),
                        "sha256": checksum,
                    }
                    for field, checksum in report["input_files"].items()
                }
                for field, path in (
                    ("job", root / "job.json"),
                    (
                        "execution",
                        root / "runs" / journal.run_id / "configuration.json",
                    ),
                ):
                    source_artifacts[field] = {
                        "path": str(path),
                        "sha256": _digest(path),
                    }
                from .host import GymExpansionHost

                needs_collection = (
                    completed < cfg.target_episodes
                    and len(prior) < cfg.max_attempts
                    and journal.time_usage()[0] < cfg.max_wall_time_s
                )
                needs_render = bool(render_pending(root, plan))
                render_startup_error = None
                if needs_collection or needs_render:
                    try:
                        host = GymExpansionHost(
                            cfg, report, root, journal, source_artifacts
                        )
                    except Exception as error:
                        render_startup_error = str(error)
                        journal.stage(
                            "collection" if needs_collection else "render",
                            "failed",
                            error=render_startup_error,
                        )
                remaining = max(0, cfg.max_attempts - len(prior))
                if needs_collection and host is not None:
                    journal.stage("collection", "running")
                    session = ExpansionSession(
                        ExpansionSessionCfg.from_mapping(
                            {
                                "collection": {
                                    "target_committed_episodes": cfg.target_episodes
                                    - completed,
                                    "max_rollout_attempts": remaining,
                                    "max_proposals": remaining,
                                    "max_wall_time_s": max(
                                        0.001,
                                        cfg.max_wall_time_s - journal.time_usage()[0],
                                    ),
                                },
                                "execution": {
                                    "ready_high_watermark": cfg.batch_size,
                                    "ready_low_watermark": 0,
                                },
                                "persistence": {
                                    "pending_episode_limit": cfg.batch_size,
                                    "pending_max_bytes": cfg.batch_size
                                    * cfg.episode_byte_budget,
                                },
                                "validation": {"require_path_collision": True},
                            }
                        ),
                        episode_namespace=journal.job_instance_id,
                    )
                    engine = ExpansionEngine(
                        ExpansionJobCfg(
                            batch_size=cfg.batch_size,
                            episode_byte_budget=cfg.episode_byte_budget,
                            candidate_budget=cfg.batch_size,
                        ),
                        session=session,
                        host=host,
                        sink=_RecordedSink(sink, journal),
                    )
                    engine.run_source(_RecipeSource(cfg, host, journal))
                    journal.stage(
                        "collection",
                        (
                            "complete"
                            if len(sink.inspect()) >= cfg.target_episodes
                            else "budget_exhausted"
                        ),
                    )
                run_postprocessing(
                    cfg, host, sink, journal, render_startup_error=render_startup_error
                )
                from .reporting import build_expansion_report

                build_expansion_report(root, write=True)
                if host is not None:
                    _atomic_json(
                        root / "runtime.json",
                        {
                            "seconds": host.timings,
                            "engine_seconds": (
                                dict(engine.timings) if engine is not None else {}
                            ),
                            "peak_rss_bytes": resource.getrusage(
                                resource.RUSAGE_SELF
                            ).ru_maxrss
                            * 1024,
                            "profiler": host.env._profiler.report(),
                        },
                    )
        finally:
            try:
                if host is not None:
                    host.close()
            finally:
                journal.checkpoint(force=True, state="closed")

        from .reporting import build_expansion_report

        build_expansion_report(root, write=True)
        return inspect_expansion_job(root)


def resume_expansion_export(directory: str | Path) -> dict[str, object]:
    """Recover frozen writes and export confirmed artifacts without source configs or a World.

    Args:
        directory: Existing job bundle, including a relocated portable v2 bundle.
    Returns:
        Current job progress with a separate stage_complete flag. Exporting a
        partial collection never marks its physical target as achieved.
    """
    root = Path(directory).resolve()
    job = json.loads((root / "job.json").read_text())
    saved = (
        json.loads((root / "execution.json").read_text())["config"]
        if (root / "execution.json").exists()
        else job["config"]
    )
    cfg = ExpansionRunCfg(**{**saved, "output_dir": str(root), "export_lerobot": True})
    report = {
        **job,
        "semantic_id": job.get("semantic_id", job["job_id"]),
        "execution_id": _hash(("frozen-export/v2", cfg.media_buffer_bytes)),
    }
    if (root / "execution.json").exists():
        report["render_profile_snapshots"] = json.loads(
            (root / "execution.json").read_text()
        ).get("render_profile_snapshots", job.get("render_profile_snapshots"))
    with (root / ".owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = ExpansionJournal(root)
        journal.open_run(report, cfg.to_dict(), resume=True, operation="export")
        try:
            sink = ArtifactEpisodeSink(root)
            journal.reconcile(sink)
            journal.stage("export", "running")
            from .export import export_expansion_lerobot

            try:
                exported = export_expansion_lerobot(None, root)
                journal.stage(
                    "export",
                    "complete",
                    dataset_path=exported["dataset_path"],
                    episode_ids=exported["episode_ids"],
                )
            except Exception as error:
                journal.stage("export", "failed", error=str(error))
                raise
        finally:
            journal.checkpoint(force=True, state="closed")
        result = inspect_expansion_job(root)
        return {
            **result,
            "operation": "export",
            "stage_complete": result["stages"]["export"]["complete"],
        }
