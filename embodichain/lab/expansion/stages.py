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

"""Postprocessing eligibility and failure isolation for confirmed parent artifacts."""

from __future__ import annotations
import json
import time
from embodichain.data_pipeline.recording.media import _atomic_json
from .visual import (
    load_visual_profiles,
    inspect_observation_variants,
    _profile_revision,
    _variant_key,
)

__all__: list[str] = []


def plan_postprocessing(cfg, root, receipts):
    execution = (
        json.loads((root / "execution.json").read_text())
        if (root / "execution.json").exists()
        else {}
    )
    snapshots = execution.get("render_profile_snapshots")
    profiles = (
        {name: value["profile"] for name, value in snapshots.items()}
        if snapshots is not None
        else (
            load_visual_profiles(cfg.visual_profile_file) if cfg.render_profiles else {}
        )
    )
    expected = [
        {
            "variant_id": _variant_key(
                parent.commit_id, name, _profile_revision(profiles[name])
            ),
            "parent_commit_id": parent.commit_id,
            "profile_id": name,
            "profile_revision": _profile_revision(profiles[name]),
        }
        for parent in receipts
        for name in cfg.render_profiles
    ]
    plan = {
        "schema_version": 2,
        "profiles": {
            name: {
                "revision": _profile_revision(profiles[name]),
                "profile": profiles[name],
            }
            for name in cfg.render_profiles
        },
        "parents": [parent.commit_id for parent in receipts],
        "render": expected,
        "export": cfg.export_lerobot,
    }
    _atomic_json(root / "postprocessing.json", plan)
    return plan


def render_pending(root, plan):
    existing = {variant["variant_id"] for variant in inspect_observation_variants(root)}
    return tuple(item for item in plan["render"] if item["variant_id"] not in existing)


def run_postprocessing(cfg, host, sink, journal, *, render_startup_error=None):
    root = journal.root
    receipts = sink.inspect()
    plan = plan_postprocessing(cfg, root, receipts)
    pending = render_pending(root, plan)
    if not receipts:
        journal.stage("render", "empty")
        journal.stage("export", "empty")
        return
    if not pending:
        journal.stage(
            "render",
            "complete",
            variants=[item["variant_id"] for item in plan["render"]],
        )
    if not cfg.export_lerobot:
        journal.stage("export", "not_requested")
    if pending:
        from .visual import render_observation_variants

        journal.stage(
            "render", "running", pending=[item["variant_id"] for item in pending]
        )
        started = time.monotonic()
        try:
            if host is None:
                raise RuntimeError(
                    render_startup_error
                    or "rendering host is unavailable; frozen parents remain exportable"
                )
            if host.appearance is not None:
                host.appearance.close()
                host.appearance = None
            result = render_observation_variants(
                host.env,
                root,
                profile_file=(
                    str(root / "runs" / journal.run_id / "render_profiles.json")
                    if (
                        root / "runs" / journal.run_id / "render_profiles.json"
                    ).exists()
                    else cfg.visual_profile_file
                ),
                profiles=cfg.render_profiles,
                buffer_bytes=cfg.media_buffer_bytes,
            )
            journal.stage(
                "render", "complete", variants=[item["variant_id"] for item in result]
            )
        except Exception as error:
            journal.stage("render", "failed", error=str(error))
        finally:
            if host is not None:
                host.timings["render_variants"] += time.monotonic() - started
            journal.checkpoint(force=True)
    if cfg.export_lerobot:
        from .export import export_expansion_lerobot

        journal.stage("export", "running")
        started = time.monotonic()
        try:
            # Export consumes only recorded schema/evidence, even after renderer failure.
            result = export_expansion_lerobot(None, root)
            journal.stage(
                "export",
                "complete",
                dataset_path=result["dataset_path"],
                episode_ids=result["episode_ids"],
            )
        except Exception as error:
            journal.stage("export", "failed", error=str(error))
        finally:
            if host is not None:
                host.timings["export"] += time.monotonic() - started
            journal.checkpoint(force=True)
