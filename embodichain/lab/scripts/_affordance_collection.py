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

"""Bounded offline Affordance collection with synchronous row receipts."""

from __future__ import annotations

import json
from hashlib import sha256
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from embodichain.lab.gym.envs.augmentation import _sampling_attempt
from embodichain.lab.gym.envs.demo import DemoExecutionCfg, execute_demo_episode
from embodichain.lab.task_program.integrations.catalog import _canonical_value
from embodichain.utils.logger import log_info

__all__: list[str] = []


def _configuration_value(value: object) -> object:
    """Canonicalize loaded config data without opaque-object repr fallbacks."""
    if isinstance(value, np.ndarray):
        return {
            "array_dtype": str(value.dtype),
            "array_shape": list(value.shape),
            "array_value": _configuration_value(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _configuration_value(value.item())
    if isinstance(value, Path):
        return {"path": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _configuration_value(getattr(value, field.name))
                for field in fields(value)
            },
        }
    if isinstance(value, Mapping):
        return _canonical_value(
            {key: _configuration_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return [_configuration_value(item) for item in value]
    return _canonical_value(value)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Atomically replace the run report after a transaction boundary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _collect_affordance_episodes(
    args: Any, env: Any, gym_config: dict[str, Any]
) -> None:
    """Collect a quota of measured episodes, counting only persistence receipts."""
    target = getattr(env, "unwrapped", env)
    cfg = target.cfg.expert_trajectory.affordance_augmentation
    quota = int(gym_config.get("max_episodes", 1))
    max_attempts = int(gym_config.get("demo_max_attempts", 3))
    if quota < 0:
        raise ValueError("max_episodes must be non-negative.")
    if max_attempts < 1:
        raise ValueError("demo_max_attempts must be positive.")
    # All policy, provider, and sink validation precedes the first scene reset.
    program = target.prepare_affordance_collection()
    manager = target.dataset_manager
    run_id = uuid4().hex
    manifest_path = (
        Path(manager.episode_commit_path) / f"affordance_collection_{run_id}.json"
    )
    previous_receipts = {
        receipt.commit_id for receipt in manager.episode_commit_receipts
    }
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "compatibility": {"accepted": "selected"},
        "run_id": run_id,
        "status": "running",
        "task_id": gym_config.get("id"),
        "gym_config_source": str(getattr(args, "gym_config", "")),
        "task_program_source": str(getattr(args, "task_program", "") or ""),
        "program_id": program.program_id,
        "seed": target.cfg.seed,
        "branches": cfg.branches,
        "num_envs": target.num_envs,
        "required_assurance": cfg.required_assurance,
        "max_episodes": quota,
        "max_batches": cfg.max_batches,
        "demo_max_attempts": max_attempts,
        "attempts": 0,
        "measured_accepted": 0,
        "selected": 0,
        "quota_discarded": 0,
        "dataset_written": 0,
        # Kept for readers of schema version 1. It has the old, selected-row
        # meaning and is intentionally not used as the collection quota.
        "accepted": 0,
        "committed": 0,
        "history": [],
        "receipts": [],
    }
    adapter = target.task_program_adapter
    registration = getattr(adapter, "_registration", None)
    # Use already-loaded declarations and their canonical registration digest.
    # Re-reading component paths would hash possibly changed, unconsumed files.
    resolved_config = {
        "gym_config": gym_config,
        "task_program": getattr(target.cfg, "task_program", None),
        "registration_sha256": getattr(registration, "fingerprint", None),
        "configured_integration_sha256": getattr(
            adapter, "_configured_integration_fingerprint", None
        ),
        "embodiment": {
            name: getattr(target.cfg, name, None)
            for name in ("robot", "sensor", "control_parts", "active_joint_ids")
        },
        "execution": {
            "joint_command_mode": getattr(
                target.cfg.expert_trajectory, "joint_command_mode", None
            ),
            "step_dt": getattr(adapter, "step_dt", None),
            "runner_cfg": getattr(adapter, "_runner_cfg", None),
        },
        "seed": target.cfg.seed,
        "branches": cfg.branches,
        "max_batches": cfg.max_batches,
        "required_assurance": cfg.required_assurance,
    }
    config_json = json.dumps(
        _configuration_value(resolved_config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    manifest["config_sha256"] = sha256(config_json.encode("utf-8")).hexdigest()
    manifest["source_sha256"] = {}
    for source in (manifest["gym_config_source"], manifest["task_program_source"]):
        if source and Path(source).is_file():
            manifest["source_sha256"][source] = sha256(
                Path(source).read_bytes()
            ).hexdigest()
    manifest["dataset_receipts"] = []
    finalized = False

    def refresh_receipts() -> None:
        receipts = [
            receipt
            for receipt in manager.episode_commit_receipts
            if receipt.commit_id not in previous_receipts
        ]
        manifest["dataset_receipts"] = [asdict(receipt) for receipt in receipts]
        manifest["dataset_written"] = len(receipts)

    try:
        _write_manifest(manifest_path, manifest)
        batch_id = 0
        while manifest["committed"] < quota:
            for attempt_id in range(max_attempts):
                if manifest["attempts"] >= cfg.max_batches:
                    raise RuntimeError(
                        f"Affordance collection exhausted max_batches={cfg.max_batches}; committed {manifest['committed']}/{quota} episodes."
                    )
                env.reset(options={"save_data": False})
                record: dict[str, Any] = {
                    "batch_id": batch_id,
                    "attempt_id": attempt_id,
                    "accepted_env_ids": [],
                    "measured_accepted": 0,
                    "selected": 0,
                    "quota_discarded": 0,
                    "receipt_ids": [],
                    "dataset_receipt_ids": [],
                }
                manifest["history"].append(record)
                manifest["attempts"] += 1
                committed = False
                attempt_error: BaseException | None = None
                try:
                    with _sampling_attempt(
                        target,
                        cfg,
                        run_id=run_id,
                        batch_id=batch_id,
                        attempt_id=attempt_id,
                    ):
                        target._affordance_collection_metadata.update(
                            config_sha256=manifest["config_sha256"],
                            source_sha256=manifest["source_sha256"],
                            program_id=program.program_id,
                        )
                        target.task_program_adapter.validate_measured_acceptance(
                            program
                        )
                        result = execute_demo_episode(
                            env,
                            episode_index=batch_id,
                            attempt_id=attempt_id,
                            execution_cfg=DemoExecutionCfg(),
                            task_program=program,
                        )
                        record["terminal_reason"] = result.terminal_reason
                        record["lengths"] = list(getattr(result, "lengths", ()))
                        rows = target.select_affordance_episode_rows(
                            result, program, quota - manifest["committed"]
                        )
                        selection = getattr(
                            target, "_affordance_selection_counts", None
                        )
                        if not isinstance(selection, Mapping):
                            # Preserve compatibility with lightweight host
                            # doubles and older environments that only return
                            # the selected row tuple.
                            selection = {
                                "measured_accepted": len(rows),
                                "selected": len(rows),
                                "quota_discarded": 0,
                            }
                        measured_accepted = int(
                            selection.get("measured_accepted", len(rows))
                        )
                        selected_count = int(selection.get("selected", len(rows)))
                        quota_discarded = int(selection.get("quota_discarded", 0))
                        if (
                            measured_accepted < 0
                            or selected_count != len(rows)
                            or quota_discarded < 0
                            or measured_accepted != selected_count + quota_discarded
                        ):
                            raise ValueError(
                                "Affordance selection counts are inconsistent with "
                                "the selected rows."
                            )
                        record["measured_accepted"] = measured_accepted
                        record["selected"] = selected_count
                        record["quota_discarded"] = quota_discarded
                        manifest["measured_accepted"] += measured_accepted
                        manifest["selected"] += selected_count
                        manifest["quota_discarded"] += quota_discarded
                        record["accepted_env_ids"] = list(rows)
                        manifest["accepted"] += selected_count
                        if rows:
                            before = {
                                receipt.commit_id
                                for receipt in manager.episode_commit_receipts
                            }
                            try:
                                receipts = target.commit_demo_rows(rows)
                            except BaseException as commit_error:
                                try:
                                    refresh_receipts()
                                    record["dataset_receipt_ids"] = [
                                        receipt.commit_id
                                        for receipt in manager.episode_commit_receipts
                                        if receipt.commit_id not in before
                                    ]
                                except BaseException as report_error:
                                    commit_error.add_note(
                                        f"Commit receipt inspection also failed: {report_error}"
                                    )
                                raise
                            refresh_receipts()
                            receipt_rows = tuple(receipt.env_id for receipt in receipts)
                            receipt_ids = {receipt.commit_id for receipt in receipts}
                            known_ids = {
                                receipt["commit_id"] for receipt in manifest["receipts"]
                            }
                            if (
                                len(receipt_rows) != len(rows)
                                or set(receipt_rows) != set(rows)
                                or len(receipt_ids) != len(rows)
                                or receipt_ids & known_ids
                            ):
                                raise RuntimeError(
                                    "Commit receipts do not uniquely confirm the accepted Affordance rows."
                                )
                            record["receipt_ids"] = [
                                receipt.commit_id for receipt in receipts
                            ]
                            record["dataset_receipt_ids"] = [
                                receipt.commit_id
                                for receipt in manager.episode_commit_receipts
                                if receipt.commit_id not in before
                            ]
                            manifest["receipts"].extend(
                                asdict(receipt) for receipt in receipts
                            )
                            manifest["committed"] += len(receipts)
                            committed = True
                            record["status"] = "committed"
                        else:
                            record["status"] = "rejected"
                except BaseException as error:
                    attempt_error = error
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    # The context has exited before discard resets the live scene.
                    if not committed:
                        try:
                            env.reset(options={"save_data": False})
                        except BaseException as cleanup_error:
                            if attempt_error is None:
                                raise
                            attempt_error.add_note(
                                f"Affordance discard also failed: {cleanup_error}"
                            )
                _write_manifest(manifest_path, manifest)
                if committed:
                    break
            else:
                raise RuntimeError(
                    f"Affordance collection batch {batch_id} exhausted {max_attempts} attempts; committed {manifest['committed']}/{quota} episodes."
                )
            batch_id += 1
        finalized = True
        manager.finalize()
        manifest["status"] = "complete"
        _write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        manifest["error"] = f"{type(error).__name__}: {error}"
        if not finalized:
            try:
                manager.finalize()
            except BaseException as cleanup_error:
                error.add_note(f"Dataset finalization also failed: {cleanup_error}")
        try:
            refresh_receipts()
            _write_manifest(manifest_path, manifest)
        except BaseException as report_error:
            error.add_note(f"Affordance manifest update also failed: {report_error}")
        raise
    log_info(
        f"Collection complete · {manifest['committed']} measured episodes saved in {manifest['attempts']} attempts. Manifest: {manifest_path}",
        color="green",
    )
