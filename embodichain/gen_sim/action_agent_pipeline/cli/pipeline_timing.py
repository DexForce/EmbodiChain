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

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["configure_pipeline_timing", "write_pipeline_timing_summary"]


def configure_pipeline_timing(
    args: argparse.Namespace,
) -> tuple[Path, Path] | None:
    if not getattr(args, "timing", True):
        from embodichain.gen_sim.action_agent_pipeline.utils.timing import (
            disable_timing_tracking,
        )

        disable_timing_tracking()
        return None

    from embodichain.gen_sim.action_agent_pipeline.utils.timing import (
        configure_timing_tracking,
    )

    output_dir = Path(args.config_output_dir).expanduser().resolve()
    timing_path = (
        Path(args.timing_output).expanduser().resolve()
        if getattr(args, "timing_output", None)
        else output_dir / "timing.jsonl"
    )
    summary_path = (
        Path(args.timing_summary_output).expanduser().resolve()
        if getattr(args, "timing_summary_output", None)
        else output_dir / "timing_summary.json"
    )
    run_id = getattr(args, "llm_usage_run_id", None) or (
        f"{args.task_name}_{_utc_run_timestamp()}"
    )
    configure_timing_tracking(
        timing_path=timing_path,
        run_id=run_id,
        process_name="run_agent_pipeline",
        reset=True,
    )
    print(f"Recording local timing: {timing_path}", flush=True)
    print(f"Local timing summary: {summary_path}", flush=True)
    return timing_path, summary_path


def write_pipeline_timing_summary(
    timing_paths: tuple[Path, Path] | None,
) -> dict[str, Any] | None:
    if timing_paths is None:
        return None

    from embodichain.gen_sim.action_agent_pipeline.utils.timing import (
        write_timing_summary,
    )

    timing_path, summary_path = timing_paths
    summary = write_timing_summary(
        timing_path=timing_path,
        summary_path=summary_path,
    )
    total = summary["total"]
    print(
        "Local timing total: "
        f"calls={total['calls']}, total={total['total_s']:.3f}s",
        flush=True,
    )
    for stage_name, bucket in _top_stages(summary, limit=5):
        print(
            "  "
            f"{stage_name}: total={bucket['total_s']:.3f}s, "
            f"calls={bucket['calls']}, max={bucket['max_s']:.3f}s",
            flush=True,
        )
    return summary


def _top_stages(
    summary: dict[str, Any],
    *,
    limit: int,
) -> list[tuple[str, dict[str, Any]]]:
    stages = summary.get("by_stage", {})
    if not isinstance(stages, dict):
        return []
    return sorted(
        stages.items(),
        key=lambda item: float(item[1].get("total_s", 0.0)),
        reverse=True,
    )[:limit]


def _utc_run_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
