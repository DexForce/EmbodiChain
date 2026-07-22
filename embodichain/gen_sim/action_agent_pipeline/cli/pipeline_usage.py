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

__all__ = ["configure_llm_usage_tracking", "write_llm_usage_summary"]


def configure_llm_usage_tracking(
    args: argparse.Namespace,
) -> tuple[Path, Path] | None:
    if not args.llm_usage:
        from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
            disable_usage_tracking,
        )

        disable_usage_tracking()
        return None

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
        configure_usage_tracking,
    )

    output_dir = Path(args.config_output_dir).expanduser().resolve()
    usage_path = (
        Path(args.llm_usage_output).expanduser().resolve()
        if args.llm_usage_output
        else output_dir / "llm_usage.jsonl"
    )
    summary_path = (
        Path(args.llm_usage_summary_output).expanduser().resolve()
        if args.llm_usage_summary_output
        else output_dir / "llm_usage_summary.json"
    )
    run_id = args.llm_usage_run_id or (f"{args.task_name}_{_utc_run_timestamp()}")
    configure_usage_tracking(
        usage_path=usage_path,
        run_id=run_id,
        process_name="run_agent_pipeline",
        reset=True,
    )
    print(f"Recording local LLM token usage: {usage_path}", flush=True)
    print(f"Local LLM token usage summary: {summary_path}", flush=True)
    return usage_path, summary_path


def write_llm_usage_summary(usage_paths: tuple[Path, Path] | None) -> None:
    if usage_paths is None:
        return

    from embodichain.gen_sim.action_agent_pipeline.utils.llm_usage import (
        write_usage_summary,
    )

    usage_path, summary_path = usage_paths
    summary = write_usage_summary(
        usage_path=usage_path,
        summary_path=summary_path,
    )
    total = summary["total"]
    print(
        "Local LLM token usage total: "
        f"calls={total['calls']}, "
        f"input={total['input_tokens']}, "
        f"output={total['output_tokens']}, "
        f"total={total['total_tokens']}",
        flush=True,
    )


def _utc_run_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
