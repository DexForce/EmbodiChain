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

"""CLI for image-to-semantic-graph planning without simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from embodichain.gen_sim.services.runtime import GenSimServices

__all__ = ["main"]


def main(
    argv: Sequence[str] | None = None,
    *,
    services: GenSimServices | None = None,
    prog: str = "embodichain task-engine",
) -> None:
    """Generate JSON declarations and an offline graph viewer from a photograph.

    Args:
        argv: CLI arguments, excluding the executable; defaults to sys.argv.
        services: Optional borrowed model-service run for an embedded caller.
        prog: Display name supplied by the unified GenSim dispatcher.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Plan image-grounded canonical Semantic Calls without simulation.",
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument(
        "--output",
        "--output-root",
        "--output_root",
        type=Path,
        required=True,
        help="Fresh output directory; existing directories are never overwritten.",
    )
    parser.add_argument("--task-id", "--task_id", default="image_task")
    args = parser.parse_args(argv)
    if not args.instruction.strip():
        parser.error("--instruction must not be empty")

    from ._image_run import _plan_image_task

    _plan_image_task(
        args.image,
        args.instruction,
        args.output,
        task_id=args.task_id,
        services=services,
    )


def _revise(argv: Sequence[str] | None = None, *, prog: str) -> None:
    import json
    from ..workflow import revise_task_plan

    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-spec-digest", required=True)
    parser.add_argument("--expected-graph-digest", required=True)
    parser.add_argument("--expected-plan-digest", required=True)
    args = parser.parse_args(argv)
    print(
        revise_task_plan(
            args.parent,
            json.loads(args.proposal.read_text()),
            args.output,
            expected_spec_digest=args.expected_spec_digest,
            expected_graph_digest=args.expected_graph_digest,
            expected_plan_digest=args.expected_plan_digest,
        )
    )


if __name__ == "__main__":
    main()
