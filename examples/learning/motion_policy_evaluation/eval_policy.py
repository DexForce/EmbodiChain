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

"""Run the public ANYmal-C checkpoint directly from the example directory."""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["example_arguments", "main"]

EXAMPLE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXAMPLE_ROOT.parents[2]
DEFAULT_CACHE = Path.home() / ".cache/embodichain/examples/anymal_c_velocity"


def example_arguments(argv: list[str]) -> list[str]:
    """Add the example Profile, checkpoint, and resource paths.

    Args:
        argv: Evaluation options accepted by ``eval-motion-policy``.

    Returns:
        Arguments ready for the EmbodiChain evaluation CLI.
    """
    cache = Path(os.environ.get("ANYMAL_C_EXAMPLE_CACHE", DEFAULT_CACHE))
    resource_root = cache / "upstream"
    checkpoint = resource_root / "anybotics_anymal_c/rl_policies/mjw_anymal.pt"
    return [
        "--profile",
        "newton-anymal-c-velocity",
        "--checkpoint",
        str(checkpoint),
        "--resource-root",
        str(resource_root),
        *argv,
    ]


def main(argv: list[str] | None = None) -> None:
    """Register the local Profile and run visual policy evaluation.

    Args:
        argv: Evaluation options. Uses command-line arguments when omitted.
    """
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))

    from anymal_c import register
    from embodichain.learning.rl.motion_policy_evaluation.cli import cli

    register()
    cli(example_arguments(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    main()
