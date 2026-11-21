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

"""Start the single-GPU wall clock before Xvfb, simulator and training imports."""

from __future__ import annotations

import os
import sys
import time


def main() -> None:
    """Replace the launcher with Xvfb and forward its monotonic origin."""
    origin = time.monotonic()
    if "--clock-origin" in sys.argv or any(
        arg.startswith("--clock-origin=") for arg in sys.argv
    ):
        raise SystemExit("The cold launcher supplies the clock origin")
    os.execvp(
        "xvfb-run",
        [
            "xvfb-run",
            "-a",
            sys.executable,
            "-Bu",
            "-m",
            "examples.learning.act.grouped",
            *sys.argv[1:],
            "--clock-origin",
            str(origin),
        ],
    )


if __name__ == "__main__":
    main()
