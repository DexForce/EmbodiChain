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

"""Execute a Python test subprocess with periodic, GIL-held stack snapshots."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import math
from pathlib import Path
import runpy
import sys
import threading
import traceback

__all__ = ["main"]


def _watch(stopped: threading.Event, interval: float) -> None:
    # The native faulthandler timer segfaults during stack traversal in the
    # simulation runtime. Collect owned Python frame objects under the GIL.
    while not stopped.wait(interval):
        print("Subprocess watchdog: Python stack snapshot", file=sys.stderr)
        names = {thread.ident: thread.name for thread in threading.enumerate()}
        for identifier, frame in sys._current_frames().items():
            if identifier == threading.get_ident():
                continue
            print(
                f"Thread {identifier} ({names.get(identifier, 'native thread')}):",
                file=sys.stderr,
            )
            traceback.print_stack(frame, file=sys.stderr)
        # A snapshot must not retain the target's locals until the next tick.
        del frame
        sys.stderr.flush()


def main(argv: Sequence[str] | None = None) -> None:
    """Run one script and stop diagnostics when it exits or raises.

    Args:
        argv: Optional watchdog options followed by the script and its arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=150.0)
    parser.add_argument("script", type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be finite and positive")

    stopped = threading.Event()
    monitor = threading.Thread(
        target=_watch,
        args=(stopped, args.interval),
        name="test-subprocess-watchdog",
        daemon=True,
    )
    previous = sys.argv
    sys.argv = [str(args.script), *args.arguments]
    monitor.start()
    try:
        runpy.run_path(str(args.script), run_name="__main__")
    finally:
        stopped.set()
        monitor.join()
        sys.argv = previous


if __name__ == "__main__":
    main()
