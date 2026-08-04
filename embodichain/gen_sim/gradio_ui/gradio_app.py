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

"""Application entry point.

The UI callbacks and pipeline services are intentionally kept out of this
module; this file only validates configuration and launches the application.
"""

from __future__ import annotations

import signal

from app_config import (
    ASSETS_DIR,
    DEBUG_ENGINE_ROOT,
    DEFAULT_CONCURRENCY_LIMIT,
)
from app_env import EMBODICHAIN_ROOT, SERVER_NAME, SERVER_PORT
from app_processes import force_stop_all_child_processes
from app_services import build_demo

__all__ = ["main"]


def _stop_child_processes() -> None:
    """Force-stop UI-owned subprocesses without masking app shutdown."""
    try:
        force_stop_all_child_processes()
    except Exception:
        # Shutdown must not be blocked by an already-exited preview process.
        pass


def _handle_shutdown_signal(signum: int, _frame: object) -> None:
    """Terminate UI subprocesses before leaving the Gradio process."""
    _stop_child_processes()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _install_shutdown_handlers() -> None:
    """Install cleanup-aware handlers for the normal Gradio stop signals."""
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)


def main() -> None:
    if not EMBODICHAIN_ROOT.is_dir():
        raise FileNotFoundError(f"EmbodiChain root not found: {EMBODICHAIN_ROOT}")
    demo = build_demo()
    demo.queue(default_concurrency_limit=DEFAULT_CONCURRENCY_LIMIT)
    _install_shutdown_handlers()
    try:
        demo.launch(
            server_name=SERVER_NAME,
            server_port=SERVER_PORT,
            allowed_paths=[
                str(EMBODICHAIN_ROOT),
                str(ASSETS_DIR),
                str(DEBUG_ENGINE_ROOT),
            ],
        )
    finally:
        _stop_child_processes()


if __name__ == "__main__":
    main()
