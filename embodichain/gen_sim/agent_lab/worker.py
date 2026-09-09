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

"""Run an ordinary Python solve(lab) function in a disposable simulator process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
from pathlib import Path
import signal
import sys
import traceback

from .catalog import write_json

__all__ = ["main"]


def main() -> int:
    """Execute the solver and preserve evidence on both normal and failed exits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"execution_completed": False, "task_success": None, "stage": "setup"}
    lab = None

    def interrupted(signum: int, frame: object) -> None:
        raise InterruptedError(f"Execution interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        from .runtime import Lab, LabCfg
        import dexsim
        import torch

        write_json(
            args.output / "environment.json",
            {
                "python": sys.version,
                "executable": sys.executable,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "dexsim_module": dexsim.__file__,
            },
        )

        sys.path.insert(0, str(args.script.resolve().parent))
        os.environ["GENSIM_LAB_OUTPUT"] = str(args.output.resolve())
        if args.config is None:
            report["stage"] = "script"
            try:
                runpy.run_path(str(args.script), run_name="__main__")
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    raise
            report["execution_completed"] = True
            report["stage"] = "complete"
            return 0
        spec = importlib.util.spec_from_file_location("lab_solution", args.script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cfg = LabCfg(**json.loads(args.config.read_text()))
        # Bind before construction so partial initialization can still be cleaned up.
        lab = Lab.__new__(Lab)
        lab.__init__(cfg, args.output)
        report["stage"] = "solve"
        report["solver_result"] = module.solve(lab)
        report["execution_completed"] = True
        report["stage"] = "complete"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        traceback.print_exc()
    finally:
        if lab is not None and hasattr(lab, "sim"):
            try:
                lab.close()
            except Exception:
                report["recording_or_cleanup_error"] = traceback.format_exc()
                report["execution_completed"] = False
                traceback.print_exc()
        video = args.output / "video.mp4"
        report["video"] = str(video) if video.is_file() else None
        write_json(args.output / "result.json", report)
    return 0 if report["execution_completed"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Match the existing GenSim subprocess boundary for native CUDA teardown.
    os._exit(code)
