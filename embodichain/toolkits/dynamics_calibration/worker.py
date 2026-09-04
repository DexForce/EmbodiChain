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

"""Private child-process entry point for a calibration evaluator."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _load_evaluator(target: str) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    module_or_path, attribute = target.rsplit(":", maxsplit=1)
    source_path = Path(module_or_path)
    if source_path.is_file():
        module_name = f"_embodichain_calibration_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load evaluator module from {source_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_or_path)
    evaluator = getattr(module, attribute)
    if not callable(evaluator):
        raise TypeError(f"evaluator target {target!r} is not callable")
    return evaluator


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=True, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    """Execute one evaluator and terminate without simulator teardown hooks.

    Args:
        argv: Input and output JSON paths. Uses process arguments when omitted.

    Raises:
        SystemExit: When called directly with invalid arguments or when a test
            invocation supplies ``argv`` and the evaluator fails.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        raise SystemExit("worker expects INPUT_JSON OUTPUT_JSON")
    input_path, output_path = map(Path, arguments)
    exit_code = 0
    try:
        request = json.loads(input_path.read_text(encoding="utf-8"))
        evaluator = _load_evaluator(request["evaluator"]["target"])
        result = evaluator(request["overlay"], request["context"])
        if not isinstance(result, dict):
            raise TypeError("evaluate() must return a dictionary")
        _write_result(output_path, {"status": "ok", "result": result})
    except BaseException as error:  # noqa: BLE001 - child must serialize all failures.
        exit_code = 1
        _write_result(
            output_path,
            {
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            },
        )
    if argv is None:
        os._exit(exit_code)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()


__all__: list[str] = []
