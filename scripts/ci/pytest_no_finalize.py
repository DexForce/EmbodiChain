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

"""Run GPU pytest to completion without DexSim's unsafe native finalizers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

import os
import sys

import pytest

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> NoReturn:
    """Return pytest's result after all hooks without unloading native libraries.

    Args:
        argv: Pytest arguments, or ``sys.argv[1:]`` when omitted.
    """
    exit_code = pytest.main(list(sys.argv[1:] if argv is None else argv))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(int(exit_code))


if __name__ == "__main__":
    main()
