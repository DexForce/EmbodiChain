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

"""G2 qualification rejects unavailable device selections before setup."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.benchmark.robotics.kinematic_solver import agibot_g2


@pytest.mark.no_sim
@pytest.mark.parametrize("devices", [["cuda"], ["cuda", "cuda"]])
def test_unavailable_cuda_selection_fails_before_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    devices: list[str],
) -> None:
    output = tmp_path / "qualification.md"
    monkeypatch.setattr(
        sys, "argv", ["agibot_g2", "--output", str(output), "--devices", *devices]
    )
    monkeypatch.setattr(agibot_g2.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        agibot_g2.wp, "init", lambda: pytest.fail("Must reject before Warp setup")
    )
    monkeypatch.setattr(
        agibot_g2.AgibotG2Cfg,
        "from_dict",
        lambda _: pytest.fail("Must reject before asset loading"),
    )
    with pytest.raises(SystemExit) as error:
        agibot_g2._main()
    assert error.value.code == 2
    assert "no usable device selected" in capsys.readouterr().err
    assert not output.exists()
