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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from embodichain.lab.sim.demo_base import DemoBase


def test_demo_base_runs_setup_run_cleanup():
    class SimpleDemo(DemoBase):
        def setup(self):
            self.sim = Mock(spec=["destroy"])

        def run(self):
            self.ran = True

    demo = SimpleDemo(SimpleNamespace())
    demo.main()
    assert demo.ran is True
    demo.sim.destroy.assert_called_once()


def test_demo_base_cleanup_runs_even_if_run_raises():
    class BrokenDemo(DemoBase):
        def setup(self):
            self.sim = Mock(spec=["destroy"])

        def run(self):
            raise RuntimeError("boom")

    demo = BrokenDemo(SimpleNamespace())
    with pytest.raises(RuntimeError, match="boom"):
        demo.main()
    demo.sim.destroy.assert_called_once()
