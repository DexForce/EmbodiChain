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

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from unittest.mock import Mock

from embodichain.lab.sim.utility.demo_utils import (
    add_demo_args,
    format_tensor,
    setup_print_options,
    shutdown_sim,
)


def test_add_demo_args_adds_expected_flags():
    parser = argparse.ArgumentParser()
    parser = add_demo_args(parser)
    args = parser.parse_args(["--headless", "--auto_play", "--record_fps", "60"])
    assert args.headless is True
    assert args.auto_play is True
    assert args.record_fps == 60
    assert args.record_steps is None
    assert args.no_vis_eef_axis is False


def test_format_tensor_rounds_and_moves_to_cpu():
    tensor = torch.tensor([1.23456789, 2.34567891])
    result = format_tensor(tensor)
    assert result == "[1.2346, 2.3457]"


def test_setup_print_options_sets_numpy_and_torch():
    setup_print_options()
    assert np.get_printoptions()["precision"] == 5
    assert np.get_printoptions()["suppress"] is True
    assert torch._tensor_str.PRINT_OPTS.precision == 5
    assert torch._tensor_str.PRINT_OPTS.sci_mode is False


def test_shutdown_sim_calls_destroy():
    sim = Mock(spec=["destroy"])
    shutdown_sim(sim)
    sim.destroy.assert_called_once()
