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

"""Tests for atomic-action tutorial helpers."""

from __future__ import annotations

from argparse import Namespace

from scripts.tutorials.atomic_action.tutorial_utils import (
    should_wait_for_tutorial_input,
)


def test_should_wait_for_tutorial_input_is_disabled_for_headless_modes() -> None:
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=True,
                diagnose_plan=False,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=True,
                headless_play=False,
            )
        )
        is False
    )
    assert (
        should_wait_for_tutorial_input(
            Namespace(
                auto_play=False,
                headless=False,
                diagnose_plan=False,
                headless_play=True,
            )
        )
        is False
    )
