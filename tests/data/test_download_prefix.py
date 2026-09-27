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

"""Tests for overriding the asset download prefix."""

from __future__ import annotations

import importlib

import pytest

from embodichain.data import constants

HF_PREFIX = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"


@pytest.fixture
def reload_constants(monkeypatch):
    def _reload():
        return importlib.reload(constants)

    yield _reload
    monkeypatch.undo()
    importlib.reload(constants)


@pytest.mark.no_sim
def test_download_prefix_defaults_to_mirror(monkeypatch, reload_constants) -> None:
    monkeypatch.delenv("EMBODICHAIN_DOWNLOAD_PREFIX", raising=False)
    assert reload_constants().EMBODICHAIN_DOWNLOAD_PREFIX == (
        "https://hf-mirror.com/datasets/DexForceAI/embodichain_data/resolve/main/"
    )


@pytest.mark.no_sim
def test_download_prefix_can_be_overridden(monkeypatch, reload_constants) -> None:
    monkeypatch.setenv("EMBODICHAIN_DOWNLOAD_PREFIX", HF_PREFIX)
    assert reload_constants().EMBODICHAIN_DOWNLOAD_PREFIX == HF_PREFIX
