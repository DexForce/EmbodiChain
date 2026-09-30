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
import json
import os
import subprocess
import sys

import pytest

from embodichain.data import constants

HF_PREFIX = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"
MIRROR = "https://mirror.example.com/embodichain_data/resolve/main/"

# Asset modules read the prefix once at import time, so the consumer URLs are
# collected in a fresh interpreter with the environment variable already set.
_COLLECT_URLS = """
import json
from embodichain.data import dataset
from embodichain.data.assets import CobotMagicArm, UnitreeGo2Locomotion

urls = {}

class Captured(Exception):
    pass

def capture(self, prefix, descriptor, path):
    urls[prefix] = list(descriptor.urls)
    raise Captured  # stop before anything is downloaded

dataset.EmbodiChainDataset.__init__ = capture
for cls in (CobotMagicArm, UnitreeGo2Locomotion):
    try:
        cls()
    except Captured:
        pass
print(json.dumps(urls))
"""


def _consumer_urls(prefix: str | None) -> dict[str, list[str]]:
    env = dict(os.environ)
    env.pop("EMBODICHAIN_DOWNLOAD_PREFIX", None)
    if prefix is not None:
        env["EMBODICHAIN_DOWNLOAD_PREFIX"] = prefix
    out = subprocess.run(
        [sys.executable, "-c", _COLLECT_URLS],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


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


@pytest.mark.no_sim
def test_download_prefix_gets_a_trailing_slash(monkeypatch, reload_constants) -> None:
    monkeypatch.setenv("EMBODICHAIN_DOWNLOAD_PREFIX", MIRROR.rstrip("/"))
    assert reload_constants().EMBODICHAIN_DOWNLOAD_PREFIX == MIRROR


@pytest.mark.no_sim
@pytest.mark.parametrize("prefix", [MIRROR, MIRROR.rstrip("/")])
def test_assets_download_from_the_configured_prefix(prefix: str) -> None:
    urls = _consumer_urls(prefix)
    assert urls["CobotMagicArm"] == [f"{MIRROR}robot_assets/CobotMagicArmV4.zip"]
    # Locomotion tries the configured source first and keeps the Hub as fallback.
    assert urls["UnitreeGo2Locomotion"] == [
        f"{MIRROR}robot_assets/UnitreeGo2Locomotion.zip",
        f"{HF_PREFIX}robot_assets/UnitreeGo2Locomotion.zip",
    ]


@pytest.mark.no_sim
def test_locomotion_does_not_repeat_the_hub_url() -> None:
    urls = _consumer_urls(HF_PREFIX)
    assert urls["UnitreeGo2Locomotion"] == [
        f"{HF_PREFIX}robot_assets/UnitreeGo2Locomotion.zip"
    ]
