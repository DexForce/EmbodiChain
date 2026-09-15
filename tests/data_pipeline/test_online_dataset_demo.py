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

"""Tests for the online-dataset example configuration."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_DEMO_PATH = _REPOSITORY_ROOT / "examples/data_pipeline/online_dataset_demo.py"


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("online_dataset_demo", _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_engine_configures_a_named_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker receives a renderer name accepted by SimulationManagerCfg."""
    demo = _load_demo_module()
    captured: dict[str, object] = {}

    class FakeEngine:
        def __init__(self, cfg) -> None:
            captured["cfg"] = cfg

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(
        "embodichain.utils.utility.load_config",
        lambda _path: {"id": "SimpleTask-v1"},
    )
    monkeypatch.setattr(demo, "OnlineDataEngine", FakeEngine)

    demo._build_engine(SimpleNamespace(device="cpu"))

    assert captured["cfg"].gym_config["renderer"] == "hybrid"
    assert captured["started"] is True


def test_main_stops_the_online_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The demo releases its worker process after every sampling mode completes."""
    demo = _load_demo_module()

    class FakeEngine:
        def __init__(self) -> None:
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

    engine = FakeEngine()
    monkeypatch.setattr(demo, "_parse_args", lambda: SimpleNamespace())
    monkeypatch.setattr(demo, "_build_engine", lambda _args: engine)
    monkeypatch.setattr(demo, "_demo_item_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(demo, "_demo_batch_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(demo, "_demo_uniform_dynamic", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(demo, "_demo_gmm_dynamic", lambda *_args, **_kwargs: None)

    demo.main()

    assert engine.stopped
