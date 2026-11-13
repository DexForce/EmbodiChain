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

"""Model-specific checkpoint downloads share authentication and proxy handling."""

from __future__ import annotations

import os
from unittest.mock import Mock

import pytest

from embodichain.data.assets import (
    download_neural_ik_checkpoint,
    download_neural_planner_checkpoint,
)
from embodichain.data.assets import _checkpoint_download


@pytest.mark.parametrize(
    "download, repository, filename",
    [
        (
            download_neural_planner_checkpoint,
            "DexForceAI/neural_motion_generator",
            "franka/franka.pt",
        ),
        (download_neural_ik_checkpoint, "DexForceAI/neural_ik_solver", "franka.pt"),
    ],
)
@pytest.mark.parametrize("token", [None, "explicit-test-token"])
def test_checkpoint_defaults_authentication_and_proxy(
    monkeypatch: pytest.MonkeyPatch,
    download,
    repository: str,
    filename: str,
    token: str | None,
) -> None:
    hub = Mock(return_value="/cache/model.pt")
    monkeypatch.setattr(_checkpoint_download, "hf_hub_download", hub)
    monkeypatch.setenv("HF_TOKEN", "environment-test-token")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    monkeypatch.delenv("https_proxy", raising=False)

    assert download(token=token) == "/cache/model.pt"

    hub.assert_called_once_with(
        repo_id=repository,
        filename=filename,
        token=token or "environment-test-token",
        endpoint="https://huggingface.co",
    )
    assert os.environ["https_proxy"] == "http://proxy.example:8080"


@pytest.mark.parametrize(
    "download", [download_neural_planner_checkpoint, download_neural_ik_checkpoint]
)
def test_checkpoint_download_preserves_failure_cause(
    monkeypatch: pytest.MonkeyPatch, download
) -> None:
    failure = OSError("test transport unavailable")
    monkeypatch.setattr(
        _checkpoint_download, "hf_hub_download", Mock(side_effect=failure)
    )

    with pytest.raises(RuntimeError, match="HF_TOKEN") as error:
        download()

    assert error.value.__cause__ is failure
