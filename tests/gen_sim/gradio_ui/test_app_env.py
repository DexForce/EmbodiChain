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

import sys
from pathlib import Path

import pytest

GRADIO_UI_ROOT = (
    Path(__file__).resolve().parents[3] / "embodichain" / "gen_sim" / "gradio_ui"
)
sys.path.insert(0, str(GRADIO_UI_ROOT))

import app_env  # noqa: E402


def test_local_gradio_server_does_not_require_authentication() -> None:
    assert app_env.get_gradio_auth("127.0.0.1", "", "") is None


def test_remote_gradio_server_requires_authentication() -> None:
    with pytest.raises(ValueError, match="requires Gradio authentication"):
        app_env.get_gradio_auth("0.0.0.0", "", "")


def test_remote_gradio_server_accepts_complete_credentials() -> None:
    assert app_env.get_gradio_auth("0.0.0.0", "workspace", "secret") == (
        "workspace",
        "secret",
    )


def test_partial_gradio_credentials_are_rejected() -> None:
    with pytest.raises(ValueError, match="Set both"):
        app_env.get_gradio_auth("127.0.0.1", "workspace", "")


def test_gradio_file_access_excludes_repository_and_blocks_dotenv(
    tmp_path: Path,
) -> None:
    generated_root = tmp_path / "generated"
    static_root = tmp_path / "static"
    external_env = tmp_path / "deployment.env"

    allowed = app_env.build_gradio_allowed_paths(generated_root, static_root)
    blocked = app_env.build_gradio_blocked_paths(external_env)

    assert str(app_env.EMBODICHAIN_ROOT.resolve()) not in allowed
    assert str(generated_root.resolve()) in allowed
    assert str(static_root.resolve()) in allowed
    assert str(external_env.resolve()) in blocked
    assert str((app_env.EMBODICHAIN_ROOT / ".git").resolve()) in blocked


def test_repository_cannot_be_configured_as_artifact_root() -> None:
    with pytest.raises(ValueError, match="dedicated artifact directory"):
        app_env.validate_gradio_artifact_root(app_env.EMBODICHAIN_ROOT)
