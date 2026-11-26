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

"""Verify link targets in the README-derived Sphinx introduction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/scripts"))
from sync_readme import convert_readme_to_rst, postprocess_rst


def test_introduction_resolves_links_for_documentation_site() -> None:
    pytest.importorskip("pypandoc")
    readme = """\
# EmbodiChain

[![Badge](https://example.com/badge.svg)](https://example.com/)

[Installation](https://dexforce.github.io/EmbodiChain/main/quick_start/install.html#verify-installation)

| Workflow |
| --- |
| [Runnable examples][examples] |

[examples]: examples/README.md

[External guide](https://example.com/guide) and [local notes](#notes).

![Teaser](assets/imgs/teaser.jpg)
"""
    rst = postprocess_rst(convert_readme_to_rst(readme), readme)

    assert "<quick_start/install.html#verify-installation>" in rst
    assert "<https://dexforce.github.io/EmbodiChain/main/" not in rst
    assert (
        "<https://github.com/DexForce/EmbodiChain/blob/main/examples/README.md>" in rst
    )
    assert "<https://example.com/guide>" in rst
    assert "<#notes>" in rst
    assert "badge.svg" not in rst
    assert ".. image:: ../../assets/imgs/teaser.jpg" in rst
