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

__all__ = ["select_default_renderer"]


def select_default_renderer(gpu_id: int = 0) -> str:
    """Select FastRT by default, honoring an explicit global renderer.

    Args:
        gpu_id: Retained for compatibility. Selection is independent of the
            platform and GPU, so this value is not queried.

    Returns:
        The explicit global renderer, or ``"fast-rt"`` when it is ``"auto"``.
    """
    from embodichain.lab.sim import cfg

    if cfg.DEFAULT_RENDERER != "auto":
        return cfg.DEFAULT_RENDERER
    return "fast-rt"
