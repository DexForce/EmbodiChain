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

"""Compatibility imports for image and simulation contact kernels.

The contact alias loads the simulation sensor package only when requested."""

from __future__ import annotations


from typing import Any

from embodichain.compute.image._warp.tiling import reshape_tiled_image

__all__ = ["reshape_tiled_image", "scatter_contact_data"]


def __getattr__(name: str) -> Any:
    if name == "scatter_contact_data":
        from embodichain.lab.sim.sensors._warp.contact import scatter_contact_data

        return scatter_contact_data
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
