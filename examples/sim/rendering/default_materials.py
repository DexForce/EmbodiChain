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

"""Export default material previews through the production preview-material CLI.

This example retains its headless default. Use ``embodichain preview-material``
for a native window, or pass ``--no-headless`` here. Use ``--all --compatible-only``
when intentionally exporting only the presets supported by Hybrid or FastRT.
"""

from __future__ import annotations

import sys

from embodichain.lab.scripts.preview_material import cli

if __name__ == "__main__":
    cli(["--headless", *sys.argv[1:]])
