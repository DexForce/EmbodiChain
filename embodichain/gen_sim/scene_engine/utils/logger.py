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

import logging

_LOGGER = logging.getLogger("embodichain.scene_engine")
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [EmbodiChain Scene Engine] %(message)s")
    )
    _LOGGER.addHandler(handler)
    _LOGGER.propagate = False
_LOGGER.setLevel(logging.INFO)


def log_stage_start(stage_name: str) -> None:
    _LOGGER.info("Starting %s", stage_name)


def log_stage_end(stage_name: str) -> None:
    _LOGGER.info("Completed %s", stage_name)
