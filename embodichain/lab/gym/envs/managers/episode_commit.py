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

"""Acknowledgements for synchronous demonstration persistence."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DemoCommitReceipt"]


@dataclass(frozen=True)
class DemoCommitReceipt:
    """One episode whose primary data and named sidecars finished saving.

    Idempotency is scoped to one recorder process, not crash recovery. An
    absent modality is not confirmed by this receipt.
    """

    env_id: int
    episode_id: str
    commit_id: str
    dataset_episode_index: int
    confirmed_modalities: tuple[str, ...]
