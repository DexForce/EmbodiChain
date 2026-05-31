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

from huggingface_hub import hf_hub_download


def download_neural_ik_checkpoint(
    repo_id: str = "dexforce/neural_ik_solver",
    filename: str = "franka.pt",
) -> str:
    """Download a neural IK solver checkpoint from HuggingFace.

    Returns:
        str: Local path to the downloaded checkpoint file.
    """
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
    )
