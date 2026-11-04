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

"""Configuration-owned peg insertion task registration."""

from __future__ import annotations
from embodichain.lab.gym.envs import EmbodiedEnv
from embodichain.lab.gym.utils.registration import register_env

__all__: list[str] = []

register_env(
    "BlindPegInsertionRL-v1",
    max_episode_steps=250,
    supports_rl=True,
    supports_handwritten_demo=False,
    scene_preparation={
        "func": "embodichain_tasks.manipulation.assembly.preparation:prepare_assembly_scene",
        "params": {"kind": "peg"},
    },
    resource_export={
        "func": "embodichain_tasks.manipulation.assembly.resources:export_assembly_resources"
    },
)(EmbodiedEnv)
