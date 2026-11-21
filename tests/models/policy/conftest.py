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

"""Real finalized SDK recordings shared by ACT integration tests."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def act_data(tmp_path: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "dataset"
    names = ["position", "velocity"]
    schema = {
        "observation": {"names": names, "dtype": "float32"},
        "action": {"names": names, "semantics": "absolute", "step_dt": 1 / 30},
    }
    writer = LeRobotDataset.create(
        repo_id="local/act",
        root=root,
        fps=30,
        use_videos=False,
        features={
            "observation.state": {"dtype": "float32", "shape": (2,), "names": names},
            "action": {"dtype": "float32", "shape": (2,), "names": names},
            "observation.images.camera": {
                "dtype": "image",
                "shape": (32, 32, 3),
                "names": ["height", "width", "channel"],
            },
        },
    )
    for episode in range(2):
        for frame in range(3):
            writer.add_frame(
                {
                    "observation.state": np.array([episode, frame], dtype=np.float32),
                    "action": np.array([frame, episode], dtype=np.float32),
                    "observation.images.camera": np.full(
                        (32, 32, 3), episode * 10 + frame, dtype=np.uint8
                    ),
                    "task": "push",
                }
            )
        writer.save_episode()
    writer.finalize()
    return root, schema
