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

"""Coverage keeps actual physical and visual conditions separate from motion geometry."""

from __future__ import annotations
from dataclasses import replace
from .test_persistence import episode
from embodichain.lab.expansion.persistence import ArtifactEpisodeSink
from embodichain.lab.expansion.reporting import build_expansion_report


def test_same_motion_keeps_distinct_physics_and_source_counts(tmp_path):
    sink = ArtifactEpisodeSink(tmp_path)
    descriptor = {
        "geometry": [[[0.0], [0.5], [1.0]]],
        "timing": [[0.0, 0.1, 0.2]],
        "phase_ids": [["whole", "free"]],
    }
    for index, (mass, scale) in enumerate(((1.0, 1.0), (2.0, 2.0), (1.0, 1.4))):
        data = episode()
        identity = replace(data.identity, candidate_id=f"candidate-{index}")
        metadata = {
            "recipe": "nominal",
            "expansion_recipe": {"physics": {"mass_scale": {"cube": scale}}},
            "physics_parameters": {"cube.mass": mass, "arm.joint_stiffness": [0.0]},
            "coverage_descriptor": descriptor,
            "source_episode_uuids": ["source-a", "source-b"],
        }
        sink.submit(
            replace(
                data,
                identity=identity,
                episode_id=f"episode-{index}",
                commit_id=f"commit-{index}",
                metadata=metadata,
            ),
            submission_id=0,
        )
    report = build_expansion_report(
        tmp_path,
        receipts=sink.inspect(),
        variants=({"profile_id": "blue", "profile_revision": "v1"},),
    )
    assert report["confirmed_physical_episodes"] == 3
    assert report["geometry"]["buckets"] == 1
    assert report["physics"]["condition_fingerprints"] == 2
    assert report["physics"]["requested_conditions"] == 3
    assert report["visual"]["physical_success_credit"] == 0
    assert report["sources"]["ancestor_groups"] == {"source-a": 3, "source-b": 3}
