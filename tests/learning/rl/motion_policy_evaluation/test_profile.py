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

import torch

from embodichain.learning.rl.motion_policy_evaluation import (
    MotionProfile,
    MotionProfileRequest,
    build_motion_profile,
    register_motion_profile,
)


def test_profile_provider_receives_checkpoint_and_training_configs(tmp_path):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    train = tmp_path / "train.yaml"
    train.write_text("trainer: {}\n", encoding="utf-8")
    requests = []

    def provider(request):
        requests.append(request)
        return MotionProfile(
            profile_id="test-profile-provider",
            checkpoint=request.checkpoint,
            policy_spec={"schema_version": 1},
        )

    register_motion_profile("test-profile-provider", provider)
    request = MotionProfileRequest(
        checkpoint=checkpoint,
        device=torch.device("cpu"),
        configs={"train": train},
    )

    profile = build_motion_profile("test-profile-provider", request)

    assert profile.checkpoint == checkpoint
    assert requests[0].configs["train"] == train
