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

"""Coverage for train-rl EnvProfiler CLI flags."""

from __future__ import annotations

import pytest

from embodichain.learning.rl.train import (
    _event_params,
    _resolve_profile_output,
    parse_args,
    train_from_config,
)


def test_parse_args_profile_flags():
    args = parse_args(
        [
            "--config",
            "dummy.yaml",
            "--profile",
            "--profile_output",
            "outputs/rl_profile.json",
        ]
    )
    assert args.profile is True
    assert args.profile_output == "outputs/rl_profile.json"

    defaults = parse_args(["--config", "dummy.yaml"])
    assert defaults.profile is False
    assert defaults.profile_output is None


def test_resolve_profile_output_disambiguates_ranks():
    assert _resolve_profile_output("out/prof.json", rank=0, world_size=1) == (
        "out/prof.json"
    )
    assert _resolve_profile_output("out/prof.json", rank=1, world_size=2) == (
        "out/prof_rank1.json"
    )


def test_learning_env_rejects_profile(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "trainer:\n  learning_env:\n    name: PointMassRL\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="learning_env"):
        train_from_config(str(config_path), profile=True)

    with pytest.raises(ValueError, match="--profile_output requires --profile"):
        train_from_config(str(config_path), profile_output="prof.json")


def test_camera_recording_defaults_to_the_run_directory(tmp_path):
    params = _event_params(
        {"func": "record_camera_data_async", "params": {"name": "main"}},
        run_base=tmp_path / "run",
        phase="eval",
    )

    assert params["save_path"] == str(tmp_path / "run" / "videos" / "eval")
