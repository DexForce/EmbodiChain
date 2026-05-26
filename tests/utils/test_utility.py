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

import json

import pytest

from embodichain.utils.utility import load_config, save_config


@pytest.fixture
def sample_config() -> dict:
    return {
        "id": "TestEnv-v1",
        "num_envs": 2,
        "env": {"events": {}},
        "robot": {"uid": "robot_1"},
    }


class TestLoadConfig:
    def test_load_json(self, tmp_path, sample_config):
        path = tmp_path / "config.json"
        path.write_text(json.dumps(sample_config), encoding="utf-8")

        loaded = load_config(path)

        assert loaded == sample_config

    def test_load_yaml(self, tmp_path, sample_config):
        path = tmp_path / "config.yaml"
        save_config(path, sample_config)

        loaded = load_config(path)

        assert loaded == sample_config

    def test_load_yml_extension(self, tmp_path, sample_config):
        path = tmp_path / "config.yml"
        save_config(path, sample_config)

        loaded = load_config(path)

        assert loaded == sample_config

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text("id = 'TestEnv-v1'", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(path)

    def test_yaml_root_must_be_mapping(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- item\n", encoding="utf-8")

        with pytest.raises(TypeError, match="Expected mapping"):
            load_config(path)

    def test_round_trip_yaml(self, tmp_path, sample_config):
        path = tmp_path / "config.yaml"
        save_config(path, sample_config)

        assert load_config(path) == sample_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
