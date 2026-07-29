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

import argparse

from embodichain.lab.visualization import (
    add_viser_args_to_parser,
    visualization_cfg_from_args,
)


def test_viser_flag_builds_visualization_configuration() -> None:
    parser = argparse.ArgumentParser()
    add_viser_args_to_parser(parser)

    args = parser.parse_args(
        [
            "--viser",
            "--viser-host",
            "0.0.0.0",
            "--viser-port",
            "9000",
            "--viser-fps",
            "12.5",
            "--viser-image-fps",
            "1.5",
            "--viser-soft-body-fps",
            "4.0",
            "--viser-env-ids",
            "1",
            "3",
        ]
    )
    visualization_cfg = visualization_cfg_from_args(args)

    assert visualization_cfg.backend == "viser"
    assert visualization_cfg.scene_fps == 12.5
    assert visualization_cfg.sensor_image_fps == 1.5
    assert visualization_cfg.soft_body_fps == 4.0
    assert visualization_cfg.env_ids == [1, 3]
    assert visualization_cfg.viser_server.host == "0.0.0.0"
    assert visualization_cfg.viser_server.port == 9000


def test_viser_is_disabled_by_default() -> None:
    parser = argparse.ArgumentParser()
    add_viser_args_to_parser(parser)

    visualization_cfg = visualization_cfg_from_args(parser.parse_args([]))

    assert visualization_cfg.backend == "none"
