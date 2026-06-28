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

"""Shared helpers for atomic-action tutorial scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import torch

from embodichain.data import get_data_path
from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.cfg import MarkerCfg, RobotCfg
from embodichain.lab.sim.robots import URRobotCfg

RECORD_WIDTH = 640
RECORD_HEIGHT = 480
VIEWER_WIDTH = 1600
VIEWER_HEIGHT = 900
AUTO_PLAY_RECORD_FPS = 20
AUTO_PLAY_RECORD_MAX_MEMORY = 2048
DEFAULT_AUTO_PLAY_LOOK_AT = (
    (-1.25, -1.15, 0.95),
    (-0.25, -0.02, 0.25),
    (0.0, 0.0, 1.0),
)
DEFAULT_AXIS_LEN = 0.06
DEFAULT_AXIS_SIZE = 0.003

GRIPPER_URDF_PATH = "DH_PGI_140_80/DH_PGI_140_80.urdf"
GRIPPER_HAND_JOINT_PATTERN = "gripper_finger1_joint_1"
GRIPPER_TCP_Z = 0.15


def get_tutorial_window_size(args: argparse.Namespace) -> tuple[int, int]:
    """Return the viewer window size used by atomic-action tutorials."""
    return VIEWER_WIDTH, VIEWER_HEIGHT


def start_auto_play_recording(
    sim: SimulationManager,
    args: argparse.Namespace,
    video_prefix: str,
    look_at: tuple[
        Sequence[float],
        Sequence[float],
        Sequence[float],
    ] = DEFAULT_AUTO_PLAY_LOOK_AT,
) -> bool:
    """Start recording for ``--auto_play`` tutorial runs."""
    if not getattr(args, "auto_play", False):
        return False

    original_width = sim.sim_config.width
    original_height = sim.sim_config.height
    try:
        sim.sim_config.width = RECORD_WIDTH
        sim.sim_config.height = RECORD_HEIGHT
        if not sim.start_window_record(
            fps=AUTO_PLAY_RECORD_FPS,
            max_memory=AUTO_PLAY_RECORD_MAX_MEMORY,
            video_prefix=video_prefix,
            look_at=look_at,
            use_sim_time=True,
        ):
            raise RuntimeError("Failed to start auto_play recording.")
    finally:
        sim.sim_config.width = original_width
        sim.sim_config.height = original_height
    return True


def stop_auto_play_recording(
    sim: SimulationManager,
    recording_started: bool,
) -> None:
    """Stop recording and wait until the mp4 has been written."""
    if not recording_started:
        return

    if sim.is_window_recording():
        sim.stop_window_record()
    sim.wait_window_record_saves()


def draw_axis_marker(
    sim: SimulationManager,
    name: str,
    xpos: torch.Tensor,
    axis_len: float = DEFAULT_AXIS_LEN,
    axis_size: float = DEFAULT_AXIS_SIZE,
) -> None:
    """Draw a named coordinate-frame marker for a semantic tutorial target."""
    sim.draw_marker(
        cfg=MarkerCfg(
            name=name,
            marker_type="axis",
            axis_xpos=xpos,
            axis_size=axis_size,
            axis_len=axis_len,
        )
    )


def create_ur5_gripper_robot_cfg(
    init_pos: Sequence[float] = (0.0, 0.0, 0.0),
) -> RobotCfg:
    """Build a UR5 arm + DH_PGI_140_80 gripper robot configuration.

    The arm is taken from :class:`~embodichain.lab.sim.robots.ur_robot.URRobotCfg`
    so the URDF, joint names and drive defaults match the canonical UR family
    config. The gripper and tool-center-point offset are added on top to match
    the existing atomic-action tutorial setups.

    .. attention::
        :class:`~embodichain.lab.sim.cfg.URDFCfg` defaults to upper-casing joint
        names during multi-component assembly. The override dict passed to
        :meth:`URRobotCfg.from_dict` sets
        ``urdf_cfg.name_case = {"joint": "lower", "link": "lower"}`` so the
        assembled robot keeps the source URDF's lowercase joint names
        (``joint1``..``joint6`` and ``gripper_finger1_joint_1``), matching the
        control parts produced by :class:`~embodichain.lab.sim.robots.ur_robot.URRobotCfg`.

    Args:
        init_pos: Initial root position of the robot in the arena.

    Returns:
        A fully populated :class:`~embodichain.lab.sim.cfg.RobotCfg`.
    """
    return URRobotCfg.from_dict(
        {
            "robot_type": "ur5",
            "uid": "UR5",
            "urdf_cfg": {
                "components": [
                    {
                        "component_type": "hand",
                        "urdf_path": GRIPPER_URDF_PATH,
                    }
                ],
            },
            "control_parts": {
                "hand": [GRIPPER_HAND_JOINT_PATTERN],
            },
            "drive_pros": {
                "stiffness": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e3,
                },
                "damping": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e2,
                },
                "max_effort": {
                    GRIPPER_HAND_JOINT_PATTERN: 1e4,
                },
            },
            "solver_cfg": {
                "arm": {
                    "tcp": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, GRIPPER_TCP_Z],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            },
            "init_qpos": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0, 0.0],
            "init_pos": init_pos,
        }
    )


__all__ = [
    "DEFAULT_AUTO_PLAY_LOOK_AT",
    "DEFAULT_AXIS_LEN",
    "DEFAULT_AXIS_SIZE",
    "GRIPPER_HAND_JOINT_PATTERN",
    "GRIPPER_TCP_Z",
    "GRIPPER_URDF_PATH",
    "create_ur5_gripper_robot_cfg",
    "get_tutorial_window_size",
    "start_auto_play_recording",
    "stop_auto_play_recording",
    "draw_axis_marker",
]
