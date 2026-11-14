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

"""Physical standalone hand qualification through the public robot catalog."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


def _run_hands() -> None:
    from pathlib import Path
    import sys
    import json
    import torch
    from embodichain.robot_registry import resolve_robot_cfg
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.cfg import RenderCfg

    models = [
        ("FrankaPandaHand", None),
        ("DHPGI14080", None),
        ("CobotMagicV100Gripper", None),
        ("PikaGripper", None),
        ("TianjiMarvinGripper", "left"),
        ("TianjiMarvinGripper", "right"),
        ("RobotiqArg2F85", None),
        ("RobotiqArg2F140", None),
    ]
    sim = SimulationManager(
        SimulationManagerCfg(
            headless=True,
            device="cpu",
            render_cfg=RenderCfg(renderer="no-render"),
            physics_dt=0.01,
        )
    )
    try:
        robots = []
        for i, (name, variant) in enumerate(models):
            cfg = resolve_robot_cfg(
                name,
                {
                    "uid": name + (variant or ""),
                    "init_pos": [i * 0.5, 0.0, 0.6],
                    **({"variant": variant} if variant else {}),
                },
            )
            robot = sim.add_robot(cfg)
            robots.append((name, variant, robot, cfg))
        sim.prepare()
        for _, _, robot, cfg in robots:
            native = cfg.control_parts["eef"]
            robot.set_qpos(
                torch.tensor(
                    [[cfg.joint_commands["open"][n] for n in native]],
                    dtype=torch.float32,
                ),
                name="eef",
            )
        sim.update(step=200)
        opened = [robot.get_qpos(name="eef").clone() for _, _, robot, _ in robots]
        for _, _, robot, cfg in robots:
            robot.set_qpos(
                torch.tensor(
                    [
                        [
                            cfg.joint_commands["grasp"][n]
                            for n in cfg.control_parts["eef"]
                        ]
                    ],
                    dtype=torch.float32,
                ),
                name="eef",
            )
        sim.update(step=200)
        results = []
        for (name, variant, robot, cfg), before in zip(robots, opened):
            after = robot.get_qpos(name="eef")
            assert torch.isfinite(after).all(), name
            assert (after - before).abs().max() > 0.001, (name, before, after)
            results.append(
                {
                    "name": name,
                    "variant": variant,
                    "opened": before.tolist(),
                    "closed": after.tolist(),
                    "max_motion": (after - before).abs().max().item(),
                }
            )
        Path(sys.argv[1]).write_text(json.dumps(results))
    finally:
        sim.destroy()
        SimulationManager.flush_cleanup_queue()


@pytest.mark.requires_sim
@pytest.mark.subprocess_sim
@pytest.mark.slow
def test_native_parallel_jaw_configs_follow_independent_open_close_commands(
    tmp_path: Path,
) -> None:
    """Every supported hand builds and follows native commands in one small world."""
    report = tmp_path / "hands.json"
    code = "import runpy,sys; module=runpy.run_path(sys.argv[1]); sys.argv=sys.argv[1:]; module['_run_hands']()"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(report)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # A native early-exit with status zero must not masquerade as a passed test.
    assert report.is_file(), result.stdout + result.stderr
    hands = json.loads(report.read_text())
    assert len(hands) == 8
    assert all(item["max_motion"] > 0.001 for item in hands)
