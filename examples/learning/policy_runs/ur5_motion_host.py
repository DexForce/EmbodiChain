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

"""Launch the dedicated World for the UR5 motion API example."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import os
from pathlib import Path
from secrets import token_bytes
import signal
from uuid import uuid4

import dexsim
from dexsim.runtime import SessionContext
from dexsim.runtime.command_queue import CommandQueue
from dexsim.runtime.config import normalize_config
from dexsim.runtime.host import _publish_json, _session_info, _world_config
from dexsim.runtime.protocol import record_path

from embodichain.lab.sim.dexsim_session._motion_jobs import executors
from ._ur5_motion import Ur5Motion
from .ur5_pick_place import _Episode

__all__: list[str] = []

_DEPLOYMENT = (
    Path(__file__).resolve().parents[2]
    / "expansion/asset_generalization/configs/cube_platform_pick_place/task.ur5.yaml"
)


def main() -> None:
    """Create a fixture World and publish the explicit writer contract."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    sessions = args.directory.resolve()
    sessions.mkdir(mode=0o700, parents=True, exist_ok=False)
    output = args.output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = normalize_config(
        dict(backend="dexsim", device="cpu", renderer="fast-rt", headless=True, dt=0.01)
    )
    world = dexsim.World(_world_config(config))
    world.set_manual_update(True)
    world.enable_physics(True)
    session_id = "dsws_" + uuid4().hex
    record = record_path(session_id, sessions)
    context = SessionContext(
        world, dexsim.spawn.SceneBuilder(world), session_id=session_id
    )
    running = True

    def stop(*_args: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    key = token_bytes(32)
    with os.fdopen(
        os.open(sessions / "writer.key", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600),
        "wb",
    ) as stream:
        stream.write(key)
    with ExitStack() as stack:
        stack.callback(context.close)
        stack.callback(record.unlink, missing_ok=True)
        episode = _Episode(context, _DEPLOYMENT, not args.no_video, stack)
        motion = Ur5Motion(episode, output)
        stack.callback(motion.close)
        with CommandQueue(
            context,
            executors(context, motion.operations(), key),
            request_timeout=config["request_timeout"],
            journal=record.with_suffix(".jsonl"),
        ) as queue:
            _publish_json(
                record, _session_info(session_id, queue.address, config, record)
            )
            while running:
                queue.poll(timeout=0.002, after_command=context.poll_updates)


if __name__ == "__main__":
    main()
