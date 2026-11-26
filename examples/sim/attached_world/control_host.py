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

"""Start a single-writer example host around a populated, manually stepped World."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
from uuid import uuid4

import dexsim
from dexsim.kit.mcp import dispatch
from dexsim.runtime import SessionContext
from dexsim.runtime.command_queue import CommandQueue
from dexsim.runtime.config import normalize_config
from dexsim.runtime.host import _publish_json, _session_info, _world_config
from dexsim.runtime.protocol import record_path

from embodichain.lab.sim.dexsim_session.control import WorldControl
from embodichain.lab.sim.dexsim_session.writer import writer_executors
from . import fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--backend", choices=["dexsim", "newton"], required=True)
    parser.add_argument("--assembled", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    cfg = fixture.robot_config(args.assembled)
    asset = Path(cfg.fpath)
    options = normalize_config(
        dict(
            backend=args.backend,
            **({"solver": "mujoco_warp"} if args.backend == "newton" else {}),
            device="cuda:0",
            headless=True,
            dt=0.01,
            asset_root=str(asset.parent),
            request_timeout=60,
        )
    )
    # Only the fixture creates the World. The controller receives finalized live objects.
    world = dexsim.World(_world_config(options))
    world.set_manual_update(True)
    world.enable_physics(True)
    session_id = "dsws_" + uuid4().hex
    context = SessionContext(
        world,
        dexsim.spawn.SceneBuilder(world),
        session_id=session_id,
        asset_root=str(asset.parent),
    )
    record = record_path(session_id, directory)
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    key = secrets.token_bytes(32)
    with os.fdopen(
        os.open(directory / "writer.key", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600),
        "wb",
    ) as stream:
        stream.write(key)
    try:
        fixture.populate_with(
            lambda method, **params: dispatch.run(
                context, method=method, params=params
            ),
            asset,
        )
        manager = fixture.attach(context, args.assembled, fixture.PATHS)
        try:
            control = WorldControl(
                context, manager, physics_dt=0.01, steps_per_control=4
            )
            with CommandQueue(
                context,
                writer_executors(control, key),
                request_timeout=60,
                journal=record.with_suffix(".jsonl"),
            ) as queue:
                _publish_json(
                    record,
                    _session_info(
                        session_id,
                        queue.address,
                        options,
                        record,
                        **({"token": queue.token} if hasattr(queue, "token") else {}),
                    ),
                )
                while running:
                    queue.poll(timeout=0.02)
        finally:
            manager.detach()
    finally:
        record.unlink(missing_ok=True)
        context.close()


if __name__ == "__main__":
    main()
