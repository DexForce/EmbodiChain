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

"""Take the independent native readbacks for a DexPilot acceptance turn."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from dexsim.runtime import Session
from embodichain.lab.sim.dexsim_session.writer import WriterClient
from .fixture import ARM, native


def main() -> None:
    directory = Path(sys.argv[1])
    (record,) = directory.glob("dsws_*.json")
    with Session.connect(
        record.stem, directory=directory, client={"kind": "native-verifier"}
    ) as s:
        result = {
            n: native(s, "articulation.get_state", object_ref=n, joint_names=ARM)
            for n in ("robot_a", "robot_b")
        }
        result["control"] = WriterClient(s, directory / "writer.key").call("status", {})
        if result["control"]["control_step"]:
            result["receipt"] = WriterClient(s, directory / "writer.key").call(
                "receipt", {"command_id": "target-b-once"}
            )
        result["runtime"] = s.info["runtime"]
        Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
