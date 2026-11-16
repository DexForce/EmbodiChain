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
"""Prepare a separate, provenance-bearing physical AgiBot input package for #1212."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from embodichain.data_pipeline.real2sim.adaptation._prepared_physics import (
    prepare_physical_inputs,
    record_native_qualification,
    seal_prepared_checksums,
    verify_prepared_checksums,
)
from embodichain_robots.agibot.g1._replay_collisions import prepare_replay_collisions


def main() -> None:
    """Prepare verified inputs and optionally exercise their native physical robot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Extracted pinned five-case input package"
    )
    parser.add_argument("output", type=Path, help="New portable package directory")
    parser.add_argument(
        "--source-bundle",
        type=Path,
        help="Optional verified offline official collision source bundle",
    )
    parser.add_argument(
        "--native-check",
        action="store_true",
        help="Initialize, step, open/close and check pad contacts using native physics",
    )
    args = parser.parse_args()
    sample_id = "03_agibot_toy"
    receipt = prepare_physical_inputs(
        args.source,
        args.output,
        sample_id=sample_id,
        prepare_robot=partial(
            prepare_replay_collisions, source_bundle=args.source_bundle
        ),
    )
    if args.native_check:
        from embodichain.gen_sim.real2sim._robot_qualification import (
            qualify_physical_robot,
        )

        sample = args.output.resolve() / "samples" / sample_id
        native = sample / "physical_validation/native"
        try:
            result = qualify_physical_robot(sample, native)
        except BaseException:
            if (native / "report.json").is_file():
                result = json.loads((native / "report.json").read_text())
                record_native_qualification(args.output, sample_id, result)
            else:
                verify_prepared_checksums(args.output, new_files_root=native)
                seal_prepared_checksums(args.output)
            raise
        record_native_qualification(args.output, sample_id, result)
        receipt["native_qualification"] = result["status"]
    verify_prepared_checksums(args.output)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
