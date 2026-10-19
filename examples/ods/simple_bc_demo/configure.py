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

"""Resolve a PickPlace ODS input/profile JSON without starting simulation.

Run with python -m examples.ods.simple_bc_demo.configure --input INPUT
--output REPORT. Optional rate overrides are explicit and saved in the report.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from embodichain.data_pipeline.configuration import (
    ODSReuseProfileCfg,
    ODSRuntimeCfg,
    configure_ods,
)
from examples.ods.simple_bc_demo.autoconfiguration import load_configuration


def main() -> None:
    """Write a reviewable configuration report; never launch worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation-throughput", type=float)
    parser.add_argument("--training-throughput", type=float)
    args = parser.parse_args()
    report = load_configuration(args.input)
    inputs = report["input"]
    overrides = {}
    for key, value in (
        ("accepted_episodes_per_second", args.generation_throughput),
        ("training_chunks_per_second", args.training_throughput),
    ):
        if value is not None:
            overrides[key] = value
    effective = {**inputs, **overrides}
    report["rate_overrides"] = overrides
    report["effective_input"] = effective
    report["configuration"] = asdict(
        configure_ods(
            effective["accepted_episodes_per_second"],
            effective["training_chunks_per_second"],
            profile=ODSReuseProfileCfg(**effective["profile"]),
            runtime=ODSRuntimeCfg(**effective["runtime"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["configuration"], indent=2))


if __name__ == "__main__":
    main()
