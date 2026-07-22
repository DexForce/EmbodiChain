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

"""Run the Image2Tabletop -> config generation -> action-agent pipeline."""

from __future__ import annotations

from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_args import build_parser
from embodichain.gen_sim.action_agent_pipeline.cli.pipeline_runner import run_pipeline

__all__ = ["main"]


def main() -> int:
    return run_pipeline(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
