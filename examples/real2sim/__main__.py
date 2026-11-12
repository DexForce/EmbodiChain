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
"""Reconstruct a real recording through the native Scene Engine.

Run ``python -m examples.real2sim --help`` for inputs and service configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embodichain.gen_sim.real2sim.image_edit import HttpImageEditor
from embodichain.gen_sim.real2sim.scene import reconstruct_video
from embodichain.gen_sim.services import (
    GenSimServices,
    ServiceRegistry,
    load_services_config,
)
from embodichain.gen_sim.services.runtime import CHAT_ROLES

__all__ = ["main"]


def main() -> None:
    """Read explicit input/configuration and invoke the packaged scene stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-frames", type=int, nargs="*", default=[])
    parser.add_argument(
        "--services",
        type=Path,
        help="Native model_services JSON; default: repository discovery",
    )
    parser.add_argument("--image-edit-url", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    cfg = load_services_config(args.services)
    registry = None
    if cfg and any(p["adapter"] == "codex_cli" for p in cfg.profiles.values()):
        from ._codex import CodexChat

        registry = ServiceRegistry.defaults()
        registry.register(
            "codex_cli",
            lambda profile: CodexChat(profile, args.output / "model_calls"),
            roles=CHAT_ROLES,
        )
    with GenSimServices(cfg, registry=registry) as services:
        try:
            result = reconstruct_video(
                args.video,
                args.output,
                services=services,
                editor=HttpImageEditor(args.image_edit_url),
                reference_frames=tuple(args.reference_frames),
            )
            print(result / "scene_export")
        finally:
            if args.output.is_dir():
                services.write_records(args.output / "service_calls.json")


if __name__ == "__main__":
    main()
