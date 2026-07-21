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
import json
from pathlib import Path

if __package__:
    from .config import load_prompt2geometry_config
    from .pipeline import Prompt2GeometryRequest, run_prompt2geometry
else:
    from config import load_prompt2geometry_config
    from pipeline import Prompt2GeometryRequest, run_prompt2geometry

__all__ = ["main"]


def main() -> None:
    """Run prompt-to-geometry from the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate one object mesh from a prompt via z-image, segmentation, "
            "and 3D-generation."
        )
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Object description. Complete single-object and pure-black background "
            "constraints are appended automatically."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prompt2geometry_output"),
        help="Local output directory.",
    )
    parser.add_argument("--target-id", default="asset_0")
    parser.add_argument("--request-id", default="prompt2geometry_asset_0")
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Final scaled GLB file name. If omitted, the LLM extracts one "
            "from the prompt."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Prompt2Geometry local config JSON path.",
    )
    parser.add_argument("--zimage-base-url", default=None)
    parser.add_argument("--sam3-base-url", default=None)
    parser.add_argument("--sam3d-base-url", default=None)
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="OpenAI-compatible API key for real-world dimension estimation.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="OpenAI-compatible model for real-world dimension estimation.",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL for real-world dimension estimation.",
    )
    parser.add_argument("--llm-timeout-s", type=float, default=None)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--zimage-seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument(
        "--zimage-prompt-suffix",
        default="a complete single object, with pure-black background",
        help="Suffix appended to the object description before z-image generation.",
    )
    parser.add_argument("--sam3d-seed", type=int, default=42)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable live progress logs.",
    )
    args = parser.parse_args()
    cfg = load_prompt2geometry_config(args.config)

    result = run_prompt2geometry(
        Prompt2GeometryRequest(
            prompt=args.prompt,
            output_root=args.output_root,
            target_id=args.target_id,
            request_id=args.request_id,
            output_name=args.output_name,
            zimage_base_url=args.zimage_base_url or cfg.zimage_base_url,
            zimage_width=args.width,
            zimage_height=args.height,
            zimage_seed=args.zimage_seed,
            zimage_num_inference_steps=args.num_inference_steps,
            zimage_prompt_suffix=args.zimage_prompt_suffix,
            sam3_base_url=args.sam3_base_url or cfg.sam3_base_url,
            sam3d_base_url=args.sam3d_base_url or cfg.sam3d_base_url,
            sam3d_seed=args.sam3d_seed,
            llm_api_key=args.llm_api_key or cfg.llm_api_key,
            llm_model=args.llm_model or cfg.llm_model,
            llm_base_url=args.llm_base_url or cfg.llm_base_url,
            llm_timeout_s=args.llm_timeout_s or cfg.llm_timeout_s,
            verbose=not args.quiet,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
