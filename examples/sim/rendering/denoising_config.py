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

"""Configure independent window and offscreen denoising pipelines."""

from __future__ import annotations

import argparse

import dexsim

from embodichain.lab.sim.cfg import DenoisingCfg, DLSSCfg, NRDCfg, RenderCfg

DENOISING_MODES = ("off", "optix", "dlss-rr", "nrd-sr")


def build_parser() -> argparse.ArgumentParser:
    """Build command-line options for the rendering configuration example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window",
        choices=DENOISING_MODES,
        default="dlss-rr",
        help="Denoising/reconstruction path used by the native window.",
    )
    parser.add_argument(
        "--offscreen",
        choices=DENOISING_MODES,
        default="dlss-rr",
        help="Denoising/reconstruction path used by offscreen cameras.",
    )
    parser.add_argument(
        "--renderer",
        choices=("hybrid", "fast-rt", "rt"),
        default="fast-rt",
        help="Ray-tracing renderer backend.",
    )
    parser.add_argument(
        "--dlss-quality",
        type=int,
        choices=range(-1, 6),
        default=2,
        help="DLSS quality preset used by dlss-rr and nrd-sr.",
    )
    parser.add_argument(
        "--nrd-history-frames",
        type=int,
        default=30,
        help="Maximum NRD history length used by nrd-sr.",
    )
    return parser


def build_render_cfg(args: argparse.Namespace) -> RenderCfg:
    """Build the public rendering configuration from parsed arguments.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Rendering configuration ready for ``SimulationManagerCfg`` or direct
        conversion to ``dexsim.WorldConfig``.
    """
    return RenderCfg(
        renderer=args.renderer,
        denoising=DenoisingCfg(
            window=args.window,
            offscreen=args.offscreen,
        ),
        dlss=DLSSCfg(dlss_quality=args.dlss_quality),
        nrd=NRDCfg(max_accumulated_frame_num=args.nrd_history_frames),
    )


def main(args: argparse.Namespace | None = None) -> None:
    """Apply the config and print the resolved native pipeline selections."""
    if args is None:
        args = build_parser().parse_args()

    render_cfg = build_render_cfg(args)
    world_cfg = dexsim.WorldConfig()
    render_cfg.apply_to_dexsim_config(world_cfg)

    print(f"renderer={world_cfg.renderer.name}")
    print(f"window={world_cfg.rt_pipeline_config.window.mode.name}")
    print(f"offscreen={world_cfg.rt_pipeline_config.offscreen.mode.name}")
    print(f"dlss_quality={world_cfg.dlss_config.dlss_quality}")
    print("nrd_history_frames=" f"{world_cfg.nrd_config.max_accumulated_frame_num}")


if __name__ == "__main__":
    main()
