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

"""Preview the measured dual-camera expert trajectory before visual BC training.

This entry point performs acquisition only. It never constructs a policy,
optimizer or learner. Use a fresh output directory and mask an idle simulation
GPU before launch. Camera poses and task physics come from acquisition.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from .acquisition import (
    camera_profile_version,
    camera_uids,
    iter_generation_episodes,
    load_acquisition_config,
    make_engine,
)

__all__ = ["capture_preview"]


def capture_preview(
    output: Path, *, device: str, seed: int, camera_profile: str = "pick_place"
) -> dict[str, Any]:
    """Save actual camera RGB frames/video from one accepted physical episode.

    Args:
        output: New directory for the preview and resolved camera configuration.
        device: Explicit simulation device within the launcher's GPU mask.
        seed: Preview-only generation seed, separate from training/evaluation.
        camera_profile: Camera calibration to review before training.

    Returns:
        Camera/frame metadata and clean producer shutdown evidence.
    """
    import imageio.v2 as imageio
    import matplotlib
    import numpy as np
    import torch

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ["EMBODICHAIN_SIM_EXIT_PROCESS"] = "0"
    config = load_acquisition_config(
        "dual", device, producer_envs=1, seed=seed, camera_profile=camera_profile
    )
    (output / "environment-config.json").write_text(json.dumps(config, indent=2))
    engine = make_engine(
        "dual",
        device,
        producer_envs=1,
        seed=seed,
        buffer_size=1,
        initialization_timeout=300,
        camera_profile=camera_profile,
    )
    started = time.monotonic()
    report: dict[str, Any] = {
        "status": "acquiring",
        "seed": seed,
        "device": device,
        "camera_profile_version": camera_profile_version(camera_profile),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training_updates": 0,
    }
    try:
        engine.start()
        episodes = list(iter_generation_episodes(engine, 1))
        if len(episodes) != 1:
            raise RuntimeError("Preview requires exactly one accepted episode")
        raw = episodes[0]
        report.update(
            frames=len(raw),
            engine_run_id=engine.run_id,
            producer_pid=engine._sim_process.pid,
            engine_stats=engine.get_stats(),
            acquisition_seconds=time.monotonic() - started,
        )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        process = engine._sim_process
        popen = None if process is None else process._popen
        engine.stop()
        report["producer_exitcode"] = None if popen is None else popen.returncode
        report["producer_closed"] = engine._sim_process is None
        (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["producer_exitcode"] != 0:
        raise RuntimeError("Preview producer failed to close normally")

    cameras = camera_uids("dual")
    images = {
        uid: raw["obs", "sensor", uid, "color"][..., :3].cpu().numpy()
        for uid in cameras
    }
    if any(value.shape != (len(raw), 256, 256, 3) for value in images.values()):
        raise ValueError("Preview camera shape differs from the training contract")
    indices = np.linspace(0, len(raw) - 1, 6).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=True)
    for ordinal, frame in enumerate(indices):
        row, pair = divmod(ordinal, 2)
        for view, uid in enumerate(cameras):
            ax = axes[row, pair * 2 + view]
            ax.imshow(images[uid][frame])
            ax.set_title(f"{uid} | frame {frame} | {frame / 30:.1f} s", fontsize=9)
            ax.axis("off")
    fig.suptitle("Franka PickPlace: measured expert cameras; no policy training")
    fig.savefig(output / "camera-preview.png", dpi=150)
    plt.close(fig)
    imageio.imwrite(
        output / "initial-cameras.png",
        np.concatenate([images[uid][0] for uid in cameras], axis=1),
    )
    with imageio.get_writer(
        output / "dual-view-expert.mp4",
        fps=30,
        codec="libx264",
        quality=8,
        ffmpeg_params=["-threads", "1"],
    ) as writer:
        for frame in range(len(raw)):
            writer.append_data(
                np.concatenate([images[uid][frame] for uid in cameras], axis=1)
            )
    report.update(
        status="complete",
        cameras=list(cameras),
        fps=30,
        observation_shape=[256, 256, 3],
        initial_cube_xy=raw["obs", "cube_pose"][0, :2].tolist(),
        maximum_cube_height=float(raw["obs", "cube_pose"][:, 2].max()),
        sensor_preview="Actual uint8 RGB camera observations; no generated imagery",
    )
    (output / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20291025)
    parser.add_argument(
        "--camera-profile", choices=("benchmark", "pick_place"), default="pick_place"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            capture_preview(
                args.output,
                device=args.device,
                seed=args.seed,
                camera_profile=args.camera_profile,
            )
        )
    )


if __name__ == "__main__":
    main()
