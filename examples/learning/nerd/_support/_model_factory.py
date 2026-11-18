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

"""Reference task assembly; model execution itself has no solver dependency."""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import torch

from embodichain_models.dynamics.nerd import load_model, NeRDStateCodec, NeRDTransition
from embodichain_solvers.physics.nerd._binding import NeRDBinding

from ._model_env import NeRDModelEnv
from ._reference_assets import reference_profile
from ._reference_model import make_reference_model
from ._scene_setup import reference_friction

__all__: list[str] = []


def make_reference_env(
    robot: str,
    num_envs: int,
    device: torch.device | str,
    *,
    checkpoint: str,
    max_episode_steps: int,
    random_reset: bool,
    seed: int | None,
    physics_profile: str | None,
    reset_source: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None,
) -> NeRDModelEnv:
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    destination = torch.empty(0, device=device).device
    path = Path(checkpoint).expanduser().resolve()
    manifest = json.loads((path / "manifest.json").read_text())
    profile = reference_profile(robot)
    if manifest.get("asset_sha256") != profile["asset_sha256"]:
        raise ValueError("NeRD export asset identity differs from the pinned reference")
    recorded_profile = manifest.get("physics_profile") or manifest.get(
        "teacher", {}
    ).get("physics_profile")
    selected_profile = recorded_profile if physics_profile is None else physics_profile
    if selected_profile not in {"native", "warp_reference"} or (
        recorded_profile is not None and selected_profile != recorded_profile
    ):
        raise ValueError("Requested physics_profile differs from the NeRD export")
    friction = reference_friction(
        robot,
        selected_profile,
        {"solver_type": "nerd", "options": {"checkpoint": str(path)}},
    )
    model = make_reference_model(
        robot, str(destination), num_envs, selected_profile, friction
    )
    binding = NeRDBinding(model, robot)
    binding.validate_manifest(manifest, profile["dt"])
    network, loaded = load_model(path, device=destination)
    if loaded != manifest:
        raise ValueError("NeRD export changed during construction")
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in network.state_dict().values()
    ):
        raise TypeError("The Newton FK reference backend requires float32 weights")
    network.requires_grad_(False)
    codec = NeRDStateCodec(**manifest["codec"])
    transition = NeRDTransition(network, codec, binding.geometry, profile["dt"])
    return NeRDModelEnv(
        transition,
        max_episode_steps=max_episode_steps,
        random_reset=random_reset,
        seed=seed,
        reset_source=reset_source,
        physics_metadata=binding.metadata(profile["dt"]),
    )
