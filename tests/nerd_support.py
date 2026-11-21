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

"""Small portable exports bound to the pinned reference geometry, without a World."""

from __future__ import annotations

from pathlib import Path

import torch

from embodichain.extensions.models.dynamics.nerd import (
    NeRDModel,
    NeRDModelCfg,
    export_model,
)


def small_nerd_export(destination: Path, robot: str) -> Path:
    from examples.learning.nerd._support._reference_assets import reference_profile
    from examples.learning.nerd._support._reference_model import make_reference_model
    from embodichain.extensions.solvers.physics.nerd._binding import NeRDBinding

    profile = reference_profile(robot)
    model = make_reference_model(robot, "cpu", 2, "native")
    binding = NeRDBinding(model, robot)
    dims = {"states_embedding": 4, "joint_f": 2}
    if robot == "ant":
        dims = {
            "states_embedding": 29,
            "contact_normals": 75,
            "contact_points_1": 75,
            "contact_depths": 25,
            "joint_f": 8,
            "gravity_dir": 3,
        }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(822)
        network = NeRDModel(
            NeRDModelCfg(
                input_dims=dims,
                output_dim=profile["q_dim"] + profile["qd_dim"],
                n_embd=12,
                n_head=3,
                n_layer=1,
                history_length=3,
                block_size=3,
                head_hidden_dims=(8,),
                attention_backend="math",
            )
        )
    with torch.no_grad():
        network.output_rms.var.fill_(1e-4)
        if robot == "ant":
            network.output_rms.mean[6] = 1.0
    export_model(
        destination,
        network,
        {
            **profile,
            **binding.metadata(profile["dt"]),
            "physics_profile": "native",
            "teacher": {
                "physics_profile": "native",
                "solver": {"solver_type": "featherstone"},
                "num_substeps": 1,
            },
        },
    )
    return destination
