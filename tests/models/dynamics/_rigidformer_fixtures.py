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

import torch
from embodichain_models.dynamics.rigidformer import RigidFormerCfg


def tiny_cfg() -> RigidFormerCfg:
    cfg = RigidFormerCfg().reference_config()
    m = cfg["model"]
    m["encoder"].update(
        output_dim=96, backbone_dim=32, hidden_dims=[32, 32], fusion_hidden_dim=32
    )
    m["state"]["token_dim"] = 96
    m["decoder"].update(
        hidden_dim=96,
        num_heads=1,
        num_layers=1,
        dropout=0.0,
        num_register_tokens=2,
        block_attn_res={"enable": False},
    )
    m["predictor"].update(hidden_dims=[96, 32], dropout=0.0)
    m["avp"]["output_dim"] = 16
    return RigidFormerCfg(architecture=cfg)


def geometry_batch(device="cpu", points=8):
    gen = torch.Generator().manual_seed(432)
    ref = torch.randn(2, 2, points, 3, generator=gen) * 0.1
    ref[..., 2] += 1.0
    pos = torch.stack([ref + i * 0.002 for i in range(4)], 1)
    v = {
        "obj_positions": pos,
        "valid_obj_mask": torch.tensor([[True, True], [True, False]]),
        "valid_vtx_mask": torch.ones(2, 4, 2, points, dtype=torch.bool),
        "ref_obj_positions": ref,
        "anchor_idx": torch.arange(4).expand(2, 2, 4).clone(),
        "phys_params": torch.tensor([1.0, 0.8, 0.3]).expand(2, 2, 3).clone(),
        "skip_step_size": torch.tensor([1.0, 5.0]),
        "dt": torch.ones(2),
        "time_step_id": torch.zeros(2, dtype=torch.long),
        "vtx_rank": torch.arange(points).expand(2, 2, points).clone(),
    }
    return {k: x.to(device) for k, x in v.items()}
