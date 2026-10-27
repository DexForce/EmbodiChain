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

import subprocess
import sys
import pytest
import torch
from ._rigidformer_fixtures import tiny_cfg, geometry_batch


def test_import_does_not_require_community_model_or_simulator():
    code = """import sys
from embodichain_models.dynamics.rigidformer import RigidFormerCfg
assert 'rigidformer' not in sys.modules
assert 'flash_attn' not in sys.modules
assert not any(n.startswith('embodichain.lab.sim') for n in sys.modules)
assert RigidFormerCfg().reference_config()['model']['kabsch_reference'] == 'fixed'
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.gpu
def test_native_forward_checkpoint_and_rollout_with_variable_objects(tmp_path):
    pytest.importorskip("roma")
    from embodichain_models.dynamics.rigidformer import (
        RigidFormerModel,
        load_rigidformer_checkpoint,
    )

    torch.manual_seed(42)
    model = RigidFormerModel(tiny_cfg()).cuda().eval()
    values = geometry_batch("cuda")
    values["valid_obj_mask"][:] = True
    for key in ("obj_positions", "valid_vtx_mask"):
        values[key] = torch.cat((values[key], values[key][:, :, :1].clone()), dim=2)
    for key in ("ref_obj_positions", "anchor_idx", "phys_params", "vtx_rank"):
        values[key] = torch.cat((values[key], values[key][:, :1].clone()), dim=1)
    values["obj_positions"][:, :, 2, :, 0] += 1
    values["ref_obj_positions"][:, 2, :, 0] += 1
    values["valid_obj_mask"] = torch.tensor(
        [[True, True, True], [True, True, False]], device="cuda"
    )
    with torch.no_grad():
        out = model(values)
    assert torch.isfinite(out["obj_positions_denorm"]).all()
    weights = tmp_path / "official-layout.pt"
    torch.save(
        {
            "config": model.reference_config,
            "model_state_dict": {
                k: v.cpu() for k, v in model.network.state_dict().items()
            },
        },
        weights,
    )
    restored = load_rigidformer_checkpoint(weights, device="cuda")
    with torch.no_grad():
        same = restored(values)
    torch.testing.assert_close(
        out["obj_positions_denorm"], same["obj_positions_denorm"], rtol=0, atol=0
    )
    rollout = restored.rollout(values, num_frames=8, warmup_frames=4)
    assert rollout["obj_positions"].shape == (2, 8, 3, 8, 3)
    assert torch.isfinite(rollout["obj_positions"]).all()


@pytest.mark.gpu
def test_native_packed_points_ignore_finite_padding_above_128():
    pytest.importorskip("roma")
    from embodichain_models.dynamics.rigidformer import RigidFormerModel

    model = RigidFormerModel(tiny_cfg()).cuda().eval()
    batch = geometry_batch("cuda", points=160)
    batch["valid_obj_mask"][:] = True
    batch["valid_vtx_mask"][:, :, 0, 96:] = False
    with torch.no_grad():
        expected = model(batch)["obj_positions_denorm"]
        changed = {k: v.clone() for k, v in batch.items()}
        changed["obj_positions"][:, :, 0, 96:] += 10000
        changed["ref_obj_positions"][:, 0, 96:] -= 10000
        actual = model(changed)["obj_positions_denorm"]
    valid = batch["valid_vtx_mask"][:, 2:]
    torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
