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

"""Pinned independent upstream reference and weights-only accuracy check."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request

import numpy as np
import torch

from embodichain_models.dynamics.lewm import LeWM, LeWMCfg, preprocess_rgb

HF_REV = "22b330c28c27ead4bfd1888615af1340e3fe9052"
SWM_REV = "21446f1ede6d5284e981bd7b47f432b994e6d812"
LEWM_REV = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"
FILES = {
    "weights.pt": "48938400ae3464c9680731287f583a9cb516f55a8ec64ea13a91be47fb15b607",
    "config.json": "2564086e961e7b5c7c04dffc451091115b389a590645ff19653c64fd0bc16e09",
    "reference_lewm.py": "10ad1c2365b569ad3f9cbda17c8154d799c3b0c4fc5d707a44dca600b5ca8315",
    "reference_module.py": "d276fb896712bed23f70f25ae9774e5ce7b3ab44fbb6bf4dea61065ba1388ef9",
    "reference_sigreg.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(root: str | Path, local_files_only: bool = False) -> dict:
    """Download and verify exact upstream source and checkpoint revisions."""
    from huggingface_hub import hf_hub_download

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("weights.pt", "config.json"):
        if not (root / name).is_file():
            hf_hub_download(
                "quentinll/lewm-pusht",
                name,
                revision=HF_REV,
                local_dir=root,
                local_files_only=local_files_only,
            )
    urls = {
        "reference_lewm.py": f"https://raw.githubusercontent.com/galilai-group/stable-worldmodel/{SWM_REV}/stable_worldmodel/wm/lewm/lewm.py",
        "reference_module.py": f"https://raw.githubusercontent.com/galilai-group/stable-worldmodel/{SWM_REV}/stable_worldmodel/wm/lewm/module.py",
        "reference_sigreg.py": f"https://raw.githubusercontent.com/lucas-maes/le-wm/{LEWM_REV}/module.py",
    }
    for name, url in urls.items():
        path = root / name
        if not path.is_file():
            if local_files_only:
                raise FileNotFoundError(path)
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != FILES[name]:
                raise ValueError(f"reference source checksum mismatch: {name}")
            path.write_bytes(data)
    for name, expected in FILES.items():
        if _sha(root / name) != expected:
            raise ValueError(f"reference resource checksum mismatch: {name}")
    return {
        "weights_revision": HF_REV,
        "stable_worldmodel_revision": SWM_REV,
        "lewm_revision": LEWM_REV,
        "sha256": FILES,
    }


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reference_model(
    model_source: Path, blocks_source: Path, weights: Path, action_dim: int
) -> torch.nn.Module:
    """Instantiate the checked upstream architecture without package side effects."""
    from transformers import ViTConfig, ViTModel

    blocks = _import_file(blocks_source, "lewm_reference_blocks")
    upstream = _import_file(model_source, "lewm_reference_model")
    vision = ViTConfig(
        image_size=224,
        patch_size=14,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )
    model = upstream.LeWM(
        encoder=ViTModel(vision, add_pooling_layer=False, use_mask_token=False),
        predictor=blocks.Predictor(
            num_frames=3,
            depth=6,
            heads=16,
            mlp_dim=2048,
            input_dim=192,
            hidden_dim=192,
            output_dim=192,
            dim_head=64,
            dropout=0.1,
        ),
        action_encoder=blocks.Embedder(input_dim=action_dim, emb_dim=192),
        projector=blocks.MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d),
        pred_proj=blocks.MLP(192, 2048, 192, norm_fn=torch.nn.BatchNorm1d),
    ).eval()
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model


def compare(
    root: str | Path,
    *,
    cube_weights: str | Path | None = None,
    cube_swm_root: str | Path | None = None,
) -> dict:
    """Compare fixed FP32 inputs and official weights against upstream code."""
    from torchvision.transforms import v2

    root = Path(root)
    action_dim, action_block = 10, 1
    history_len = 3
    model_source = root / "reference_lewm.py"
    blocks_source = root / "reference_module.py"
    if cube_swm_root is not None and cube_weights is None:
        raise ValueError("the original Cube reference requires Cube weights")
    if cube_weights is not None and cube_swm_root is None:
        raise ValueError("Cube comparison requires the pinned cube_swm_root checkout")
    weights_path = root / "weights.pt"
    if cube_weights is None:
        identity = prepare(root, local_files_only=True)
    else:
        weights_path = Path(cube_weights)
        expected = "2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89"
        if _sha(weights_path) != expected:
            raise ValueError("official Cube checkpoint checksum mismatch")
        from ._official_cube import CUBE_SWM_REV, _revision

        source_root = Path(cube_swm_root)
        _revision(source_root, CUBE_SWM_REV)
        model_source = source_root / "stable_worldmodel/wm/lewm/lewm.py"
        blocks_source = source_root / "stable_worldmodel/wm/lewm/module.py"
        sources = {path.name: _sha(path) for path in (model_source, blocks_source)}
        # PlanConfig's history_len defaults to one in the original eval.
        history_len = 1
        identity = {
            "task": "ogbench_cube_single",
            "weights_revision": "b0747c5002e86d2ce8f3cd8178004b97524c587d",
            "stable_worldmodel_revision": CUBE_SWM_REV,
            "lewm_revision": LEWM_REV,
            "sha256": {**sources, "weights.pt": expected},
        }
        action_dim, action_block = 25, 5
    reference = _load_reference_model(
        model_source, blocks_source, weights_path, action_dim
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = LeWM(LeWMCfg(action_dim=action_dim, action_block=action_block)).eval()
    model.load_reference_weights(state)
    torch.manual_seed(42)
    raw = torch.randint(0, 256, (1, history_len, 224, 224, 3), dtype=torch.uint8)
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            v2.Resize((224, 224)),
        ]
    )
    reference_pixels = torch.stack([transform(x.numpy()) for x in raw[0]])[None]
    pixels = preprocess_rgb(raw)
    actions = torch.randn(1, history_len, action_dim)
    candidates = torch.randn(1, 2, 5, action_dim)
    past = actions[:, : history_len - 1]
    with torch.no_grad():
        embedding = model.encode(pixels)
        ref_embedding = reference.encode({"pixels": reference_pixels})["emb"]
        encoded_actions = model.encode_actions(actions)
        ref_actions = reference.action_encoder(actions)
        predicted = model.predict(embedding, actions)
        ref_predicted = reference.predict(ref_embedding, ref_actions)
        future = model.rollout(embedding, past, candidates)
        info = {
            "pixels": reference_pixels[:, None].expand(-1, 2, -1, -1, -1, -1).clone(),
            "action_history": past[:, None].expand(-1, 2, -1, -1).clone(),
        }
        ref_future = reference.rollout(info, candidates)["predicted_emb"][
            :, :, history_len:
        ]
        goal = ref_embedding[:, -1]
        cost = model.goal_cost(future, goal)
        ref_cost = (ref_future[:, :, -1] - goal[:, None]).square().sum(-1)
    values = {
        "preprocess": (pixels, reference_pixels),
        "encoder": (embedding, ref_embedding),
        "action_encoder": (encoded_actions, ref_actions),
        "predictor": (predicted, ref_predicted),
        "rollout": (future, ref_future),
        "goal_cost": (cost, ref_cost),
    }
    errors = {}
    for name, (actual, expected) in values.items():
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        errors[name] = float((actual - expected).abs().max())
    return {
        "passed": True,
        "max_absolute_error": errors,
        "atol": 1e-5,
        "rtol": 1e-4,
        "torch": torch.__version__,
        "device": "cpu",
        "observed_history_len": history_len,
        "resources": identity,
    }
