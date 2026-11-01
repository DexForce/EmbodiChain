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

"""Native SDK datasets and official training configuration contracts."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from embodichain_models.policy.act import build_act_train_config, dataset_fingerprint

pytestmark = pytest.mark.skipif(
    version("lerobot") != "0.6.1",
    reason="Official ACT integration requires LeRobot 0.6.1",
)


def test_config_uses_native_training_and_declared_features(act_data, tmp_path: Path):
    from lerobot.datasets.factory import make_dataset

    root, schema = act_data
    cfg = build_act_train_config(
        root,
        tmp_path / "run",
        schema=schema,
        image_key="observation.images.camera",
        device="cpu",
        smoke=True,
        num_workers=0,
    )
    assert cfg.dataset.eval_split == 0
    assert cfg.env_eval_freq == 0
    assert cfg.policy.chunk_size == 30
    assert cfg.policy.n_action_steps == 1
    assert cfg.policy.temporal_ensemble_coeff == 0.01
    dataset = make_dataset(cfg)
    last = dataset[2]
    np.testing.assert_array_equal(last["action_is_pad"].numpy(), [False] + [True] * 29)
    np.testing.assert_array_equal(last["action"][0].numpy(), [2, 0])
    assert tuple(last["observation.images.camera"].shape) == (3, 32, 32)


def test_config_rejects_wrong_feature_order_and_cadence(act_data, tmp_path: Path):
    import copy

    root, schema = act_data
    bad = copy.deepcopy(schema)
    bad["action"]["names"].reverse()
    with pytest.raises(ValueError, match="order"):
        build_act_train_config(root, tmp_path / "run", schema=bad)
    bad = copy.deepcopy(schema)
    bad["action"]["step_dt"] = 1 / 20
    with pytest.raises(ValueError, match="control period"):
        build_act_train_config(root, tmp_path / "run", schema=bad)


def test_dataset_fingerprint_binds_rgb_and_ignores_sdk_cache(act_data):
    root, _ = act_data
    expected = dataset_fingerprint(root)
    (root / ".cache").mkdir()
    (root / ".cache/temporary").write_text("cache")
    assert dataset_fingerprint(root) == expected
    import io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    shard = next((root / "data").rglob("*.parquet"))
    table = pq.read_table(shard)
    key = "observation.images.camera"
    values = table[key].to_pylist()
    image = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(image, format="PNG")
    values[0]["bytes"] = image.getvalue()
    table = table.set_column(
        table.schema.get_field_index(key), key, pa.array(values, type=table[key].type)
    )
    pq.write_table(table, shard)
    assert dataset_fingerprint(root) != expected


def test_official_trainer_publishes_and_resumes_native_checkpoint(
    act_data, tmp_path: Path
):
    import json
    import os
    import subprocess
    import sys

    root, schema = act_data
    output = tmp_path / "native"
    cfg = build_act_train_config(
        root,
        output,
        schema=schema,
        image_key="observation.images.camera",
        device="cpu",
        smoke=True,
        updates=2,
        batch_size=2,
        save_freq=1,
        num_workers=0,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={path}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    checkpoint = output / "checkpoints/000002"
    policy = checkpoint / "pretrained_model"
    for name in [
        "model.safetensors",
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ]:
        assert (policy / name).is_file()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={policy / 'train_config.json'}",
            "--resume=true",
            "--steps=3",
            f"--output_dir={output}",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    state = json.loads(
        (output / "checkpoints/000003/training_state/training_step.json").read_text()
    )
    assert state["step"] == 3
