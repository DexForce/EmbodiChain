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

import json
from types import SimpleNamespace

import pytest
from requests.exceptions import ConnectionError

from embodichain.data.assets import policy_assets
from embodichain.learning.rl.policy_evaluation.manifest import RunManifest

MODEL = "g1-flat-ppo-newton"
REVISION = "a" * 40
OTHER_REVISION = "b" * 40


@pytest.fixture
def hub(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": {
                    MODEL: {
                        "format": "embodichain-rl-run-v1",
                        "task_package": "embodichain_tasks",
                    }
                },
            }
        )
    )
    calls = []
    monkeypatch.setattr(policy_assets, "_DEFAULT_REVISION", REVISION)
    monkeypatch.setattr(
        policy_assets.importlib.util, "find_spec", lambda name: object()
    )

    def download_file(**kwargs):
        calls.append((kwargs["filename"], kwargs))
        if kwargs["filename"] == "index.json":
            return str(index)
        path = kwargs["local_dir"] / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = {
            "checkpoint.pt": b"policy",
            "configs/train.yaml": b"trainer: {}",
            "configs/env.yaml": b"env: {}",
            "evaluation.json": b"{}",
            "run-manifest.json": json.dumps(
                {
                    "schema_version": 1,
                    "configs": {
                        "train": "configs/train.yaml",
                        "gym": "configs/env.yaml",
                    },
                    "checkpoints": {"latest": "checkpoint.pt"},
                }
            ).encode(),
        }
        relative = kwargs["filename"].removeprefix(f"policies/{MODEL}/")
        if not path.exists():
            path.write_bytes(contents[relative])
        return str(path)

    monkeypatch.setattr(policy_assets, "hf_hub_download", download_file)
    return index, calls


def test_download_materializes_one_model_and_reuses_its_revision(tmp_path, hub):
    _, calls = hub
    root, source = policy_assets.download_pretrained_policy(MODEL, cache_dir=tmp_path)
    manifest = RunManifest.load(root)
    checkpoint = manifest.select_checkpoint()[1]
    original_mtime = checkpoint.stat().st_mtime_ns
    repeated, _ = policy_assets.download_pretrained_policy(MODEL, cache_dir=tmp_path)
    assert repeated == root
    assert checkpoint.stat().st_mtime_ns == original_mtime
    assert source == {
        "repo_id": "DexForceAI/embodichain_model",
        "revision": REVISION,
        "model_id": MODEL,
    }
    assert {name for name, _ in calls if name != "index.json"} == {
        f"policies/{MODEL}/{name}"
        for name in (
            "run-manifest.json",
            "checkpoint.pt",
            "configs/train.yaml",
            "configs/env.yaml",
            "evaluation.json",
        )
    }
    assert all(kwargs["revision"] == REVISION for _, kwargs in calls)


def test_named_revision_is_resolved_before_any_file_download(
    tmp_path, hub, monkeypatch
):
    _, calls = hub
    resolutions = []

    def model_info(repo_id, *, revision):
        resolutions.append((repo_id, revision))
        return SimpleNamespace(sha=OTHER_REVISION)

    monkeypatch.setattr(
        policy_assets, "HfApi", lambda: SimpleNamespace(model_info=model_info)
    )
    root, source = policy_assets.download_pretrained_policy(
        MODEL, revision="release", cache_dir=tmp_path
    )
    assert resolutions == [("DexForceAI/embodichain_model", "release")]
    assert source["revision"] == OTHER_REVISION
    assert OTHER_REVISION in root.parts
    assert all(kwargs["revision"] == OTHER_REVISION for _, kwargs in calls)


@pytest.mark.parametrize(
    "model_id", ("../g1", "/tmp/model", "g1/*", "G1", "g1..newton")
)
def test_invalid_model_id_is_rejected_before_network(model_id, hub):
    _, calls = hub
    with pytest.raises(ValueError, match="Model ID"):
        policy_assets.download_pretrained_policy(model_id)
    assert calls == []


@pytest.mark.parametrize(
    "value, message",
    [
        ({"schema_version": 2}, "schema"),
        ({"schema_version": 1, "models": []}, "mapping"),
        ({"schema_version": 1, "models": {}}, "Unknown pretrained"),
        ({"schema_version": 1, "models": {MODEL: {"format": "other"}}}, "format"),
        (
            {
                "schema_version": 1,
                "models": {MODEL: {"format": "embodichain-rl-run-v1"}},
            },
            "task package",
        ),
    ],
)
def test_invalid_index_does_not_download_weights(value, message, hub):
    index, calls = hub
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=message):
        policy_assets.download_pretrained_policy(MODEL)
    assert [name for name, _ in calls] == ["index.json"]


def test_missing_task_package_is_reported_before_weights(hub, monkeypatch):
    _, calls = hub
    monkeypatch.setattr(policy_assets.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match="embodichain_tasks"):
        policy_assets.download_pretrained_policy(MODEL)
    assert [name for name, _ in calls] == ["index.json"]


def test_download_failure_reports_model_context(hub, monkeypatch):
    index, _ = hub

    def fail(**kwargs):
        if kwargs["filename"] == "index.json":
            return str(index)
        raise ConnectionError("offline")

    monkeypatch.setattr(policy_assets, "hf_hub_download", fail)
    with pytest.raises(
        RuntimeError, match=f"Unable to download pretrained policy '{MODEL}'"
    ):
        policy_assets.download_pretrained_policy(MODEL)
