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

"""Real analytical collection and an SDK-free generate/train/export/eval path."""

from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest
import torch

from embodichain.data_pipeline.dynamics.generation import (
    DynamicsGenerationCfg,
    _sample,
    generate_dataset,
)
from embodichain.data_pipeline.dynamics._newton import _NewtonTeacher
from embodichain.data_pipeline.dynamics.shards import load_manifest, trajectory_split


@pytest.fixture
def deterministic_generation(monkeypatch):
    """Exercise persistence with reproducible accepted and rejected candidates."""
    from embodichain.data_pipeline.dynamics import generation

    class Teacher:
        def __init__(self, robot, *, device, num_envs, **kwargs):
            self.device = torch.device("cpu")
            self.gravity_dir = torch.tensor([0.0, 0.0, -1.0]).expand(num_envs, -1)
            self.manifest = {
                "physics_signature": "fake-physical-v1",
                "teacher_origin": {"collector": "fake"},
            }

        def reset(self, q, qd):
            self.q, self.qd = q.clone(), qd.clone()
            self.rejected = torch.tensor([int(seed) in {11, 13} for seed in q[:, 0]])
            self.index = 0

        def snapshot(self):
            q = self.q + self.index * 0.01
            q[self.rejected] = float("nan")
            return {"q": q, "qd": self.qd.clone()}, {}

        def step(self, force):
            self.index += 1

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def sample(robot, seed, horizon, *args):
        return torch.tensor([float(seed), 0.0]), torch.zeros(2), torch.zeros(horizon, 2)

    monkeypatch.setattr(generation, "_NewtonTeacher", Teacher)
    monkeypatch.setattr(generation, "_sample", sample)


def test_resume_retains_rejections_when_interrupted_after_shard_manifest(
    tmp_path, monkeypatch, deterministic_generation
):
    from embodichain.data_pipeline.dynamics import shards

    config = {
        "output_dir": str(tmp_path),
        "seed": 11,
        "trajectories": 2,
        "num_envs": 2,
        "horizon": 2,
        "trajectories_per_shard": 1,
    }
    atomic_json = shards._atomic_json

    def interrupt_after_publish(path, value):
        atomic_json(path, value)
        if path.name == "manifest.json" and value["shards"]:
            raise RuntimeError("interrupted-after-manifest-publication")

    monkeypatch.setattr(shards, "_atomic_json", interrupt_after_publish)
    with pytest.raises(RuntimeError, match="interrupted-after-manifest-publication"):
        generate_dataset(config)
    committed = load_manifest(tmp_path, verify=True)
    assert [
        row["id"] for shard in committed["shards"] for row in shard["trajectories"]
    ] == ["11:1"]
    # This legacy report can lag behind the committed shard; it cannot be the
    # recovery authority once progress is included in the shard manifest.
    (tmp_path / "generation-progress.json").write_text(
        json.dumps({"next_candidate": 0, "rejected": 0, "rejections": []})
    )
    monkeypatch.setattr(shards, "_atomic_json", atomic_json)
    result = generate_dataset({**config, "resume": True})
    assert result["trajectories"] == 2
    assert result["rejected"] == 2
    evidence = json.loads(Path(result["rejection_evidence"]).read_text())
    assert evidence["next_candidate"] == 4
    assert [record["id"] for record in evidence["rejections"]] == ["11:0", "11:2"]
    final = load_manifest(tmp_path, verify=True)
    assert final["metadata"] == committed["metadata"]
    assert final["shards"][0] == committed["shards"][0]
    baseline = tmp_path / "uninterrupted"
    generate_dataset({**config, "output_dir": str(baseline)})
    expected = load_manifest(baseline, verify=True)
    assert final["metadata"] == expected["metadata"]
    for actual_shard, expected_shard in zip(final["shards"], expected["shards"]):
        assert actual_shard["trajectories"] == expected_shard["trajectories"]
        with (
            h5py.File(tmp_path / actual_shard["file"]) as actual_file,
            h5py.File(baseline / expected_shard["file"]) as expected_file,
        ):
            for group in ("states", "transitions"):
                for field in actual_file[f"0/{group}"]:
                    assert torch.equal(
                        torch.from_numpy(actual_file[f"0/{group}/{field}"][()]),
                        torch.from_numpy(expected_file[f"0/{group}/{field}"][()]),
                    )


def test_resume_reads_legacy_generation_progress(tmp_path, deterministic_generation):
    config = {
        "output_dir": str(tmp_path),
        "seed": 11,
        "trajectories": 1,
        "num_envs": 1,
        "horizon": 2,
        "trajectories_per_shard": 1,
        "max_attempts": 1,
    }
    with pytest.raises(RuntimeError, match="Generation attempt limit"):
        generate_dataset(config)
    # Even a batch with no accepted trajectories advances a recoverable cursor.
    manifest = load_manifest(tmp_path)
    assert not manifest["shards"]
    assert manifest["generation_progress"]["next_candidate"] == 1
    assert manifest["generation_progress"]["rejected"] == 1
    # Older collectors store progress only in the separate JSON report.
    progress = manifest.pop("generation_progress")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "generation-progress.json").write_text(json.dumps(progress))
    result = generate_dataset({**config, "resume": True, "max_attempts": 2})
    assert result["trajectories"] == result["rejected"] == 1
    evidence = json.loads(Path(result["rejection_evidence"]).read_text())
    assert [record["id"] for record in evidence["rejections"]] == ["11:0"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_envs", True),
        ("horizon", 2.5),
        ("resume", "false"),
        ("seed", False),
        ("use_cuda_graph", 1),
        ("effort_scale", float("nan")),
        ("physics_profile", "unregistered"),
    ],
)
def test_generation_rejects_invalid_config(field, value):
    with pytest.raises((ValueError, TypeError)):
        DynamicsGenerationCfg(**{field: value})


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_real_generation_partial_batch_resume_and_teacher_replay(tmp_path, robot):
    config = {
        "robot": robot,
        "output_dir": str(tmp_path / "data"),
        "num_envs": 2,
        "trajectories": 3,
        "horizon": 3,
        "num_substeps": 2,
        "seed": 7,
        "split_seed": 11,
        "trajectories_per_shard": 2,
        "initial_states": "default",
        "effort_scale": 0.01,
    }
    assert generate_dataset(config)["transitions"] == 9
    first = load_manifest(tmp_path / "data", verify=True)
    original = [row for shard in first["shards"] for row in shard["trajectories"]]
    assert (
        generate_dataset({**config, "trajectories": 5, "resume": True})["transitions"]
        == 15
    )
    manifest = load_manifest(tmp_path / "data", verify=True)
    rows = [row for shard in manifest["shards"] for row in shard["trajectories"]]
    assert [row["id"] for row in rows] == [f"7:{i}" for i in range(5)]
    assert rows[:3] == original
    assert [row["split"] for row in rows] == [
        trajectory_split(row["id"], 11) for row in rows
    ]
    assert manifest["metadata"]["teacher_origin"]["collector"] == "direct_newton"
    shard = manifest["shards"][0]
    with h5py.File(tmp_path / "data" / shard["file"]) as file:
        row = file[shard["trajectories"][0]["group"]]
        q, qd, force = [
            torch.tensor(row[key][()])
            for key in ("states/q", "states/qd", "transitions/joint_f")
        ]
        contacts = {
            key: torch.tensor(value[:1])
            for key, value in row["transitions"].items()
            if key.startswith("contact_")
        }
    assert q.shape[0] == force.shape[0] + 1 == 4
    with _NewtonTeacher(robot, num_substeps=2) as teacher:
        assert (
            teacher.manifest["physics_signature"]
            == manifest["metadata"]["physics_signature"]
        )
        teacher.reset(q[:1], qd[:1])
        previous, actual_contacts = teacher.snapshot()
        for key, value in contacts.items():
            torch.testing.assert_close(actual_contacts[key], value)
        teacher.step(force[:1])
        actual, _ = teacher.snapshot()
        torch.testing.assert_close(actual["q"], q[1:2], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(actual["qd"], qd[1:2], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(previous["q"], q[:1], atol=0, rtol=0)
        teacher.reset(q[:1], qd[:1])
        teacher.step(force[:1])
        repeated, _ = teacher.snapshot()
        torch.testing.assert_close(repeated["q"], actual["q"], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("robot", ["cartpole", "ant"])
def test_actual_cli_pipeline_with_dexsim_imports_forbidden(tmp_path, robot):
    """A native SDK or its custom solver extension cannot mask a dependency."""
    script = """
import importlib.abc
import json
from pathlib import Path
import sys

class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'dexsim' or fullname.startswith(('dexsim.', 'embodichain.lab.')):
            raise AssertionError('SDK-free workflow imported ' + fullname)
sys.meta_path.insert(0, NoSDK())
from embodichain.cli.main import main
from embodichain.data_pipeline.dynamics.shards import load_manifest
root, robot = Path(sys.argv[1]), sys.argv[2]
def run(operation, config):
    path = root / (operation + '.json')
    path.write_text(json.dumps(config))
    main(['dynamics', operation, '--config', str(path)])
run('generate', dict(robot=robot, output_dir='data', num_envs=2, trajectories=64,
    horizon=4, num_substeps=2, initial_states='default', effort_scale=0.01))
import yaml
model_cfg = yaml.safe_load((Path.cwd() / 'extensions/embodichain_tasks/configs/tasks/classic_control' / robot / 'dynamics.train.smoke.yaml').read_text())['model']
model_cfg.update(n_embd=12, history_length=2, head_hidden_dims=[12])
run('train', dict(dataset='data', output_dir='train',
    model=model_cfg,
    train=dict(total_steps=2, batch_size=2, validation_interval=1, checkpoint_interval=1,
               device='cpu')))
main(['dynamics', 'export', '--checkpoint', str(root/'train/best.pt'),
      '--output', str(root/'export')])
run('eval', dict(dataset='data', model='export', split='test', device='cpu',
                output='reports/evaluation.json'))
report=json.loads((root/'reports/evaluation.json').read_text())
assert report['windows'] > 0
assert not any(name == 'dexsim' or name.startswith('embodichain.lab.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), robot],
        capture_output=True,
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize(
    "robot,substeps,profile",
    [
        ("cartpole", 5, "warp_reference"),
        ("ant", 16, "warp_reference"),
        ("cartpole", 10, "native"),
    ],
)
def test_cuda_graph_and_direct_teacher_replay(robot, substeps, profile):
    if not torch.cuda.is_available():
        pytest.skip("CUDA teacher required")
    q, qd, effort = _sample(robot, 5, 3, "random_effort", "default", 0.01)
    q, qd, effort = q[None].cuda(), qd[None].cuda(), effort.cuda()
    trajectories = []
    for capture in (False, True):
        with _NewtonTeacher(
            robot,
            device="cuda:0",
            num_substeps=substeps,
            physics_profile=profile,
            use_cuda_graph=capture,
        ) as teacher:
            assert (teacher.graph is not None) is capture
            teacher.reset(q, qd)
            rows = []
            for force in effort:
                teacher.step(force[None])
                state, _ = teacher.snapshot()
                rows.append(torch.cat((state["q"], state["qd"]), -1))
            trajectories.append(torch.stack(rows))
            teacher.reset(q, qd)
            teacher.step(effort[:1])
            repeated, _ = teacher.snapshot()
            torch.testing.assert_close(
                torch.cat((repeated["q"], repeated["qd"]), -1),
                rows[0],
                atol=1e-6,
                rtol=1e-5,
            )
    torch.testing.assert_close(trajectories[0], trajectories[1], atol=1e-5, rtol=1e-4)
