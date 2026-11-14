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

"""Reference motion boundaries, ownership and archive mapping."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from embodichain.data_pipeline.datasets.motion import MotionLibrary


def data():
    qpos = np.zeros((5, 8), np.float32)
    qpos[:, 6] = 1
    qpos[:, 0] = np.arange(5)
    return qpos, np.zeros((5, 7), np.float32)


def test_future_hold_cannot_cross_either_clip_boundary():
    library = MotionLibrary(*data(), [3, 2], ["first", "second"], ["hinge"], 50)
    future = library.future(torch.tensor([2, 3]), torch.tensor([-10, 0, 1, 20]))
    torch.testing.assert_close(
        future["pos"][..., 0],
        torch.tensor([[0.0, 2.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]]),
    )
    assert library.remaining(torch.tensor([0, 2, 3, 4])).tolist() == [2, 0, 1, 0]


def test_scalar_and_batch_queries_own_storage():
    qpos, qvel = data()
    library = MotionLibrary(qpos, qvel, [3, 2], ["a", "b"], ["hinge"], 50)
    qpos[:] = 99
    library.frame(torch.tensor(0))["pos"].fill_(33)
    library.frame(torch.tensor([0, 1]))["q"].fill_(20)
    assert library.frame(torch.tensor([0]))["pos"][0, 0] == 0
    assert library.frame(torch.tensor([0]))["q"][0, 0] == 0


def test_archive_selection_and_explicit_quaternion_order(tmp_path):
    qpos, qvel = data()
    qpos[:, 3:7] = [1, 0, 0, 0]
    path = tmp_path / "motions.npz"
    np.savez(
        path,
        qpos=qpos,
        qvel=qvel,
        lengths=[3, 2],
        starts=[0, 3],
        names=["a", "b"],
        fps=50,
        joint_names=["hinge"],
    )
    library = MotionLibrary.from_npz(
        path, ["hinge"], quaternion_order="wxyz", clips=["b"]
    )
    state = library.frame(torch.tensor([0]))
    assert library.names == ("b",) and library.num_frames == 2
    torch.testing.assert_close(state["quat"], torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
    assert state["pos"][0, 0] == 3
    with pytest.raises(ValueError, match="joint order"):
        MotionLibrary.from_npz(path, ["other"], quaternion_order="wxyz")


@pytest.mark.parametrize("source_frames, target_frames", [(70, 116), (71, 117)])
def test_csv_resampling_constant_translation_and_rotation(
    tmp_path, source_frames, target_frames
):
    times = np.arange(source_frames, dtype=np.float32) / 30.0
    qpos = np.zeros((source_frames, 8), dtype=np.float32)
    qpos[:, 0] = times
    qpos[:, 5] = np.sin(0.3 * times / 2)
    qpos[:, 6] = np.cos(0.3 * times / 2)
    qpos[::2, 3:7] *= 2
    qpos[1::2, 3:7] *= -0.5
    qpos[:, 7] = 2 * times
    source = tmp_path / "turn.csv"
    np.savetxt(source, qpos, delimiter=",")
    library = MotionLibrary.from_csv([source], ["joint"])
    assert library.num_frames == target_frames
    state = library.frame(torch.arange(target_frames))
    assert state["pos"][-1, 0] == pytest.approx((target_frames - 1) / 50)
    torch.testing.assert_close(
        state["v"],
        torch.tensor([[1.0, 0.0, 0.0]]).expand(target_frames, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        state["w"],
        torch.tensor([[0.0, 0.0, 0.3]]).expand(target_frames, 3),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        state["qd"], torch.full((target_frames, 1), 2.0), atol=2e-5, rtol=1e-5
    )
    archive = tmp_path / "portable.npz"
    library.save_npz(archive)
    loaded = MotionLibrary.from_npz(archive, ["joint"], quaternion_order="xyzw")
    torch.testing.assert_close(
        loaded.frame(torch.arange(target_frames))["rot"], state["rot"]
    )
    with pytest.raises(ValueError, match="quaternion convention"):
        MotionLibrary.from_npz(archive, ["joint"], quaternion_order="wxyz")


def test_motion_start_sampling_merges_duplicate_returns_and_restores_distribution():
    from embodichain.data_pipeline.datasets.motion_sampling import (
        MotionStartSampler,
        MotionStartSamplerCfg,
    )

    library = MotionLibrary(*data(), [3, 2], ["a", "b"], ["hinge"], 50)
    sampler = MotionStartSampler(library, MotionStartSamplerCfg(probability_cap=1.0))
    sampler.update(torch.tensor([0, 1, 3]), torch.tensor([0.2, 0.6, 0.9]))
    torch.testing.assert_close(sampler.scores, torch.tensor([0.4, 0.9]))
    torch.testing.assert_close(sampler.visits, torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(sampler.probabilities(), torch.tensor([0.5, 0.5]))
    sampler.update(torch.tensor([0]), torch.tensor([1.0]))
    assert sampler.scores[0] == pytest.approx(0.46)
    restored = MotionStartSampler(library, sampler.cfg)
    restored.load_state_dict(sampler.state_dict())
    first = sampler.sample(100, torch.Generator().manual_seed(7))
    second = restored.sample(100, torch.Generator().manual_seed(7))
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert set(first.tolist()) <= {0, 1, 3}


@pytest.mark.parametrize("change", ["qpos", "qvel", "missing_identity"])
def test_sampler_rejects_changed_motion_before_mutating_state(tmp_path, change):
    from copy import deepcopy
    from embodichain.data_pipeline.datasets.motion_sampling import (
        MotionStartSampler,
        MotionStartSamplerCfg,
    )

    path = tmp_path / "motions.npz"
    qpos, qvel = data()

    def load():
        np.savez(path, qpos=qpos, qvel=qvel, lengths=[3, 2], names=["a", "b"], fps=50)
        return MotionLibrary.from_npz(
            path, ["hinge"], quaternion_order="xyzw", clips=["b"]
        )

    original = load()
    source = MotionStartSampler(original, MotionStartSamplerCfg())
    source.update(torch.tensor([0]), torch.tensor([0.7]))
    saved = source.state_dict()
    compatible = MotionStartSampler(load(), source.cfg)
    compatible.load_state_dict(saved)
    torch.testing.assert_close(compatible.scores, source.scores, rtol=0, atol=0)
    # Unselected content has no effect on the selected library identity.
    qpos[0, 7] += 0.5
    assert load().content_digest == original.content_digest
    if change == "qpos":
        qpos[3, 7] += 0.5
    elif change == "qvel":
        qvel[3, 6] += 0.5
    else:
        saved.pop("content_digest")
    restored = MotionStartSampler(load(), source.cfg)
    before = deepcopy(restored.state_dict())
    with pytest.raises(ValueError, match="content differs"):
        restored.load_state_dict(saved)
    for name in ("scores", "visits", "past"):
        torch.testing.assert_close(
            getattr(restored, name), before[name], rtol=0, atol=0
        )
