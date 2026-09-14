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

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from embodichain.lab.sim.motion.workspace.analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.configs import (
    CacheConfig,
    DimensionConstraint,
    SamplingConfig,
    SamplingStrategy,
)
from embodichain.lab.sim.motion.workspace.samplers import (
    RandomSampler,
    SobolSampler,
    LatinHypercubeSampler,
)
from embodichain.lab.sim.motion.workspace.runtime import RobotWorkspace
from embodichain.lab.sim.motion.workspace.caches.results_cache import ResultsCache


def _analyzer(**kwargs) -> WorkspaceAnalyzer:
    """Use deterministic kinematics to isolate analysis and sample alignment."""
    robot = Mock()
    robot.device = torch.device("cpu")
    robot.num_envs = 3
    robot.control_parts = {"arm": ["j0", "j1", "j2"]}
    robot.get_joint_ids.return_value = [0, 1, 2]
    robot.body_data = SimpleNamespace(qpos_limits=torch.tensor([[[-1.0, 1.0]] * 3]))
    robot.get_qpos.return_value = torch.zeros(3, 3)
    robot.cfg = SimpleNamespace(uid="test", fpath=None, solver_cfg={})
    robot._solvers = {}

    def fk(qpos, **kw):
        pose = torch.eye(4).expand(*qpos.shape[:-1], 4, 4).clone()
        pose[..., :3, 3] = qpos
        return pose

    def ik(pose, joint_seed, **kw):
        # Mix successful and unsuccessful points, independent of batch splitting.
        success = pose[..., 0, 3] >= 0
        return success, pose[..., :3, 3].clone()

    robot.compute_batch_fk.side_effect = fk
    robot.compute_fk.side_effect = fk
    robot.compute_batch_ik.side_effect = ik
    sampling = kwargs.pop("sampling", SamplingConfig(num_samples=64, batch_size=7))
    constraint = kwargs.pop(
        "constraint",
        DimensionConstraint(
            min_bounds=np.array([-1.0, -1.0, 0.0]),
            max_bounds=np.ones(3),
        ),
    )
    cfg = WorkspaceAnalyzerConfig(
        sampling=sampling,
        constraint=constraint,
        cache=CacheConfig(enabled=False),
        control_part_name="arm",
        reference_pose=torch.eye(4),
        **kwargs,
    )
    result = WorkspaceAnalyzer(robot, cfg)
    result._log_analysis_summary = Mock()
    result._create_optimized_tqdm = lambda it, **kw: _Progress(it)
    result._update_progress_with_stats = Mock()
    return result


class _Progress:
    def __init__(self, it):
        self.it = it

    def __iter__(self):
        return iter(self.it)

    def close(self):
        pass


def test_dynamic_bounds_use_one_batch_and_env_zero():
    analyzer = _analyzer()
    expected_q = RandomSampler(seed=42).sample(1000, analyzer.qpos_limits)
    lo, hi = expected_q.amin(0), expected_q.amax(0)
    expected = torch.stack((lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)), 1)
    torch.testing.assert_close(analyzer._compute_dynamic_workspace_bounds(), expected)
    assert analyzer.robot.compute_batch_fk.call_count == 1
    assert analyzer.robot.compute_batch_fk.call_args.kwargs["env_ids"] == [0]
    analyzer.robot.compute_fk.assert_not_called()


@pytest.mark.parametrize("seeds", [1, 4])
def test_reachability_preserves_order_with_filtered_and_tail_batches(seeds):
    analyzer = _analyzer(ik_samples_per_point=seeds)
    points = torch.tensor(
        [[0.2, 0.0, 0.1], [-0.2, 0.0, 0.1], [0.3, 0.0, -1.0], [0.4, 0.0, 0.2]]
    )
    all_points, reachable, scores, mask, qpos = analyzer.compute_reachability(
        points, batch_size=3
    )
    torch.testing.assert_close(all_points, points)
    assert mask.tolist() == [True, False, False, True]
    torch.testing.assert_close(scores, torch.tensor([1.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(reachable, points[[0, 3]])
    torch.testing.assert_close(qpos, reachable)
    calls = analyzer.robot.compute_batch_ik.call_args_list
    assert [c.kwargs["pose"].shape[1] for c in calls] == [2 * seeds, seeds]
    assert all(c.kwargs["env_ids"] == [0] for c in calls)


def test_empty_analysis_entry_points():
    analyzer = _analyzer()
    points, qpos = analyzer.compute_workspace_points(torch.empty(0, 3))
    assert points.shape == qpos.shape == (0, 3)
    assert analyzer.compute_reachability(torch.empty(0, 3))[3].numel() == 0


@pytest.mark.parametrize(
    "sampler_cls", [RandomSampler, SobolSampler, LatinHypercubeSampler]
)
def test_sampler_is_reproducible_without_touching_global_rng(sampler_cls):
    bounds = torch.tensor([[-1.0, 1.0]] * 3)
    global_state = torch.random.get_rng_state().clone()
    sampler = sampler_cls(seed=12)
    first = sampler.sample(num_samples=16, bounds=bounds)
    second = sampler.sample(num_samples=16, bounds=bounds)
    torch.testing.assert_close(torch.random.get_rng_state(), global_state)
    torch.testing.assert_close(
        sampler_cls(seed=12).sample(num_samples=16, bounds=bounds), first
    )
    assert not torch.equal(first, second)
    assert ((first >= -1) & (first <= 1)).all()


def test_sobol_chunks_match_one_draw_and_plane_sampling_works():
    bounds = torch.tensor([[-1.0, 1.0]] * 3)
    sampler = SobolSampler(seed=9)
    chunks = torch.cat([sampler.sample(n, bounds) for n in [7, 9, 16]])
    torch.testing.assert_close(chunks, SobolSampler(seed=9).sample(32, bounds))
    points = _analyzer().sample_plane(
        32, plane_point=torch.tensor([0.0, 0.0, 0.5]), plane_bounds=bounds[:2]
    )
    torch.testing.assert_close(points[:, 2], torch.full((32,), 0.5))


def test_constrained_box_refills_excluded_zone():
    analyzer = _analyzer(
        sample_within_constraints=True,
        constraint_type="box",
        constraint_bounds=torch.tensor([[0.1, 0.8], [-0.5, 0.5], [0.2, 0.8]]),
        constraint=DimensionConstraint(
            min_bounds=[-1, -1, 0],
            max_bounds=[1, 1, 1],
            exclude_zones=[([0.3, -1, 0], [0.6, 1, 1])],
        ),
    )
    points = analyzer.sample_cartesian_space(128)
    assert points.shape == (128, 3)
    assert analyzer._check_constraints(points).all()
    assert ((points[:, 0] < 0.3) | (points[:, 0] > 0.6)).all()


def test_constrained_sphere_has_volume_distribution():
    analyzer = _analyzer(
        sample_within_constraints=True,
        constraint_type="sphere",
        sphere_center=torch.tensor([0.0, 0.0, 0.5]),
        sphere_radius=0.4,
    )
    points = analyzer.sample_cartesian_space(4096)
    radii = torch.linalg.vector_norm(points - torch.tensor([0.0, 0.0, 0.5]), dim=1)
    assert radii.max() <= 0.4 + 1e-6
    assert float(radii.mean()) == pytest.approx(0.4 * 0.75, abs=0.005)


def test_constrained_plane_and_exhaustion():
    analyzer = _analyzer(sample_within_constraints=True)
    bounds = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])
    points = analyzer.sample_plane(
        64, plane_point=torch.tensor([0.0, 0.0, 0.5]), plane_bounds=bounds
    )
    assert analyzer._check_constraints(points).all()
    torch.testing.assert_close(points[:, 2], torch.full((64,), 0.5))
    analyzer.config.max_sampling_rounds = 2
    with pytest.raises(ValueError, match="exhausted"):
        analyzer.sample_plane(
            64, plane_point=torch.tensor([0.0, 0.0, -1.0]), plane_bounds=bounds
        )


@pytest.mark.parametrize(
    "mode", [AnalysisMode.CARTESIAN_SPACE, AnalysisMode.PLANE_SAMPLING]
)
def test_compact_results_cache_and_visualization(tmp_path, mode):
    kwargs = dict(mode=mode, retain_diagnostics=False)
    if mode == AnalysisMode.PLANE_SAMPLING:
        kwargs.update(
            plane_point=torch.tensor([0.0, 0.0, 0.5]),
            plane_bounds=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
        )
    analyzer = _analyzer(**kwargs)
    results = analyzer.analyze(visualize=False)
    assert not {"workspace_points", "all_points", "reachability_mask"} & results.keys()
    assert len(results["success_rates"]) == len(results["joint_configurations"])
    cache = ResultsCache(tmp_path)
    path = cache.save("compact", results, {"mode": mode.value})
    runtime = RobotWorkspace.from_cache(path)
    torch.testing.assert_close(runtime.qpos, results["joint_configurations"])
    restored = cache.load("compact")
    analyzer._restore_analysis_state(restored)
    colors = analyzer._generate_point_colors(analyzer.workspace_points.numpy())
    assert colors.shape == (len(runtime.qpos), 3)
    analyzer.config.visualization.enabled = True
    analyzer.config.visualization.show_unreachable_points = False
    visualizer = Mock()
    analyzer._create_visualizer_with_config = Mock(return_value=visualizer)
    analyzer.visualize(show=False, backend="matplotlib")
    assert visualizer.visualize.call_args.args[0].shape == (len(runtime.qpos), 3)


def test_cache_identity_tracks_new_options_and_exclusions():
    from embodichain.lab.sim.motion.workspace.caches.results_cache import (
        compute_cache_key,
    )

    analyzer = _analyzer()
    key = lambda: compute_cache_key(analyzer._build_cache_key_metadata(64))
    initial = key()
    analyzer.config.retain_diagnostics = False
    assert key() != initial
    compact = key()
    analyzer.config.sample_within_constraints = True
    assert key() != compact
    before_exclusion = key()
    analyzer.config.constraint.exclude_zones = [([0, 0, 0], [1, 1, 1])]
    assert key() != before_exclusion


def test_reference_pose_uses_env_zero_and_accepts_serialized_pose():
    analyzer = _analyzer()
    analyzer.config.reference_pose = None
    pose = analyzer._get_reference_pose()
    assert pose.shape == (1, 4, 4)
    assert analyzer.robot.compute_fk.call_args.kwargs["env_ids"] == [0]
    analyzer.config.reference_pose = torch.eye(4).tolist()
    torch.testing.assert_close(analyzer._get_reference_pose(), torch.eye(4)[None])


def test_cli_exposes_compact_constrained_sobol_defaults():
    from embodichain.lab.scripts.analyze_workspace import (
        parse_args,
        build_analyzer_config,
        _preview_points_and_colors,
    )

    args = parse_args(
        ["--robot", "cobotmagic", "--compact-results", "--sample-within-constraints"]
    )
    cfg = build_analyzer_config(args, "left_arm")
    assert cfg.sampling.strategy == SamplingStrategy.SOBOL
    assert cfg.sample_within_constraints and not cfg.retain_diagnostics
    points = np.zeros((5, 3))
    # Array insertion order must not make preview choose qpos as XYZ.
    xyz, colors = _preview_points_and_colors(
        {"joint_configurations": np.zeros((5, 6)), "reachable_points": points},
        "cartesian_space",
        False,
    )
    assert xyz.shape == colors.shape == (5, 3)
