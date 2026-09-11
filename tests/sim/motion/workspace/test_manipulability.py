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
"""Per-point Yoshikawa manipulability in workspace analysis."""

from __future__ import annotations

import numpy as np
import torch

from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.objects import Robot
from embodichain.lab.sim.robots import CobotMagicCfg
from embodichain.lab.sim.motion.workspace.analyzer import (
    AnalysisMode,
    WorkspaceAnalyzer,
    WorkspaceAnalyzerConfig,
)
from embodichain.lab.sim.motion.workspace.caches.results_cache import (
    deserialize_results,
    serialize_results,
)
from embodichain.lab.sim.motion.workspace.configs import (
    MetricConfig,
    MetricType,
    SamplingConfig,
)
from embodichain.lab.sim.motion.workspace.metrics.manipulability_metric import (
    ManipulabilityMetric,
)


class TestManipulabilityMetricUnit:
    """Metric-class behaviour without a simulation."""

    def test_exact_yoshikawa_from_jacobians(self):
        # J = [I3 | 0] gives J @ J^T = I -> w = 1 exactly.
        jac = np.zeros((5, 6, 7))
        for i in range(5):
            jac[i, :6, :6] = np.eye(6)
        metric = ManipulabilityMetric()
        out = metric.compute(np.zeros((5, 3)), jacobians=jac)
        assert abs(out["mean_manipulability"] - 1.0) < 1e-9
        assert out["num_valid_points"] == 5

    def test_precomputed_scores_take_precedence(self):
        metric = ManipulabilityMetric()
        scores = np.array([0.2, 0.4, 0.6])
        out = metric.compute(np.zeros((3, 3)), manipulability_scores=scores)
        assert abs(out["mean_manipulability"] - 0.4) < 1e-9
        assert abs(out["min_manipulability"] - 0.2) < 1e-9

    def test_no_inputs_produces_no_fabricated_statistics(self):
        """The centroid-distance placeholder was anti-correlated with truth
        (measured corr = -0.37 on Franka) and must stay removed."""
        metric = ManipulabilityMetric()
        out = metric.compute(np.random.rand(100, 3))
        assert out == {}

    def test_precomputed_condition_numbers_feed_isotropy(self):
        metric = ManipulabilityMetric()
        out = metric.compute(
            np.zeros((3, 3)),
            manipulability_scores=np.array([0.5, 0.5, 0.5]),
            condition_numbers=np.array([2.0, 4.0, 6.0]),
        )
        assert abs(out["mean_condition"] - 4.0) < 1e-9


class TestAnalyzerManipulability:
    """End-to-end score plumbing on the library CobotMagic robot."""

    def setup_method(self):
        config = SimulationManagerCfg(headless=True, sim_device="cpu")
        self.sim = SimulationManager(config)
        cfg_dict = {
            "uid": "CobotMagic",
            "solver_cfg": {
                "left_arm": {
                    "class_type": "OPWSolver",
                    "end_link_name": "left_link6",
                    "root_link_name": "left_arm_base",
                    "tcp": [
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0.143],
                        [0, 0, 0, 1],
                    ],
                },
            },
        }
        self.robot: Robot = self.sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    def teardown_method(self):
        self.sim.destroy()
        SimulationManager.flush_cleanup_queue()

    def _analyze(self, metric_cfg: MetricConfig | None = None):
        cfg = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=100),
            control_part_name="left_arm",
            metric=metric_cfg,
        )
        analyzer = WorkspaceAnalyzer(robot=self.robot, config=cfg, sim_manager=self.sim)
        return analyzer.analyze(num_samples=100, force_recompute=True), analyzer

    def test_scores_aligned_and_positive(self):
        results, analyzer = self._analyze()
        scores = results.get("manipulability_scores")
        assert scores is not None
        assert scores.shape == (len(results["joint_configurations"]),)
        assert bool((scores >= 0).all())
        assert float(scores.mean()) > 0.0
        assert analyzer.manipulability_scores is not None

    def test_metrics_contain_true_aggregates(self):
        results, _ = self._analyze()
        manip = results["metrics"].get("manipulability")
        assert manip is not None
        assert manip["mean_manipulability"] > 0.0
        assert manip["max_manipulability"] >= manip["min_manipulability"]

    def test_disabled_metric_skips_scores(self):
        results, analyzer = self._analyze(
            MetricConfig(enabled_metrics=[MetricType.REACHABILITY])
        )
        assert "manipulability_scores" not in results
        assert analyzer.manipulability_scores is None

    def test_isotropy_condition_stats_honored(self):
        results, _ = self._analyze()
        manip = results["metrics"]["manipulability"]
        # Default ManipulabilityConfig has compute_isotropy=True; the
        # documented condition statistics must actually be produced.
        assert manip["mean_condition"] >= 1.0
        assert manip["std_condition"] >= 0.0

    def test_cache_hit_repairs_entry_from_different_metric_config(self, tmp_path):
        from embodichain.lab.sim.motion.workspace.configs.cache_config import (
            CacheConfig,
        )

        def run(metric_cfg):
            cfg = WorkspaceAnalyzerConfig(
                mode=AnalysisMode.JOINT_SPACE,
                sampling=SamplingConfig(num_samples=100),
                control_part_name="left_arm",
                metric=metric_cfg,
                cache=CacheConfig(enabled=True, cache_dir=tmp_path),
            )
            analyzer = WorkspaceAnalyzer(
                robot=self.robot, config=cfg, sim_manager=self.sim
            )
            return analyzer.analyze(num_samples=100)

        # First run writes a cache entry WITHOUT manipulability.
        first = run(MetricConfig(enabled_metrics=[MetricType.REACHABILITY]))
        assert "manipulability_scores" not in first

        # Second run (default metrics) hits the same key — metric settings are
        # not part of the cache identity — and must repair the entry on load.
        second = run(None)
        assert "manipulability_scores" in second
        assert second["manipulability_scores"].shape == (
            len(second["joint_configurations"]),
        )
        assert second["metrics"]["manipulability"]["mean_manipulability"] > 0.0
        assert second["metrics"]["manipulability"]["mean_condition"] >= 1.0

    def test_cache_hit_strips_fields_when_metric_disabled(self, tmp_path):
        """Symmetric direction: an enabled run writes a score-bearing entry;
        a later disabled run hitting the same key must strip the fields so
        cached and fresh analyses expose the same result contract."""
        from embodichain.lab.sim.motion.workspace.configs.cache_config import (
            CacheConfig,
        )

        def run(metric_cfg):
            cfg = WorkspaceAnalyzerConfig(
                mode=AnalysisMode.JOINT_SPACE,
                sampling=SamplingConfig(num_samples=100),
                control_part_name="left_arm",
                metric=metric_cfg,
                cache=CacheConfig(enabled=True, cache_dir=tmp_path),
            )
            analyzer = WorkspaceAnalyzer(
                robot=self.robot, config=cfg, sim_manager=self.sim
            )
            return analyzer.analyze(num_samples=100)

        # Enabled run computes scores and persists them in the entry.
        first = run(None)
        assert "manipulability_scores" in first

        # Disabled run hits the score-bearing entry: fields must be stripped.
        second = run(MetricConfig(enabled_metrics=[MetricType.REACHABILITY]))
        assert "manipulability_scores" not in second
        assert "manipulability" not in second.get("metrics", {})

    def test_scores_survive_cache_serialization(self):
        results, _ = self._analyze()
        arrays, meta = serialize_results(results)
        assert "manipulability_scores" in arrays
        restored = deserialize_results(arrays, meta)
        original = results["manipulability_scores"].cpu().numpy()
        np.testing.assert_allclose(
            np.asarray(restored["manipulability_scores"]), original, rtol=1e-6
        )
