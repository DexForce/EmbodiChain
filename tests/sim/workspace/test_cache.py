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

import argparse

import numpy as np
import pytest
import torch

from embodichain.lab.sim.workspace.caches.results_cache import (
    ResultsCache,
    compute_cache_key,
    deserialize_results,
    serialize_results,
)

# ---------------------------------------------------------------------------
# Pure cache tests (no simulation)
# ---------------------------------------------------------------------------


def _joint_results(n: int = 10, j: int = 3) -> dict:
    return {
        "mode": "joint_space",
        "workspace_points": torch.rand(n, 3),
        "joint_configurations": torch.rand(n, j),
        "num_samples": n,
        "num_valid": n,
        "metrics": {
            "bounding_box": {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
            "bounding_box_volume": 1.0,
            "centroid": [0.5, 0.5, 0.5],
            "dimensions": [1.0, 1.0, 1.0],
        },
        "analysis_time": 1.23,
        "constraint_statistics": {
            "num_points": n,
            "bounds_pass_rate": 100.0,
            "overall_pass_rate": 100.0,
            "collision_pass_rate": 100.0,
        },
    }


def _cartesian_results(
    n: int = 12, m: int = 5, j: int = 3, mode: str = "cartesian_space"
) -> dict:
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:m] = True
    return {
        "mode": mode,
        "all_points": torch.rand(n, 3),
        "workspace_points": torch.rand(n, 3),
        "reachable_points": torch.rand(m, 3),
        "joint_configurations": torch.rand(m, j),
        "success_rates": torch.rand(n),
        "reachability_mask": mask,
        "num_samples": n,
        "num_reachable": m,
        "metrics": {"bounding_box_volume": 2.0},
        "analysis_time": 4.56,
        "constraint_statistics": {"all_points": {"overall_pass_rate": 50.0}},
    }


@pytest.mark.parametrize(
    "results_factory,mode",
    [
        (_joint_results, "joint_space"),
        (_cartesian_results, "cartesian_space"),
        (lambda: _cartesian_results(mode="plane_sampling"), "plane_sampling"),
    ],
)
def test_results_cache_roundtrip(tmp_path, results_factory, mode):
    """save -> load preserves all tensor and scalar fields."""
    results = results_factory()
    cache = ResultsCache(tmp_path / "wa_cache")
    key = compute_cache_key({"mode": mode, "num_samples": results["num_samples"]})
    assert not cache.exists(key)

    entry = cache.save(key, results, metadata={"mode": mode})
    assert cache.exists(key)
    assert (entry / "results.npz").is_file()
    assert (entry / "meta.json").is_file()

    loaded = cache.load(key)
    assert loaded is not None
    assert loaded["mode"] == mode

    # Tensor fields round-trip with matching values and dtype.
    for field in ("workspace_points", "joint_configurations"):
        assert torch.allclose(loaded[field], results[field])

    if mode in ("cartesian_space", "plane_sampling"):
        assert torch.allclose(loaded["all_points"], results["all_points"])
        assert torch.allclose(loaded["reachable_points"], results["reachable_points"])
        assert torch.allclose(loaded["success_rates"], results["success_rates"])
        assert loaded["reachability_mask"].dtype == torch.bool
        assert torch.equal(loaded["reachability_mask"], results["reachability_mask"])

    # Scalar fields preserved.
    assert loaded["num_samples"] == results["num_samples"]
    assert loaded["analysis_time"] == results["analysis_time"]
    assert (
        loaded["metrics"]["bounding_box_volume"]
        == results["metrics"]["bounding_box_volume"]
    )


def test_results_cache_load_missing_returns_none(tmp_path):
    """Loading a non-existent key returns None."""
    cache = ResultsCache(tmp_path / "wa_cache")
    assert cache.load("nonexistent_key") is None


def test_serialize_deserialize_plane_config(tmp_path):
    """Plane-sampling config is preserved through JSON serialization."""
    results = _cartesian_results(mode="plane_sampling")
    results["plane_sampling_config"] = {
        "plane_normal": torch.tensor([0.0, 0.0, 1.0]),
        "plane_point": torch.tensor([0.0, 0.0, 1.2]),
        "plane_bounds": None,
    }
    arrays, meta = serialize_results(results)
    back = deserialize_results(arrays, meta)
    np.testing.assert_allclose(
        back["plane_sampling_config"]["plane_normal"], [0.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(
        back["plane_sampling_config"]["plane_point"], [0.0, 0.0, 1.2]
    )


def test_compute_cache_key_stability_and_sensitivity():
    """Keys are readable and remain stable/sensitive to complete inputs."""
    base = {
        "mode": "joint_space",
        "num_samples": 100,
        "robot": {
            "name": "URRobot",
            "parameters": {"robot_type": "ur5"},
            "control_part": "arm",
            "fpath": "/a/b.urdf",
            "joint_names": ["j1", "j2"],
        },
        "sampling": {"strategy": "sobol", "seed": 42},
    }
    k1 = compute_cache_key(base)
    k2 = compute_cache_key(dict(base))
    assert k1 == k2  # stable / order-independent
    assert k1.startswith(
        "urrobot__robot_type-ur5__part-arm__mode-joint_space__"
        "sampler-sobol__samples-100__seed-42__"
    )

    assert compute_cache_key({**base, "num_samples": 200}) != k1
    assert compute_cache_key({**base, "mode": "cartesian_space"}) != k1
    assert (
        compute_cache_key({**base, "robot": {**base["robot"], "fpath": "/a/c.urdf"}})
        != k1
    )
    # Key insertion order must not matter.
    reordered = {
        "sampling": {"seed": 42, "strategy": "sobol"},
        "robot": {
            "joint_names": ["j1", "j2"],
            "fpath": "/a/b.urdf",
            "control_part": "arm",
            "parameters": {"robot_type": "ur5"},
            "name": "URRobot",
        },
        "num_samples": 100,
        "mode": "joint_space",
    }
    assert compute_cache_key(reordered) == k1


# ---------------------------------------------------------------------------
# Preview-cache helpers (no simulation)
# ---------------------------------------------------------------------------


def _write_cache_entry(tmp_path, results, mode):
    from embodichain.lab.sim.workspace.caches import (
        ResultsCache,
        compute_cache_key,
    )

    cache = ResultsCache(tmp_path / "robot_workspace")
    key = compute_cache_key({"mode": mode, "num_samples": results["num_samples"]})
    entry = cache.save(key, results, metadata={"mode": mode})
    return cache.cache_dir, key, entry


def test_load_preview_data_from_dir(tmp_path):
    """Loading from a cache entry directory returns arrays + mode."""
    from embodichain.lab.scripts.analyze_workspace import _load_preview_data

    results = _joint_results()
    cache_dir, _key, entry = _write_cache_entry(tmp_path, results, "joint_space")
    arrays, mode = _load_preview_data(str(entry), str(cache_dir))
    assert mode == "joint_space"
    assert "workspace_points" in arrays
    assert arrays["workspace_points"].shape[0] == results["num_samples"]


def test_load_preview_data_from_npz_file(tmp_path):
    """Loading directly from a results.npz file works (meta from sibling)."""
    from embodichain.lab.scripts.analyze_workspace import _load_preview_data

    results = _cartesian_results()
    cache_dir, _key, entry = _write_cache_entry(tmp_path, results, "cartesian_space")
    arrays, mode = _load_preview_data(str(entry / "results.npz"), str(cache_dir))
    assert mode == "cartesian_space"
    assert "reachable_points" in arrays
    assert "reachability_mask" in arrays


def test_load_preview_data_from_key(tmp_path):
    """A bare cache key is resolved under the cache dir."""
    from embodichain.lab.scripts.analyze_workspace import _load_preview_data

    results = _joint_results()
    cache_dir, key, _entry = _write_cache_entry(tmp_path, results, "joint_space")
    arrays, mode = _load_preview_data(key, str(cache_dir))
    assert mode == "joint_space"
    assert "workspace_points" in arrays


def test_load_preview_data_missing_raises(tmp_path):
    """An unknown path/key raises FileNotFoundError."""
    from embodichain.lab.scripts.analyze_workspace import _load_preview_data

    with pytest.raises(FileNotFoundError):
        _load_preview_data("nonexistent_key", str(tmp_path / "robot_workspace"))


def test_preview_colors_joint_all_green():
    """Joint-space preview colors all points green."""
    from embodichain.lab.scripts.analyze_workspace import _preview_points_and_colors

    arrays = {"workspace_points": np.random.rand(5, 3)}
    points, colors = _preview_points_and_colors(arrays, "joint_space", False)
    assert points.shape == (5, 3)
    assert np.allclose(colors, np.array([[0.0, 1.0, 0.0]] * 5))


def test_preview_colors_cartesian_mask():
    """Cartesian preview colors reachable green, unreachable red."""
    from embodichain.lab.scripts.analyze_workspace import _preview_points_and_colors

    arrays = {
        "workspace_points": np.random.rand(3, 3),
        "reachability_mask": np.array([True, False, True]),
        "reachable_points": np.random.rand(2, 3),
    }
    _points, colors = _preview_points_and_colors(arrays, "cartesian_space", False)
    assert np.allclose(colors[0], [0.0, 1.0, 0.0])  # reachable green
    assert np.allclose(colors[1], [1.0, 0.0, 0.0])  # unreachable red
    assert np.allclose(colors[2], [0.0, 1.0, 0.0])


def test_preview_colors_hide_unreachable():
    """hide_unreachable shows only reachable points (green)."""
    from embodichain.lab.scripts.analyze_workspace import _preview_points_and_colors

    arrays = {
        "workspace_points": np.random.rand(3, 3),
        "reachability_mask": np.array([True, False, True]),
        "reachable_points": np.random.rand(2, 3),
    }
    points, colors = _preview_points_and_colors(arrays, "cartesian_space", True)
    assert points.shape == (2, 3)
    assert np.allclose(colors, np.array([[0.0, 1.0, 0.0]] * 2))


def test_parse_args_preview_cache_requires_robot_source():
    """--preview-cache is combined with the robot source for in-sim preview."""
    from embodichain.lab.scripts.analyze_workspace import parse_args

    a = parse_args(["--robot", "franka_panda", "--preview-cache", "/tmp/foo"])
    assert a.preview_cache == "/tmp/foo"
    assert a.asset is None and a.robot == "franka_panda"

    with pytest.raises(SystemExit):
        parse_args(["--preview-cache", "/tmp/foo"])


def test_parse_args_and_sim_cfg_enable_viser_workspace():
    """The workspace CLI wires standard Viser arguments into simulation."""
    from embodichain.lab.scripts.analyze_workspace import (
        build_analyzer_config,
        build_sim_cfg,
        parse_args,
    )

    args = parse_args(
        [
            "--robot",
            "franka_panda",
            "--viser",
            "--viser-port",
            "9000",
            "--viser-point-size",
            "0.02",
        ]
    )
    sim_cfg = build_sim_cfg(args)
    analyzer_cfg = build_analyzer_config(args, "arm")

    assert sim_cfg.headless
    assert sim_cfg.visualization.backend == "viser"
    assert sim_cfg.visualization.viser_server.port == 9000
    assert analyzer_cfg.visualization.enabled
    assert analyzer_cfg.visualization.viser_point_size == 0.02


def test_preview_cache_uses_embodichain_sim_backend(tmp_path):
    """Cached points are restored and rendered in the robot simulation."""
    from embodichain.lab.scripts.analyze_workspace import preview_cache

    results = _cartesian_results()
    cache_dir, _key, entry = _write_cache_entry(tmp_path, results, "cartesian_space")

    class AnalyzerStub:
        restored_results = None
        visualize_kwargs = None

        def _restore_analysis_state(self, restored_results):
            self.restored_results = restored_results

        def visualize(self, **kwargs):
            self.visualize_kwargs = kwargs

    analyzer = AnalyzerStub()
    args = argparse.Namespace(
        preview_cache=str(entry),
        cache_dir=str(cache_dir),
        hide_unreachable=False,
        vis_type="point_cloud",
        viser=False,
    )
    preview_cache(args, analyzer)

    assert analyzer.restored_results["mode"] == "cartesian_space"
    assert isinstance(analyzer.restored_results["workspace_points"], torch.Tensor)
    assert analyzer.visualize_kwargs == {
        "vis_type": "point_cloud",
        "show": False,
        "backend": "sim_manager",
    }


def test_preview_cache_uses_viser_backend(tmp_path):
    """Cached points use the Viser backend when requested."""
    from embodichain.lab.scripts.analyze_workspace import preview_cache

    results = _cartesian_results()
    cache_dir, _key, entry = _write_cache_entry(tmp_path, results, "cartesian_space")

    class AnalyzerStub:
        restored_results = None
        visualize_kwargs = None

        def _restore_analysis_state(self, restored_results):
            self.restored_results = restored_results

        def visualize(self, **kwargs):
            self.visualize_kwargs = kwargs

    analyzer = AnalyzerStub()
    args = argparse.Namespace(
        preview_cache=str(entry),
        cache_dir=str(cache_dir),
        hide_unreachable=False,
        vis_type="point_cloud",
        viser=True,
    )
    preview_cache(args, analyzer)

    assert analyzer.visualize_kwargs == {
        "vis_type": "point_cloud",
        "show": False,
        "backend": "viser",
    }


def test_point_cloud_visualizer_publishes_persistent_viser_overlay():
    """Workspace colors and points are published through SimulationManager."""
    from embodichain.lab.sim.workspace.visualizers import (
        PointCloudVisualizer,
    )

    class SimStub:
        overlays = None

        def set_visualization_overlays(self, overlays):
            self.overlays = overlays

    sim = SimStub()
    visualizer = PointCloudVisualizer(
        backend="viser",
        point_size=0.02,
        sim_manager=sim,
        control_part_name="arm",
    )
    points = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32)
    colors = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

    overlay = visualizer.visualize(points, colors)

    assert sim.overlays.point_clouds == (overlay,)
    assert overlay.overlay_id == "workspace_arm"
    assert overlay.point_size == 0.02
    np.testing.assert_array_equal(
        overlay.colors,
        np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8),
    )


def test_point_cloud_visualizer_uses_sim_manager_point_cloud_api():
    """The native workspace backend delegates point-cloud creation to the manager."""
    from embodichain.lab.sim.workspace.visualizers import (
        PointCloudVisualizer,
    )

    class SimStub:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.handle = object()

        def visualize_point_cloud(self, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return self.handle

    sim = SimStub()
    visualizer = PointCloudVisualizer(
        backend="sim_manager",
        point_size=0.02,
        sim_manager=sim,
        control_part_name="arm",
    )
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    colors = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

    assert visualizer.visualize(points, colors) is sim.handle
    assert len(sim.calls) == 1
    call = sim.calls[0]
    np.testing.assert_array_equal(call["points"], points)
    np.testing.assert_array_equal(call["colors"], colors)
    assert call["point_size"] == 0.02
    assert call["name"] == "workspace_pcd_arm"


def test_workspace_analyzer_prefers_configured_viser_backend():
    """Automatic workspace visualization selects Viser before local backends."""
    from types import SimpleNamespace

    from embodichain.lab.sim.workspace.analyzer import (
        WorkspaceAnalyzer,
    )

    analyzer = object.__new__(WorkspaceAnalyzer)
    analyzer.sim_manager = SimpleNamespace(
        sim_config=SimpleNamespace(visualization=SimpleNamespace(backend="viser"))
    )

    assert analyzer._get_backend_priority_list() == [
        "viser",
        "open3d",
        "matplotlib",
    ]


# ---------------------------------------------------------------------------
# CLI config builder tests (no simulation)
# ---------------------------------------------------------------------------


def _robot_ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        asset="/tmp/panda.urdf",
        robot=None,
        robot_params=None,
        urdf=None,
        ee_link="fr3_hand_tcp",
        joints="fr3_joint[1-7]",
        root_link="base",
        control_part="arm",
        solver="pytorch",
        tcp=None,
        uid=None,
        init_pos=[0.0, 0.0, 0.0],
        init_rot=[0.0, 0.0, 0.0],
        fix_base=True,
        use_usd_properties=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _analyzer_ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        mode="cartesian_space",
        num_samples=500,
        seed=42,
        batch_size=1000,
        sampler="sobol",
        visualize=True,
        headless=False,
        vis_type="point_cloud",
        point_size=4.0,
        voxel_size=0.05,
        hide_unreachable=False,
        no_cache=False,
        cache_dir="/tmp/wa_results",
        bounds=[-0.5, 0.5, -0.5, 0.5, 0.0, 1.5],
        joint_limits_scale=1.0,
        ik_samples_per_point=4,
        plane_normal=[0.0, 0.0, 1.0],
        plane_point=[0.0, 0.0, 1.2],
        plane_bounds=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_robot_cfg_urdf_defaults_solver_urdf():
    """For a URDF asset, the solver URDF defaults to the asset path."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    cfg, part, urdf = build_robot_cfg(_robot_ns())
    assert part == "arm"
    assert cfg.fpath == "/tmp/panda.urdf"
    assert urdf == "/tmp/panda.urdf"
    assert cfg.control_parts == {"arm": ["fr3_joint[1-7]"]}
    assert cfg.solver_cfg["arm"].end_link_name == "fr3_hand_tcp"
    assert cfg.solver_cfg["arm"].urdf_path == "/tmp/panda.urdf"


def test_build_robot_cfg_usd_requires_urdf():
    """A USD asset without --urdf raises a clear error."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    with pytest.raises(ValueError, match="USD/non-URDF assets require --urdf"):
        build_robot_cfg(_robot_ns(asset="/tmp/robot.usd"))


def test_build_robot_cfg_usd_with_urdf():
    """A USD asset with --urdf uses the companion URDF for the solver."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    cfg, _part, urdf = build_robot_cfg(
        _robot_ns(asset="/tmp/robot.usd", urdf="/tmp/robot.urdf")
    )
    assert cfg.fpath == "/tmp/robot.usd"
    assert urdf == "/tmp/robot.urdf"
    assert cfg.solver_cfg["arm"].urdf_path == "/tmp/robot.urdf"


def test_build_robot_cfg_asset_requires_ee_link():
    """--asset without --ee-link raises a clear error."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    with pytest.raises(ValueError, match="--ee-link is required when using --asset"):
        build_robot_cfg(_robot_ns(ee_link=None))


def test_build_robot_cfg_asset_control_part_defaults_to_arm():
    """--asset without --control-part defaults the part name to 'arm'."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    cfg, part, _urdf = build_robot_cfg(_robot_ns(control_part=None))
    assert part == "arm"
    assert "arm" in cfg.control_parts


def test_build_preset_robot_cfg_franka():
    """--robot franka_panda loads the preset control parts (no ee-link/joints)."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg
    from embodichain.lab.sim.robots import FrankaPandaCfg

    cfg, part, urdf = build_robot_cfg(
        _robot_ns(
            asset=None,
            robot="franka_panda",
            ee_link=None,
            joints=None,
            control_part=None,
        )
    )
    assert isinstance(cfg, FrankaPandaCfg)
    assert "arm" in cfg.control_parts and "hand" in cfg.control_parts
    # Solver + ee-link come from the preset, not the CLI.
    assert cfg.solver_cfg["arm"].end_link_name == "fr3_hand_tcp"
    assert part is None  # auto-resolved later by _resolve_control_part
    assert urdf is None


def test_build_preset_robot_cfg_cobotmagic_control_part_and_params():
    """--robot cobotmagic honors --control-part and --robot-params overrides."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg
    from embodichain.lab.sim.robots import CobotMagicCfg

    cfg, part, _urdf = build_robot_cfg(
        _robot_ns(
            asset=None,
            robot="cobotmagic",
            ee_link=None,
            joints=None,
            control_part="right_arm",
            robot_params='{"uid": "CM2"}',
        )
    )
    assert isinstance(cfg, CobotMagicCfg)
    assert cfg.uid == "CM2"  # --robot-params override
    assert part == "right_arm"
    assert "right_arm" in cfg.control_parts
    assert cfg.control_parts["right_arm"][0] == "right_joint1"


def test_build_preset_robot_cfg_invalid_control_part():
    """An unknown --control-part for a preset raises a clear error."""
    from embodichain.lab.scripts.analyze_workspace import build_robot_cfg

    with pytest.raises(ValueError, match="not a control part of robot"):
        build_robot_cfg(
            _robot_ns(
                asset=None, robot="franka_panda", ee_link=None, control_part="nope"
            )
        )


def test_build_analyzer_config_cartesian_with_bounds():
    """Cartesian config wires bounds, sampler, ik_samples and cache_dir."""
    from embodichain.lab.scripts.analyze_workspace import build_analyzer_config
    from embodichain.lab.sim.workspace.analyzer import (
        AnalysisMode,
    )

    cfg = build_analyzer_config(_analyzer_ns(), "arm")
    assert cfg.mode == AnalysisMode.CARTESIAN_SPACE
    assert cfg.ik_samples_per_point == 4
    assert cfg.control_part_name == "arm"
    assert cfg.sampling.strategy.value == "sobol"
    assert cfg.cache.cache_dir == "/tmp/wa_results"
    # bounds [XMIN XMAX YMIN YMAX ZMIN ZMAX] -> min/max split.
    np.testing.assert_allclose(cfg.constraint.min_bounds, [-0.5, -0.5, 0.0])
    np.testing.assert_allclose(cfg.constraint.max_bounds, [0.5, 0.5, 1.5])


def test_build_analyzer_config_plane_sets_plane_params():
    """Plane mode populates plane_normal / plane_point tensors."""
    from embodichain.lab.scripts.analyze_workspace import build_analyzer_config
    from embodichain.lab.sim.workspace.analyzer import (
        AnalysisMode,
    )

    cfg = build_analyzer_config(_analyzer_ns(mode="plane_sampling", bounds=None), "arm")
    assert cfg.mode == AnalysisMode.PLANE_SAMPLING
    assert torch.allclose(cfg.plane_normal, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(cfg.plane_point, torch.tensor([0.0, 0.0, 1.2]))


# ---------------------------------------------------------------------------
# Integration: cache hit with a real simulation
# ---------------------------------------------------------------------------


def _make_cobotmagic_sim(tmp_path):
    """Build a headless sim + CobotMagic robot for integration tests."""
    from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
    from embodichain.lab.sim.robots import CobotMagicCfg

    config = SimulationManagerCfg(headless=True, sim_device="cpu")
    sim = SimulationManager(config)
    sim.set_manual_update(False)

    cfg_dict = {
        "uid": "CobotMagic",
        "init_pos": [0.0, 0.0, 0.7775],
        "init_qpos": [
            -0.3,
            0.3,
            1.0,
            1.0,
            -1.2,
            -1.2,
            0.0,
            0.0,
            0.6,
            0.6,
            0.0,
            0.0,
            0.05,
            0.05,
            0.05,
            0.05,
        ],
        "solver_cfg": {
            "left_arm": {
                "class_type": "OPWSolver",
                "end_link_name": "left_link6",
                "root_link_name": "left_arm_base",
                "tcp": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.143], [0, 0, 0, 1]],
            },
        },
    }
    robot = sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))
    return sim, robot


class TestWorkspaceResultsCacheIntegration:
    """Cache write/hit against the real analyzer (headless)."""

    def setup_method(self):
        self.sim = None

    def teardown_method(self):
        if self.sim is not None:
            try:
                self.sim.destroy()
            except Exception:
                pass
            from embodichain.lab.sim.sim_manager import SimulationManager

            SimulationManager.flush_cleanup_queue()

    def test_cache_hit_skips_recompute(self, tmp_path):
        """A second analyze() with force_recompute=False loads from cache."""
        from embodichain.lab.sim.workspace.configs import (
            SamplingConfig,
            VisualizationConfig,
        )
        from embodichain.lab.sim.workspace.analyzer import (
            AnalysisMode,
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
        )

        self.sim, robot = _make_cobotmagic_sim(tmp_path)

        config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=80),
            visualization=VisualizationConfig(enabled=False),
            cache=_cache_cfg(tmp_path),
            control_part_name="left_arm",
        )
        analyzer = WorkspaceAnalyzer(robot=robot, config=config, sim_manager=self.sim)

        first = analyzer.analyze(num_samples=80, visualize=False)
        assert first["workspace_points"].shape[0] > 0
        cache_path = analyzer.get_results_cache_path()
        assert cache_path is not None and cache_path.exists()
        assert cache_path.name.startswith(
            "cobotmagic__part-left_arm__mode-joint_space__"
        )

        # Force the recompute path to fail; a cache hit must not reach it.
        def _boom(*_args, **_kwargs):
            raise AssertionError("sample_joint_space should not run on a cache hit")

        analyzer.sample_joint_space = _boom  # type: ignore[assignment]

        second = analyzer.analyze(
            num_samples=80, force_recompute=False, visualize=False
        )
        assert second["mode"] == "joint_space"
        # Cached points are returned verbatim.
        assert torch.equal(second["workspace_points"], first["workspace_points"])
        assert analyzer.get_results_cache_path() == cache_path

    def test_force_recompute_ignores_cache(self, tmp_path):
        """force_recompute=True recomputes even when a cache entry exists."""
        from embodichain.lab.sim.workspace.configs import (
            SamplingConfig,
            VisualizationConfig,
        )
        from embodichain.lab.sim.workspace.analyzer import (
            AnalysisMode,
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
        )

        self.sim, robot = _make_cobotmagic_sim(tmp_path)

        config = WorkspaceAnalyzerConfig(
            mode=AnalysisMode.JOINT_SPACE,
            sampling=SamplingConfig(num_samples=60),
            visualization=VisualizationConfig(enabled=False),
            cache=_cache_cfg(tmp_path),
            control_part_name="left_arm",
        )
        analyzer = WorkspaceAnalyzer(robot=robot, config=config, sim_manager=self.sim)

        analyzer.analyze(num_samples=60, visualize=False)
        assert analyzer.get_results_cache_path() is not None

        # With force_recompute=True, sampling must run (no early cache return).
        called = {"count": 0}
        original = analyzer.sample_joint_space

        def _spy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        analyzer.sample_joint_space = _spy  # type: ignore[assignment]
        analyzer.analyze(num_samples=60, force_recompute=True, visualize=False)
        assert called["count"] == 1

    def test_different_params_use_different_cache_keys(self, tmp_path):
        """Changing num_samples produces a different cache entry."""
        from embodichain.lab.sim.workspace.configs import (
            SamplingConfig,
            VisualizationConfig,
        )
        from embodichain.lab.sim.workspace.analyzer import (
            AnalysisMode,
            WorkspaceAnalyzer,
            WorkspaceAnalyzerConfig,
        )

        self.sim, robot = _make_cobotmagic_sim(tmp_path)

        def run(n):
            config = WorkspaceAnalyzerConfig(
                mode=AnalysisMode.JOINT_SPACE,
                sampling=SamplingConfig(num_samples=n),
                visualization=VisualizationConfig(enabled=False),
                cache=_cache_cfg(tmp_path),
                control_part_name="left_arm",
            )
            analyzer = WorkspaceAnalyzer(
                robot=robot, config=config, sim_manager=self.sim
            )
            analyzer.analyze(num_samples=n, visualize=False)
            return analyzer.get_results_cache_path()

        path_a = run(50)
        path_b = run(70)
        assert path_a is not None and path_b is not None
        assert path_a != path_b


def _cache_cfg(tmp_path):
    from embodichain.lab.sim.workspace.configs import CacheConfig

    return CacheConfig(enabled=True, cache_dir=str(tmp_path / "results"))


# ---------------------------------------------------------------------------
# Integration: full CLI main() with a --robot preset
# ---------------------------------------------------------------------------


@pytest.mark.requires_sim
def test_main_robot_preset_runs_and_caches(tmp_path):
    """The full CLI pipeline works for a --robot preset (headless)."""
    from embodichain.lab.scripts.analyze_workspace import main, parse_args

    cache_dir = tmp_path / "preset_results"
    args = parse_args(
        [
            "--robot",
            "cobotmagic",
            "--control-part",
            "left_arm",
            "--mode",
            "joint_space",
            "--num-samples",
            "50",
            "--sampler",
            "random",
            "--cache-dir",
            str(cache_dir),
            "--headless",
            "--no-visualize",
        ]
    )
    main(args)

    # The run should have written a results cache entry.
    npz_files = list(cache_dir.glob("*/results.npz"))
    assert npz_files, f"no results.npz under {cache_dir}"
    assert npz_files[0].parent.name.startswith(
        "cobotmagic__part-left_arm__mode-joint_space__"
    )
    with np.load(npz_files[0]) as data:
        assert "workspace_points" in data.files
        assert data["workspace_points"].shape[0] > 0


@pytest.mark.requires_sim
def test_main_asset_path_runs_and_caches(tmp_path):
    """The full CLI pipeline works for a --asset URDF (headless, Franka)."""
    from embodichain.data import get_data_path
    from embodichain.lab.scripts.analyze_workspace import main, parse_args

    urdf = get_data_path("Franka/Panda/PandaWithHand.urdf")
    cache_dir = tmp_path / "asset_results"
    args = parse_args(
        [
            "--asset",
            urdf,
            "--ee-link",
            "fr3_hand_tcp",
            "--joints",
            "fr3_joint[1-7]",
            "--root-link",
            "base",
            "--mode",
            "joint_space",
            "--num-samples",
            "50",
            "--sampler",
            "random",
            "--cache-dir",
            str(cache_dir),
            "--headless",
            "--no-visualize",
        ]
    )
    main(args)

    npz_files = list(cache_dir.glob("*/results.npz"))
    assert npz_files, f"no results.npz under {cache_dir}"
    with np.load(npz_files[0]) as data:
        assert "workspace_points" in data.files
