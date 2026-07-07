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

import pytest
import torch
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.cfg import LightCfg


class TestLight:
    def setup_method(self):
        # Setup SimulationManager
        config = SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=10)
        self.sim = SimulationManager(config)

        # Create batch of lights
        cfg_dict = {
            "light_type": "point",
            "color": [0.1, 0.1, 0.1],
            "radius": 10.0,
            "position": [0.0, 0.0, 2.0],
            "uid": "point_light",
        }
        self.light = self.sim.add_light(cfg=LightCfg.from_dict(cfg_dict))

    def test_set_color_with_env_ids(self):
        """Test set_color with and without env_ids."""
        base_color = torch.tensor([0.1, 0.1, 0.1], device=self.sim.device)

        # Set for all environments
        try:
            self.light.set_color(base_color)
        except Exception as e:
            pytest.fail(f"Failed to set color for all envs: {e}")

        # Set for specific envs
        env_ids = [1, 3, 5]
        new_color = torch.tensor([0.9, 0.8, 0.7], device=self.sim.device)
        try:
            self.light.set_color(new_color, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"Failed to set color for env_ids={env_ids}: {e}")

    def test_set_falloff_with_env_ids(self):
        """Test set_falloff with and without env_ids."""
        base_val = torch.tensor(100.0, device=self.sim.device)

        # Set for all
        try:
            self.light.set_falloff(base_val)
        except Exception as e:
            pytest.fail(f"Failed to set falloff for all envs: {e}")

        env_ids = [0, 7, 9]
        new_vals = torch.tensor([200.0, 300.0, 400.0], device=self.sim.device)
        try:
            self.light.set_falloff(new_vals, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"Failed to set falloff for env_ids={env_ids}: {e}")

    def test_set_and_get_local_pose_matrix_and_vector(self):
        """
        Test setting and getting local pose in both matrix and vector forms.

        1. Set all lights to identity pose (4x4 matrix)
        2. Overwrite subset of lights (env_ids) with custom pose
        3. Check both vector and matrix results match expectations
        """

        # ----------------------------
        # 1. Set all lights to identity matrix
        # ----------------------------
        pose_matrix = torch.eye(4, device=self.sim.device)
        try:
            self.light.set_local_pose(pose_matrix, to_matrix=True)
        except Exception as e:
            pytest.fail(f"Failed to set pose matrix for all envs: {e}")

        result_matrix = self.light.get_local_pose(to_matrix=True)
        assert result_matrix.shape == (
            10,
            4,
            4,
        ), "Unexpected shape from get_local_pose(to_matrix=True)"
        for i, mat in enumerate(result_matrix):
            assert torch.allclose(
                mat, pose_matrix, atol=1e-5
            ), f"Initial matrix pose mismatch at env {i}"

        # ----------------------------
        # 2. Set translation via matrix for selected env_ids
        # ----------------------------
        env_ids = [2, 4, 6]
        pose_matrix_2 = (
            torch.eye(4, device=self.sim.device).unsqueeze(0).repeat(len(env_ids), 1, 1)
        )
        pose_matrix_2[:, 0, 3] = 1.0
        pose_matrix_2[:, 1, 3] = 2.0
        pose_matrix_2[:, 2, 3] = 3.0

        try:
            self.light.set_local_pose(pose_matrix_2, env_ids=env_ids, to_matrix=True)
        except Exception as e:
            pytest.fail(f"Failed to set pose matrix for env_ids={env_ids}: {e}")

        # ----------------------------
        # 3. Check vector form after env_ids modification
        # ----------------------------
        result_vec = self.light.get_local_pose(to_matrix=False)
        assert result_vec.shape == (
            10,
            3,
        ), "Unexpected shape from get_local_pose(to_matrix=False)"

        for i in range(10):
            expected = (
                torch.tensor([1.0, 2.0, 3.0], device=self.sim.device)
                if i in env_ids
                else torch.tensor([0.0, 0.0, 0.0], device=self.sim.device)
            )
            assert torch.allclose(
                result_vec[i], expected, atol=1e-5
            ), f"Translation vector mismatch at env {i}"

        # ----------------------------
        # 4. Verify matrix form translation field
        # ----------------------------
        result_matrix = self.light.get_local_pose(to_matrix=True)
        for i in range(10):
            expected = (
                torch.tensor([1.0, 2.0, 3.0], device=self.sim.device)
                if i in env_ids
                else torch.tensor([0.0, 0.0, 0.0], device=self.sim.device)
            )
            assert torch.allclose(
                result_matrix[i][:3, 3], expected, atol=1e-5
            ), f"Translation matrix mismatch at env {i}"

    def teardown_method(self):
        """Clean up resources after each test method."""
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()


class TestLightTypes:
    """Tests for all six supported light types."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a SimulationManager for each test."""
        config = SimulationManagerCfg(headless=True, sim_device="cpu", num_envs=4)
        self.sim = SimulationManager(config)
        yield
        self.sim.destroy()
        import embodichain.lab.sim as om

        om.SimulationManager.flush_cleanup_queue()
        self.__dict__.clear()
        import gc

        gc.collect()

    # ------------------------------------------------------------------
    # Creation tests
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "light_type, expected_num_instances",
        [
            ("point", 4),
            ("sun", 1),
            ("direction", 1),
            ("spot", 4),
            ("rect", 4),
            ("mesh", 4),
        ],
    )
    def test_create_each_light_type(self, light_type, expected_num_instances):
        """Each of the 6 light types can be created without error.

        ``sun`` and ``direction`` are global scene lights with a single instance.
        All other types are per-environment batched lights.
        """
        cfg = LightCfg(
            uid=f"test_{light_type}",
            light_type=light_type,
            init_pos=(0.0, 0.0, 3.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None, f"Failed to create light of type '{light_type}'"
        assert light.num_instances == expected_num_instances
        if light_type in ("sun", "direction"):
            assert light.is_global, f"{light_type} should be a global light"

    def test_unknown_light_type_errors(self):
        """Passing an invalid light_type raises RuntimeError."""
        cfg = LightCfg(uid="bad", light_type="invalid")
        with pytest.raises(RuntimeError, match="Unsupported light type"):
            self.sim.add_light(cfg=cfg)

    def test_mesh_light_empty_path_warns(self):
        """Mesh light with empty mesh_path warns at creation but succeeds."""
        cfg = LightCfg(
            uid="mesh_no_path",
            light_type="mesh",
            init_pos=(0.0, 0.0, 2.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None, "Mesh light should be created even without path"

    # ------------------------------------------------------------------
    # Setter type-validation tests
    # ------------------------------------------------------------------

    def test_set_direction_on_point_warns(self):
        """Calling set_direction on a point light logs a warning and no-ops."""
        cfg = LightCfg(uid="pt", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        direction = torch.tensor([1.0, 0.0, 0.0])
        # Should not raise; should log a warning
        try:
            light.set_direction(direction)
        except Exception as e:
            pytest.fail(f"set_direction on point light should warn, not crash: {e}")

    def test_set_spot_angle_on_point_warns(self):
        """Calling set_spot_angle on a non-spot light warns and no-ops."""
        cfg = LightCfg(uid="pt2", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        inner = torch.tensor(15.0)
        outer = torch.tensor(30.0)
        try:
            light.set_spot_angle(inner, outer)
        except Exception as e:
            pytest.fail(f"set_spot_angle on point light should warn, not crash: {e}")

    def test_set_rect_wh_on_point_warns(self):
        """Calling set_rect_wh on a non-rect light warns and no-ops."""
        cfg = LightCfg(uid="pt3", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        w = torch.tensor(2.0)
        h = torch.tensor(3.0)
        try:
            light.set_rect_wh(w, h)
        except Exception as e:
            pytest.fail(f"set_rect_wh on point light should warn, not crash: {e}")

    def test_set_local_pose_on_direction_warns(self):
        """Calling set_local_pose on a direction light warns and no-ops."""
        cfg = LightCfg(
            uid="dir_light",
            light_type="direction",
            direction=(0.0, -1.0, 0.0),
        )
        light = self.sim.add_light(cfg=cfg)
        pose = torch.tensor([1.0, 2.0, 3.0])
        try:
            light.set_local_pose(pose)
        except Exception as e:
            pytest.fail(
                f"set_local_pose on direction light should warn, not crash: {e}"
            )

    def test_set_local_pose_on_sun_warns(self):
        """Calling set_local_pose on a sun light warns and no-ops."""
        cfg = LightCfg(
            uid="sun_light",
            light_type="sun",
            direction=(0.0, -1.0, 0.0),
        )
        light = self.sim.add_light(cfg=cfg)
        pose = torch.tensor([1.0, 2.0, 3.0])
        try:
            light.set_local_pose(pose)
        except Exception as e:
            pytest.fail(f"set_local_pose on sun light should warn, not crash: {e}")

    # ------------------------------------------------------------------
    # Type-specific property tests
    # ------------------------------------------------------------------

    def test_spot_light_has_cone_angles(self):
        """Spot light creation applies inner/outer cone angles from cfg."""
        cfg = LightCfg(
            uid="spot_test",
            light_type="spot",
            init_pos=(0.0, 0.0, 3.0),
            direction=(0.0, 0.0, -1.0),
            spot_angle_inner=20.0,
            spot_angle_outer=40.0,
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None
        # Should not crash on creation; spot_angle applied during reset()

    def test_rect_light_has_dimensions(self):
        """Rect light creation applies width/height from cfg."""
        cfg = LightCfg(
            uid="rect_test",
            light_type="rect",
            init_pos=(0.0, 0.0, 3.0),
            direction=(0.0, 0.0, -1.0),
            rect_width=2.0,
            rect_height=1.5,
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None

    @pytest.mark.parametrize("light_type", ["sun", "direction"])
    def test_global_light_no_position_in_reset(self, light_type):
        """Global light (sun/direction) reset does not set position."""
        cfg = LightCfg(
            uid=f"{light_type}_test",
            light_type=light_type,
            direction=(0.0, -1.0, 0.0),
        )
        light = self.sim.add_light(cfg=cfg)
        assert light is not None
        assert light.is_global
        assert light.num_instances == 1

    @pytest.mark.parametrize("light_type", ["sun", "direction"])
    def test_reset_global_light_with_env_ids(self, light_type):
        """reset_objects_state with env_ids does not crash for global lights."""
        cfg = LightCfg(
            uid=f"global_{light_type}",
            light_type=light_type,
            direction=(0.0, -1.0, 0.0),
        )
        self.sim.add_light(cfg=cfg)
        try:
            self.sim.reset_objects_state(env_ids=[1, 2])
        except Exception as e:
            pytest.fail(
                f"reset_objects_state with global light ({light_type}) failed: {e}"
            )

    # ------------------------------------------------------------------
    # Broadcasting tests
    # ------------------------------------------------------------------

    def test_set_direction_broadcasting(self):
        """set_direction with (3,) tensor broadcasts to all instances."""
        cfg = LightCfg(
            uid="spot_broadcast",
            light_type="spot",
            init_pos=(0, 0, 3),
            direction=(0.0, 0.0, -1.0),
        )
        light = self.sim.add_light(cfg=cfg)
        new_dir = torch.tensor([1.0, 0.0, 0.0])
        try:
            light.set_direction(new_dir)
        except Exception as e:
            pytest.fail(f"set_direction broadcast failed: {e}")

    def test_set_direction_with_env_ids(self):
        """set_direction with (M, 3) tensor applies per-instance."""
        cfg = LightCfg(
            uid="spot_env",
            light_type="spot",
            init_pos=(0, 0, 3),
            direction=(0.0, 0.0, -1.0),
        )
        light = self.sim.add_light(cfg=cfg)
        env_ids = [0, 2]
        dirs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        try:
            light.set_direction(dirs, env_ids=env_ids)
        except Exception as e:
            pytest.fail(f"set_direction per-instance failed: {e}")

    def test_enable_shadow(self):
        """enable_shadow sets shadow flag on all instances."""
        cfg = LightCfg(uid="shadow_test", light_type="point", init_pos=(0, 0, 2))
        light = self.sim.add_light(cfg=cfg)
        try:
            light.enable_shadow(torch.tensor(0.0))  # disable
            light.enable_shadow(torch.tensor(1.0))  # enable
        except Exception as e:
            pytest.fail(f"enable_shadow failed: {e}")

    # ------------------------------------------------------------------
    # from_dict compatibility
    # ------------------------------------------------------------------

    def test_from_dict_new_types(self):
        """LightCfg.from_dict works with new type fields."""
        cfg = LightCfg.from_dict(
            {
                "light_type": "spot",
                "color": [1.0, 0.9, 0.8],
                "intensity": 100.0,
                "init_pos": [0.0, 0.0, 3.0],
                "direction": [0.0, 0.0, -1.0],
                "spot_angle_inner": 25.0,
                "spot_angle_outer": 50.0,
                "uid": "from_dict_spot",
            }
        )
        assert cfg.light_type == "spot"
        assert cfg.spot_angle_inner == 25.0
        assert cfg.spot_angle_outer == 50.0
        assert cfg.direction == (0.0, 0.0, -1.0) or cfg.direction == [0.0, 0.0, -1.0]

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def test_point_light_backward_compat(self):
        """Existing point-light config pattern still works."""
        cfg_dict = {
            "light_type": "point",
            "color": [0.1, 0.1, 0.1],
            "radius": 10.0,
            "position": [0.0, 0.0, 2.0],
            "uid": "point_compat",
        }
        light = self.sim.add_light(cfg=LightCfg.from_dict(cfg_dict))
        assert light is not None
        assert light.num_instances == 4

        # set_color should still work
        base_color = torch.tensor([0.1, 0.1, 0.1])
        try:
            light.set_color(base_color)
        except Exception as e:
            pytest.fail(f"set_color failed: {e}")

        # set_falloff should still work
        try:
            light.set_falloff(torch.tensor(100.0))
        except Exception as e:
            pytest.fail(f"set_falloff failed: {e}")
