# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
...
# ----------------------------------------------------------------------------
import torch
import pytest
import numpy as np

from embodichain.lab.sim.planners.toppra_planner import (
    ToppraPlanner,
    ToppraPlannerCfg,
    ToppraOptions,
)
from embodichain.lab.sim.planners.utils import PlanState, TrajectorySampleMethod
from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
from embodichain.lab.sim.robots import CobotMagicCfg


class TestToppraPlanner:
    @classmethod
    def setup_class(cls):
        cls.sim_config = SimulationManagerCfg(headless=True, sim_device="cpu")
        cls.sim = SimulationManager(cls.sim_config)

        cfg_dict = {
            "uid": "CobotMagic_toppra",
            "init_pos": [0.0, 0.0, 0.7775],
            "init_qpos": [0.0] * 16,
        }
        cls.robot = cls.sim.add_robot(cfg=CobotMagicCfg.from_dict(cfg_dict))

    @classmethod
    def teardown_class(cls):
        cls.sim.destroy()

    def setup_method(self):
        cfg = ToppraPlannerCfg(
            robot_uid="CobotMagic_toppra",
            constraints={"velocity": 1.0, "acceleration": 2.0},
        )
        self.planner = ToppraPlanner(cfg=cfg)

    def test_initialization(self):
        assert self.planner.device == torch.device("cpu")

    def test_plan_basic(self):
        current_state = PlanState(qpos=np.zeros(6))
        target_states = [PlanState(qpos=np.ones(6))]

        plan_option = ToppraOptions(
            start_qpos=torch.zeros(
                size=(6,), dtype=torch.float32, device=self.planner.device
            ),
            is_linear=False,
            is_pre_interpolate=True,
            control_part="left_arm",
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=0.1,
        )
        result = self.planner.plan(target_states, plan_option=plan_option)
        assert result.success is True
        assert result.positions is not None
        assert result.velocities is not None
        assert result.accelerations is not None

        # Check constraints
        is_satisfied = self.planner.is_satisfied_constraint(
            result.velocities, result.accelerations
        )
        assert is_satisfied is True

    def test_trivial_trajectory(self):
        target_states = [PlanState(qpos=np.zeros(6))]

        plan_option = ToppraOptions(
            start_qpos=torch.zeros(
                size=(6,), dtype=torch.float32, device=self.planner.device
            ),
            is_linear=False,
            is_pre_interpolate=True,
            control_part="left_arm",
            sample_method=TrajectorySampleMethod.TIME,
            sample_interval=0.1,
        )
        result = self.planner.plan(target_states, plan_option=plan_option)
        assert result.success is True
        assert len(result.positions) == 2
        assert result.duration == 0.0
