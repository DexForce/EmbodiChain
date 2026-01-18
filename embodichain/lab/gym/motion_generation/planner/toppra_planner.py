# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------

import numpy as np
from embodichain.lab.gym.motion_generation.planner.utils import (
    TrajectorySampleMethod,
)

from typing import TYPE_CHECKING, Union

try:
    import toppra as ta
    import toppra.constraint as constraint
except ImportError:
    raise ImportError("toppra not installed. Install with `pip install toppra==0.6.3`")

ta.setup_logging(level="WARN")


class ToppraPlanner:
    def __init__(self, DOFs, max_constraints):
        r"""Initialize the TOPPRA trajectory planner.

        Args:
            DOFs: Number of degrees of freedom
            max_constraints: Dictionary containing 'velocity' and 'acceleration' constraints
        """

        self.DOFs = DOFs
        self.time_step = 0.01
        self.max_constraints = max_constraints

        # Create TOPPRA constraints
        self.vlims = np.array([[-v, v] for v in max_constraints["velocity"]])
        self.alims = np.array([[-a, a] for a in max_constraints["acceleration"]])

    def plan(
        self,
        current_state: dict,
        target_states: list[dict],
        sample_method: TrajectorySampleMethod = TrajectorySampleMethod.TIME,
        sample_interval: Union[float, int] = 0.01,
    ):
        r"""Execute trajectory planning.

        Args:
            current_state: Dictionary containing 'position', 'velocity', 'acceleration' for current state
            target_states: List of dictionaries containing target states

        Returns:
            Tuple of (success, positions, velocities, accelerations, times, duration)
        """
        if not isinstance(sample_interval, (float, int)):
            raise TypeError(
                f"sample_interval must be float/int, got {type(sample_interval)}"
            )
        if sample_method == TrajectorySampleMethod.TIME and sample_interval <= 0:
            raise ValueError("Time interval must be positive")
        elif sample_method == TrajectorySampleMethod.QUANTITY and sample_interval < 2:
            raise ValueError("At least 2 sample points required")

        # Check waypoints
        if len(current_state["position"]) != self.DOFs:
            print(f"Current wayponit does not align")
            return False, None, None, None, None, None
        for target in target_states:
            if len(target["position"]) != self.DOFs:
                print(f"Target Wayponits does not align")
                return False, None, None, None, None, None

        if (
            len(target_states) == 1
            and np.sum(
                np.abs(
                    np.array(target_states[0]["position"])
                    - np.array(current_state["position"])
                )
            )
            < 1e-3
        ):
            print(f"Only two same waypoints, do not plan")
            return (
                True,
                np.array([current_state["position"], target_states[0]["position"]]),
                np.array([[0.0] * self.DOFs, [0.0] * self.DOFs]),
                np.array([[0.0] * self.DOFs, [0.0] * self.DOFs]),
                0,
                0,
            )

        # Build waypoints
        waypoints = [np.array(current_state["position"])]
        for target in target_states:
            waypoints.append(np.array(target["position"]))
        waypoints = np.array(waypoints)

        # Create spline interpolation
        # NOTE(fsh)：适合密集的点
        ss = np.linspace(0, 1, len(waypoints))

        # NOTE(fsh)：适合稀疏的点，密集点容易不满足CubicSpline严格递增的条件
        # len_total = 0
        # len_from_start = [0]
        # for i in range(len(waypoints)-1):
        #     len_total += np.sum(np.abs(waypoints[i+1] - waypoints[i]))
        #     len_from_start.append(len_total)
        # ss = np.array([cur/len_total for cur in len_from_start])

        path = ta.SplineInterpolator(ss, waypoints)

        # Set constraints
        pc_vel = constraint.JointVelocityConstraint(self.vlims)
        pc_acc = constraint.JointAccelerationConstraint(self.alims)

        # Create TOPPRA instance
        instance = ta.algorithm.TOPPRA(
            [pc_vel, pc_acc],
            path,
            parametrizer="ParametrizeConstAccel",
            gridpt_min_nb_points=max(100, 10 * len(waypoints)),
        )
        # NOTES:合理设置gridpt_min_nb_points对加速度约束很重要

        # Compute parameterized trajectory
        jnt_traj = instance.compute_trajectory()
        if jnt_traj is None:
            # raise RuntimeError("Unable to find feasible trajectory")
            print(f"Unable to find feasible trajectory")
            return False, None, None, None, None, None

        duration = jnt_traj.duration
        # Sample trajectory points
        if duration <= 0:
            raise ValueError(f"Duration must be positive, got {duration}")
        if sample_method == TrajectorySampleMethod.TIME:
            n_points = max(2, int(np.ceil(duration / sample_interval)) + 1)
            ts = np.linspace(0, duration, n_points)
        else:
            ts = np.linspace(0, duration, num=int(sample_interval))

        positions = []
        velocities = []
        accelerations = []

        for t in ts:
            positions.append(jnt_traj.eval(t))
            velocities.append(jnt_traj.evald(t))
            accelerations.append(jnt_traj.evaldd(t))

        return (
            True,
            np.array(positions),
            np.array(velocities),
            np.array(accelerations),
            ts,
            duration,
        )

    def is_satisfied_constraint(self, velocities, accelerations) -> bool:
        r"""Check if the trajectory satisfies velocity and acceleration constraints.

        Args:
            velocities: array
            accelerations: array
        """
        # NOTE(fsh)：密集点过多的情况下，当前实现容易求解无法严格满足约束，会有一定的越界
        vlims = self.vlims * (1 + 0.1)  # 允许10%误差
        alims = self.alims * (1 + 0.25)  # 允许25%误差

        vel_check = np.all((velocities >= vlims[:, 0]) & (velocities <= vlims[:, 1]))
        acc_check = np.all(
            (accelerations >= alims[:, 0]) & (accelerations <= alims[:, 1])
        )

        # 超限情况
        if not vel_check:
            vel_exceed_info = []
            min_vel = np.min(velocities, axis=0)
            max_vel = np.max(velocities, axis=0)
            for i in range(self.DOFs):
                exceed_percentage = 0
                if min_vel[i] < self.vlims[i, 0]:
                    exceed_percentage = (min_vel[i] - self.vlims[i, 0]) / self.vlims[
                        i, 0
                    ]
                if max_vel[i] > self.vlims[i, 1]:
                    temp = (max_vel[i] - self.vlims[i, 1]) / self.vlims[i, 1]
                    if temp > exceed_percentage:
                        exceed_percentage = temp
                vel_exceed_info.append(exceed_percentage * 100)
            print(f"Velocity exceed info: {vel_exceed_info} percentage")

        if not acc_check:
            acc_exceed_info = []
            min_acc = np.min(accelerations, axis=0)
            max_acc = np.max(accelerations, axis=0)
            for i in range(self.DOFs):
                exceed_percentage = 0
                if min_acc[i] < self.alims[i, 0]:
                    exceed_percentage = (min_acc[i] - self.alims[i, 0]) / self.alims[
                        i, 0
                    ]
                if max_acc[i] > self.alims[i, 1]:
                    temp = (max_acc[i] - self.alims[i, 1]) / self.alims[i, 1]
                    if temp > exceed_percentage:
                        exceed_percentage = temp
                acc_exceed_info.append(exceed_percentage * 100)
            print(f"Acceleration exceed info: {acc_exceed_info} percentage")

        return vel_check and acc_check

    def plot_trajectory(self, positions, velocities, accelerations):
        r"""Plot trajectory data.

        Args:
            positions: Position array
            velocities: Velocity array
            accelerations: Acceleration array
        """
        import matplotlib.pyplot as plt

        time_steps = np.arange(positions.shape[0]) * self.time_step
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        for i in range(self.DOFs):
            axs[0].plot(time_steps, positions[:, i], label=f"Joint {i+1}")
            axs[1].plot(time_steps, velocities[:, i], label=f"Joint {i+1}")
            axs[2].plot(time_steps, accelerations[:, i], label=f"Joint {i+1}")

        axs[1].plot(
            time_steps,
            [self.vlims[0][0]] * len(time_steps),
            "k--",
            label="Max Velocity",
        )
        axs[1].plot(time_steps, [self.vlims[0][1]] * len(time_steps), "k--")
        axs[2].plot(
            time_steps,
            [self.alims[0][0]] * len(time_steps),
            "k--",
            label="Max Accleration",
        )
        axs[2].plot(time_steps, [self.alims[0][1]] * len(time_steps), "k--")

        axs[0].set_title("Position")
        axs[1].set_title("Velocity")
        axs[2].set_title("Acceleration")

        for ax in axs:
            ax.set_xlabel("Time [s]")
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()
