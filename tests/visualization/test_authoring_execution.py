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

"""Tests for host-driven, non-blocking execution of a compiled sequence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.visualization.authoring import (
    AuthoringBridge,
    AuthoringSession,
    ExecuteSequence,
    ExecutionProgress,
    SkillCard,
    SkillCardState,
    SkillSequencePanel,
    StepwiseExecution,
)
from embodichain.lab.visualization.protocol import PanelCommand

WAYPOINT_COUNT = 8
ROBOT_DOF = 3
PHYSICS_DT = 0.01
SEGMENTS = ((0, 3), (3, WAYPOINT_COUNT))


# ----------------------------------------------------------------------------
# Simulation-free doubles
# ----------------------------------------------------------------------------


class _Robot:
    """Robot double recording every commanded configuration."""

    def __init__(self) -> None:
        self.commands: list[torch.Tensor] = []

    def set_qpos(self, qpos: torch.Tensor) -> None:
        self.commands.append(qpos.clone())


class _Sim:
    """Simulation double recording the physics steps it was asked for."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.sim_config = SimpleNamespace(physics_dt=PHYSICS_DT)
        self.steps: list[int] = []
        self._fail_at = fail_at

    def update(self, step: int = 1) -> None:
        if self._fail_at is not None and len(self.steps) == self._fail_at:
            raise RuntimeError("simulated physics failure")
        self.steps.append(int(step))


def _compiled_double() -> SimpleNamespace:
    """Return a duck-typed compilation with distinguishable waypoints."""
    positions = torch.arange(WAYPOINT_COUNT * ROBOT_DOF, dtype=torch.float32).reshape(
        1, WAYPOINT_COUNT, ROBOT_DOF
    )
    return SimpleNamespace(
        trajectory=SimpleNamespace(
            positions=positions,
            dt=torch.full((1, WAYPOINT_COUNT), PHYSICS_DT),
            waypoint_count=WAYPOINT_COUNT,
        )
    )


class _CompiledSession(AuthoringSession):
    """Session preloaded with a compilation so execution needs no planner.

    Only tests may install a compilation directly; production code always
    reaches this state through :meth:`AuthoringSession.compile`.
    """

    def install_compilation(self) -> None:
        """Install two compiled cards covering the whole trajectory."""
        self._compiled = _compiled_double()
        self._cards = [
            SkillCard(
                card_id=f"card_{index + 1}",
                skill_id="move_end_effector",
                params={"position": (0.1, 0.2, 0.3)},
                state=SkillCardState.READY,
                segment_start=start,
                segment_stop=stop,
            )
            for index, (start, stop) in enumerate(SEGMENTS)
        ]


def _session(fail_at: int | None = None) -> tuple[_CompiledSession, _Robot, _Sim]:
    robot = _Robot()
    sim = _Sim(fail_at=fail_at)
    session = _CompiledSession(
        robot=robot,
        engine=SimpleNamespace(),
        sim=sim,
        control_parts={"motion": "arm"},
    )
    session.install_compilation()
    return session, robot, sim


def _states(session: AuthoringSession) -> list[SkillCardState]:
    return [card.state for card in session.cards]


# ----------------------------------------------------------------------------
# Progress value
# ----------------------------------------------------------------------------


class TestExecutionProgress:
    """Validation of the immutable progress value."""

    def test_reports_the_completed_fraction(self) -> None:
        assert ExecutionProgress(step_index=0, total_steps=4).fraction == 0.25
        assert ExecutionProgress(step_index=3, total_steps=4).fraction == 1.0

    def test_rejects_out_of_range_and_mistyped_fields(self) -> None:
        with pytest.raises(ValueError, match="total_steps"):
            ExecutionProgress(step_index=0, total_steps=0)
        with pytest.raises(ValueError, match="step_index"):
            ExecutionProgress(step_index=4, total_steps=4)
        with pytest.raises(TypeError, match="step_index"):
            ExecutionProgress(step_index=True, total_steps=4)
        with pytest.raises(TypeError, match="holding"):
            ExecutionProgress(step_index=0, total_steps=4, holding=1)


# ----------------------------------------------------------------------------
# Session iterator
# ----------------------------------------------------------------------------


class TestExecuteStepwise:
    """The iterator performs the same work as the blocking entry point."""

    def test_yields_one_progress_per_simulation_update(self) -> None:
        session, robot, sim = _session()

        progress = list(session.execute_stepwise(hold_steps=2))

        assert [item.step_index for item in progress if not item.holding] == list(
            range(WAYPOINT_COUNT)
        )
        assert [item.step_index for item in progress if item.holding] == [0, 1]
        assert all(item.total_steps == WAYPOINT_COUNT for item in progress[:-2])
        assert len(sim.steps) == len(progress)
        assert len(robot.commands) == WAYPOINT_COUNT + 2

    def test_card_states_advance_while_the_iterator_is_paused(self) -> None:
        session, _, _ = _session()
        steps = session.execute_stepwise()

        assert _states(session) == [SkillCardState.READY] * 2

        next(steps)
        assert _states(session) == [SkillCardState.RUNNING, SkillCardState.READY]

        for _ in range(SEGMENTS[0][1]):
            next(steps)
        assert _states(session) == [SkillCardState.SUCCEEDED, SkillCardState.RUNNING]

        with pytest.raises(StopIteration) as stop:
            while True:
                next(steps)
        assert stop.value.value is True
        assert _states(session) == [SkillCardState.SUCCEEDED] * 2

    def test_matches_the_blocking_entry_point(self) -> None:
        blocking_session, blocking_robot, blocking_sim = _session()
        stepwise_session, stepwise_robot, stepwise_sim = _session()

        assert blocking_session.execute(hold_steps=3) is True
        assert StepwiseExecution(stepwise_session, hold_steps=3).run_to_completion()

        assert blocking_sim.steps == stepwise_sim.steps
        assert len(blocking_robot.commands) == len(stepwise_robot.commands)
        for expected, actual in zip(blocking_robot.commands, stepwise_robot.commands):
            assert torch.equal(expected, actual)
        assert _states(blocking_session) == _states(stepwise_session)

    def test_forwards_the_on_step_callback_for_trajectory_ticks_only(self) -> None:
        session, _, _ = _session()
        observed: list[tuple[int, int]] = []

        StepwiseExecution(
            session,
            lambda step_index, total: observed.append((step_index, total)),
            hold_steps=4,
        ).run_to_completion()

        assert observed == [(index, WAYPOINT_COUNT) for index in range(WAYPOINT_COUNT)]

    def test_failure_marks_the_running_card_and_stops(self) -> None:
        session, _, _ = _session(fail_at=SEGMENTS[0][1] + 1)

        steps = session.execute_stepwise()
        with pytest.raises(StopIteration) as stop:
            while True:
                next(steps)

        assert stop.value.value is False
        assert _states(session) == [SkillCardState.SUCCEEDED, SkillCardState.FAILED]
        assert "simulated physics failure" in session.cards[1].failure_message

    def test_validates_eagerly_before_any_simulation_update(self) -> None:
        session, _, sim = _session()

        with pytest.raises(ValueError, match="hold_steps"):
            session.execute_stepwise(hold_steps=-1)
        assert sim.steps == []

        empty = AuthoringSession(
            robot=SimpleNamespace(),
            engine=SimpleNamespace(),
            sim=SimpleNamespace(),
            control_parts={"motion": "arm"},
        )
        with pytest.raises(RuntimeError, match="no successful compilation"):
            empty.execute_stepwise()

    def test_blocking_execute_still_reports_a_raising_callback(self) -> None:
        session, _, _ = _session()

        def _raise(step_index: int, total_steps: int) -> None:
            raise RuntimeError("callback boom")

        assert session.execute(_raise) is False
        assert session.cards[0].state is SkillCardState.FAILED
        assert "callback boom" in session.cards[0].failure_message


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


class TestStepwiseExecution:
    """The driver hands pacing control to the host loop."""

    def test_advance_performs_a_bounded_number_of_ticks(self) -> None:
        session, _, sim = _session()
        execution = StepwiseExecution(session)

        progress = execution.advance(3)

        assert progress is not None
        assert progress.step_index == 2
        assert execution.progress is progress
        assert execution.is_active
        assert len(sim.steps) == 3

    def test_advance_reports_completion_exactly_once(self) -> None:
        session, _, _ = _session()
        execution = StepwiseExecution(session)

        assert execution.advance(WAYPOINT_COUNT + 1) is None
        assert execution.is_finished
        assert execution.succeeded
        assert execution.advance() is None

    def test_close_abandons_the_run_and_keeps_card_states(self) -> None:
        session, _, sim = _session()
        execution = StepwiseExecution(session)
        execution.advance()

        execution.close()

        assert execution.is_finished
        assert not execution.succeeded
        assert len(sim.steps) == 1
        assert _states(session) == [SkillCardState.RUNNING, SkillCardState.READY]

    def test_advance_rejects_a_non_positive_count(self) -> None:
        session, _, _ = _session()
        execution = StepwiseExecution(session)

        with pytest.raises(ValueError, match="at least one"):
            execution.advance(0)
        with pytest.raises(TypeError, match="integer"):
            execution.advance(True)


# ----------------------------------------------------------------------------
# Bridge integration
# ----------------------------------------------------------------------------


class _Runtime:
    """Visualization runtime double exposing only the bridge's dependencies."""

    def __init__(self) -> None:
        self.exporter = SimpleNamespace(run_id="run", scene_revision=1)
        self.panel_commands: list[PanelCommand] = []
        self.published: list[tuple[str, object]] = []

    def register_panel(self, spec: object) -> None:
        pass

    def unregister_panel(self, panel_id: str) -> None:
        pass

    def publish_panel_state(self, panel_id: str, state: object) -> None:
        self.published.append((panel_id, state))

    def drain_panel_commands(
        self,
        panel_id: str | None = None,
    ) -> tuple[PanelCommand, ...]:
        commands = tuple(
            command
            for command in self.panel_commands
            if panel_id is None or command.panel_id == panel_id
        )
        self.panel_commands = [
            command
            for command in self.panel_commands
            if panel_id is not None and command.panel_id != panel_id
        ]
        return commands

    def drain_pick_commands(self) -> tuple[object, ...]:
        return ()

    def queue(self, value: object) -> None:
        self.panel_commands.append(
            PanelCommand(
                run_id="run",
                scene_revision=1,
                sequence=len(self.panel_commands) + 1,
                panel_id="skill_sequence",
                client_id="client",
                value=value,
            )
        )


def _bridge(**kwargs: object) -> tuple[AuthoringBridge, _CompiledSession, _Runtime]:
    session, _, _ = _session()
    runtime = _Runtime()
    bridge = AuthoringBridge(
        session,
        runtime,
        SkillSequencePanel(),
        process_picks=False,
        **kwargs,
    )
    return bridge, session, runtime


class _Preview:
    """Playback surface the bridge drives, recording suppression changes."""

    def __init__(self, length: int = WAYPOINT_COUNT) -> None:
        self.length = length
        self.is_playing = True
        self.suppressed = False
        self.suppression_history: list[bool] = []
        self.cursor = 0

    def set_suppressed(self, suppressed: bool) -> None:
        self.suppressed = bool(suppressed)
        self.suppression_history.append(self.suppressed)

    def pause(self) -> None:
        self.is_playing = False

    def seek(self, index: int) -> int:
        self.cursor = index
        return index

    def state(self) -> None:
        return None


class TestBridgeStepwiseExecution:
    """``ExecuteSequence`` no longer has to block the host loop."""

    def test_blocking_execution_remains_the_default(self) -> None:
        bridge, session, runtime = _bridge()
        runtime.queue(ExecuteSequence())

        bridge.update()

        assert not bridge.execution_active
        assert _states(session) == [SkillCardState.SUCCEEDED] * 2
        assert bridge.status == "Execution finished."

    def test_stepwise_execution_publishes_running_cards(self) -> None:
        bridge, session, runtime = _bridge(
            stepwise_execution=True,
            execution_steps_per_update=2,
        )
        runtime.queue(ExecuteSequence(hold_steps=2))

        view = bridge.update()

        assert bridge.execution_active
        assert view.snapshot.cards[0].state is SkillCardState.RUNNING
        assert view.snapshot.cards[1].state is SkillCardState.READY
        assert view.status == f"Executing {WAYPOINT_COUNT} waypoints."

        updates = 1
        while bridge.execution_active:
            view = bridge.update()
            updates += 1
            if bridge.execution_active:
                assert "Executing: " in view.status

        assert updates == (WAYPOINT_COUNT + 2) // 2 + 1
        assert _states(session) == [SkillCardState.SUCCEEDED] * 2
        assert bridge.status == "Execution finished."
        running_states = [
            state.snapshot.cards[0].state for _, state in runtime.published
        ]
        assert SkillCardState.RUNNING in running_states

    def test_edits_are_refused_while_an_execution_runs(self) -> None:
        from embodichain.lab.visualization.authoring import RemoveCard

        bridge, session, runtime = _bridge(stepwise_execution=True)
        runtime.queue(ExecuteSequence())
        bridge.update()

        runtime.queue(RemoveCard(card_id="card_1"))
        bridge.update()

        assert "an execution is in progress" in bridge.status
        assert len(session.cards) == 2

    def test_cancel_stops_the_run_and_releases_the_loop(self) -> None:
        bridge, session, runtime = _bridge(stepwise_execution=True)
        runtime.queue(ExecuteSequence())
        bridge.update()

        bridge.cancel_execution()

        assert not bridge.execution_active
        assert bridge.status == "Execution cancelled."
        assert _states(session) == [SkillCardState.RUNNING, SkillCardState.READY]

    def test_execution_on_step_runs_on_the_simulation_thread(self) -> None:
        observed: list[int] = []
        bridge, _, runtime = _bridge(
            stepwise_execution=True,
            execution_steps_per_update=WAYPOINT_COUNT,
            execution_on_step=lambda step_index, _: observed.append(step_index),
        )
        runtime.queue(ExecuteSequence())

        while True:
            bridge.update()
            if not bridge.execution_active:
                break

        assert observed == list(range(WAYPOINT_COUNT))

    def test_rejects_a_non_positive_step_budget(self) -> None:
        with pytest.raises(ValueError, match="execution_steps_per_update"):
            _bridge(execution_steps_per_update=0)


class TestBridgePreviewSuppression:
    """The translucent preview steps aside while the real robot executes."""

    def test_execution_hides_the_preview_robot(self) -> None:
        preview = _Preview()
        bridge, _, runtime = _bridge(stepwise_execution=True, preview=preview)
        runtime.queue(ExecuteSequence())

        bridge.update()

        assert bridge.execution_active
        assert preview.suppressed

    def test_execution_pauses_the_hidden_preview(self) -> None:
        """A preview nobody can see must not keep reporting that it plays."""
        preview = _Preview()
        bridge, _, runtime = _bridge(stepwise_execution=True, preview=preview)
        assert preview.is_playing
        runtime.queue(ExecuteSequence())

        bridge.update()

        assert not preview.is_playing

    def test_finishing_restores_the_preview_robot(self) -> None:
        preview = _Preview()
        bridge, _, runtime = _bridge(
            stepwise_execution=True,
            execution_steps_per_update=WAYPOINT_COUNT,
            preview=preview,
        )
        runtime.queue(ExecuteSequence())

        while True:
            bridge.update()
            if not bridge.execution_active:
                break

        assert not preview.suppressed
        # It really was hidden for the run rather than never suppressed at all.
        assert True in preview.suppression_history

    def test_cancelling_restores_the_preview_robot(self) -> None:
        """Cancel restores immediately; a host may capture before the next update."""
        preview = _Preview()
        bridge, _, runtime = _bridge(stepwise_execution=True, preview=preview)
        runtime.queue(ExecuteSequence())
        bridge.update()
        assert preview.suppressed

        bridge.cancel_execution()

        assert not preview.suppressed

    def test_an_idle_bridge_leaves_the_preview_visible(self) -> None:
        preview = _Preview()
        bridge, _, _ = _bridge(stepwise_execution=True, preview=preview)

        bridge.update()

        assert not preview.suppressed
