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

"""Tests for the browser skill-sequence authoring protocol and session."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.visualization.authoring import (
    AddCard,
    AuthoringSession,
    CompileSequence,
    ExecuteSequence,
    MoveCard,
    RemoveCard,
    SequenceSnapshot,
    SkillCard,
    SkillCardState,
    UpdateCard,
)

MEE_HOVER_POSITION = (-0.42, -0.08, 0.36)
PLACE_POSITION = (-0.20, 0.28, 0.10)
UNREACHABLE_POSITION = (2.5, 0.0, 2.5)
OBJECT_XY = (-0.42, -0.08)
OBJECT_SIZE = 0.05
MIN_PLACE_XY_DISPLACEMENT_M = 0.10


def _make_pure_session() -> AuthoringSession:
    """Create a session with inert stubs for simulation-free logic tests."""
    return AuthoringSession(
        robot=SimpleNamespace(),
        engine=SimpleNamespace(),
        sim=SimpleNamespace(),
        control_parts={"motion": "arm", "grasp": "hand"},
    )


class TestAuthoringProtocol:
    """Validation behavior of the immutable protocol objects."""

    def test_skill_card_rejects_empty_identifiers(self) -> None:
        with pytest.raises(ValueError, match="card_id"):
            SkillCard(card_id="", skill_id="move_end_effector")
        with pytest.raises(ValueError, match="skill_id"):
            SkillCard(card_id="card_1", skill_id="")
        with pytest.raises(ValueError, match="entity_uid"):
            SkillCard(card_id="card_1", skill_id="pick_up", entity_uid="")

    def test_skill_card_failure_message_matches_state(self) -> None:
        with pytest.raises(ValueError, match="failure_message"):
            SkillCard(
                card_id="card_1",
                skill_id="move_end_effector",
                state=SkillCardState.FAILED,
            )
        with pytest.raises(ValueError, match="FAILED"):
            SkillCard(
                card_id="card_1",
                skill_id="move_end_effector",
                state=SkillCardState.READY,
                failure_message="planning failed",
            )

    def test_skill_card_segment_metadata_is_validated(self) -> None:
        with pytest.raises(ValueError, match="together"):
            SkillCard(
                card_id="card_1",
                skill_id="move_end_effector",
                segment_start=0,
            )
        with pytest.raises(ValueError, match="greater than"):
            SkillCard(
                card_id="card_1",
                skill_id="move_end_effector",
                segment_start=4,
                segment_stop=4,
            )
        card = SkillCard(
            card_id="card_1",
            skill_id="move_end_effector",
            segment_start=2,
            segment_stop=7,
        )
        assert card.segment_waypoint_count == 5

    def test_skill_card_params_are_frozen_recursively(self) -> None:
        card = SkillCard(
            card_id="card_1",
            skill_id="place",
            params={"position": [0.1, 0.2, 0.3], "meta": {"nested": [1, 2]}},
        )

        with pytest.raises(TypeError):
            card.params["position"] = (0.0, 0.0, 0.0)  # type: ignore[index]
        assert card.params["position"] == (0.1, 0.2, 0.3)
        assert card.params["meta"]["nested"] == (1, 2)

    def test_sequence_snapshot_validates_cards_and_waypoints(self) -> None:
        card = SkillCard(card_id="card_1", skill_id="move_end_effector")
        with pytest.raises(ValueError, match="duplicate"):
            SequenceSnapshot(cards=(card, card), compiled=False)
        with pytest.raises(ValueError, match="zero"):
            SequenceSnapshot(cards=(card,), compiled=False, trajectory_waypoint_count=3)
        with pytest.raises(ValueError, match="non-negative"):
            SequenceSnapshot(cards=(card,), compiled=True, trajectory_waypoint_count=-1)

    def test_commands_validate_their_fields(self) -> None:
        with pytest.raises(ValueError, match="skill_id"):
            AddCard(skill_id="")
        with pytest.raises(ValueError, match="non-negative"):
            AddCard(skill_id="pick_up", index=-1)
        with pytest.raises(ValueError, match="card_id"):
            RemoveCard(card_id="")
        with pytest.raises(ValueError, match="non-negative"):
            MoveCard(card_id="card_1", new_index=-2)
        with pytest.raises(ValueError, match="change"):
            UpdateCard(card_id="card_1")
        with pytest.raises(ValueError, match="set and clear"):
            UpdateCard(card_id="card_1", entity_uid="cube", clear_entity_uid=True)
        with pytest.raises(ValueError, match="non-negative"):
            ExecuteSequence(hold_steps=-1)


class TestAuthoringSessionLogic:
    """Card management and the configuration state machine without a sim."""

    def test_add_card_tracks_configuration_state(self) -> None:
        session = _make_pure_session()

        move = session.add_card("move_end_effector")
        pick = session.add_card("pick_up")
        place = session.add_card("place", params={"position": PLACE_POSITION})

        assert move.state is SkillCardState.UNCONFIGURED
        assert pick.state is SkillCardState.UNCONFIGURED
        assert place.state is SkillCardState.READY

        move = session.update_card(
            move.card_id, params={"position": MEE_HOVER_POSITION}
        )
        pick = session.update_card(pick.card_id, entity_uid="cube")
        assert move.state is SkillCardState.READY
        assert pick.state is SkillCardState.READY

        pick = session.update_card(pick.card_id, clear_entity_uid=True)
        assert pick.state is SkillCardState.UNCONFIGURED

    def test_add_card_rejects_invalid_configuration(self) -> None:
        session = _make_pure_session()

        with pytest.raises(ValueError, match="Unsupported skill"):
            session.add_card("open_door")
        with pytest.raises(ValueError, match="Unknown parameters"):
            session.add_card("move_end_effector", params={"speed": 1.0})
        with pytest.raises(ValueError, match="position"):
            session.add_card("move_end_effector", params={"position": (1.0, 2.0)})
        with pytest.raises(ValueError, match="approach_direction"):
            session.add_card("pick_up", params={"approach_direction": (0.0, 0.0, 0.0)})
        session.add_card("place", card_id="card_a")
        with pytest.raises(ValueError, match="already"):
            session.add_card("place", card_id="card_a")
        with pytest.raises(IndexError):
            session.add_card("place", index=5)

    def test_move_remove_and_reorder_cards(self) -> None:
        session = _make_pure_session()
        first = session.add_card("move_end_effector", card_id="first")
        second = session.add_card("pick_up", card_id="second")
        third = session.add_card("place", card_id="third")

        session.move_card(third.card_id, 0)
        assert [card.card_id for card in session.cards] == [
            "third",
            "first",
            "second",
        ]

        session.remove_card(first.card_id)
        assert [card.card_id for card in session.cards] == ["third", "second"]

        with pytest.raises(KeyError):
            session.remove_card("missing")
        with pytest.raises(IndexError):
            session.move_card(second.card_id, 5)

    def test_handle_command_dispatches_card_operations(self) -> None:
        session = _make_pure_session()

        snapshot = session.handle_command(
            AddCard(
                skill_id="move_end_effector",
                card_id="move",
                params={"position": MEE_HOVER_POSITION},
            )
        )
        assert isinstance(snapshot, SequenceSnapshot)
        assert snapshot.cards[0].state is SkillCardState.READY

        snapshot = session.handle_command(AddCard(skill_id="pick_up", card_id="pick"))
        assert snapshot.cards[1].state is SkillCardState.UNCONFIGURED

        snapshot = session.handle_command(UpdateCard(card_id="pick", entity_uid="cube"))
        assert snapshot.cards[1].state is SkillCardState.READY

        snapshot = session.handle_command(MoveCard(card_id="pick", new_index=0))
        assert [card.card_id for card in snapshot.cards] == ["pick", "move"]

        snapshot = session.handle_command(RemoveCard(card_id="move"))
        assert [card.card_id for card in snapshot.cards] == ["pick"]
        assert not snapshot.compiled
        assert snapshot.trajectory_waypoint_count == 0

        with pytest.raises(TypeError, match="Unknown authoring command"):
            session.handle_command(object())  # type: ignore[arg-type]

    def test_compile_guards_reject_invalid_sequences(self) -> None:
        session = _make_pure_session()

        with pytest.raises(ValueError, match="empty"):
            session.compile()

        session.add_card("move_end_effector", card_id="unset")
        with pytest.raises(ValueError, match="not fully configured"):
            session.compile()

    def test_preview_and_execute_require_successful_compilation(self) -> None:
        session = _make_pure_session()

        assert session.preview_length == 0
        assert session.compiled_trajectory is None
        with pytest.raises(RuntimeError, match="no successful compilation"):
            session.preview_qpos(0)
        with pytest.raises(RuntimeError, match="no successful compilation"):
            session.execute()

    def test_session_requires_motion_control_part(self) -> None:
        with pytest.raises(ValueError, match="motion"):
            AuthoringSession(
                robot=SimpleNamespace(),
                engine=SimpleNamespace(),
                sim=SimpleNamespace(),
                control_parts={"grasp": "hand"},
            )


@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.gpu
class TestAuthoringSessionWithSimulation:
    """End-to-end compile, preview, and execute against a real simulation."""

    sim = None

    @classmethod
    def setup_class(cls) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.atomic_actions import (
            AtomicActionEngine,
            ControlPartCommandProfile,
        )
        from embodichain.lab.sim.cfg import RigidBodyAttributesCfg, RigidObjectCfg
        from embodichain.lab.sim.shapes import CubeCfg
        from scripts.tutorials.atomic_action.tutorial_utils import (
            add_ur5_gripper_robot,
            clone_local_pose_from_first_env,
            create_parallel_jaw_grasp_pose_generator,
            create_toppra_motion_generator,
            get_hand_open_close_qpos,
        )

        cls.sim = SimulationManager(
            SimulationManagerCfg(
                headless=True,
                sim_device="cuda",
                num_envs=1,
                physics_dt=1.0 / 100.0,
            )
        )
        cls.robot = add_ur5_gripper_robot(cls.sim, tcp_z=0.15)
        cls.initial_qpos = cls.robot.get_qpos().clone()
        cls.obj = cls.sim.add_rigid_object(
            cfg=RigidObjectCfg(
                uid="cube",
                shape=CubeCfg(size=[OBJECT_SIZE] * 3),
                attrs=RigidBodyAttributesCfg(
                    mass=0.05,
                    dynamic_friction=0.97,
                    static_friction=0.99,
                    enable_ccd=True,
                ),
                max_convex_hull_num=16,
                init_pos=[*OBJECT_XY, OBJECT_SIZE],
            )
        )
        cls.sim.update(step=10)
        clone_local_pose_from_first_env(cls.obj)
        cls.obj.clear_dynamics()
        cls.initial_obj_pose = cls.obj.get_local_pose(to_matrix=True).clone()
        cls.hand_open, cls.hand_close = get_hand_open_close_qpos(cls.robot)
        cls.motion_gen = create_toppra_motion_generator(cls.robot)
        cls.engine = AtomicActionEngine(
            motion_generator=cls.motion_gen,
            control_profiles={
                "hand": ControlPartCommandProfile.joint_positions(
                    open=cls.hand_open,
                    grasp=cls.hand_close,
                )
            },
            grasp_pose_generators={
                "hand": create_parallel_jaw_grasp_pose_generator(
                    n_sample=1000,
                    force_refresh=False,
                )
            },
        )

    @classmethod
    def teardown_class(cls) -> None:
        from embodichain.lab.sim import SimulationManager

        cls.engine = None
        cls.motion_gen = None
        if cls.sim is not None:
            cls.sim.destroy()
            SimulationManager.flush_cleanup_queue()
            cls.sim = None

    @classmethod
    def _reset_scene(cls) -> None:
        from scripts.tutorials.atomic_action.tutorial_utils import (
            initialize_pre_pick_robot_pose,
        )

        for target in (False, True):
            cls.robot.set_qpos(cls.initial_qpos.clone(), target=target)
        cls.robot.clear_dynamics()
        cls.obj.set_local_pose(cls.initial_obj_pose.clone())
        cls.obj.clear_dynamics()
        initialize_pre_pick_robot_pose(cls.robot, cls.obj, cls.hand_open)

    def _make_session(self) -> AuthoringSession:
        return AuthoringSession(
            robot=self.robot,
            engine=self.engine,
            sim=self.sim,
            control_parts={"motion": "arm", "grasp": "hand"},
        )

    def test_full_sequence_compile_preview_execute(self) -> None:
        self._reset_scene()
        session = self._make_session()
        session.add_card(
            "move_end_effector",
            card_id="move",
            params={"position": MEE_HOVER_POSITION},
        )
        session.add_card("pick_up", card_id="pick", entity_uid="cube")
        session.add_card("place", card_id="place", params={"position": PLACE_POSITION})
        assert all(card.state is SkillCardState.READY for card in session.cards)

        compiled_ok = session.compile()
        assert compiled_ok, [card.failure_message for card in session.cards]

        snapshot = session.snapshot()
        assert snapshot.compiled
        cards = snapshot.cards
        assert cards[0].segment_start == 0
        assert cards[0].segment_stop == cards[1].segment_start
        assert cards[1].segment_stop == cards[2].segment_start
        assert cards[2].segment_stop == snapshot.trajectory_waypoint_count
        assert all(card.segment_waypoint_count > 0 for card in cards)

        assert session.preview_length == snapshot.trajectory_waypoint_count
        preview = session.preview_qpos(0)
        assert preview.shape == (1, self.robot.get_qpos().shape[1])
        assert torch.equal(
            session.preview_qpos(5),
            session.compiled_trajectory.trajectory.positions[:, 5, :],
        )
        with pytest.raises(IndexError):
            session.preview_qpos(session.preview_length)

        lift_start = session.compiled_trajectory.segment(1, "lift").start
        cleared = False

        def _on_step(step_index: int, total_steps: int) -> None:
            nonlocal cleared
            assert 0 <= step_index < total_steps
            if not cleared and step_index + 1 >= lift_start:
                self.obj.clear_dynamics()
                cleared = True

        position_before = self.obj.get_local_pose(to_matrix=True)[0, :3, 3].clone()
        executed_ok = session.execute(on_step=_on_step, hold_steps=30)
        assert executed_ok
        assert all(card.state is SkillCardState.SUCCEEDED for card in session.cards)

        position_after = self.obj.get_local_pose(to_matrix=True)[0, :3, 3]
        xy_displacement = torch.linalg.norm(
            position_after[:2] - position_before[:2]
        ).item()
        assert xy_displacement > MIN_PLACE_XY_DISPLACEMENT_M

        # Any later edit invalidates the compilation and resets card states.
        session.add_card("move_end_effector", card_id="tail")
        snapshot = session.snapshot()
        assert not snapshot.compiled
        assert snapshot.trajectory_waypoint_count == 0
        assert snapshot.cards[0].segment_start is None
        assert snapshot.cards[0].state is SkillCardState.READY
        assert snapshot.cards[-1].state is SkillCardState.UNCONFIGURED

    def test_unreachable_pose_marks_card_failed(self) -> None:
        self._reset_scene()
        session = self._make_session()
        session.add_card(
            "move_end_effector",
            card_id="bad",
            params={"position": UNREACHABLE_POSITION},
        )
        session.add_card(
            "move_end_effector",
            card_id="after",
            params={"position": MEE_HOVER_POSITION},
        )

        compiled_ok = session.handle_command(CompileSequence()).compiled
        assert not compiled_ok

        cards = session.snapshot().cards
        assert cards[0].state is SkillCardState.FAILED
        assert cards[0].failure_message
        assert cards[0].segment_start is None
        assert cards[1].state is SkillCardState.READY
        assert cards[1].segment_start is None

        assert session.preview_length == 0
        with pytest.raises(RuntimeError, match="no successful compilation"):
            session.execute()
