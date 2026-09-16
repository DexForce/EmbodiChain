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

"""Tests for translucent preview nodes and authoring preview playback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from embodichain.lab.visualization import (
    PreviewGroupCfg,
    PreviewNodeUpdate,
    SceneExporter,
    SceneNode,
    VisualizationCfg,
)
from embodichain.lab.visualization.authoring import (
    PreviewPlaybackCfg,
    PreviewPlaybackState,
    SequencePreview,
    SkillCard,
    SkillCardState,
)

MEE_HOVER_POSITION = (-0.42, -0.08, 0.36)
MEE_SECOND_POSITION = (-0.35, 0.12, 0.42)
PREVIEW_OPACITY = 0.3
FK_POSITION_TOLERANCE_M = 2.0e-3


# ----------------------------------------------------------------------------
# Simulation-free fakes
# ----------------------------------------------------------------------------


class _Articulation:
    """Minimal articulation exposing the link geometry the exporter reads."""

    link_names = ["base", "tool"]
    uid = "robot"

    def __init__(self) -> None:
        self._meshes = {
            "base": (
                np.array(
                    [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
            "tool": (
                np.array(
                    [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0]],
                    dtype=np.float32,
                ),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
        }
        self.body_data = SimpleNamespace(
            body_link_pose=np.array(
                [
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                    ],
                    [
                        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0],
                    ],
                ],
                dtype=np.float32,
            )
        )

    def get_link_vert_face(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        return self._meshes[link_name]


class _Simulation:
    """Two-environment simulation stub with one robot and no other assets."""

    num_envs = 2
    arena_offsets = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)

    def __init__(self) -> None:
        self.robot = _Articulation()

    def get_robot_uid_list(self) -> list[str]:
        return ["robot"]

    def get_robot(self, uid: str) -> _Articulation | None:
        return self.robot if uid == "robot" else None

    def get_rigid_object_uid_list(self) -> list[str]:
        return []

    def get_rigid_object_group_uid_list(self) -> list[str]:
        return []

    def get_articulation_uid_list(self) -> list[str]:
        return []

    def get_articulation(self, uid: str) -> None:
        return None

    def get_deformable_object_uid_list(self) -> list[str]:
        return []

    def get_deformable_object(self, uid: str) -> None:
        raise AssertionError(f"Unexpected deformable-object lookup: {uid}")

    def get_sensor_uid_list(self) -> list[str]:
        return []

    def get_sensor(self, uid: str) -> None:
        raise AssertionError(f"Unexpected sensor lookup: {uid}")


def _make_exporter(env_ids: list[int] | None = None) -> SceneExporter:
    return SceneExporter(
        _Simulation(),
        VisualizationCfg(env_ids=env_ids if env_ids is not None else [0, 1]),
    )


class _StubExporter:
    """Preview-group registry surface consumed by :class:`SequencePreview`."""

    def __init__(self, link_names: tuple[str, ...] = ("base", "tool")) -> None:
        self._groups: tuple[PreviewGroupCfg, ...] = ()
        self._link_names = link_names

    @property
    def preview_groups(self) -> tuple[PreviewGroupCfg, ...]:
        return self._groups

    def set_preview_groups(self, groups: object = ()) -> None:
        self._groups = tuple(groups)

    def preview_link_names(self, group_id: str) -> tuple[str, ...]:
        if not any(group.group_id == group_id for group in self._groups):
            raise KeyError(group_id)
        return self._link_names


class _StubSession:
    """Compiled-trajectory surface consumed by :class:`SequencePreview`."""

    def __init__(self) -> None:
        self.cards: tuple[SkillCard, ...] = ()
        self.compiled_trajectory: object | None = None
        self._length = 0
        self.robot = SimpleNamespace(uid="robot")

    def recompile(self, length: int, cards: tuple[SkillCard, ...] = ()) -> None:
        """Simulate a fresh compilation of ``length`` waypoints."""
        self._length = length
        self.cards = cards
        self.compiled_trajectory = object() if length else None

    @property
    def preview_length(self) -> int:
        return self._length

    def preview_qpos(self, step_index: int) -> torch.Tensor:
        raise AssertionError("Simulation-free tests must not evaluate kinematics.")


def _make_driver(
    cfg: PreviewPlaybackCfg | None = None,
) -> tuple[SequencePreview, _StubSession, _StubExporter]:
    session = _StubSession()
    exporter = _StubExporter()
    return SequencePreview(session, exporter, cfg), session, exporter


# ----------------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------------


class TestPreviewProtocol:
    """Validation of the preview protocol additions."""

    def test_scene_node_opacity_is_validated(self) -> None:
        node = SceneNode(
            node_id="node",
            path="/node",
            parent_id=None,
            env_id=0,
            kind="rigid_object",
            geometry_id="sha256:mesh",
        )
        assert node.opacity == 1.0

        with pytest.raises(ValueError, match="opacity"):
            SceneNode(
                node_id="node",
                path="/node",
                parent_id=None,
                env_id=0,
                kind="rigid_object",
                geometry_id="sha256:mesh",
                opacity=1.5,
            )

    def test_preview_node_update_validates_and_normalizes(self) -> None:
        with pytest.raises(ValueError, match="group_id"):
            PreviewNodeUpdate(
                group_id="",
                positions=np.zeros((1, 3), dtype=np.float32),
                wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            PreviewNodeUpdate(
                group_id="preview",
                positions=np.zeros((3,), dtype=np.float32),
                wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        with pytest.raises(ValueError, match="wxyz"):
            PreviewNodeUpdate(
                group_id="preview",
                positions=np.zeros((2, 3), dtype=np.float32),
                wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            )
        with pytest.raises(ValueError, match="degenerate"):
            PreviewNodeUpdate(
                group_id="preview",
                positions=np.zeros((1, 3), dtype=np.float32),
                wxyz=np.zeros((1, 4), dtype=np.float32),
            )

        update = PreviewNodeUpdate(
            group_id="preview",
            positions=np.zeros((1, 3), dtype=np.float32),
            wxyz=np.array([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(update.wxyz, [[1.0, 0.0, 0.0, 0.0]])
        assert update.visible

    def test_preview_node_update_from_matrices(self) -> None:
        poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        poses[1, :3, 3] = (0.1, 0.2, 0.3)

        update = PreviewNodeUpdate.from_matrices("preview", poses, visible=False)

        assert update.group_id == "preview"
        assert not update.visible
        np.testing.assert_allclose(update.positions[1], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(update.wxyz[0], [1.0, 0.0, 0.0, 0.0])

    def test_playback_state_validates_cursor_range(self) -> None:
        with pytest.raises(ValueError, match="below length"):
            PreviewPlaybackState(
                group_id="preview",
                length=3,
                cursor=3,
                playing=False,
                loop=True,
                step_stride=1,
            )
        with pytest.raises(ValueError, match="empty preview"):
            PreviewPlaybackState(
                group_id="preview",
                length=0,
                cursor=1,
                playing=False,
                loop=True,
                step_stride=1,
            )


# ----------------------------------------------------------------------------
# Exporter preview nodes
# ----------------------------------------------------------------------------


class TestScenePreviewNodes:
    """Generic preview-node registration, manifests, and frame poses."""

    def test_manifest_adds_translucent_preview_nodes(self) -> None:
        exporter = _make_exporter()
        exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="authoring_preview",
                    articulation_uid="robot",
                    env_id=1,
                    opacity=PREVIEW_OPACITY,
                )
            ]
        )

        manifest = exporter.build_manifest()

        preview_nodes = [node for node in manifest.nodes if node.kind == "preview_link"]
        assert [node.node_id for node in preview_nodes] == [
            "preview:authoring_preview/link:base",
            "preview:authoring_preview/link:tool",
        ]
        assert [node.path for node in preview_nodes] == [
            "/previews/authoring_preview/links/base",
            "/previews/authoring_preview/links/tool",
        ]
        assert all(node.env_id == 1 for node in preview_nodes)
        assert all(node.opacity == PREVIEW_OPACITY for node in preview_nodes)
        assert not any(node.visible for node in preview_nodes)
        assert all(
            node.opacity == 1.0
            for node in manifest.nodes
            if node.kind != "preview_link"
        )
        assert exporter.preview_link_names("authoring_preview") == ("base", "tool")
        assert exporter.preview_node_ids("authoring_preview") == tuple(
            node.node_id for node in preview_nodes
        )

    def test_preview_color_controls_geometry_reuse(self) -> None:
        baseline = _make_exporter().build_manifest()

        tinted_exporter = _make_exporter()
        tinted_exporter.set_preview_groups(
            [PreviewGroupCfg(group_id="tinted", articulation_uid="robot")]
        )
        tinted = tinted_exporter.build_manifest()

        shared_exporter = _make_exporter()
        shared_exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="shared",
                    articulation_uid="robot",
                    color=None,
                )
            ]
        )
        shared = shared_exporter.build_manifest()

        assert len(tinted.geometries) == len(baseline.geometries) + 2
        assert len(shared.geometries) == len(baseline.geometries)
        preview_geometry_ids = {
            node.geometry_id for node in shared.nodes if node.kind == "preview_link"
        }
        link_geometry_ids = {
            node.geometry_id for node in shared.nodes if node.kind == "robot_link"
        }
        assert preview_geometry_ids <= link_geometry_ids

    def test_preview_rows_are_parked_until_an_update_arrives(self) -> None:
        exporter = _make_exporter()
        exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="preview",
                    articulation_uid="robot",
                    env_id=1,
                )
            ]
        )
        exporter.build_manifest()

        frame = exporter.capture(sim_step=1, sim_time=0.01).frame

        indices = [
            index
            for index, node_id in enumerate(frame.node_ids)
            if node_id.startswith("preview:")
        ]
        assert indices
        assert not frame.visible[indices].any()
        np.testing.assert_allclose(
            frame.positions[indices],
            np.tile(np.array([[2.0, 0.0, 0.0]], dtype=np.float32), (len(indices), 1)),
        )
        np.testing.assert_allclose(
            frame.wxyz[indices],
            np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(indices), 1)
            ),
        )

    def test_preview_update_offsets_poses_into_the_arena(self) -> None:
        exporter = _make_exporter()
        exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="preview",
                    articulation_uid="robot",
                    env_id=1,
                )
            ]
        )
        exporter.build_manifest()
        baseline = exporter.capture(sim_step=1, sim_time=0.01).frame
        update = PreviewNodeUpdate(
            group_id="preview",
            positions=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            wxyz=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)),
        )

        frame = exporter.capture(
            sim_step=2,
            sim_time=0.02,
            preview_updates=(update,),
        ).frame

        indices = [
            index
            for index, node_id in enumerate(frame.node_ids)
            if node_id.startswith("preview:")
        ]
        assert frame.visible[indices].all()
        np.testing.assert_allclose(
            frame.positions[indices],
            update.positions + np.array([2.0, 0.0, 0.0], dtype=np.float32),
        )
        # Simulation-backed nodes keep the poses they had without previews.
        simulated = [
            index
            for index, node_id in enumerate(frame.node_ids)
            if not node_id.startswith("preview:")
        ]
        np.testing.assert_allclose(
            frame.positions[simulated], baseline.positions[simulated]
        )
        np.testing.assert_array_equal(
            frame.visible[simulated], baseline.visible[simulated]
        )

    def test_preview_registration_and_update_errors(self) -> None:
        exporter = _make_exporter(env_ids=[0])

        with pytest.raises(ValueError, match="not visualized"):
            exporter.set_preview_groups(
                [
                    PreviewGroupCfg(
                        group_id="preview",
                        articulation_uid="robot",
                        env_id=1,
                    )
                ]
            )
        with pytest.raises(ValueError, match="Duplicate preview group"):
            exporter.set_preview_groups(
                [
                    PreviewGroupCfg(group_id="preview", articulation_uid="robot"),
                    PreviewGroupCfg(group_id="preview", articulation_uid="robot"),
                ]
            )
        with pytest.raises(TypeError, match="Missing values"):
            exporter.set_preview_groups([PreviewGroupCfg(group_id="preview")])

        exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="preview",
                    articulation_uid="robot",
                    link_names=["tool"],
                )
            ]
        )
        exporter.build_manifest()
        assert exporter.preview_link_names("preview") == ("tool",)

        pose = PreviewNodeUpdate(
            group_id="preview",
            positions=np.zeros((1, 3), dtype=np.float32),
            wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        with pytest.raises(ValueError, match="unregistered group"):
            exporter.capture(
                sim_step=1,
                sim_time=0.01,
                preview_updates=(
                    PreviewNodeUpdate(
                        group_id="other",
                        positions=np.zeros((1, 3), dtype=np.float32),
                        wxyz=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    ),
                ),
            )
        with pytest.raises(ValueError, match="Duplicate preview update"):
            exporter.capture(
                sim_step=1,
                sim_time=0.01,
                preview_updates=(pose, pose),
            )
        with pytest.raises(ValueError, match="expects 1 node poses"):
            exporter.capture(
                sim_step=1,
                sim_time=0.01,
                preview_updates=(
                    PreviewNodeUpdate(
                        group_id="preview",
                        positions=np.zeros((2, 3), dtype=np.float32),
                        wxyz=np.tile(
                            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (2, 1)
                        ),
                    ),
                ),
            )

    def test_unknown_preview_source_is_rejected(self) -> None:
        exporter = _make_exporter()
        exporter.set_preview_groups(
            [PreviewGroupCfg(group_id="preview", articulation_uid="missing")]
        )

        with pytest.raises(ValueError, match="unknown robot"):
            exporter.build_manifest()

        exporter.set_preview_groups(
            [
                PreviewGroupCfg(
                    group_id="preview",
                    articulation_uid="robot",
                    link_names=["wrist"],
                )
            ]
        )
        with pytest.raises(ValueError, match="unknown links"):
            exporter.build_manifest()


# ----------------------------------------------------------------------------
# Playback cursor
# ----------------------------------------------------------------------------


class TestSequencePreviewPlayback:
    """Cursor arithmetic and scene registration without a simulation."""

    def test_empty_sequence_keeps_the_cursor_parked(self) -> None:
        driver, _, _ = _make_driver()

        assert driver.length == 0
        assert driver.cursor == 0
        assert not driver.is_playing
        assert not driver.play()
        assert driver.advance() == 0
        assert driver.seek(4) == 0
        assert driver.active_card_id() is None

    def test_seek_and_step_clamp_to_the_compiled_range(self) -> None:
        driver, session, _ = _make_driver()
        session.recompile(10)

        assert driver.seek(20) == 9
        assert driver.seek(-3) == 0
        assert driver.step(4) == 4
        assert not driver.is_playing
        assert driver.step(-9) == 0
        assert driver.step(99) == 9

        with pytest.raises(TypeError):
            driver.seek(1.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            driver.step(True)  # type: ignore[arg-type]

    def test_advance_respects_stride_looping_and_pause(self) -> None:
        driver, session, _ = _make_driver(PreviewPlaybackCfg(step_stride=4))
        session.recompile(10)

        assert driver.advance() == 0, "A paused driver must not move."
        assert driver.play()
        assert driver.advance() == 4
        assert driver.advance() == 8
        assert driver.advance() == 2, "Looping wraps past the final waypoint."
        assert driver.is_playing

        driver.cfg.loop = False
        driver.seek(8)
        driver.play()
        assert driver.advance() == 9
        assert not driver.is_playing
        assert driver.advance() == 9

    def test_toggle_flips_playback(self) -> None:
        driver, session, _ = _make_driver()
        session.recompile(5)

        assert driver.toggle()
        assert driver.is_playing
        assert not driver.toggle()
        assert not driver.is_playing

    def test_recompilation_resets_the_cursor(self) -> None:
        driver, session, _ = _make_driver(PreviewPlaybackCfg(autoplay=True))
        session.recompile(10)
        driver.seek(7)
        assert not driver.sync()

        session.recompile(4)

        assert driver.sync()
        assert driver.cursor == 0
        assert driver.is_playing, "autoplay starts a freshly compiled sequence."

        session.recompile(0)
        assert driver.length == 0
        assert not driver.is_playing

    def test_shrinking_compilation_clamps_the_cursor(self) -> None:
        driver, session, _ = _make_driver()
        session.recompile(10)
        driver.seek(9)

        # Same compilation object, fewer waypoints: the cursor is clamped.
        session._length = 3

        assert driver.cursor == 2

    def test_active_card_tracks_the_cursor(self) -> None:
        driver, session, _ = _make_driver()
        cards = (
            SkillCard(
                card_id="move",
                skill_id="move_end_effector",
                params={"position": MEE_HOVER_POSITION},
                state=SkillCardState.READY,
                segment_start=0,
                segment_stop=4,
            ),
            SkillCard(
                card_id="place",
                skill_id="place",
                params={"position": MEE_SECOND_POSITION},
                state=SkillCardState.READY,
                segment_start=4,
                segment_stop=10,
            ),
        )
        session.recompile(10, cards)

        assert driver.active_card_id() == "move"
        driver.seek(4)
        assert driver.active_card_id() == "place"
        driver.seek(9)
        assert driver.active_card_id() == "place"

    def test_register_preserves_other_preview_groups(self) -> None:
        driver, _, exporter = _make_driver()
        other = PreviewGroupCfg(group_id="other", articulation_uid="robot")
        exporter.set_preview_groups([other])

        assert not driver.is_registered
        group = driver.register()

        assert group.group_id == "authoring_preview"
        assert group.articulation_uid == "robot"
        assert group.opacity == driver.cfg.opacity
        assert not group.visible, "Preview groups stay hidden until a pose arrives."
        assert [item.group_id for item in exporter.preview_groups] == [
            "other",
            "authoring_preview",
        ]
        assert driver.is_registered

        # Re-registering replaces the group instead of duplicating it.
        driver.register()
        assert [item.group_id for item in exporter.preview_groups] == [
            "other",
            "authoring_preview",
        ]

        driver.unregister()
        assert [item.group_id for item in exporter.preview_groups] == ["other"]
        assert not driver.is_registered

    def test_state_snapshot_is_immutable_and_complete(self) -> None:
        driver, session, _ = _make_driver()
        session.recompile(6)
        driver.register()
        driver.seek(2)
        driver.play()

        state = driver.state()

        assert isinstance(state, PreviewPlaybackState)
        assert state.group_id == "authoring_preview"
        assert (state.length, state.cursor, state.playing) == (6, 2, True)
        assert state.registered
        assert state.step_stride == 1
        with pytest.raises(AttributeError):
            state.cursor = 3  # type: ignore[misc]

    def test_pose_production_requires_a_published_manifest(self) -> None:
        driver, session, _ = _make_driver()
        session.recompile(4)

        with pytest.raises(RuntimeError, match="not part of the current scene"):
            driver.preview_update()

    def test_uncompiled_sequence_hides_the_preview(self) -> None:
        driver, _, _ = _make_driver()
        driver.register()

        update = driver.preview_update()

        assert update.group_id == "authoring_preview"
        assert not update.visible
        assert update.positions.shape == (2, 3)
        assert driver.overlays().trajectories == ()


# ----------------------------------------------------------------------------
# Simulation-backed behavior
# ----------------------------------------------------------------------------


@pytest.mark.requires_sim
@pytest.mark.slow
@pytest.mark.gpu
class TestSequencePreviewWithSimulation:
    """Preview kinematics and side-effect freedom against a real simulation."""

    sim = None

    @classmethod
    def setup_class(cls) -> None:
        from embodichain.lab.sim import SimulationManager, SimulationManagerCfg
        from embodichain.lab.sim.atomic_actions import AtomicActionEngine
        from scripts.tutorials.atomic_action.tutorial_utils import (
            add_ur5_gripper_robot,
            create_toppra_motion_generator,
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
        cls.sim.update(step=10)
        cls.initial_qpos = cls.robot.get_qpos().clone()
        cls.motion_gen = create_toppra_motion_generator(cls.robot)
        cls.engine = AtomicActionEngine(motion_generator=cls.motion_gen)

    @classmethod
    def teardown_class(cls) -> None:
        from embodichain.lab.sim import SimulationManager

        cls.engine = None
        cls.motion_gen = None
        if cls.sim is not None:
            cls.sim.destroy()
            SimulationManager.flush_cleanup_queue()
            cls.sim = None

    def _make_preview(self) -> tuple[object, SequencePreview, SceneExporter]:
        from embodichain.lab.visualization.authoring import AuthoringSession

        session = AuthoringSession(
            robot=self.robot,
            engine=self.engine,
            sim=self.sim,
            control_parts={"motion": "arm"},
        )
        exporter = SceneExporter(self.sim, VisualizationCfg(env_ids=[0]))
        preview = SequencePreview(
            session,
            exporter,
            PreviewPlaybackCfg(opacity=PREVIEW_OPACITY),
        )
        preview.register()
        exporter.build_manifest()
        return session, preview, exporter

    def test_preview_matches_simulated_link_poses_at_the_current_qpos(self) -> None:
        _, preview, exporter = self._make_preview()
        link_names = exporter.preview_link_names(preview.cfg.group_id)
        assert link_names

        update = preview.preview_update_for(self.robot.get_qpos()[0])

        simulated = self.robot.body_data.body_link_pose[0].detach().cpu().numpy()
        expected = np.stack(
            [simulated[self.robot.link_names.index(name), :3] for name in link_names]
        )
        np.testing.assert_allclose(
            update.positions,
            expected,
            atol=FK_POSITION_TOLERANCE_M,
        )

    def test_preview_frames_leave_the_simulation_untouched(self) -> None:
        session, preview, exporter = self._make_preview()
        session.add_card(
            "move_end_effector",
            card_id="move",
            params={"position": MEE_HOVER_POSITION},
        )
        session.add_card(
            "move_end_effector",
            card_id="back",
            params={"position": MEE_SECOND_POSITION},
        )
        assert session.compile(), [card.failure_message for card in session.cards]
        assert preview.length > 1

        qpos_before = self.robot.get_qpos().clone()
        target_qpos_before = self.robot.get_qpos(target=True).clone()
        link_pose_before = self.robot.body_data.body_link_pose.clone()

        preview.play()
        updates = []
        for _ in range(5):
            updates.append(preview.preview_update())
            preview.advance()
        overlays = preview.overlays()
        preview.seek(preview.length - 1)
        final_update = preview.preview_update()

        assert torch.equal(self.robot.get_qpos(), qpos_before)
        assert torch.equal(self.robot.get_qpos(target=True), target_qpos_before)
        assert torch.equal(self.robot.body_data.body_link_pose, link_pose_before)

        # The preview really moved and stayed finite.
        assert all(update.visible for update in updates)
        assert not np.allclose(updates[0].positions, final_update.positions)
        assert np.isfinite(final_update.positions).all()
        assert final_update.positions.shape == (
            len(exporter.preview_link_names(preview.cfg.group_id)),
            3,
        )

        # The end-effector polyline ends on the final previewed link pose.
        polyline = overlays.trajectories[0]
        assert polyline.overlay_id == preview.cfg.trajectory_overlay_id
        assert polyline.points.shape[0] >= 2
        np.testing.assert_allclose(
            polyline.points[-1],
            final_update.positions[-1],
            atol=FK_POSITION_TOLERANCE_M,
        )

    def test_captured_frame_publishes_translucent_preview_nodes(self) -> None:
        session, preview, exporter = self._make_preview()
        session.add_card(
            "move_end_effector",
            card_id="move",
            params={"position": MEE_HOVER_POSITION},
        )
        assert session.compile(), [card.failure_message for card in session.cards]

        manifest = exporter.build_manifest()
        preview_nodes = [node for node in manifest.nodes if node.kind == "preview_link"]
        assert preview_nodes
        assert all(node.opacity == PREVIEW_OPACITY for node in preview_nodes)

        preview.seek(preview.length - 1)
        update = preview.preview_update()
        frame = exporter.capture(
            sim_step=0,
            sim_time=0.0,
            preview_updates=(update,),
        ).frame

        node_ids = exporter.preview_node_ids(preview.cfg.group_id)
        indices = [frame.node_ids.index(node_id) for node_id in node_ids]
        assert frame.visible[indices].all()
        np.testing.assert_allclose(
            frame.positions[indices],
            update.positions,
            atol=FK_POSITION_TOLERANCE_M,
        )
