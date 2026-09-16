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

"""Translucent preview playback for compiled skill sequences.

:class:`SequencePreview` is the simulation-thread glue between an
:class:`~embodichain.lab.visualization.authoring.session.AuthoringSession` and
the scene exporter's generic preview-node capability. It owns a playback
cursor over the compiled trajectory and turns the session's
:meth:`~embodichain.lab.visualization.authoring.session.AuthoringSession.preview_qpos`
waypoints into detached link poses for a translucent copy of the robot.

The driver is strictly read-only with respect to simulation state. Link poses
come from the robot's analytic forward kinematics evaluated on candidate joint
positions, never from writing joint targets, so generating a preview frame
leaves robot joints, object poses, and the physics clock untouched. It also
never touches viser: it returns immutable protocol values that the caller
hands to
:meth:`~embodichain.lab.visualization.scene_exporter.SceneExporter.capture`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from embodichain.utils import configclass

from ..cfg import PreviewGroupCfg
from ..protocol import PreviewNodeUpdate, SceneOverlays, TrajectoryOverlay

if TYPE_CHECKING:
    from embodichain.lab.sim.objects import Robot

    from ..scene_exporter import SceneExporter
    from .session import AuthoringSession

__all__ = [
    "PreviewPlaybackCfg",
    "PreviewPlaybackState",
    "SequencePreview",
]


@configclass
class PreviewPlaybackCfg:
    """Configure translucent preview playback of a compiled sequence.

    Args:
        group_id: Preview group identifier registered on the scene exporter.
        env_id: Environment whose compiled waypoints are previewed. The same
            index selects the batch row of ``preview_qpos`` and the
            environment the preview meshes are drawn in.
        opacity: Constant transparency of the preview meshes in ``(0, 1]``.
        color: RGB tint of the preview meshes. ``None`` reuses the real robot's
            link geometry entries instead of uploading a tinted copy.
        link_names: Robot links included in the preview. ``None`` previews
            every link that has renderable geometry.
        step_stride: Waypoints advanced by one :meth:`SequencePreview.advance`
            tick while playing.
        loop: Whether playback wraps to the first waypoint at the end.
        autoplay: Whether a freshly compiled sequence starts playing.
        trajectory_overlay_id: Overlay identifier of the end-effector polyline.
            ``None`` disables the polyline.
        trajectory_link_name: Link traced by the polyline. ``None`` traces the
            last previewed link.
        trajectory_color: RGB color of the polyline.
        trajectory_line_width: Polyline width in pixels.
        trajectory_max_points: Upper bound on polyline samples. Longer
            trajectories are uniformly subsampled.
    """

    group_id: str = "authoring_preview"
    env_id: int = 0
    opacity: float = 0.35
    color: tuple[int, int, int] | None = (120, 190, 255)
    link_names: list[str] | None = None
    step_stride: int = 1
    loop: bool = True
    autoplay: bool = False
    trajectory_overlay_id: str | None = "authoring_preview_path"
    trajectory_link_name: str | None = None
    trajectory_color: tuple[int, int, int] = (255, 170, 30)
    trajectory_line_width: float = 3.0
    trajectory_max_points: int = 512

    def __post_init__(self) -> None:
        """Validate the playback and appearance settings."""
        if not self.group_id:
            raise ValueError("group_id must not be empty.")
        if self.env_id < 0:
            raise ValueError("env_id must be non-negative.")
        if not 0.0 < self.opacity <= 1.0:
            raise ValueError("opacity must be greater than zero and at most one.")
        if self.step_stride < 1:
            raise ValueError("step_stride must be at least one.")
        if self.trajectory_overlay_id is not None and not self.trajectory_overlay_id:
            raise ValueError("trajectory_overlay_id must be None or non-empty.")
        if self.trajectory_line_width <= 0.0:
            raise ValueError("trajectory_line_width must be greater than zero.")
        if self.trajectory_max_points < 2:
            raise ValueError("trajectory_max_points must be at least two.")


@dataclass(frozen=True)
class PreviewPlaybackState:
    """Immutable playback snapshot published to browser-facing UI layers."""

    group_id: str
    length: int
    cursor: int
    playing: bool
    loop: bool
    step_stride: int
    registered: bool = False
    active_card_id: str | None = None

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ValueError("PreviewPlaybackState.group_id must not be empty.")
        if self.length < 0:
            raise ValueError("PreviewPlaybackState.length must be non-negative.")
        if self.cursor < 0:
            raise ValueError("PreviewPlaybackState.cursor must be non-negative.")
        if self.length == 0:
            if self.cursor != 0:
                raise ValueError("An empty preview must keep the cursor at zero.")
        elif self.cursor >= self.length:
            raise ValueError("PreviewPlaybackState.cursor must be below length.")
        if self.step_stride < 1:
            raise ValueError("PreviewPlaybackState.step_stride must be at least one.")


@dataclass
class _TrajectoryCache:
    """Polyline points cached for one compiled trajectory identity."""

    trajectory_key: int
    link_name: str
    points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )


def _invert_rigid(transform: torch.Tensor) -> torch.Tensor:
    """Return the exact inverse of a batch of homogeneous rigid transforms."""
    rotation = transform[..., :3, :3].transpose(-1, -2)
    translation = transform[..., :3, 3].unsqueeze(-1)
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation
    inverse[..., :3, 3] = -(rotation @ translation).squeeze(-1)
    inverse[..., 3, 3] = 1.0
    return inverse


class SequencePreview:
    """Play a compiled skill sequence back on a translucent preview robot.

    The driver keeps a playback cursor over the session's compiled trajectory
    and converts the selected waypoint into preview-node poses. Frames are
    produced without stepping physics or writing joint state:
    :meth:`preview_update` evaluates analytic forward kinematics on candidate
    joint positions and only reads the robot's current link poses to anchor the
    result in the environment frame. Two small forward-kinematics evaluations
    are therefore performed per preview frame, which keeps mobile bases correct
    and the preview aligned with the rendered robot at a negligible cost
    compared with a mutate-and-restore approach.

    Args:
        session: Authoring session owning the compiled trajectory.
        exporter: Scene exporter that materializes the preview nodes.
        cfg: Playback and appearance settings. Defaults are used when omitted.
    """

    def __init__(
        self,
        session: AuthoringSession,
        exporter: SceneExporter,
        cfg: PreviewPlaybackCfg | None = None,
    ) -> None:
        self._session = session
        self._exporter = exporter
        self.cfg = cfg if cfg is not None else PreviewPlaybackCfg()
        self._cursor = 0
        self._playing = False
        self._trajectory_key: int | None = None
        self._trajectory_cache: _TrajectoryCache | None = None
        self._fk_kwargs: dict[str, object] | None = None

    # ------------------------------------------------------------------
    # Scene registration
    # ------------------------------------------------------------------

    @property
    def group_cfg(self) -> PreviewGroupCfg:
        """Preview group configuration derived from the playback settings."""
        return PreviewGroupCfg(
            group_id=self.cfg.group_id,
            articulation_uid=str(self._session.robot.uid),
            source_kind="robot",
            env_id=self.cfg.env_id,
            link_names=(
                None if self.cfg.link_names is None else list(self.cfg.link_names)
            ),
            opacity=self.cfg.opacity,
            color=self.cfg.color,
            visible=False,
        )

    @property
    def is_registered(self) -> bool:
        """Whether the exporter's current manifest holds this preview group."""
        try:
            self._exporter.preview_link_names(self.cfg.group_id)
        except KeyError:
            return False
        return True

    def register(self) -> PreviewGroupCfg:
        """Add this preview group to the exporter, keeping other groups.

        Registration only changes the exporter's pending topology. The caller
        must publish a new manifest, for example through
        :meth:`~embodichain.lab.visualization.runtime.VisualizationRuntime.refresh_scene`,
        before :meth:`preview_update` can produce poses.

        Returns:
            The registered preview group configuration.
        """
        group = self.group_cfg
        groups = [
            existing
            for existing in self._exporter.preview_groups
            if existing.group_id != group.group_id
        ]
        groups.append(group)
        self._exporter.set_preview_groups(groups)
        return group

    def unregister(self) -> None:
        """Remove this preview group from the exporter's pending topology."""
        self._exporter.set_preview_groups(
            [
                existing
                for existing in self._exporter.preview_groups
                if existing.group_id != self.cfg.group_id
            ]
        )

    # ------------------------------------------------------------------
    # Playback cursor
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Waypoint count of the session's compiled trajectory, or zero."""
        return int(self._session.preview_length)

    @property
    def cursor(self) -> int:
        """Current waypoint index, clamped into the compiled range."""
        self.sync()
        return self._cursor

    @property
    def is_playing(self) -> bool:
        """Whether :meth:`advance` moves the cursor."""
        self.sync()
        return self._playing

    def sync(self) -> bool:
        """Reconcile the cursor with the session's current compilation.

        Returns:
            Whether the compiled trajectory changed since the last call.
        """
        compiled = self._session.compiled_trajectory
        key = None if compiled is None else id(compiled)
        changed = key != self._trajectory_key
        if changed:
            self._trajectory_key = key
            self._trajectory_cache = None
            self._cursor = 0
            self._playing = bool(self.cfg.autoplay) and compiled is not None
        length = self.length
        if length == 0:
            self._cursor = 0
            self._playing = False
        elif self._cursor >= length:
            self._cursor = length - 1
        return changed

    def play(self) -> bool:
        """Start playback and return whether it is now running."""
        self.sync()
        if self.length == 0:
            return False
        self._playing = True
        return True

    def pause(self) -> None:
        """Stop playback without moving the cursor."""
        self.sync()
        self._playing = False

    def toggle(self) -> bool:
        """Flip playback and return whether it is now running."""
        if self.is_playing:
            self.pause()
            return False
        return self.play()

    def seek(self, index: int) -> int:
        """Move the cursor to ``index``, clamped into the compiled range.

        Args:
            index: Requested waypoint index.

        Returns:
            The resulting cursor position.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("seek index must be an integer.")
        self.sync()
        length = self.length
        if length == 0:
            return 0
        self._cursor = max(0, min(index, length - 1))
        return self._cursor

    def step(self, delta: int = 1) -> int:
        """Nudge the cursor by ``delta`` waypoints and pause playback.

        Stepping always clamps at the trajectory bounds, even when ``loop`` is
        enabled, so a browser frame-by-frame control never wraps unexpectedly.

        Args:
            delta: Signed waypoint offset.

        Returns:
            The resulting cursor position.
        """
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise TypeError("step delta must be an integer.")
        self.sync()
        self._playing = False
        return self.seek(self._cursor + delta)

    def advance(self) -> int:
        """Apply one playback tick and return the resulting cursor.

        The cursor only moves while playing. Without ``loop`` the cursor stops
        on the final waypoint and playback pauses.
        """
        self.sync()
        length = self.length
        if not self._playing or length == 0:
            return self._cursor
        target = self._cursor + int(self.cfg.step_stride)
        if target <= length - 1:
            self._cursor = target
        elif self.cfg.loop:
            self._cursor = target % length
        else:
            self._cursor = length - 1
            self._playing = False
        return self._cursor

    def active_card_id(self) -> str | None:
        """Return the card owning the current waypoint, if any."""
        self.sync()
        if self.length == 0:
            return None
        for card in self._session.cards:
            if card.segment_start is None or card.segment_stop is None:
                continue
            if card.segment_start <= self._cursor < card.segment_stop:
                return card.card_id
        return None

    def state(self) -> PreviewPlaybackState:
        """Return the immutable playback snapshot for browser-facing layers."""
        self.sync()
        return PreviewPlaybackState(
            group_id=self.cfg.group_id,
            length=self.length,
            cursor=self._cursor,
            playing=self._playing,
            loop=bool(self.cfg.loop),
            step_stride=int(self.cfg.step_stride),
            registered=self.is_registered,
            active_card_id=self.active_card_id(),
        )

    # ------------------------------------------------------------------
    # Frame production
    # ------------------------------------------------------------------

    def preview_update(self) -> PreviewNodeUpdate:
        """Return the preview-node poses of the current waypoint.

        Returns:
            An update holding detached CPU poses. The update hides the preview
            nodes when the sequence has no successful compilation.

        Raises:
            RuntimeError: If the preview group is not part of the exporter's
                current manifest.
        """
        link_names = self._preview_link_names()
        self.sync()
        if self.length == 0 or not link_names:
            return self._hidden_update(len(link_names))
        qpos = self._session.preview_qpos(self._cursor)
        return self.preview_update_for(qpos[self._env_row(qpos)])

    def preview_update_for(self, qpos: torch.Tensor) -> PreviewNodeUpdate:
        """Return preview-node poses for an arbitrary robot configuration.

        This bypasses the playback cursor, letting a UI layer preview a hovered
        waypoint or an externally proposed configuration. Like
        :meth:`preview_update`, it only reads simulation state.

        Args:
            qpos: Full-robot joint positions with shape ``(robot_dof,)`` or
                ``(1, robot_dof)``.

        Returns:
            An update holding detached CPU poses.

        Raises:
            RuntimeError: If the preview group is not part of the exporter's
                current manifest.
            ValueError: If ``qpos`` does not describe one configuration.
        """
        link_names = self._preview_link_names()
        candidate = qpos if qpos.dim() == 2 else qpos.unsqueeze(0)
        if candidate.dim() != 2 or candidate.shape[0] != 1:
            raise ValueError(
                "preview qpos must have shape (robot_dof,) or (1, robot_dof), "
                f"received {tuple(qpos.shape)}."
            )
        if not link_names:
            return self._hidden_update(0)
        poses = self._link_poses(candidate, link_names)
        return PreviewNodeUpdate.from_matrices(
            self.cfg.group_id,
            poses.detach().to("cpu").numpy(),
            visible=True,
        )

    def overlays(self) -> SceneOverlays:
        """Return the optional end-effector polyline for the compiled path.

        Returns:
            Overlays holding at most one trajectory polyline. The result is
            empty when the polyline is disabled or nothing is compiled.
        """
        overlay_id = self.cfg.trajectory_overlay_id
        if overlay_id is None:
            return SceneOverlays()
        points = self._trajectory_points()
        if points.shape[0] < 2:
            return SceneOverlays()
        return SceneOverlays(
            trajectories=(
                TrajectoryOverlay(
                    overlay_id=overlay_id,
                    points=points,
                    color=tuple(self.cfg.trajectory_color),
                    line_width=float(self.cfg.trajectory_line_width),
                ),
            ),
        )

    def capture_inputs(self) -> tuple[tuple[PreviewNodeUpdate, ...], SceneOverlays]:
        """Return the ``(preview_updates, overlays)`` pair for one capture.

        This is the convenience entry point for a per-frame visualization loop
        that has no other preview groups or overlays to merge.
        """
        return (self.preview_update(),), self.overlays()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hidden_update(self, node_count: int) -> PreviewNodeUpdate:
        """Return an update that parks and hides every preview node."""
        return PreviewNodeUpdate(
            group_id=self.cfg.group_id,
            positions=np.zeros((node_count, 3), dtype=np.float32),
            wxyz=np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                (node_count, 1),
            ),
            visible=False,
        )

    def _preview_link_names(self) -> tuple[str, ...]:
        """Return the manifest link order for this preview group."""
        try:
            return self._exporter.preview_link_names(self.cfg.group_id)
        except KeyError as error:
            raise RuntimeError(
                f"Preview group {self.cfg.group_id!r} is not part of the current "
                "scene manifest. Call register() and publish a new manifest first."
            ) from error

    def _env_row(self, qpos: torch.Tensor) -> int:
        """Return the batch row of ``qpos`` previewed by this driver."""
        env_id = int(self.cfg.env_id)
        if env_id >= int(qpos.shape[0]):
            raise IndexError(
                f"Preview env_id {env_id} is outside the compiled batch of "
                f"{int(qpos.shape[0])} environments."
            )
        return env_id

    def _forward_kinematics_kwargs(self, robot: Robot) -> dict[str, object]:
        """Return FK keyword arguments that align qpos with the chain order."""
        if self._fk_kwargs is not None:
            return self._fk_kwargs
        kwargs: dict[str, object] = {}
        chain = getattr(robot, "pk_chain", None)
        joint_names = getattr(robot, "joint_names", None)
        if chain is not None and joint_names is not None:
            chain_names = tuple(chain.get_joint_parameter_names())
            names = tuple(joint_names)
            if (
                chain_names
                and len(names) == len(chain_names)
                and set(names) == set(chain_names)
                and names != chain_names
            ):
                kwargs["qpos_joint_names"] = list(names)
        self._fk_kwargs = kwargs
        return kwargs

    def _chain_poses(
        self,
        qpos: torch.Tensor,
        link_names: Sequence[str],
    ) -> torch.Tensor:
        """Return chain-root-relative link poses for candidate joint positions.

        Args:
            qpos: Candidate joint positions with shape ``(count, robot_dof)``.
            link_names: Link names in the requested order.

        Returns:
            Homogeneous poses with shape ``(count, len(link_names), 4, 4)``.
        """
        robot = self._session.robot
        names = list(link_names)
        poses = robot.compute_fk(
            qpos=qpos,
            link_names=names,
            **self._forward_kinematics_kwargs(robot),
        )
        return poses.reshape(-1, len(names), 4, 4)

    def _anchor_transforms(self, link_names: Sequence[str]) -> torch.Tensor:
        """Return per-link chain-frame to environment-frame transforms.

        The robot's kinematic chain reports poses relative to its chain root,
        and a chain's fixed offsets can differ slightly from the loaded
        simulation model. Anchoring every link separately, by comparing its
        current analytic pose with its current simulated pose, makes the
        preview coincide exactly with the rendered robot whenever the previewed
        configuration equals the current one, and keeps mobile bases correct.
        Reading simulated link poses is the only simulation access a preview
        frame performs; nothing is written.

        Args:
            link_names: Link names in the requested order.

        Returns:
            Homogeneous transforms with shape ``(len(link_names), 4, 4)``.
        """
        robot = self._session.robot
        names = list(link_names)
        env_id = int(self.cfg.env_id)
        current_qpos = robot.get_qpos()[env_id].unsqueeze(0)
        chain_poses = self._chain_poses(current_qpos, names)[0]
        local_poses = torch.stack(
            [
                robot.get_link_pose(
                    link_name=name,
                    env_ids=[env_id],
                    to_matrix=True,
                )[0]
                for name in names
            ]
        )
        return local_poses.to(chain_poses.dtype) @ _invert_rigid(chain_poses)

    def _link_poses(
        self,
        qpos: torch.Tensor,
        link_names: Sequence[str],
    ) -> torch.Tensor:
        """Return environment-local link poses for one candidate configuration.

        Args:
            qpos: Candidate joint positions with shape ``(1, robot_dof)``.
            link_names: Previewed link names in preview-node order.

        Returns:
            Homogeneous poses with shape ``(len(link_names), 4, 4)``.
        """
        names = list(link_names)
        chain_poses = self._chain_poses(qpos, names)[0]
        return self._anchor_transforms(names) @ chain_poses

    def _trajectory_points(self) -> np.ndarray:
        """Return the cached end-effector polyline of the compiled trajectory."""
        self.sync()
        compiled = self._session.compiled_trajectory
        if compiled is None:
            return np.empty((0, 3), dtype=np.float32)
        link_names = self._preview_link_names()
        if not link_names:
            return np.empty((0, 3), dtype=np.float32)
        traced = self.cfg.trajectory_link_name or link_names[-1]
        cache = self._trajectory_cache
        if (
            cache is not None
            and cache.trajectory_key == id(compiled)
            and cache.link_name == traced
        ):
            return cache.points
        positions = compiled.trajectory.positions
        env_row = self._env_row(positions)
        waypoints = positions[env_row]
        count = int(waypoints.shape[0])
        limit = int(self.cfg.trajectory_max_points)
        if count > limit:
            indices = torch.from_numpy(
                np.linspace(0, count - 1, num=limit).astype(np.int64)
            ).to(waypoints.device)
            waypoints = waypoints.index_select(0, indices)
        chain_poses = self._chain_poses(waypoints, [traced])[:, 0]
        poses = self._anchor_transforms([traced]) @ chain_poses
        points = poses[:, :3, 3].detach().to("cpu").numpy().astype(np.float32)
        self._trajectory_cache = _TrajectoryCache(
            trajectory_key=id(compiled),
            link_name=traced,
            points=points,
        )
        return points
