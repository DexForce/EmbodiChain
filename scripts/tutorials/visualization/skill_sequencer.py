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

"""Author, preview, and run an Atomic Skill sequence from the browser.

The tutorial builds a UR5 with a parallel-jaw gripper, a work surface, and a
graspable cube, then registers the browser skill-sequence authoring panel on the
Viser visualization runtime. Everything is driven from the browser; the terminal
only prints the server endpoint.

Example:

.. code-block:: bash

    python scripts/tutorials/visualization/skill_sequencer.py --viser

Open the printed endpoint, normally ``http://127.0.0.1:8080``, and use the
**Skill sequence** panel on the right:

1. **Pick a target.** Click the cube in the 3-D view. The panel's
   *Picked entity* line shows its UID.
2. **Choose a skill.** Select ``move_end_effector``, ``pick_up``, or ``place``
   in the *Skill* dropdown and press **Add card**. The new card appears in the
   *Cards* list as 🟡 ``unconfigured``.
3. **Configure the card.** Select it in *Selected card*, then press
   **Bind selected entity** for a ``pick_up`` card, or type a target position
   and press **Apply target position** for the other two skills. A configured
   card turns ⚪ ``ready``.
4. **Compile.** Press **Compile sequence**. Every card receives its waypoint
   range, or turns 🔴 ``failed`` with the planner's diagnostic.
5. **Preview.** A translucent copy of the robot replays the compiled
   trajectory with an orange end-effector polyline. *Play / Pause*, *Step*, and
   the *Frame* slider drive the cursor. Previewing never touches the physics
   world: the real robot and the cube do not move.
6. **Execute.** Press **Execute sequence**. The real robot replays the
   trajectory and the cards advance 🔵 ``running`` → 🟢 ``succeeded`` live,
   because the host loop drives the run one slice at a time.

A suggested first sequence is ``move_end_effector`` above the cube, ``pick_up``
bound to the cube, then ``place`` at a free spot on the surface.

Use ``--headless_smoke`` to run the same pipeline programmatically, without a
browser. It adds the three cards, compiles, advances the preview, executes, and
asserts the end-to-end invariants, which makes it usable as a regression check:

.. code-block:: bash

    python scripts/tutorials/visualization/skill_sequencer.py --headless_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Callable

import torch

from embodichain.lab.sim import SimulationManager
from embodichain.lab.sim.atomic_actions import (
    AtomicActionEngine,
    ControlPartCommandProfile,
)
from embodichain.lab.sim.cfg import RigidObjectCfg
from embodichain.lab.sim.objects import RigidObject, Robot
from embodichain.lab.sim.shapes import CubeCfg
from embodichain.lab.visualization import SceneExporter, VisualizationCfg
from embodichain.lab.visualization.authoring import (
    AddCard,
    AuthoringBridge,
    AuthoringSession,
    CompileSequence,
    PreviewPlaybackCfg,
    SequencePreview,
    SkillCardState,
    SkillSequencePanel,
    SkillSequencePanelCfg,
    StepwiseExecution,
    UpdateCard,
)
from embodichain.utils import logger
from scripts.tutorials.atomic_action.tutorial_utils import (
    add_ur5_gripper_robot,
    clone_local_pose_from_first_env,
    create_parallel_jaw_grasp_pose_generator,
    create_toppra_motion_generator,
    create_tutorial_argument_parser,
    create_tutorial_rigid_body_physics,
    create_tutorial_simulation,
    get_hand_open_close_qpos,
    initialize_pre_pick_robot_pose,
    run_tutorial,
)

OBJECT_UID = "cube"
OBJECT_SIZE = 0.05
OBJECT_XY = (-0.42, -0.08)
SURFACE_UID = "work_surface"
SURFACE_SIZE = (0.70, 0.70, 0.02)
SURFACE_CENTER_XY = (-0.31, 0.10)
SURFACE_TOP_Z = -0.002
HOVER_POSITION = (-0.42, -0.08, 0.36)
PLACE_POSITION = (-0.20, 0.28, 0.10)
GRIPPER_TCP_Z = 0.15
GRASP_SAMPLE_COUNT = 1000
EXECUTE_HOLD_STEPS = 30
SMOKE_PREVIEW_FRAMES = 24
MIN_PLACE_XY_DISPLACEMENT_M = 0.10
PREVIEW_TOLERANCE = 1.0e-6
SMOKE_ENV_ID = 0


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the skill-sequencer tutorial."""
    parser = create_tutorial_argument_parser(
        "Author and run an Atomic Skill sequence from the Viser browser.",
        features=("grasp_sampling",),
    )
    parser.set_defaults(n_sample=GRASP_SAMPLE_COUNT)
    parser.add_argument(
        "--headless_smoke",
        action="store_true",
        help=(
            "Drive the whole authoring pipeline programmatically without a "
            "browser, assert its invariants, and exit."
        ),
    )
    parser.add_argument(
        "--execute_hold_steps",
        type=int,
        default=EXECUTE_HOLD_STEPS,
        help="Final-pose hold updates appended to every execution.",
    )
    parser.add_argument(
        "--execution_steps_per_update",
        type=int,
        default=2,
        help=(
            "Trajectory waypoints replayed per browser frame while executing. "
            "Larger values replay faster at a coarser browser update rate."
        ),
    )
    parser.add_argument(
        "--smoke_preview_frames",
        type=int,
        default=SMOKE_PREVIEW_FRAMES,
        help="Preview frames produced by --headless_smoke.",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------


def create_work_surface(sim: SimulationManager) -> RigidObject:
    """Add a static slab that reads as a workbench under the skill targets.

    The slab's top face sits just below the ground plane, so it is a purely
    visual reference and every planning assumption of the atomic-action
    tutorials still holds.

    Args:
        sim: Simulation manager that owns the surface.

    Returns:
        The added rigid object.
    """
    return sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=SURFACE_UID,
            shape=CubeCfg(size=list(SURFACE_SIZE)),
            body_type="kinematic",
            init_pos=[
                *SURFACE_CENTER_XY,
                SURFACE_TOP_Z - 0.5 * SURFACE_SIZE[2],
            ],
        )
    )


def create_pick_object(sim: SimulationManager) -> RigidObject:
    """Add and settle the graspable cube used by the ``pick_up`` card.

    Args:
        sim: Simulation manager that owns the cube.

    Returns:
        The settled rigid object with its dynamics cleared.
    """
    obj = sim.add_rigid_object(
        cfg=RigidObjectCfg(
            uid=OBJECT_UID,
            shape=CubeCfg(size=[OBJECT_SIZE] * 3),
            attrs=create_tutorial_rigid_body_physics(
                mass=0.05,
                dynamic_friction=0.97,
                static_friction=0.99,
                enable_ccd=True,
                newton_contact=sim.is_newton_backend,
            ),
            init_pos=[*OBJECT_XY, OBJECT_SIZE],
        )
    )
    sim.prepare()
    sim.update(step=10)
    clone_local_pose_from_first_env(obj)
    obj.clear_dynamics()
    return obj


def create_engine(
    robot: Robot,
    args: argparse.Namespace,
) -> AtomicActionEngine:
    """Create the atomic-action engine backing the authoring session.

    Args:
        robot: Robot whose trajectories are planned.
        args: Parsed tutorial arguments carrying the grasp-sampling options.

    Returns:
        An engine with the tutorial gripper profile and grasp-pose generator.
    """
    hand_open, hand_close = get_hand_open_close_qpos(robot)
    return AtomicActionEngine(
        motion_generator=create_toppra_motion_generator(robot),
        control_profiles={
            "hand": ControlPartCommandProfile.joint_positions(
                open=hand_open,
                grasp=hand_close,
            )
        },
        grasp_pose_generators={
            "hand": create_parallel_jaw_grasp_pose_generator(
                n_sample=args.n_sample,
                force_refresh=args.force_reannotate,
            )
        },
    )


def build_scene(
    args: argparse.Namespace,
) -> tuple[SimulationManager, Robot, RigidObject, AuthoringSession]:
    """Assemble the tutorial scene and its authoring session.

    Args:
        args: Parsed tutorial arguments.

    Returns:
        The simulation manager, robot, graspable cube, and authoring session.
    """
    sim = create_tutorial_simulation(args)
    robot = add_ur5_gripper_robot(sim, tcp_z=GRIPPER_TCP_Z)
    create_work_surface(sim)
    obj = create_pick_object(sim)
    hand_open, _ = get_hand_open_close_qpos(robot)
    initialize_pre_pick_robot_pose(robot, obj, hand_open)
    session = AuthoringSession(
        robot=robot,
        engine=create_engine(robot, args),
        sim=sim,
        control_parts={"motion": "arm", "grasp": "hand"},
    )
    return sim, robot, obj, session


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------


def pick_lift_start(session: AuthoringSession) -> int | None:
    """Return the compiled waypoint where the first pick starts lifting.

    Args:
        session: Session holding a successful compilation.

    Returns:
        The global waypoint index, or ``None`` when the sequence has no
        ``pick_up`` card with a ``lift`` segment.
    """
    compiled = session.compiled_trajectory
    if compiled is None:
        return None
    for index, card in enumerate(session.cards):
        if card.skill_id != "pick_up":
            continue
        try:
            return int(compiled.segment(index, "lift").start)
        except (IndexError, KeyError, ValueError):
            return None
    return None


def make_hold_object_callback(
    session: AuthoringSession,
    obj: RigidObject,
) -> Callable[[int, int], None]:
    """Create an execution callback that freezes the grasped object once.

    Replaying joint positions does not simulate a real grasp force, so the
    residual velocity accumulated while the fingers close would throw the cube
    away at lift-off. Clearing its dynamics at the lift waypoint is the same
    trick the atomic-action tutorials use. The callback re-arms itself whenever
    a new run starts, so the panel's Execute button stays repeatable.

    Args:
        session: Session whose compiled segments locate the lift waypoint.
        obj: Object attached by the ``pick_up`` card.

    Returns:
        A callback matching the execution ``on_step`` signature.
    """
    cleared = False
    lift_start: int | None = None

    def on_step(step_index: int, total_steps: int) -> None:
        nonlocal cleared, lift_start
        if step_index == 0:
            cleared = False
            lift_start = pick_lift_start(session)
        if lift_start is None or cleared or step_index + 1 < lift_start:
            return
        obj.clear_dynamics()
        cleared = True

    return on_step


def create_preview(
    session: AuthoringSession,
    exporter: SceneExporter,
) -> SequencePreview:
    """Create and register the translucent preview driver.

    Args:
        session: Session owning the compiled trajectory.
        exporter: Scene exporter that materializes the preview nodes.

    Returns:
        The registered preview driver. The caller must still publish a new
        manifest before requesting a preview frame.
    """
    preview = SequencePreview(
        session,
        exporter,
        PreviewPlaybackCfg(autoplay=True, env_id=SMOKE_ENV_ID),
    )
    preview.register()
    return preview


# ----------------------------------------------------------------------------
# Interactive browser session
# ----------------------------------------------------------------------------


def run_interactive(
    args: argparse.Namespace,
    sim: SimulationManager,
    obj: RigidObject,
    session: AuthoringSession,
) -> None:
    """Serve the authoring panel until the user interrupts the process.

    Args:
        args: Parsed tutorial arguments.
        sim: Simulation manager stepped by the host loop.
        obj: Graspable cube frozen at the pick lift waypoint.
        session: Authoring session driven from the browser.
    """
    runtime = sim.visualization_runtime
    if runtime is None:
        raise RuntimeError(
            "The Viser runtime is not available. Re-run with --viser, or use "
            "--headless_smoke for the browser-free path."
        )
    preview = create_preview(session, runtime.exporter)
    runtime.refresh_scene()
    bridge = AuthoringBridge(
        session,
        runtime,
        SkillSequencePanel(
            SkillSequencePanelCfg(execute_hold_steps=args.execute_hold_steps)
        ),
        preview,
        process_picks=True,
        stepwise_execution=True,
        execution_steps_per_update=args.execution_steps_per_update,
        execution_on_step=make_hold_object_callback(session, obj),
    )
    bridge.register()

    logger.log_info(
        f"Skill sequencer ready at {runtime.endpoint}. "
        "Use the 'Skill sequence' panel; press Ctrl+C to exit.",
        color="green",
    )
    step = 0
    physics_dt = float(sim.sim_config.physics_dt)
    try:
        while True:
            # SimulationManager.update() drains the shared browser pick queue
            # through its Gizmo processing, so the bridge must take the picks
            # first or the panel never sees a click-pick.
            bridge.drain_picks()
            # A stepwise execution owns the simulation clock while it runs, so
            # the host loop must not step physics a second time.
            if not bridge.execution_active:
                sim.update(step=1)
                step += 1
            preview.advance()
            bridge.update()
            if bridge.execution_active:
                step += 1
            preview_updates, overlays = preview.capture_inputs()
            runtime.capture(
                sim_step=step,
                sim_time=step * physics_dt,
                overlays=overlays,
                preview_updates=preview_updates,
            )
    except KeyboardInterrupt:
        logger.log_info("Stopping the skill sequencer.")
    finally:
        bridge.unregister()
        preview.unregister()


# ----------------------------------------------------------------------------
# Headless smoke run
# ----------------------------------------------------------------------------


def _report(message: str) -> None:
    """Print one line of the headless smoke report."""
    print(f"[smoke] {message}", flush=True)


def _format_position(position: torch.Tensor) -> str:
    """Format a 3-vector for the smoke report."""
    return "(" + ", ".join(f"{float(value):+.3f}" for value in position[:3]) + ")"


def smoke_add_cards(session: AuthoringSession) -> None:
    """Add the demonstration sequence through the browser command path."""
    _report("2/7 adding cards through AuthoringSession.handle_command")
    commands = (
        AddCard(
            skill_id="move_end_effector",
            card_id="hover",
            params={"position": HOVER_POSITION},
        ),
        AddCard(skill_id="pick_up", card_id="pick"),
        UpdateCard(card_id="pick", entity_uid=OBJECT_UID),
        AddCard(
            skill_id="place",
            card_id="drop",
            params={"position": PLACE_POSITION},
        ),
    )
    for command in commands:
        snapshot = session.handle_command(command)
        states = ", ".join(
            f"{card.card_id}={card.state.value}" for card in snapshot.cards
        )
        _report(f"      {type(command).__name__:<12} -> [{states}]")
    assert [card.card_id for card in session.cards] == ["hover", "pick", "drop"]
    assert all(card.state is SkillCardState.READY for card in session.cards)


def smoke_compile(session: AuthoringSession) -> None:
    """Compile the sequence and assert contiguous per-card waypoint ranges."""
    snapshot = session.handle_command(CompileSequence())
    failures = [
        f"{card.card_id}: {card.failure_message}"
        for card in snapshot.cards
        if card.state is SkillCardState.FAILED
    ]
    assert snapshot.compiled, f"compilation failed: {failures}"
    _report(
        f"3/7 compiled: {len(snapshot.cards)} cards -> "
        f"{snapshot.trajectory_waypoint_count} waypoints"
    )
    expected_start = 0
    for card in snapshot.cards:
        _report(
            f"      {card.card_id:<6} {card.skill_id:<18} "
            f"segment [{card.segment_start}, {card.segment_stop}) "
            f"= {card.segment_waypoint_count} waypoints"
        )
        assert card.segment_start == expected_start
        assert card.segment_waypoint_count > 0
        expected_start = card.segment_stop
    assert expected_start == snapshot.trajectory_waypoint_count
    assert session.preview_length == snapshot.trajectory_waypoint_count


def smoke_preview(
    args: argparse.Namespace,
    robot: Robot,
    obj: RigidObject,
    session: AuthoringSession,
    preview: SequencePreview,
) -> None:
    """Advance the preview and assert that it leaves the world untouched."""
    qpos_before = robot.get_qpos().clone()
    pose_before = obj.get_local_pose(to_matrix=True).clone()
    link_count = 0
    cursors: list[int] = []
    active_cards: list[str] = []
    for _ in range(args.smoke_preview_frames):
        update = preview.preview_update()
        assert update.visible
        link_count = int(update.positions.shape[0])
        cursors.append(preview.cursor)
        card_id = preview.active_card_id()
        if card_id is not None and card_id not in active_cards:
            active_cards.append(card_id)
        preview.advance()
    seeks: list[tuple[int, str | None]] = []
    for card in session.cards:
        cursor = preview.seek(int(card.segment_start))
        resolved = preview.active_card_id()
        assert resolved == card.card_id
        preview.preview_update()
        seeks.append((cursor, resolved))
    preview.seek(0)
    overlays = preview.overlays()
    polyline_points = (
        0
        if not overlays.trajectories
        else int(overlays.trajectories[0].points.shape[0])
    )
    qpos_delta = float((robot.get_qpos() - qpos_before).abs().max().item())
    pose_delta = float(
        (obj.get_local_pose(to_matrix=True) - pose_before).abs().max().item()
    )
    _report(
        f"4/7 preview: {len(cursors)} frames, cursor {cursors[0]} -> {cursors[-1]}, "
        f"{link_count} preview links, {polyline_points} polyline points"
    )
    _report(
        f"      cards reached while previewing: {', '.join(active_cards) or 'none'}"
    )
    _report(
        "      seek per card: "
        + ", ".join(f"{cursor} -> {card_id}" for cursor, card_id in seeks)
    )
    _report(
        f"      physics untouched: max |dqpos| = {qpos_delta:.3e}, "
        f"max |dpose| = {pose_delta:.3e}"
    )
    assert len(cursors) > 1 and cursors[-1] > cursors[0]
    assert link_count > 0
    assert polyline_points >= 2
    assert qpos_delta < PREVIEW_TOLERANCE
    assert pose_delta < PREVIEW_TOLERANCE


def smoke_execute(
    args: argparse.Namespace,
    obj: RigidObject,
    session: AuthoringSession,
) -> None:
    """Execute the sequence stepwise and assert the live card transitions."""
    total = session.preview_length
    _report(
        f"5/7 executing stepwise: {total} waypoints + "
        f"{args.execute_hold_steps} hold steps"
    )
    execution = StepwiseExecution(
        session,
        make_hold_object_callback(session, obj),
        hold_steps=args.execute_hold_steps,
    )
    first_running: dict[str, int] = {}
    first_succeeded: dict[str, int] = {}
    ticks = 0
    while execution.is_active:
        progress = execution.advance()
        if progress is None:
            break
        ticks += 1
        if progress.holding:
            continue
        for card in session.cards:
            if card.state is SkillCardState.RUNNING:
                first_running.setdefault(card.card_id, progress.step_index)
            elif card.state is SkillCardState.SUCCEEDED:
                first_succeeded.setdefault(card.card_id, progress.step_index)
    assert execution.is_finished
    assert execution.succeeded, [card.failure_message for card in session.cards]
    for card in session.cards:
        _report(
            f"      {card.card_id:<6} running@{first_running.get(card.card_id, -1):<4} "
            f"succeeded@{first_succeeded.get(card.card_id, total)}"
        )
        assert card.card_id in first_running
    _report(f"      {ticks} host-driven ticks, browser never blocked")
    assert ticks == total + args.execute_hold_steps


def run_headless_smoke(
    args: argparse.Namespace,
    sim: SimulationManager,
    robot: Robot,
    obj: RigidObject,
    session: AuthoringSession,
) -> None:
    """Drive the whole authoring pipeline without a browser and check it.

    Args:
        args: Parsed tutorial arguments.
        sim: Simulation manager backing the scene exporter.
        robot: Robot commanded by the session.
        obj: Graspable cube.
        session: Authoring session under test.

    Raises:
        AssertionError: If any end-to-end invariant does not hold.
    """
    position_before = obj.get_local_pose(to_matrix=True)[0, :3, 3].clone()
    _report(
        f"1/7 scene ready: robot={robot.uid}, {OBJECT_UID} at "
        f"{_format_position(position_before)}"
    )
    smoke_add_cards(session)
    smoke_compile(session)

    exporter = SceneExporter(
        sim,
        VisualizationCfg(backend="none", env_ids=[SMOKE_ENV_ID]),
    )
    preview = create_preview(session, exporter)
    exporter.build_manifest()
    smoke_preview(args, robot, obj, session, preview)

    smoke_execute(args, obj, session)

    position_after = obj.get_local_pose(to_matrix=True)[0, :3, 3]
    displacement = float(
        torch.linalg.norm(position_after[:2] - position_before[:2]).item()
    )
    _report(
        f"6/7 object moved {_format_position(position_before)} -> "
        f"{_format_position(position_after)}, xy displacement {displacement:.3f} m"
    )
    assert displacement > MIN_PLACE_XY_DISPLACEMENT_M

    states = ", ".join(f"{card.card_id}={card.state.value}" for card in session.cards)
    _report(f"7/7 final card states: [{states}]")
    assert all(card.state is SkillCardState.SUCCEEDED for card in session.cards)
    preview.unregister()
    _report("PASS")


def main() -> None:
    """Build the scene and serve or self-check the skill sequencer."""
    args = parse_arguments()
    if args.headless_smoke and getattr(args, "viser", False):
        raise SystemExit("--headless_smoke cannot be combined with --viser.")
    if not args.headless_smoke and not getattr(args, "viser", False):
        raise SystemExit(
            "This tutorial is browser-driven: pass --viser to serve the panel, "
            "or --headless_smoke for the browser-free self-check."
        )
    sim, robot, obj, session = build_scene(args)
    if args.headless_smoke:
        run_headless_smoke(args, sim, robot, obj, session)
        return
    run_interactive(args, sim, obj, session)


if __name__ == "__main__":
    run_tutorial(main)
