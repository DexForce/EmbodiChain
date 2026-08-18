.. _tutorial_robot_articulation:

Opening a Drawer with Contact-Rich Motion
=========================================

This tutorial combines a driven Franka Panda with a passive drawer. The robot
approaches a handle, pulls the drawer open through gripper contact, and then
pushes it back to half of the measured opening. Arm motion is generated with
``MotionGenerator``; the drawer joint is never commanded directly.

.. raw:: html

   <figure style="margin: 1.5rem auto; max-width: 960px;">
     <video controls playsinline preload="metadata" style="width: 100%;"
            poster="../_static/tutorials/open_drawer_poster.jpg"
            aria-label="Franka opens a drawer and pushes it halfway closed">
       <source src="../_static/tutorials/open_drawer.mp4" type="video/mp4">
       Your browser does not support embedded MP4 video.
     </video>
     <figcaption style="text-align: center; margin-top: 0.5rem;">
       Franka approaches the handle, pulls the passive drawer open, and pushes
       it halfway back using three generated arm trajectories.
     </figcaption>
   </figure>

:download:`Download the MP4 video <../_static/tutorials/open_drawer.mp4>`.

Before starting, it helps to be familiar with :doc:`robot`,
:doc:`articulation`, and :doc:`motion_gen`.


Learning objectives
~~~~~~~~~~~~~~~~~~~

After completing this tutorial, you should understand how to:

- derive robot targets from a moving articulation link rather than fixed world
  coordinates;
- convert Cartesian task poses into joint waypoints and time-parameterized
  trajectories;
- separate planning, robot control, and physics interaction;
- use measured articulation state to plan the next contact phase;
- verify task success from object state instead of commanded robot motion.

The complete example is ``scripts/tutorials/sim/open_drawer.py``.

.. dropdown:: Complete open_drawer.py
   :icon: code

   .. literalinclude:: ../../../scripts/tutorials/sim/open_drawer.py
      :language: python
      :linenos:


Understand what is controlled
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The robot and drawer are both articulations, but they play different roles:

.. list-table:: Control boundary
   :header-rows: 1
   :widths: 22 31 47

   * - Entity
     - Command
     - Effect
   * - Franka arm
     - Arm joint-position targets
     - Tracks the generated trajectory through its joint drives.
   * - Franka gripper
     - Finger joint-position targets
     - Creates and maintains contact with the handle.
   * - Drawer
     - No joint target
     - Its passive prismatic joint moves only in response to contact forces.

The drawer uses ``drive_type="none"`` and a fixed base. Fixing the base does
not lock the slide joint; it only prevents the cabinet body from moving. Contact
friction is increased so the fingertips can retain the narrow handle during
the pull.

The full data flow is:

.. code-block:: text

   handle link pose
          ↓
   Cartesian TCP waypoints
          ↓  inverse kinematics
   arm joint waypoints
          ↓  MotionGenerator + TOPPRA
   time-sampled joint trajectory
          ↓  position drives + physics updates
   gripper contact moves the passive drawer

This distinction is important: ``MotionGenerator`` generates robot motion. It
does not generate a drawer trajectory or directly change drawer state.


Build targets in the handle frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The drawer URDF provides a link named ``handle_xpos``. Treating this link as the
task frame keeps all targets attached to the drawer when it moves.

The asset's handle frame already points TCP +Z along the approach axis. The
example post-multiplies its orientation by a 90-degree local-Z rotation:

.. math::

   R_{grasp} = R_{handle} R_z(\pi / 2)

Post-multiplication matters here: it rolls the gripper in the handle frame. The
TCP Z axis and approach direction remain unchanged, while the finger-closing
direction rotates to grip the handle vertically. A world-frame rotation would
also change the approach direction.

.. literalinclude:: ../../../scripts/tutorials/sim/open_drawer.py
   :language: python
   :start-at: def get_handle_grasp_pose(
   :end-before: def open_drawer(

For this asset, the three translations are:

.. code-block:: text

   pre-grasp = handle position - TCP_Z × 0.10 m
   pull      = live handle position - TCP_Z × 0.16 m
   push      = live handle position + TCP_Z × half the measured opening

The signs come from the bundled handle frame convention. For another asset,
inspect its task-frame axes before reusing them. The handle pose is read again
before the pull and push because previous contact phases may have moved the
drawer.


Turn task poses into trajectories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Trajectory generation is a three-stage process:

1. ``robot.compute_ik`` converts each sparse TCP pose into an arm joint
   waypoint. The previous IK result seeds the next solve, which encourages a
   continuous solution instead of switching kinematic branches.
2. ``MotionGenerator`` passes the joint waypoints to TOPPRA, applying velocity
   and acceleration constraints and sampling a smooth trajectory.
3. The script sends each sample to the arm position drives and advances
   physics. Generating a trajectory alone does not move the robot.

The tensors stay batched throughout this process. For ``B`` environments, the
generated arm positions have shape ``(B, samples, arm_dof)``. Consequently,
each environment can use a different measured drawer opening while sharing the
same planning code.

.. attention::

   TOPPRA time-parameterizes the supplied path but does not collision-check it.
   Keep the pre-grasp outside the object, inspect the motion in the viewer, and
   add collision-aware planning when obstacles make a straight approach unsafe.


Plan contact phases separately
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The example uses three arm trajectories instead of planning the entire task at
once:

1. **Approach:** move through the pre-grasp and handle poses with the gripper
   open.
2. **Pull:** close the fingers, let contact settle, then move 16 cm along TCP
   -Z.
3. **Push:** measure the achieved drawer opening and move half that distance
   back along TCP +Z while keeping the gripper closed.

Splitting the task is necessary because closing the gripper changes the contact
state, and the push target is not known until the pull has physically executed.
This is phase-level feedback: each trajectory is played open-loop, but the next
phase is planned from newly measured simulator state.

The push target uses the achieved opening rather than the requested 16 cm:

.. literalinclude:: ../../../scripts/tutorials/sim/open_drawer.py
   :language: python
   :start-at:     # Push the drawer back by half of its measured opening.
   :end-at:     push_pose[:, :3, 3] += pushed_handle_pose[:, :3, 2] * push_distance.unsqueeze(-1)

``pulled_opening`` is cloned before further simulation updates so the 50%
target remains fixed while the push executes.


Synchronize execution and verify the object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The approach trajectory is generated before execution starts. By default the
script then pauses at the terminal:

.. code-block:: text

   [READY]: Trajectory planned. Press Enter to start execution...

Pressing Enter starts the complete approach, grasp, pull, and push sequence.
This breakpoint is useful for checking the initial scene and robot state before
any arm target is applied. Use ``--auto-start`` only for unattended runs.

Success is evaluated from ``drawer.get_qpos()`` rather than the final TCP pose.
The script checks two facts:

- the pull opened every drawer by at least 10 cm;
- the push finished within 2 cm of half the opening actually achieved by that
  environment.

This catches contact failures that a robot-only trajectory check would miss.


Run the tutorial
~~~~~~~~~~~~~~~~

From the repository root, run with the native viewer:

.. code-block:: bash

   python scripts/tutorials/sim/open_drawer.py

For a non-interactive CPU run:

.. code-block:: bash

   python scripts/tutorials/sim/open_drawer.py \
       --headless \
       --device cpu \
       --hold-steps 0 \
       --auto-start

CUDA physics and multiple environments use the common launcher arguments:

.. code-block:: bash

   python scripts/tutorials/sim/open_drawer.py \
       --headless \
       --device cuda \
       --num_envs 4 \
       --auto-start

The embedded video was recorded directly from a fixed camera in headless mode.
You can reproduce it with:

.. code-block:: bash

   python scripts/tutorials/sim/open_drawer.py \
       --headless \
       --device cuda \
       --hold-steps 100 \
       --auto-start \
       --record-fps 30 \
       --record-save-path outputs/videos/open_drawer.mp4

Typical output is similar to:

.. code-block:: text

   [INFO]: Drawer opening after pull (m): [0.1606]
   [INFO]: Drawer opening after half push (m): [0.0811]

Exact values vary slightly because the drawer is moved through simulated
contact. ``--hold-steps`` controls how long the final pose remains visible.


Diagnose common failures
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Symptom
     - What to inspect
   * - IK fails before execution
     - Check handle-frame orientation, reachability, and the seeded arm
       configuration. Shorten the approach distance when necessary.
   * - The arm moves but the drawer does not
     - Confirm the drawer drive is passive, the fingertips close around the
       handle, and the contact materials provide enough friction.
   * - Pull succeeds but the push misses halfway
     - Re-read the handle pose after pulling and calculate the push from measured
       drawer position, not the requested pull distance.
   * - The arm intersects the cabinet
     - Add safe Cartesian waypoints or use a collision-aware planner; TOPPRA
       alone does not change the geometric path.


Adapt the pattern
~~~~~~~~~~~~~~~~~

For another prismatic mechanism, provide a task frame whose approach axis and
joint axis have known directions, then adjust the signed translation distances.

A revolute door needs a different geometric path: sample handle poses along an
arc around the hinge, keep the gripper orientation consistent with the door,
solve the poses sequentially with seeded IK, and pass those joint waypoints to
``MotionGenerator``. The remaining structure—contact transition, state
measurement, replanning, and object-state validation—stays the same.
