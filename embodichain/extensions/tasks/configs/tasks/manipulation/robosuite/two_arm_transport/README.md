# Robosuite TwoArmTransport

This native two-arm workcell adapts the
[TwoArmTransport benchmark](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/environments/manipulation/two_arm_transport.py).
The opposed Franka arms use the repository's Robotiq 2F-140 embodiment assets.
The tables, bins, lid, trash and hammer use native primitive geometry. The
hammer has a flat-sided handle for parallel-jaw contact.

The Task Program lifts and parks the lid, clears the target bin, lifts the
hammer, runs the HandOver skill from the source arm to the receiver, and
delivers the hammer. Final validation waits for stable support from the
respective bin floors and checks that both objects are inside. Separating
the floors from the walls makes the support test specific to the bottom of
each bin.
