# DexUni: Unified Physics for Embodied AI

Developed by the **DexForce Simulation Engine Team**, **DexUni** is a
GPU-accelerated physics solver for robot interaction and embodied AI. It brings
articulated robots, rigid objects, cloth, volume soft bodies, and pneumatic
membranes into a common scene workflow, with additional support for cables and
prototype liquid simulation.

DexUni is designed for tasks where contact changes an object's motion or shape:
folding a garment, squeezing a soft object, loading a flexible bag, or pouring
liquid between containers. It is available through DexSim's Newton backend and
is the default Newton solver configuration in EmbodiChain.

## Core advantages

- **Robot interaction with deformable objects.** Build manipulation scenes with
  cloth, thin membranes, and volume soft bodies. Contact, friction, and cloth
  self-collision support folding, dragging, pressing, and wrapping interactions.
- **Motion control suited to the task.** Use prescribed robot trajectories for
  replay, or dynamic joint control with optional contact feedback when the
  environment should affect robot motion.
- **Configurable physical behavior.** Set elastic, bending, friction, and damping
  properties for the supported objects. Pneumatic surfaces add pressure and
  volume controls for sealed, inflatable structures.
- **GPU execution for repeated interaction.** Combine CUDA simulation, batched
  environments, and CUDA Graph execution for compatible scenes to support
  repeated rollouts and data collection.
- **Integration with the task workflow.** Use EmbodiChain's Python APIs for scene
  creation, robot control, batched state access, and reset. Connect these scenes
  to the platform's rendering, sensors, task execution, and recording tools.

## Capability matrix

The matrix describes the current DexUni integration in EmbodiChain. Supported
scene combinations are summarized at the end of this page.

| Capability | Support | What it enables |
| :--- | :--- | :--- |
| Articulated robots | Supported | Joint-driven robot motion and prescribed trajectory replay |
| Rigid objects | Supported | Dynamic objects, collision, and interaction with deformables |
| Cloth and thin membranes | Supported | Stretching, shearing, bending, and cloth self-collision |
| Volume soft bodies | Supported | Elastic deformation and contact with robots and rigid objects |
| Pneumatic surfaces | Supported | Closed-cavity pressure response, pressure/volume controls, and state readback |
| Cables | Supported | Stretching, shearing, bending, twisting, and fixed endpoint attachments |
| Liquids | Prototype | Container interaction and robot-driven pouring |
| Robot contact feedback | Configurable | One-way interaction or two-way feedback for compatible dynamic-joint scenes |
| Object material settings | Supported | Object-specific elasticity, friction, and damping |
| GPU execution and CUDA Graphs | Supported | CUDA simulation and repeated execution for compatible scenes |
| Batched environments | Supported | Independent environments, batched state access, and selected-environment reset |

## Typical applications

| Application | Simulation focus | Starting point |
| :--- | :--- | :--- |
| Garment and cloth manipulation | Grasping, lifting, dragging, folding, and self-contact | [Cloth pickup](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/pick_up_cloth.py), [T-shirt folding](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/fold_tshirt.py) |
| Soft-object interaction | Pressing, deformation, and recovery after release | [Soft-body pressing](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/press_softbody.py) |
| Flexible packaging and inflatable structures | Bag deformation, object loading, and sealed-cavity pressure response | [Surface and cloth configuration](../../../sim_cloth.md) |
| Cable interaction | Bending, pulling, external loads, and fixed endpoint motion | [Cable creation](https://github.com/DexForce/EmbodiChain/blob/main/scripts/tutorials/sim/create_cable.py) |
| Robot pouring | Moving containers and transferring liquid | [Franka pouring demo](https://github.com/DexForce/EmbodiChain/blob/main/examples/sim/demo/franka_dexuni_pour_demo.py) |

These scenes can be used for manipulation development, repeated policy
evaluation, demonstration replay, and simulation data generation through
EmbodiChain's task and recording tools.

## Getting started

`NewtonPhysicsCfg()` selects DexUni by default. To select it explicitly:

```python
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

physics_cfg = NewtonPhysicsCfg(
    device="cuda:0",
    solver_cfg={"solver_type": "dexuni"},
    collision_cfg=None,
)
```

Pass this configuration to `SimulationManagerCfg.physics_cfg`. Declare scene
objects and register any prescribed trajectories before `sim.prepare()`, then
advance the simulation and read object states through `SimulationManager`.
Keep `collision_cfg=None` when selecting DexUni explicitly to use its contact
handling.

For asset setup and state access, see [volume soft bodies](../../../sim_soft_object.md)
and [surface cloth](../../../sim_cloth.md). The
[Newton Solver Guide](https://newton-physics.github.io/newton/stable/solvers/index.html)
describes Newton's native solver options.

## Common parameters

Set timing and execution parameters on `NewtonPhysicsCfg`. DexUni's interaction
parameters belong inside `solver_cfg`.

| Parameter | Purpose | Selection guidance |
| :--- | :--- | :--- |
| `physics_dt` | Duration of one physics step, in seconds | Choose it with the task's control timing; changing it can also change the control period. |
| `num_substeps` | Number of solver substeps per physics step | The solver interval is `physics_dt / num_substeps`. More substeps increase computation; assess stability and contact quality for the scene. |
| `use_cuda_graph` | Request CUDA Graph execution | Enable it for compatible CUDA scenes; check runtime status to confirm capture. |
| `solver_cfg.joint_mode` | Robot motion mode | Use `"dynamic"` for joint-driven dynamics or `"kinematic"` for prescribed motion. |
| `solver_cfg.coupling` | Robot/environment feedback | `"one_way"` lets the environment respond without feeding its reaction back to robot motion. Use `"two_way"` with dynamic joints for supported contact feedback. |
| `solver_cfg.contact_mode` | Contact configuration | Keep `"auto"` for general scene setup. |

The following is an illustrative starting point for a dynamic robot scene,
rather than a universal tuning preset:

```python
from embodichain.lab.sim.cfg import NewtonPhysicsCfg

physics_cfg = NewtonPhysicsCfg(
    device="cuda:0",
    physics_dt=1.0 / 100.0,
    num_substeps=10,
    use_cuda_graph=True,
    solver_cfg={
        "solver_type": "dexuni",
        "joint_mode": "dynamic",
        "coupling": "one_way",
        "contact_mode": "auto",
    },
    collision_cfg=None,
)
```

This example uses a solver interval of `0.001 s`. For prescribed trajectory
replay, select `joint_mode="kinematic"`. For compatible dynamic-joint scenes
that need contact feedback, select `coupling="two_way"`.

Elasticity, friction, and damping belong to the object or articulation material
configuration. See the asset guides linked above for material settings.

## Current limitations

- Cloth, soft bodies, cables, and liquids require CUDA and `requires_grad=False`.
  DexUni does not currently provide end-to-end differentiable simulation.
- Two-way robot feedback requires dynamic joints and compatible contact
  settings. Prescribed trajectories do not respond to reaction forces.
- Liquid support is a prototype with one-way rigid-to-liquid interaction and
  cannot be combined with cloth or volume soft bodies.
- Cables cannot share an arena with cloth, soft bodies, or liquids. Endpoint
  attachments are fixed at creation.
- Declare physical topology before `prepare()`; topology changes require a new
  scene. CUDA Graph availability depends on the scene and runtime controls.
- Contact geometry and measured impulses have different availability; honor
  per-row validity flags. See [runtime qualification](runtime_qualification.md)
  for detailed solver/device and contact-query requirements.
