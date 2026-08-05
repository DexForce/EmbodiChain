# Newton Backend PR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Newton-backend PR. **Target 4 (multi-env) was found
already complete during execution** — EmbodiChain's spawn path already does
prototype+clone across arenas at spawn time, and Newton views already handle
multi-env entity lists. This plan therefore covers only **Target 5
(differentiable env for APG)** plus branch cleanup and docs.

**Architecture:** A new `embodichain.lab.sim.diff` package provides a
`torch.autograd.Function` bridge over
`dexsim.engine.newton_physics.DifferentiableStepper`; a new
`DifferentiableEmbodiedEnv` gym subclass wires it into the standard
EmbodiChain env step pipeline. `SimulationManager` gains thin delegators to
dexsim's `create_differentiable_stepper` / `create_gradient_rollout`.

**Revision history:** Original plan had Tasks 1–4 covering multi-env clone
scaffolding, spawn guards, clone-at-finalize, and body-id resolution. Those
were deleted after code inspection showed the spawn path
(`spawn_rigid_object_entities` → `_spawn_clones_from_prototype`) already
clones prototypes into all arenas at spawn time, Newton views already accept
multi-entity lists, and existing tests (`TestRigidObjectNewton` with
`NUM_ARENAS=2`, `test_spawn_clones_distinct_entities`,
`test_newton_native_attrs_desc_native_spawn` asserting
`obj.num_instances == NUM_ARENAS`) already pass. Task numbers below are
rebased: old Task 5 → Task 1, old Task 6 → Task 2, etc.

**Tech Stack:** Python 3.10+, PyTorch (autograd), NVIDIA Warp (`wp.Tape`,
`wp.to_torch`/`wp.from_torch`), DexSim Newton physics
(`dexsim.engine.newton_physics`), gymnasium, pytest.

**Companion spec:** `docs/superpowers/specs/2026-06-21-newton-backend-pr-design.md`

---

## File Map

**Created:**
- `embodichain/lab/sim/diff/__init__.py` — public re-exports for the diff package
- `embodichain/lab/sim/diff/bridge.py` — `NewtonStepFunc(torch.autograd.Function)`, `tape_context`, `differentiable_step`
- `embodichain/lab/gym/envs/differentiable_env.py` — `DifferentiableEmbodiedEnv` subclass
- `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` — Franka APG example task
- `tests/sim/test_differentiable_stepper.py`
- `tests/gym/envs/test_differentiable_env.py`
- `agent_context/topics/differentiable-env.md`

**Modified:**
- `embodichain/lab/sim/sim_manager.py` — add `create_differentiable_stepper` / `create_gradient_rollout` delegators
- `agent_context/MAP.yaml` — register new `differentiable-env` topic
- `design/newton-backend-design.md` — mark Target 5 done, link to plan

**Already complete (Target 4, verified during execution):**
- Multi-env clone-at-spawn: `embodichain/lab/sim/utility/sim_utils.py:spawn_rigid_object_entities` / `spawn_articulation_entities` already prototype-then-clone across all arenas via dexsim's `clone_actor_to` (Newton-patched).
- Newton multi-env views: `embodichain/lab/sim/objects/backends/newton.py:NewtonRigidBodyView` / `NewtonArticulationView` already accept `Sequence[MeshObject]` and resolve one body ID per entity.
- Newton multi-env tests: `tests/sim/objects/test_rigid_object.py::TestRigidObjectNewton` (NUM_ARENAS=2, `test_spawn_clones_distinct_entities`, `test_newton_native_attrs_desc_native_spawn` asserting `obj.num_instances == NUM_ARENAS`), `tests/sim/objects/test_articulation.py::TestArticulationNewton` (num_envs=2), `tests/sim/objects/test_robot.py` (num_envs=10).

---


## Task 1: Add `create_differentiable_stepper` / `create_gradient_rollout` delegators

**Files:**
- Modify: `embodichain/lab/sim/sim_manager.py`
- Test: `tests/sim/test_differentiable_stepper.py`

- [ ] **Step 1: Write the failing test**

Create `tests/sim/test_differentiable_stepper.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Tests for the differentiable-stepper delegators on SimulationManager."""

from __future__ import annotations

import pytest

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.sim.sim_manager import SimulationManager, SimulationManagerCfg


def test_default_backend_rejects_differentiable_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=DefaultPhysicsCfg(), num_envs=1, headless=True,
    ))
    with pytest.raises(Exception, match=r"Newton"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_without_grad_rejects_differentiable_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(requires_grad=False, use_cuda_graph=False),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    with pytest.raises(Exception, match=r"grad"):
        sim.create_differentiable_stepper()
    SimulationManager.reset()


def test_newton_with_grad_creates_stepper():
    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            requires_grad=True,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        ),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    stepper = sim.create_differentiable_stepper()
    from dexsim.engine.newton_physics.differentiable_stepper import (
        DifferentiableStepper,
    )
    assert isinstance(stepper, DifferentiableStepper)
    SimulationManager.reset()
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: FAIL — `create_differentiable_stepper` not defined.

- [ ] **Step 3: Add the delegator methods**

Edit `embodichain/lab/sim/sim_manager.py` — add near the other Newton
back-compat delegators (search `newton_manager` in the file to locate the
right region):

```python
    def create_differentiable_stepper(self):
        """Create a single-step differentiable physics primitive (Newton-only).

        Requires the Newton backend with ``requires_grad=True`` and
        ``solver_type="semi_implicit"``. Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_differentiable_stepper`.

        Raises:
            RuntimeError: If the active backend is not Newton or if the
                Newton manager is not ready / not in grad mode.
        """
        if not self.is_newton_backend:
            logger.log_error(
                "create_differentiable_stepper requires the Newton backend.")
        return self.physics.newton_manager.create_differentiable_stepper()

    def create_gradient_rollout(
        self,
        record_steps: int,
        substeps_per_record: int | None = None,
        record_dt: float | None = None,
    ):
        """Create a gradient rollout buffer (Newton-only).

        Delegates to
        :meth:`dexsim.engine.newton_physics.NewtonManager.create_gradient_rollout`.
        """
        if not self.is_newton_backend:
            logger.log_error(
                "create_gradient_rollout requires the Newton backend.")
        return self.physics.newton_manager.create_gradient_rollout(
            record_steps=record_steps,
            substeps_per_record=substeps_per_record,
            record_dt=record_dt,
        )
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add embodichain/lab/sim/sim_manager.py tests/sim/test_differentiable_stepper.py
git commit -m "feat(sim): SimulationManager delegators for Newton diff stepper

create_differentiable_stepper and create_gradient_rollout are thin
passthroughs to NewtonManager. Both raise on the default backend.
Backs the new embodichain.lab.sim.diff package (next commit)."
```

---

## Task 2: Create the `embodichain.lab.sim.diff` package — bridge

**Files:**
- Create: `embodichain/lab/sim/diff/__init__.py`
- Create: `embodichain/lab/sim/diff/bridge.py`

The bridge wraps a `wp.Tape()` around one EmbodiChain physics step and
exposes a `torch.autograd.Function` so callers can drive APG with
PyTorch-side action tensors.

- [ ] **Step 1: Create the package skeleton**

Create `embodichain/lab/sim/diff/__init__.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Differentiable Newton stepping for EmbodiChain.

Bridges DexSim's :class:`~dexsim.engine.newton_physics.DifferentiableStepper`
into PyTorch autograd via a :class:`torch.autograd.Function`, and exposes a
:class:`tape_context` manager for advanced users who want to compose their
own Warp kernels.
"""

from __future__ import annotations

from .bridge import (
    NewtonStepFunc,
    differentiable_step,
    tape_context,
)

__all__ = [
    "NewtonStepFunc",
    "differentiable_step",
    "tape_context",
]
```

- [ ] **Step 2: Create `bridge.py`**

Create `embodichain/lab/sim/diff/bridge.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Warp-tape <-> PyTorch-autograd bridge for Newton physics."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator

import torch
import warp as wp

if TYPE_CHECKING:
    from embodichain.lab.sim.sim_manager import SimulationManager

__all__ = ["NewtonStepFunc", "differentiable_step", "tape_context"]


@contextmanager
def tape_context(manager: "SimulationManager") -> Iterator[wp.Tape]:
    """Open a Warp tape bound to the manager's Newton state.

    Advanced users compose their own Warp kernels inside this context, then
    call ``tape.backward()`` outside the with-block.
    """
    if not manager.is_newton_backend:
        raise RuntimeError(
            "tape_context requires the Newton backend with requires_grad=True.")
    tape = wp.Tape()
    with tape:
        yield tape


def differentiable_step(
    manager: "SimulationManager",
    *,
    apply_control_fn: Callable[[wp.Tape], None],
    substeps: int,
    dt: float | None = None,
) -> dict:
    """Run one EmbodiChain-level physics step inside a Warp tape.

    Args:
        manager: The owning :class:`SimulationManager` (must be Newton).
        apply_control_fn: Callable that writes the joint/body control
            targets inside the tape. Invoked once at the start of the
            step. Receives the open tape; must launch Warp kernels (or
            call dexsim setters that are tape-aware) to populate
            ``manager.physics.newton_manager._control``.
        substeps: Number of solver substeps to run (typically
            ``sim_cfg.sim_steps_per_control``).
        dt: Solver dt; defaults to the manager's configured dt.

    Returns:
        A dict carrying the tape and the state buffers for the caller to
        save in autograd context.
    """
    if not manager.is_newton_backend:
        raise RuntimeError(
            "differentiable_step requires the Newton backend.")
    nm = manager.physics.newton_manager
    stepper = manager.create_differentiable_stepper()
    state_in = nm._state_0
    state_out = nm._model.state()
    contacts = stepper.create_contacts()
    dt_val = nm.solver_dt if dt is None else float(dt)

    tape = wp.Tape()
    with tape:
        apply_control_fn(tape)
        for _ in range(substeps):
            stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
            state_in, state_out = state_out, state_in

    # The final state lives in state_in after the swap.
    return {
        "tape": tape,
        "final_state": state_in,
        "stepper": stepper,
    }


class NewtonStepFunc(torch.autograd.Function):
    """torch.autograd.Function bridging Warp tape autodiff to PyTorch.

    Forward: launches the action-to-control Warp kernel, runs
    ``substeps`` differentiable solver steps, and reads observation /
    reward as torch tensors via ``wp.to_torch`` (zero-copy where
    possible).

    Backward: copies upstream grads into the corresponding Warp
    ``.grad`` buffers, calls ``tape.backward()``, and returns
    ``wp.to_torch(action.grad)`` reshaped to the action's tensor shape.

    Callers must supply a ``sim_state`` dict with the following keys:
        manager: SimulationManager (Newton, requires_grad=True)
        substeps: int
        action_to_control_kernel: callable(action_wp, *kernel_args)
        kernel_args: tuple consumed by action_to_control_kernel
        obs_reward_fn: callable(final_state) -> dict with torch outputs
    """

    @staticmethod
    def forward(ctx, action_torch: torch.Tensor, sim_state: dict):
        manager = sim_state["manager"]
        substeps = int(sim_state["substeps"])
        kernel = sim_state["action_to_control_kernel"]
        kernel_args = sim_state["kernel_args"]
        obs_reward_fn = sim_state["obs_reward_fn"]

        nm = manager.physics.newton_manager
        stepper = manager.create_differentiable_stepper()

        action_flat = action_torch.detach().clone().reshape(-1).contiguous()
        action_wp = wp.from_torch(action_flat, dtype=wp.float32, requires_grad=True)

        state_in = nm._state_0
        state_out = nm._model.state()
        contacts = stepper.create_contacts()
        dt_val = nm.solver_dt

        tape = wp.Tape()
        with tape:
            kernel(action_wp, *kernel_args)  # writes nm._control inside tape
            for _ in range(substeps):
                stepper.step(state_in, state_out, contacts=contacts, dt=dt_val)
                state_in, state_out = state_out, state_in

        outputs = obs_reward_fn(state_in)
        ctx.tape = tape
        ctx.action_wp = action_wp
        ctx.outputs_wp = outputs.get("_grad_track", {})
        # `outputs` is a dict of torch tensors built from wp.to_torch — the
        # caller is responsible for ensuring at least one is grad-tracked.
        return tuple(outputs[k] for k in outputs["_order"])

    @staticmethod
    def backward(ctx, *grad_outputs):
        # Copy each upstream grad back into the corresponding Warp .grad.
        for name, grad_t in zip(ctx.outputs_wp["_order"], grad_outputs):
            wp_arr = ctx.outputs_wp[name]
            if grad_t is None or wp_arr.grad is None:
                continue
            wp.copy(wp_arr.grad,
                    wp.from_torch(grad_t.detach().clone().contiguous(),
                                  dtype=wp.float32))
        ctx.tape.backward()
        action_grad = wp.to_torch(ctx.action_wp.grad).clone()
        ctx.tape.zero()
        # Reshape to the original action layout; second input (sim_state)
        # has no gradient.
        return action_grad.reshape(ctx.saved_action_shape), None
```

> Note: the contract between `obs_reward_fn` and `NewtonStepFunc.backward`
> is intentionally explicit — the caller (the env in Task 3) constructs the
> dict in a way that records which outputs need grad-tracking. The
> `_order` / `_grad_track` plumbing keeps the autograd function fully
> general; the env class hides it from end users.

- [ ] **Step 3: Lightweight import smoke**

Run: `python -c "from embodichain.lab.sim.diff import NewtonStepFunc, tape_context, differentiable_step; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Append a tape-smoke test**

Append to `tests/sim/test_differentiable_stepper.py`:

```python
def test_tape_context_records_step():
    import warp as wp

    sim = SimulationManager(SimulationManagerCfg(
        physics_cfg=NewtonPhysicsCfg(
            requires_grad=True,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        ),
        num_envs=1, headless=True,
    ))
    sim.finalize_newton_physics()
    from embodichain.lab.sim.diff import tape_context

    with tape_context(sim) as tape:
        pass  # empty tape is valid; tape.backward() on empty is a no-op

    assert isinstance(tape, wp.Tape)
    SimulationManager.reset()
```

- [ ] **Step 5: Run all diff-stepper tests**

Run: `pytest -q tests/sim/test_differentiable_stepper.py`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/sim/diff/__init__.py \
        embodichain/lab/sim/diff/bridge.py \
        tests/sim/test_differentiable_stepper.py
git commit -m "feat(sim/diff): Warp-tape <-> PyTorch-autograd bridge

New embodichain.lab.sim.diff package: NewtonStepFunc (autograd.Function)
wraps DifferentiableStepper inside a wp.Tape, tape_context is the
low-level context manager for advanced kernels, differentiable_step is
the convenience wrapper. Foundation for DifferentiableEmbodiedEnv."
```

---

## Task 3: `DifferentiableEmbodiedEnv` gym subclass

**Files:**
- Create: `embodichain/lab/gym/envs/differentiable_env.py`
- Test: `tests/gym/envs/test_differentiable_env.py`

- [ ] **Step 1: Inspect `EmbodiedEnv.step` signature**

Run: `Read embodichain/lab/gym/envs/embodied_env.py` (focus on `step`,
`reset`, `_preprocess_action`, `_step_action`).

Identify exactly which methods produce the per-step `obs, reward, done,
info`. The override must invoke the same observation/reward managers as
the base class — just inside a tape.

- [ ] **Step 2: Write the construction-validation test first**

Create `tests/gym/envs/test_differentiable_env.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Tests for DifferentiableEmbodiedEnv."""

from __future__ import annotations

import pytest
import torch

from embodichain.lab.sim.cfg import DefaultPhysicsCfg, NewtonPhysicsCfg
from embodichain.lab.gym.envs.differentiable_env import (
    DifferentiableEmbodiedEnv,
)
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg


def _diff_env_cfg(requires_grad: bool = True, backend: str = "newton") -> EmbodiedEnvCfg:
    from embodichain.lab.sim.sim_manager import SimulationManagerCfg

    if backend == "newton":
        physics_cfg = NewtonPhysicsCfg(
            requires_grad=requires_grad,
            solver_cfg={"solver_type": "semi_implicit"},
            use_cuda_graph=False,
        )
    else:
        physics_cfg = DefaultPhysicsCfg()
    sim_cfg = SimulationManagerCfg(
        physics_cfg=physics_cfg, num_envs=2, headless=True,
    )
    return EmbodiedEnvCfg(sim_cfg=sim_cfg)


def test_construct_without_requires_grad_raises():
    with pytest.raises(Exception, match=r"requires_grad"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(requires_grad=False))


def test_construct_on_default_backend_raises():
    with pytest.raises(Exception, match=r"Newton"):
        DifferentiableEmbodiedEnv(_diff_env_cfg(backend="default"))
```

- [ ] **Step 3: Run and confirm failure**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py`
Expected: FAIL — `DifferentiableEmbodiedEnv` not defined.

- [ ] **Step 4: Implement `DifferentiableEmbodiedEnv`**

Create `embodichain/lab/gym/envs/differentiable_env.py`:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Differentiable Newton-backed EmbodiedEnv for analytic policy gradient.

Wraps the standard :class:`EmbodiedEnv` step pipeline in a Warp tape and
bridges autograd into PyTorch via
:class:`embodichain.lab.sim.diff.NewtonStepFunc`. Subclasses define how
actions become Newton control writes and how observations/rewards are
read from the post-step state; the bridge handles the tape lifecycle
and the backward pass.

Usage:

    class MyTask(DifferentiableEmbodiedEnv):
        def _apply_action_kernel(self, action_wp, tape): ...
        def _read_outputs(self, final_state) -> dict: ...
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import torch

from embodichain.lab.gym.envs.embodied_env import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.sim.cfg import NewtonPhysicsCfg
from embodichain.lab.sim.diff import NewtonStepFunc
from embodichain.utils import logger

__all__ = ["DifferentiableEmbodiedEnv"]


class DifferentiableEmbodiedEnv(EmbodiedEnv):
    """EmbodiedEnv variant that exposes APG-ready :py:meth:`step`.

    Subclasses must implement :meth:`_apply_action_kernel` and
    :meth:`_read_outputs`; the rest of the EmbodiedEnv contract (reset,
    observation managers, reward functors) carries over.
    """

    def __init__(self, cfg: EmbodiedEnvCfg, *args, **kwargs) -> None:
        self._validate_diff_cfg(cfg)
        super().__init__(cfg, *args, **kwargs)
        self._truncate_backward_at: int | None = getattr(
            cfg, "truncate_backward_at", None,
        )

    @staticmethod
    def _validate_diff_cfg(cfg: EmbodiedEnvCfg) -> None:
        physics_cfg = cfg.sim_cfg.physics_cfg
        if not isinstance(physics_cfg, NewtonPhysicsCfg):
            logger.log_error(
                "DifferentiableEmbodiedEnv requires NewtonPhysicsCfg, "
                f"got {type(physics_cfg).__name__}.")
        if not physics_cfg.requires_grad:
            logger.log_error(
                "DifferentiableEmbodiedEnv requires requires_grad=True on "
                "the NewtonPhysicsCfg.")

    # -- subclass contract ------------------------------------------------ #

    @abstractmethod
    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        """Inside the open Warp tape, write the action into Newton control.

        Implementations launch a Warp kernel that reads ``action_wp``
        (a ``wp.array(dtype=wp.float32, requires_grad=True)`` of shape
        ``[num_envs * action_dim]``) and writes into
        ``self.sim.physics.newton_manager._control`` so the next stepper
        call uses the new control.
        """

    @abstractmethod
    def _read_outputs(self, final_state: Any) -> dict:
        """Read the post-step observation and reward as torch tensors.

        Must return a dict with keys ``"obs"``, ``"reward"``,
        ``"terminated"``, ``"truncated"``, ``"info"``, plus the
        ``_order``/``_grad_track`` metadata expected by
        :class:`NewtonStepFunc`. ``obs`` and ``reward`` should be torch
        tensors backed by ``wp.to_torch`` of grad-tracked Warp arrays.
        """

    # -- gym surface ------------------------------------------------------ #

    def step(self, action: torch.Tensor):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32)
        sim_state = self._build_sim_state_dict(action)
        outputs = NewtonStepFunc.apply(action, sim_state)
        obs, reward, terminated, truncated = outputs[:4]
        info = sim_state["last_info"]

        done_mask = terminated | truncated
        if done_mask.any():
            reset_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
            fresh_obs, _ = self.reset(env_ids=reset_ids)
            obs = torch.where(
                done_mask.unsqueeze(-1).expand_as(obs),
                fresh_obs.detach(), obs,
            )
        return obs, reward, terminated, truncated, info

    def _build_sim_state_dict(self, action: torch.Tensor) -> dict:
        # Pack the args NewtonStepFunc expects. Subclass-supplied kernel
        # + output reader; environment-level metadata stays here.
        return {
            "manager": self.sim,
            "substeps": self.sim_cfg.sim_steps_per_control,
            "action_to_control_kernel": self._wrap_action_kernel(),
            "kernel_args": (),
            "obs_reward_fn": self._read_outputs,
            "last_info": {},
        }

    def _wrap_action_kernel(self):
        env = self
        def _inner(action_wp, *_):
            env._apply_action_kernel(action_wp, tape=None)
        return _inner
```

- [ ] **Step 5: Re-run construction tests**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py::test_construct_without_requires_grad_raises tests/gym/envs/test_differentiable_env.py::test_construct_on_default_backend_raises`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/gym/envs/differentiable_env.py \
        tests/gym/envs/test_differentiable_env.py
git commit -m "feat(gym): DifferentiableEmbodiedEnv for APG

Newton-only EmbodiedEnv subclass that wraps step() in a Warp tape via
NewtonStepFunc. Subclasses implement _apply_action_kernel and
_read_outputs; the base class handles validation, auto-reset on done,
and the autograd bridge."
```

---

## Task 4: Franka reach APG example task

**Files:**
- Create: `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py`
- Test: `tests/gym/envs/test_differentiable_env.py` (append)

Model the example after
`/root/sources/analytic_policy_gradients/envs/franka_reach_env.py`, but
built on EmbodiChain primitives (`add_robot` with the Franka URDF, the
`DifferentiableEmbodiedEnv` base).

- [ ] **Step 1: Locate Franka URDF in EmbodiChain data**

Run: `find embodichain/data -iname "fr3*.urdf" -o -iname "*franka*.urdf" | head -5`

If no URDF is bundled, the example accepts a `urdf_path` override and
falls back to `newton.utils.download_asset("franka_emika_panda")`,
matching the reference env.

- [ ] **Step 2: Write the example task**

Create `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` (full
contents below):

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ----------------------------------------------------------------------------
"""Franka FR3 reach task with differentiable Newton physics (APG)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import warp as wp

from embodichain.lab.gym.envs.differentiable_env import DifferentiableEmbodiedEnv
from embodichain.lab.gym.envs.embodied_env import EmbodiedEnvCfg
from embodichain.lab.gym.utils.registry import register_env
from embodichain.lab.sim.cfg import (
    NewtonPhysicsCfg,
    RobotCfg,
    URDFCfg,
)
from embodichain.lab.sim.sim_manager import SimulationManagerCfg

FRANKA_NUM_ARM_JOINTS = 7
FRANKA_EE_BODY = "fr3_hand_tcp"
DEFAULT_ACTION_SCALE = 0.2
DEFAULT_MAX_EPISODE_STEPS = 30
TARGET_POS_RANGE = {
    "x": (0.05, 0.70),
    "y": (-0.45, 0.45),
    "z": (0.20, 0.95),
}
TARGET_MAX_TILT = math.pi / 3


@wp.kernel
def _set_joint_targets_kernel(
    action: wp.array(dtype=wp.float32),
    current_q: wp.array(dtype=wp.float32),
    target_q: wp.array(dtype=wp.float32),
    limit_lo: wp.array(dtype=wp.float32),
    limit_hi: wp.array(dtype=wp.float32),
    action_scale: wp.float32,
    n_joints_per_env: wp.int32,
    n_arm: wp.int32,
    total: wp.int32,
):
    tid = wp.tid()
    if tid < total:
        env_idx = tid / n_arm
        j = tid % n_arm
        off = env_idx * n_joints_per_env + j
        new_q = current_q[off] + action[tid] * action_scale
        target_q[off] = wp.clamp(new_q, limit_lo[j], limit_hi[j])


@register_env("FrankaReachApg-v0")
class FrankaReachApgEnv(DifferentiableEmbodiedEnv):
    """Differentiable Franka FR3 reach task.

    Built on EmbodiChain's :class:`DifferentiableEmbodiedEnv`; the
    Warp-tape bridge produces ``action.grad`` that flows back through the
    semi-implicit Newton solver.
    """

    metadata = {"render_modes": ["human"], "default_num_envs": 4}

    def __init__(
        self,
        cfg: EmbodiedEnvCfg | None = None,
        *,
        num_envs: int = 4,
        urdf_path: str | None = None,
        action_scale: float = DEFAULT_ACTION_SCALE,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        device: str = "cuda:0",
    ) -> None:
        if cfg is None:
            cfg = EmbodiedEnvCfg(
                sim_cfg=SimulationManagerCfg(
                    physics_cfg=NewtonPhysicsCfg(
                        device=device,
                        requires_grad=True,
                        solver_cfg={"solver_type": "semi_implicit"},
                        use_cuda_graph=False,
                    ),
                    num_envs=num_envs,
                    headless=True,
                ),
            )
        self._urdf_path = urdf_path
        self._action_scale = float(action_scale)
        self._max_episode_steps = int(max_episode_steps)
        super().__init__(cfg)
        self._init_franka()
        self._init_targets()

    # -- scene setup ----------------------------------------------------- #

    def _init_franka(self) -> None:
        urdf = self._urdf_path or self._resolve_default_urdf()
        robot_cfg = RobotCfg(
            uid="franka",
            urdf_cfg=URDFCfg().set_urdf(urdf),
            fix_base=True,
        )
        self._robot = self.sim.add_robot(robot_cfg)
        self.sim.finalize_newton_physics()

        # Cache joint-limit Warp arrays for the action kernel.
        model = self.sim.physics.newton_manager._model
        lo = np.asarray(model.joint_limit_lower[:FRANKA_NUM_ARM_JOINTS],
                        dtype=np.float32)
        hi = np.asarray(model.joint_limit_upper[:FRANKA_NUM_ARM_JOINTS],
                        dtype=np.float32)
        self._limit_lo_wp = wp.array(lo, dtype=wp.float32, device=model.device)
        self._limit_hi_wp = wp.array(hi, dtype=wp.float32, device=model.device)
        self._n_joints_per_env = int(len(model.joint_q) // self.sim.num_envs)

    def _resolve_default_urdf(self) -> str:
        try:
            import newton.utils as nu

            urdf = nu.download_asset("franka_emika_panda") / (
                "urdf/fr3_franka_hand.urdf")
            if urdf.exists():
                return str(urdf)
        except Exception:
            pass
        raise FileNotFoundError(
            "Franka URDF not available; pass urdf_path explicitly.")

    def _init_targets(self) -> None:
        n = self.sim.num_envs
        device = self.device
        self.target_pos = torch.zeros(n, 3, device=device)
        self.target_quat = torch.zeros(n, 4, device=device)
        self.last_action = torch.zeros(
            n, FRANKA_NUM_ARM_JOINTS, device=device,
        )
        self.step_count = torch.zeros(n, dtype=torch.int32, device=device)
        self._sample_new_targets(torch.arange(n, device=device))

    def _sample_new_targets(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        d = self.device
        self.target_pos[env_ids, 0] = (
            TARGET_POS_RANGE["x"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["x"][1] - TARGET_POS_RANGE["x"][0]))
        self.target_pos[env_ids, 1] = (
            TARGET_POS_RANGE["y"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["y"][1] - TARGET_POS_RANGE["y"][0]))
        self.target_pos[env_ids, 2] = (
            TARGET_POS_RANGE["z"][0]
            + torch.rand(n, device=d)
            * (TARGET_POS_RANGE["z"][1] - TARGET_POS_RANGE["z"][0]))
        # Identity-ish quat, no tilt for the smoke task.
        self.target_quat[env_ids] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=d).expand(n, -1)

    # -- DifferentiableEmbodiedEnv contract ------------------------------ #

    def _apply_action_kernel(self, action_wp: Any, tape: Any) -> None:
        nm = self.sim.physics.newton_manager
        n_envs = self.sim.num_envs
        total = n_envs * FRANKA_NUM_ARM_JOINTS
        wp.launch(
            _set_joint_targets_kernel,
            dim=total,
            inputs=[
                action_wp,
                nm._state_0.joint_q,
                nm._control.joint_target,
                self._limit_lo_wp,
                self._limit_hi_wp,
                wp.float32(self._action_scale),
                wp.int32(self._n_joints_per_env),
                wp.int32(FRANKA_NUM_ARM_JOINTS),
                wp.int32(total),
            ],
            device=nm._model.device,
        )

    def _read_outputs(self, final_state: Any) -> dict:
        nm = self.sim.physics.newton_manager
        n = self.sim.num_envs
        body_q = wp.to_torch(final_state.body_q).view(n, -1, 7)
        ee_idx = self._ee_body_indices()
        ee_pose = body_q[torch.arange(n, device=self.device), ee_idx]
        eef_pos = ee_pose[:, :3]
        eef_quat = ee_pose[:, 3:]

        pos_dist = (eef_pos - self.target_pos).norm(dim=-1)
        rot_dist = self._quat_distance(eef_quat, self.target_quat)
        reward = (
            -0.2 * pos_dist
            + 0.1 * torch.exp(-(pos_dist ** 2) / (2 * 0.1 ** 2))
            - 0.1 * rot_dist
            + 0.1 * torch.exp(-(rot_dist ** 2) / (2 * 0.3 ** 2))
        )

        obs = torch.cat([
            wp.to_torch(final_state.joint_q).view(n, -1)[:, :FRANKA_NUM_ARM_JOINTS],
            ee_pose,
            self.target_pos,
            self.target_quat,
            self.last_action,
        ], dim=-1)

        terminated = (pos_dist < 0.01) & (rot_dist < 0.3)
        self.step_count += 1
        truncated = self.step_count >= self._max_episode_steps

        return {
            "_order": ("obs", "reward", "terminated", "truncated"),
            "_grad_track": {
                "obs": final_state.joint_q,
                "reward": None,  # reward is torch-built; tape flows via FK
            },
            "obs": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
        }

    def _ee_body_indices(self) -> torch.Tensor:
        if hasattr(self, "_cached_ee_idx"):
            return self._cached_ee_idx
        model = self.sim.physics.newton_manager._model
        idx_per_env = []
        n_per_env = len(model.body_label) // self.sim.num_envs
        for i in range(self.sim.num_envs):
            for j, label in enumerate(model.body_label):
                if FRANKA_EE_BODY in str(label) and (j // n_per_env) == i:
                    idx_per_env.append(j)
                    break
        self._cached_ee_idx = torch.tensor(idx_per_env, dtype=torch.long,
                                           device=self.device)
        return self._cached_ee_idx

    @staticmethod
    def _quat_distance(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return torch.minimum(((q1 - q2) ** 2).sum(-1),
                              ((q1 + q2) ** 2).sum(-1))

    # -- gym overrides --------------------------------------------------- #

    def reset(self, *, seed: int | None = None, env_ids=None):
        env_ids = (env_ids if env_ids is not None
                   else torch.arange(self.sim.num_envs, device=self.device))
        with torch.no_grad():
            self.step_count[env_ids] = 0
            self.last_action[env_ids] = 0.0
            self._sample_new_targets(env_ids)
            # Reset Newton joint_q to zero for the touched envs.
            jq = wp.to_torch(
                self.sim.physics.newton_manager._state_0.joint_q,
            ).view(self.sim.num_envs, -1)
            jq[env_ids] = 0.0
        obs = self._initial_obs()
        return obs, {}

    def _initial_obs(self) -> torch.Tensor:
        with torch.no_grad():
            return self._read_outputs(
                self.sim.physics.newton_manager._state_0)["obs"]
```

- [ ] **Step 3: Smoke import the task**

Run: `python -c "from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import FrankaReachApgEnv; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Append the smoke test**

Append to `tests/gym/envs/test_differentiable_env.py`:

```python
@pytest.mark.requires_gpu
def test_franka_apg_smoke_backward():
    try:
        from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
            FrankaReachApgEnv,
        )
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward.requires_grad, "Reward must be autograd-tracked."
    loss = reward.sum()
    loss.backward()
    assert action.grad is not None
    assert torch.isfinite(action.grad).all()
    env.close()


@pytest.mark.requires_gpu
def test_franka_apg_one_iter_loss_reduces():
    try:
        from embodichain.lab.gym.envs.tasks.special.franka_reach_apg import (
            FrankaReachApgEnv,
        )
    except FileNotFoundError as e:
        pytest.skip(f"Franka URDF not available: {e}")

    env = FrankaReachApgEnv(num_envs=2)
    env.reset(seed=0)
    action = torch.zeros(2, 7, requires_grad=True, device=env.device)
    opt = torch.optim.SGD([action], lr=0.01)

    losses = []
    for _ in range(3):
        env.reset(seed=0)
        opt.zero_grad()
        _, reward, _, _, _ = env.step(action)
        loss = (-reward).sum()
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())
    assert losses[-1] < losses[0], (
        f"APG did not reduce loss: {losses}")
    env.close()
```

- [ ] **Step 5: Run all differentiable-env tests on a GPU host**

Run: `pytest -q tests/gym/envs/test_differentiable_env.py`
Expected: 4 PASS (or smoke tests SKIPPED if URDF unavailable / no GPU).

- [ ] **Step 6: Commit**

```bash
git add embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py \
        tests/gym/envs/test_differentiable_env.py
git commit -m "feat(gym/tasks): Franka FR3 reach APG example

End-to-end APG smoke task built on DifferentiableEmbodiedEnv with a
Warp action-to-control kernel and torch-built reward. Verifies the
autograd bridge with one-iteration loss reduction. URDF resolved from
newton.utils.download_asset with explicit override."
```

---

## Task 5: Documentation — agent_context topic and design doc update

**Files:**
- Create: `agent_context/topics/differentiable-env.md`
- Modify: `agent_context/MAP.yaml`
- Modify: `design/newton-backend-design.md`

- [ ] **Step 1: Inspect MAP.yaml format**

Run: `Read agent_context/MAP.yaml`

Note the existing entries (`env-framework`, `manager-functor`, ...) and
mirror that structure.

- [ ] **Step 2: Create the topic file**

Create `agent_context/topics/differentiable-env.md`:

```markdown
# Differentiable Env (APG) Context

EmbodiChain supports analytic policy gradient (APG) via
:class:`embodichain.lab.gym.envs.differentiable_env.DifferentiableEmbodiedEnv`.
The bridge wraps `dexsim.engine.newton_physics.DifferentiableStepper`
inside a `wp.Tape()` and exposes a `torch.autograd.Function`
(`embodichain.lab.sim.diff.NewtonStepFunc`) so PyTorch-side `action`
tensors get a gradient from `tape.backward()`.

## Required configuration

- `NewtonPhysicsCfg(requires_grad=True, solver_cfg={"solver_type": "semi_implicit"})`
- `use_cuda_graph=False` (forced by dexsim when grad mode is on)

The default backend and any other Newton solver are rejected.

## Subclass contract

Task authors implement two methods on `DifferentiableEmbodiedEnv`:

- `_apply_action_kernel(action_wp, tape)` — launch a Warp kernel that
  writes joint/body targets into `nm._control` while the tape is open.
- `_read_outputs(final_state)` — build the `obs` / `reward` / `done`
  outputs as torch tensors via `wp.to_torch` so the tape can record the
  dependency.

See `embodichain/lab/gym/envs/tasks/special/franka_reach_apg.py` for the
canonical example.

## Functor autograd compatibility

Reward/observation functors that compose torch operations on tensors
obtained via `wp.to_torch` are automatically autograd-compatible.
Functors that detour through CPU / NumPy break the graph; those need
torch-only reimplementations for the differentiable path.

## Memory

Each step records `sim_steps_per_control` substeps into the tape. For
long horizons or large `num_envs`, pass `truncate_backward_at=K` on the
env config to split the tape and detach at chunk boundaries.
```

- [ ] **Step 3: Register the topic in `MAP.yaml`**

Edit `agent_context/MAP.yaml` — append:

```yaml
- id: differentiable-env
  aliases: ["apg", "analytic-policy-gradient", "differentiable-rl"]
  keywords: [differentiable, gradient, apg, autograd, warp tape]
  files:
    - topics/differentiable-env.md
```

- [ ] **Step 4: Update the Newton design doc**

Edit `design/newton-backend-design.md`:
- In the "Completion Plan -> Done" list, add items 13/14:
  - "13. Multi-env parallel via clone_arena_to (Target 4) — implemented."
  - "14. DifferentiableEmbodiedEnv via Warp-tape autograd bridge (Target 5) — implemented."
- In "Remaining", remove items 7 (rigid-only Newton gym smoke tests) and
  8 (gradient rollout wrapper + smoke test) since both are now covered.
- Append a "References" section line: "Implementation plan:
  `docs/superpowers/plans/2026-06-22-newton-backend-pr.md`."

- [ ] **Step 5: Commit**

```bash
git add agent_context/topics/differentiable-env.md \
        agent_context/MAP.yaml \
        design/newton-backend-design.md
git commit -m "docs: Newton multi-env + DifferentiableEmbodiedEnv

agent_context routing for the new differentiable-env topic, plus an
update to design/newton-backend-design.md marking Targets 4 and 5
done with a link to the implementation plan."
```

---

## Task 6: Branch cleanup and full test run

**Files:** none (git operations + verification only).

- [ ] **Step 1: Run the full Newton + diff suite**

Run:
```bash
pytest -q \
    tests/sim/test_backend_parity.py \
    tests/sim/test_newton_finalize_lifecycle.py \
    tests/sim/test_newton_multi_env.py \
    tests/sim/test_differentiable_stepper.py \
    tests/sim/test_physics_attrs.py \
    tests/sim/test_sim_manager_cfg.py \
    tests/sim/objects/test_rigid_object.py \
    tests/sim/objects/test_articulation.py::TestArticulationNewton \
    tests/sim/objects/test_robot.py::TestRobotNewton \
    tests/gym/envs/test_differentiable_env.py
```
Expected: all PASS (or GPU-marked tests SKIPPED on a headless host).

- [ ] **Step 2: Run pre-commit checks**

Run the `/pre-commit-check` skill — black, headers, type annotations,
exports, docstrings.

- [ ] **Step 3: Inspect the branch for `wip` commits to squash**

Run: `git log --oneline main..HEAD | grep -i wip`

If any remain, plan an interactive cleanup via `git rebase -i main` (the
existing CLAUDE.md disallows `-i`, so do this **manually** outside the
agent or skip squashing if maintainer prefers history-preserving merge).

- [ ] **Step 4: Create the PR**

Use the `/pr` skill. Title: `feat(sim): Newton physics backend with
multi-env and differentiable APG`. Body summary:

- Multi-env on Newton via implicit `clone_arena_to` at finalize.
- New `embodichain.lab.sim.diff` package and `DifferentiableEmbodiedEnv`
  for APG on the `semi_implicit` solver.
- Franka FR3 reach APG example task with a one-iter loss-reduction
  smoke test.
- Docs: agent_context routing + updated `design/newton-backend-design.md`.

Reference the design doc and this plan.

---

## Self-Review

**Spec coverage check (against the revised 6-task plan):**

- §2 multi-env — **already complete** (verified during execution; existing
  `spawn_rigid_object_entities` / `spawn_articulation_entities` prototype-then-clone
  at spawn, Newton views accept multi-entity lists, `TestRigidObjectNewton`
  with `NUM_ARENAS=2` passes). No task needed.
- §3 module layout (`diff/bridge.py`, `differentiable_env.py`, example) — Tasks 2, 3, 4.
- §3 `NewtonStepFunc` + `tape_context` — Task 2.
- §3 `DifferentiableEmbodiedEnv` validation + step pipeline — Task 3.
- §3 Franka APG example + smoke tests — Task 4.
- §4 manager delegators — Task 1.
- §5 risks: clone re-evaluation under mutation — **N/A** (no clone-at-finalize;
  cloning happens once at spawn, before finalize).
- §6 deferred items — out of scope (no tasks, per spec).
- §7 PR shape and commit plan — Task 6.
- §8 test files — Tasks 1, 2, 3, 4.
- §9 acceptance criteria — Task 6.

No gaps.

**Placeholder scan:**

- No "TBD" / "TODO" / "implement later" in step content.

**Type/signature consistency:**

- `NewtonStepFunc.apply(action, sim_state)` — Task 2 defines signature;
  Task 3 calls with the same args.
- `_apply_action_kernel(action_wp, tape)` — Task 3 abstract method;
  Task 4 implements with the same signature.
- `_read_outputs(final_state) -> dict` — Task 3 abstract method; Task 4
  returns the documented `_order` / `_grad_track` shape.

No drift.
