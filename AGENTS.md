# EmbodiChain — Developer Reference

## Package Name

**IMPORTANT**: The Python package name is `embodichain` (all lowercase, one word).
- Repository folder: `EmbodiChain` (PascalCase)
- Python package: `embodichain` (lowercase)

## Project Structure

```
EmbodiChain/
├── embodichain/                  # Main Python package
│   ├── agents/                   # AI agents
│   │   ├── datasets/             # Datasets and data loaders for model training
│   │   ├── engine/               # Online Data Streaming Engine
│   │   ├── hierarchy/            # LLM-based hierarchical agents (task, code, validation)
│   │   ├── mllm/                 # Multimodal LLM prompt scaffolding
│   │   └── rl/                   # RL agents: PPO algo, rollout buffer, actor-critic models
│   ├── data/                     # Assets, datasets, constants, enums
│   ├── lab/                      # Simulation lab
│   │   ├── gym/                  # OpenAI Gym-compatible environments
│   │   │   ├── envs/             # BaseEnv, EmbodiedEnv
│   │   │   │   ├── managers/     # Observation, event, reward, record, dataset managers
│   │   │   │   │   └── randomization/  # Physics, geometry, spatial, visual randomizers
│   │   │   │   ├── tasks/        # Task implementations (tableware, RL, special)
│   │   │   │   ├── action_bank/  # Configurable action primitives
│   │   │   │   └── wrapper/      # Env wrappers (e.g. no_fail)
│   │   │   └── utils/            # Gym registration, misc helpers
│   │   ├── sim/                  # Simulation core
│   │   │   ├── objects/          # Robot, RigidObject, Articulation, Light, Gizmo, SoftObject
│   │   │   ├── sensors/          # Camera, StereoCamera, BaseSensor
│   │   │   ├── robots/           # Robot-specific configs and params (dexforce_w1, cobotmagic)
│   │   │   ├── planners/         # Motion planners (TOPPRA, motion generator)
│   │   │   └── solvers/          # IK solvers (SRS, OPW, pink, pinocchio, pytorch)
│   │   ├── devices/              # Real-device controllers
│   │   └── scripts/              # Entry-point scripts (run_env, run_agent)
│   ├── toolkits/                 # Standalone tools
│   │   ├── graspkit/pg_grasp/    # Parallel-gripper grasp sampling
│   │   └── urdf_assembly/        # URDF builder utilities
│   └── utils/                    # Shared utilities
│       ├── configclass.py        # @configclass decorator
│       ├── logger.py             # Project logger
│       ├── math/                 # Tensor math helpers
│       └── warp/kinematics/      # GPU kinematics via Warp
├── configs/                      # Agent configs and task prompts (text/YAML)
├── docs/                         # Sphinx documentation source + build
│   └── source/                   # .md doc pages (overview, quick_start, features, resources)
├── tests/                        # Test suite
├── .github/                      # CI workflows and issue/PR templates
├── setup.py                      # Package setup
└── VERSION                       # Package version file
```

---

## Code Style

### Formatting

- **Formatter**: `black==24.3.0` — run before every commit.
  ```bash
  black .
  ```
- Use the `/pre-commit-check` skill before committing to catch all CI violations locally.

### File Headers

Every source file begins with the Apache 2.0 copyright header:

```python
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
```

### Type Annotations

- Use full type hints on all public APIs.
- Use `from __future__ import annotations` at the top of every file.
- Use `TYPE_CHECKING` guards for circular-import-safe imports.
- Prefer `A | B` over `Union[A, B]`.

### Configuration Pattern (`@configclass`)

All configuration objects use the `@configclass` decorator (similar to Isaac Lab's pattern):

```python
from embodichain.utils import configclass
from dataclasses import MISSING

@configclass
class MyManagerCfg:
    param_a: float = 1.0
    param_b: str = MISSING  # required — must be set by caller
```

### Functor / Manager Pattern

Managers (observation, event, reward, randomization) use a `Functor`/`FunctorCfg` pattern with two styles:

- **Function-style**: a plain function with signature `(env, env_ids, ...) -> None`.
- **Class-style**: a class inheriting `Functor`, with `__init__(cfg, env)` and `__call__(env, env_ids, ...)`.

Registered in a manager config via `FunctorCfg(func=..., params={...})`.

Use the `/add-functor` skill to scaffold new functors with the correct signature and module placement.

### Docstrings

Use Google-style docstrings with Sphinx directives:

```python
def my_function(env, env_ids, scale: float = 1.0) -> None:
    """Short one-line summary.

    Longer description if needed.

    .. attention::
        Note a non-obvious behavior here.

    .. tip::
        Helpful usage hint.

    Args:
        env: The environment instance.
        env_ids: Target environment IDs.
        scale: Scaling factor applied to the result.

    Returns:
        Description if not None.

    Raises:
        ValueError: If the entity type is unsupported.
    """
```

### Module Exports

Define `__all__` in every public module to declare the exported API:

```python
__all__ = ["MyClass", "my_function"]
```

### Documentation

- Docs are built with **Sphinx** using **Markdown** source files (`docs/source/`).
- Build locally:
  ```bash
  pip install -r docs/requirements.txt
  cd docs && make html
  # Preview at docs/build/html/index.html
  ```
- If you encounter locale errors: `export LC_ALL=C.UTF-8; export LANG=C.UTF-8`

---

## Contributing Guide

### Bug Reports

Use the **Bug Report** issue template (`.github/ISSUE_TEMPLATE/bug.md`). Title format: `[Bug Report] Short description`.

Include:
- Clear description of the bug
- Minimal reproduction steps and stack trace
- System info: commit hash, OS, GPU model, CUDA version, GPU driver version
- Confirm you checked for duplicate issues

### Feature Requests / Proposals

Use the **Proposal** issue template (`.github/ISSUE_TEMPLATE/proposal.md`). Title format: `[Proposal] Short description`.

Include:
- Description of the feature and its core capabilities
- Motivation and problem it solves
- Any related existing issues

### Pull Requests

1. **Fork** the repository and create a focused branch.
2. **Keep PRs small** — one logical change per PR.
3. **Format** the code with `black==24.3.0` before submitting.
4. **Update documentation** for any public API changes.
5. **Add tests** that prove your fix or feature works.
6. Use the `/pr` skill to create PRs following the project's template and label conventions.

### Adding a New Robot

Refer to `docs/source/tutorial/add_robot.rst` for a detailed guide. The basic structure requires:

- A config class (inheriting from `RobotCfg`)
- URDF configuration for the robot
- Control parts definition
- IK solver configuration
- Drive properties for joint physics

For complex robots with multiple variants (like `dexforce_w1`), use a package structure with `types.py`, `params.py`, `utils.py`, and `cfg.py`.

Also add robot documentation in `docs/source/resources/robot/` (see existing examples: `cobotmagic.md`, `dexforce_w1.md`) and update `docs/source/resources/robot/index.rst` to include the new robot.

### Adding a New Task Environment

Use the `/add-task-env` skill to scaffold a new task with the correct file structure, `@register_env` decorator, base class, and test stub.

### Adding Functors

Use the `/add-functor` skill to scaffold observation, reward, event, action, dataset, or randomization functors with the correct signature, style, and module placement.

### Writing Tests

Use the `/add-test` skill to scaffold tests with the correct file placement, style (pytest vs class), mock patterns, and project conventions.

---

## Skills Quick Reference

| Skill | Command | Purpose |
|-------|---------|---------|
| Add Task Env | `/add-task-env` | Scaffold a new `EmbodiedEnv` task |
| Add Functor | `/add-functor` | Scaffold observation/reward/event/action/dataset/randomization functors |
| Add Test | `/add-test` | Scaffold tests following project conventions |
| Pre-Commit Check | `/pre-commit-check` | Run all local CI checks before committing |
| Create PR | `/pr` | Create a PR following the project template |
| Benchmark | `/benchmark` | Write benchmark scripts for EmbodiChain modules |
