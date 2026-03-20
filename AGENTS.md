# EmbodiChain — Developer Reference

## Package Name

**IMPORTANT**: The Python package name is `embodichain` (all lowercase, one word).
- Repository folder: `EmbodiChain` (PascalCase)
- Python package: `embodichain` (lowercase)
- NEVER use: `embodiedchain`, `embodyichain`, or any other variant

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

Managers (observation, event, reward, randomization) use a `Functor`/`FunctorCfg` pattern:

- **Function-style**: a plain function with signature `(env, env_ids, ...) -> None`.
- **Class-style**: a class inheriting `Functor`, with `__init__(cfg, env)` and `__call__(env, env_ids, ...)`.
- Registered in a manager config via `FunctorCfg(func=..., params={...})`.

```python
from embodichain.lab.gym.envs.managers import Functor, FunctorCfg

class my_randomizer(Functor):
    def __init__(self, cfg: FunctorCfg, env):
        super().__init__(cfg, env)

    def __call__(self, env, env_ids, my_param: float = 0.5):
        ...
```

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
6. **Submit** using the PR template (`.github/PULL_REQUEST_TEMPLATE.md`):
   - Summarize changes and link the related issue (`Fixes #123`).
   - Specify the type of change (bug fix / enhancement / new feature / breaking change / docs).
   - Attach before/after screenshots for visual changes.
   - Complete the checklist:
     - [ ] `black .` has been run
     - [ ] Documentation updated
     - [ ] Tests added
     - [ ] Dependencies updated (if applicable)

> It is recommended to open an issue and discuss the design before opening a large PR.

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

Refer to `embodichain/lab/gym/envs/tasks/` for existing examples. Tasks subclass `EmbodiedEnv` or `BaseAgentEnv` and implement `_setup_scene`, `_reset_idx`, and evaluation logic.

---

## Unit Tests

### Structure

Tests live in `tests/` and mirror the source tree:

```text
tests/
├── toolkits/
│   └── test_pg_grasp.py
├── gym/
│   └── action_bank/
│       └── test_configurable_action.py
└── sim/
    ├── objects/
    │   ├── test_light.py
    │   └── test_rigid_object_group.py
    ├── sensors/
    │   ├── test_camera.py
    │   └── test_stereo.py
    └── planners/
        └── test_motion_generator.py
```

Place new test files at `tests/<subpackage>/test_<module>.py`, matching the layout of `embodichain/`.

### Two accepted styles

**pytest style** — for pure-Python logic with no test ordering dependency:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ----------------------------------------------------------------------------

from embodichain.my_module import my_function


def test_expected_output():
    result = my_function(input_value)
    assert result == expected_value


def test_edge_case():
    result = my_function(edge_input)
    assert result is not None
```

**`Class` style** — when tests must run in a specific order or share `setup_method`/`teardown_method` state:

```python
# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
# ----------------------------------------------------------------------------

from embodichain.my_module import MyClass


class TestMyClass():
    def setup_method(self):
        self.obj = MyClass(param=1.0)

    def teardown_method(self):
        pass

    def test_basic_behavior(self):
        result = self.obj.run()
        assert result == expected_result

    def test_raises_on_bad_input(self):
        with pytest.raises(ValueError):
            self.obj.run(bad_input)

### Conventions

- **File header**: include the standard Apache 2.0 copyright block (same as all source files).
- **Naming**: test files are `test_<module>.py`; test functions/methods are `test_<scenario>`.
- **Simulation-dependent tests**: tests that require a running `SimulationManager` (GPU, sensors, robots) must initialize and teardown the sim inside `setUp`/`tearDown` or a pytest fixture. Keep them isolated from pure-logic tests.
- **No magic numbers**: define expected values as named constants or comments explaining their origin.
- **`if __name__ == "__main__"`**: include this block for tests that support optional visual/interactive output (pass `is_visual=True` manually when debugging).

### Running tests

```bash
# Run all tests
pytest tests/

# Run a specific file
pytest tests/toolkits/test_pg_grasp.py

# Run a specific test function
pytest tests/toolkits/test_pg_grasp.py::test_antipodal_score_selector

# Run with verbose output
pytest -v tests/
```
