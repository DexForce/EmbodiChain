# @configclass Pattern

## Entry Points

| What | Path |
|------|------|
| Decorator + helpers | `embodichain/utils/configclass.py` |
| Public exports | `embodichain/utils/__init__.py` — exports `configclass`, `is_configclass`, `set_seed`, `GLOBAL_SEED` |
| MISSING sentinel | Re-exported from `dataclasses.MISSING` (used inside configclass files) |

Import:
```python
from embodichain.utils import configclass
from dataclasses import MISSING
```

## Overview

`@configclass` is a decorator wrapping Python's `@dataclass` that fixes two pain points for configuration objects:

1. **Missing type annotations** — automatically infers annotations from default values so unannotated members are not silently ignored.
2. **Mutable defaults** — wraps mutable defaults (lists, dicts, nested objects) in `field(default_factory=...)` automatically, avoiding the standard dataclass `ValueError`.

It also adds runtime utility methods and deep-copies all mutable members in `__post_init__` to prevent shared-state bugs.

**Origin**: Adapted from Isaac Lab's `configclass` implementation (`isaaclab.utils.configclass`).

## @configclass Decorator — What It Adds

When `@configclass` decorates a class, the following happens (in order):

1. **`_add_annotation_types(cls)`** — scans the MRO and adds type annotations for any class member that lacks one (deduced from the default value's type). Raises `TypeError` if a `MISSING` field has no annotation.

2. **`_process_mutable_types(cls)`** — converts mutable default values (lists, dicts, class instances) into `field(default_factory=lambda: deepcopy(default))` to satisfy dataclass rules.

3. **`__post_init__` injection** — installs (or augments) `__post_init__` with `custom_post_init`, which deep-copies every non-callable, non-property, non-dunder attribute. This prevents instances from sharing mutable state.

4. **Utility methods attached**:
   - `to_dict()` → recursive dict conversion (handles nested configclasses, torch tensors, callables).
   - `replace(**kwargs)` → returns a new instance with specified fields replaced (delegates to `dataclasses.replace`).
   - `copy(**kwargs)` → alias for `replace`.
   - `validate()` → recursively checks for `MISSING` values; raises `TypeError` listing all missing fields.

5. **Wrapped with `@dataclass`** — the class is finally passed through the standard `dataclass()` call.

### Difference vs Plain @dataclass

| Feature | `@dataclass` | `@configclass` |
|---------|-------------|----------------|
| Unannotated members | Silently ignored | Auto-annotated from default type |
| Mutable defaults (list, dict) | `ValueError` | Auto-wrapped in `default_factory` |
| Nested configclass defaults | Shared across instances | Deep-copied per instance |
| `to_dict()` | Not available | Recursive conversion |
| `validate()` | Not available | Checks for MISSING fields |
| `replace()` / `copy()` | `dataclasses.replace()` | Attached as method |

## MISSING Sentinel

`MISSING` (from `dataclasses`) marks required fields that must be provided at instantiation:

```python
@configclass
class RobotCfg:
    name: str = MISSING       # caller MUST provide
    num_joints: int = MISSING  # caller MUST provide
    speed: float = 1.0         # optional with default
```

- Constructing `RobotCfg()` without `name` or `num_joints` raises `TypeError` from dataclass.
- Calling `cfg.validate()` after construction checks for any `MISSING` values that slipped through (e.g., in nested configs) and raises `TypeError` listing them.

## Nested Configs

Configclasses compose hierarchically:

```python
@configclass
class SensorCfg:
    width: int = 640
    height: int = 480

@configclass
class RobotCfg:
    name: str = MISSING
    camera: SensorCfg = SensorCfg()  # auto-wrapped in default_factory
```

Each `RobotCfg()` instance gets its own deep copy of `SensorCfg()` — no shared state.

### update_class_from_dict

`update_class_from_dict(obj, data, _ns="")` performs recursive in-place updates from a dict:

- Nested `Mapping` values → recurse into sub-object.
- Iterable values → matched by length; recursed if elements are mappings.
- Callable members → resolved from string via `string_to_callable()`.
- Type mismatches → `ValueError`.
- Unknown keys → `KeyError`.

This powers config loading from YAML/JSON files.

## Usage Patterns

### Algorithm config (RL)

```python
# embodichain/learning/rl/utils/config.py
@configclass
class AlgorithmCfg:
    device: str = "cuda"
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

# embodichain/learning/rl/algo/ppo.py
@configclass
class PPOCfg(AlgorithmCfg):
    n_epochs: int = 10
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
```

### Manager/functor configs (env framework)

```python
@configclass
class MyObservationCfg:
    sensor_name: str = MISSING
    scale: float = 1.0
    noise_std: float = 0.0
```

### Checking for configclass

```python
from embodichain.utils import is_configclass

is_configclass(PPOCfg)   # True — has 'validate' method
is_configclass(dict)      # False
```

Note: `is_configclass` simply checks for the presence of a `validate` attribute.

## Common Mistakes

| Mistake | What Happens | Fix |
|---------|-------------|-----|
| Forgetting `MISSING` annotation | `TypeError: Missing type annotation for 'X'` at class definition time | Add explicit type annotation: `x: int = MISSING` |
| Using `MISSING` without type hint | Decorator can't infer type from `MISSING` | Always annotate MISSING fields |
| Sharing mutable nested defaults across subclasses | Usually safe due to `__post_init__` deepcopy, but watch out for very large objects (perf cost) | Acceptable for configs; avoid storing large tensors as defaults |
| Overriding `__post_init__` | Your custom logic runs first, then `custom_post_init` deepcopies everything | Use `combined_function` behavior — both run; order: yours → deepcopy |
| `validate()` not called | MISSING fields in nested configs go undetected until attribute access | Call `cfg.validate()` after construction to fail fast |
| `to_dict()` on torch.Tensor | Returns the tensor directly (not converted to list/scalar) | Intentional — tensors are preserved as-is in the dict |
| Callable fields in `to_dict()` | Converted to string via `callable_to_string()` | Use `string_to_callable()` to reverse when loading |
| `update_class_from_dict` with extra keys | Raises `KeyError` | Only pass keys that exist on the target config |
| `update_class_from_dict` length mismatch on lists | Raises `ValueError` | Source and target iterables must have the same length |
