# Configclass pattern

## Entry points

`embodichain/utils/configclass.py` owns the decorator, `class_to_dict()`,
`update_class_from_dict()` and validation. Public imports use
`from embodichain.utils import configclass`; the sentinel is
`from dataclasses import MISSING`.

`@configclass` wraps `dataclass`, infers annotations from ordinary defaults,
copies mutable/nested defaults per instance, and adds `validate()`, `to_dict()`,
`replace()` and `copy()`. Public configuration fields should still be annotated.

## Required values and validation

Construction may leave annotated `MISSING` fields unresolved. The constructor
does not itself enforce that these values were supplied. Call `validate()` at
the completed-configuration boundary; it recursively raises `TypeError` for
unresolved fields. An unannotated `MISSING` field fails during decoration.

```python
from dataclasses import MISSING
from embodichain.utils import configclass

@configclass
class ExampleCfg:
    name: str = MISSING
    sizes: list[int] = [1, 2]

cfg = ExampleCfg()              # construction succeeds
assert cfg.name is MISSING
cfg.name = "example"
cfg.validate()                  # now complete; succeeds
other = ExampleCfg(name="other")
cfg.sizes.append(3)
assert other.sizes == [1, 2]     # independent mutable defaults
```

`validate()` checks missing values; it is not a universal runtime type, range or
domain-contract validator. Domain loaders add their own checks.

## Serialization and updates

| Operation | Contract |
|---|---|
| `to_dict()` | Recursively converts configs; tensors remain tensors, callables become strings |
| Custom `to_dict()` | A custom method anywhere in the MRO is preserved, including on decorated subclasses |
| `replace(**kwargs)` / `copy(**kwargs)` | Construct a replacement through the dataclass helper and initialization hooks |
| Custom `__post_init__` | Runs before configclass's copying hook |
| `is_configclass(value)` | Checks for a `validate` attribute; it is a convenience heuristic |

`update_class_from_dict(obj, data)` updates in place:

- Nested mappings recurse into existing fields.
- Flat iterables without mapping elements replace the field wholesale and may
  change length. Existing tuple fields keep tuple type.
- Lists containing mappings merge into existing objects; their lengths must
  agree. Such a merge into `None` is rejected. Tuple handling differs: inspect
  the helper before relying on a nested-object merge into tuples.
- Callable fields accept resolvable callable strings; unknown keys raise
  `KeyError`; incompatible scalar values raise `ValueError`.

This helper is not an automatically installed `from_dict()` method. Inspect the
owning production loader for JSON/YAML behavior; robot configs, for example,
use the [robot config protocol](../robot-system/robot-system.md) and may apply
variant-specific defaults and serialization transforms.

## Change sites and validation

Change the domain config or loader for one subsystem. Change `configclass.py`
only for behavior intended to apply across config families. Preserve sentinel
identity, independent mutable defaults, and custom serialization inheritance.
Validate round-trips through the affected production loader and its focused
tests; config construction alone does not prove a valid deployment.

## Common failure modes

- Unresolved required fields survive construction: call `validate()` after assembly.
- Tensor output is assumed JSON-ready: serialization preserves tensors.
- A flat list unexpectedly changes length: wholesale replacement is intentional.
- Robot serialization applies derived transforms twice: follow the robot's
  `from_dict(to_dict())` protocol, not a generic attribute-copy implementation.
- Large defaults cause costly copies: keep runtime tensors/resources outside configs.
