# Dual-Manipulator Composition from a Single-Arm Robot

**Date:** 2026-06-28
**Status:** Approved (implementation in progress)
**Branch:** `feat/ur-robot`

## Goal

Make it easy to define a dual-arm (bimanual) robot from **any** single-arm
specific robot cfg — today the UR family, tomorrow e.g. Franka — by composition,
with the result constructible from a dict/YAML like the existing
`URRobotCfg.from_dict({"robot_type": "ur5"})`.

Adding "two-arm Franka" in the future must require writing **only** Franka's
single-arm cfg — no new dual-arm class, no mixin, no per-robot boilerplate.

## Verified background (existing infrastructure)

| Primitive | Location | Role |
|---|---|---|
| `RobotCfg` | `embodichain/lab/sim/cfg.py:1487` | Arm-agnostic base: `control_parts: Dict[str, List[str]] \| None`, `urdf_cfg: URDFCfg \| None`, `solver_cfg: Union[SolverCfg, Dict[str, SolverCfg], None]`, `drive_pros: JointDrivePropertiesCfg`. |
| `URDFCfg` | `cfg.py:1003` | Multi-component assembly: `components=[{component_type, urdf_path, transform}]`, `assemble_urdf()`. Default `component_prefix` includes `("left_arm","left_"), ("right_arm","right_")`. |
| `URDFAssemblyManager` | `toolkits/urdf_assembly/urdf_assembly_manager.py` | Auto-prefixes (`left_arm`→`left_`, `right_arm`→`right_`), case-normalizes (joints UPPER, links lower), and connects **orphan** arms (no chassis/torso) to a synthetic `base_link`. |
| `URRobotCfg` | `lab/sim/robots/ur_robot.py` | Single `"arm"` component; `control_parts={"arm":[Joint1..6]}` (`joint1..6` for ur5); `solver_cfg={"arm": URSolverCfg(root_link_name="base_link", end_link_name="ee_link")}`; **scalar** `drive_pros`; `build_pk_serial_chain()->{"arm": chain}`. |
| `CobotMagicCfg` (reference dual-arm) | `lab/sim/robots/cobotmagic.py` | Two `left_arm`/`right_arm` components; solver per arm with **prefixed** link names operating on the assembled URDF; pk chains built from the **single-arm** URDF with **unprefixed** names (`base_link`/`link6`), keyed per arm. |
| `merge_robot_cfg` | `lab/sim/utility/cfg_utils.py:75` | Field-level merge of `urdf_cfg`/`control_parts`/`solver_cfg`/`drive_pros`/`attrs` onto a base cfg. |
| `create_pk_serial_chain` | `lab/sim/utility/solver_utils.py:63` | Builds a `pk.SerialChain` from a URDF + root/end link. |

**Key gap:** there is **no** name→class robot registry today; `SimulationManager.add_robot(cfg: RobotCfg)` takes a cfg *instance*, and each robot has its own `from_dict`. So a dict key like `base_robot: "ur5"` needs a small registry to resolve.

**Confirmed two-path kinematics** (from `cobotmagic.py:160-188`): the solver operates on the assembled (prefixed) URDF, while the `pk.SerialChain` is built from the single-arm (unprefixed) URDF. The engine replicates this split.

## Design

### Architecture — two layers, one entry point

1. **Engine** — `build_dual_arm_cfg(base_cfg, mounts, *, dual_part=True) -> DualArmRobotCfg`.
   A (mostly) pure function. Takes a *constructed* single-arm cfg (e.g. a
   `URRobotCfg`) and emits a fully-populated dual-arm cfg. Generic over any
   single-arm `RobotCfg` that follows the existing `"arm"` convention.
2. **Wrapper / dict entry** — `DualArmRobotCfg(RobotCfg)` whose `from_dict`
   resolves `base_robot` → single-arm class (via a tiny registry), constructs
   that single-arm cfg, calls the engine, and returns a `DualArmRobotCfg` that
   plugs straight into `sim.add_robot(cfg=...)` and round-trips via
   `to_dict()`/`from_dict()`.
3. **Registry** — `_BASE_ROBOT_REGISTRY: dict[str, tuple[type[RobotCfg], dict]]`.
   Adding Franka later = one line: `"franka": (FrankaRobotCfg, {})`.

### Mechanism — option (a): duck-type the `"arm"` convention

The engine reads from the base cfg, with **no edits** to `ur_robot.py`:

- `base_cfg.urdf_cfg.components` — the `"arm"` component (`urdf_path`, `transform`).
  A dict may override the part name via `arm_part` (default `"arm"`) for robots
  whose single-arm part is named differently.
- `base_cfg.control_parts[arm_part]` — joint names.
- `base_cfg.solver_cfg[arm_part]` — the solver (`root_link_name`, `end_link_name`, `tcp`, …).
- `base_cfg.drive_pros` — drive properties.
- `base_cfg._pk_urdf_path` (or the arm component's `urdf_path`) + the base
  solver's unprefixed `root_link_name`/`end_link_name` — for the pk chain.

### Prefix + case naming

Assembled joint/link names are produced by `URDFAssemblyManager`:
`prefix` (`left_`/`right_`) then case-normalize (joints UPPER, links lower),
skipping double-prefix when the name already starts with the prefix.

The engine must **predict** these names to build `control_parts`/`solver_cfg`.
To avoid drift, the engine reuses the manager's exact logic via a small shared
helper (`prefixed_name(name, prefix, kind)`), extracted from the existing
`_generate_unique_name` + `NameNormalizer`. A round-trip test assembles a real
dual UR and asserts `control_parts` resolves to real joints — the safety net
that proves prediction matches assembly.

Concretely for UR: joint `Joint1` → `left_Joint1` → `LEFT_JOINT1`; link
`base_link` → `left_base_link`; link `ee_link` → `left_ee_link`. (Both `ur5`
lowercase `joint1` and others' `Joint1` normalize to `LEFT_JOINT1`.)

### Mounts — preset + override (the "easy to define" surface)

`resolve_mounts({preset, separation, left?, right?}) -> {left: T4x4, right: T4x4}`:

- `side_by_side` — left at `+separation/2` in Y, right at `-separation/2`, same
  orientation. Default preset.
- `facing_inward` — same ±Y separation, yawed ±90° so the arms face each other.
- Optional per-arm `{xyz, rpy}` override → 4×4 transform (escape hatch for any
  non-mirrored setup).

```yaml
# Easy case (≈90% of uses):
base_robot: ur5
mount: {preset: side_by_side, separation: 0.6}

# Full form:
base_robot: ur5
mount:
  preset: side_by_side
  separation: 0.6
  left:  {xyz: [0.0, 0.3, 0.0], rpy: [0, 0, 0]}    # optional override
  right: {xyz: [0.0, -0.3, 0.0], rpy: [0, 0, 0]}   # optional override
dual_part: true            # include "dual_arm" composite part (default true)
```

`base_robot` accepts either a registry key (`"ur5"`) or an explicit
`{type: "ur5", init: {robot_type: "ur5", …}}` for passing extra base-robot init
params. Both route through the base robot's own `from_dict`.

### What the engine produces

- **`urdf_cfg`** — two components `left_arm`/`right_arm`, each the base arm URDF
  + the resolved mount transform. (Assembly runs lazily at `add_robot` time, as
  for every robot.)
- **`control_parts`** — every base part duplicated: `arm`→`left_arm`+`right_arm`,
  `eef`→`left_eef`+`right_eef` (generic over **all** base parts, not just `arm`).
  Plus `dual_arm` = `concat(left_arm, right_arm)` when `dual_part=True`.
- **`solver_cfg`** — each base solver duplicated with **prefixed**
  `root_link_name`/`end_link_name` (`left_base_link`/`left_ee_link`), keyed
  `left_arm`/`right_arm`. `tcp` copied unchanged.
- **`drive_pros`** — if **scalar** (UR's case): copied verbatim (both arms
  identical, applies to all joints uniformly). If **dict/regex**: each
  `(pattern, value)` mirrored into left- and right-prefixed pattern variants.
- **`build_pk_serial_chain`** — overridden to return `{"left_arm": chain,
  "right_arm": chain}` built from the base arm URDF with **unprefixed**
  `root_link_name`/`end_link_name` (CobotMagic-style arm-local FK).
- **`_pk_urdf_path`** — the base arm URDF path.

### Data flow

```
dict/YAML
  │
  ▼
DualArmRobotCfg.from_dict
  │  1. registry: "ur5" -> (URRobotCfg, {robot_type: "ur5"})
  │  2. base_cfg = URRobotCfg.from_dict(merged_init)
  │  3. mounts  = resolve_mounts(dict["mount"])
  ▼
build_dual_arm_cfg(base_cfg, mounts)   ← the one reusable engine
  │  urdf_cfg / control_parts / solver_cfg / drive_pros / pk-chain override
  ▼
DualArmRobotCfg  →  sim.add_robot(cfg=...)   (assemble_urdf at spawn)
                   cfg.to_dict() -> from_dict() reproduces cfg
```

### Error handling

- Unknown `base_robot` key → `ValueError` listing registered names.
- Base cfg has no `arm` part (and no `arm_part` override) → `ValueError`
  naming the `arm_part` override key and the base's actual `control_parts` keys.
- Unknown `mount.preset` → `ValueError` listing supported presets.
- Mount override present for only one side → `ValueError` (both or neither).
- Scalar vs dict `drive_pros` — both handled; mismatched solver/part keys caught
  by the existing `Robot.init_solver` checks.

### Testing (TDD)

Tests in `tests/sim/objects/test_dual_arm.py` (alongside `test_robot_cfg.py`).

- **Unit**
  - `resolve_mounts`: `side_by_side`/`facing_inward` symmetry; per-arm override;
    separation sign; raises on unknown preset / one-sided override.
  - `prefixed_name`: joints upper, links lower, no double-prefix; matches UR
    names (`Joint1`→`LEFT_JOINT1`, `base_link`→`left_base_link`).
  - `build_dual_arm_cfg` on a `URRobotCfg`: produces `left_arm`/`right_arm`/
    `dual_arm` parts, per-arm solver with prefixed links, scalar `drive_pros`
    preserved.
- **Integration** (real UR5 dual arm)
  - `assemble_urdf()` succeeds; assembled URDF has `LEFT_JOINT1`..`LEFT_JOINT6`
    and `RIGHT_JOINT1`..`RIGHT_JOINT6` (predict == assemble).
  - `control_parts` resolve to 12 joints total.
  - `sim.add_robot(cfg)` succeeds; DOF == 12.
  - Per-arm `compute_fk`/`compute_ik` round-trip.
  - `to_dict()` → `from_dict()` reproduces the cfg (mirror `URRobotCfg.__main__`
    round-trip check), plus a DOF-drift guard (mirror the existing UR test).
- **Registry**: unknown key raises with a helpful message; resolves `"ur5"`.

### File placement

- `embodichain/lab/sim/robots/dual_arm.py` — `build_dual_arm_cfg`,
  `DualArmRobotCfg`, `resolve_mounts`, `_BASE_ROBOT_REGISTRY`,
  `prefixed_name` (or the shared helper's import site).
- Export `DualArmRobotCfg` (and `build_dual_arm_cfg`) from
  `embodichain/lab/sim/robots/__init__.py`.
- Tests in `tests/sim/objects/test_dual_arm.py`.
- Docs: add a "Dual-arm composition" section to
  `docs/source/resources/robot/` (new `dual_arm.md`) and link from
  `docs/source/resources/robot/index.rst`; add a short subsection to
  `docs/source/tutorial/add_robot.rst`.
- Shared naming helper: extract from `toolkits/urdf_assembly/` if a clean seam
  exists; otherwise replicate the 3-line logic in `dual_arm.py` with the
  round-trip test as the drift guard. (Decision deferred to implementation —
  prefer reuse; fall back to replication if extraction would touch too much.)

### Out of scope (v1)

- Arbitrary N arms (only 2 — YAGNI).
- A shared torso/chassis/rail between the arms (orphan-on-shared-base only).
- Hand/EEF composition beyond what the base cfg already carries.
- Non-UR robots validated end-to-end (Franka is future; the engine is generic,
  but only UR is integration-tested).

### Why this satisfies the constraints

- **Easy to define from dict/YAML:** one-liner `base_robot + mount.preset +
  separation`; full escape hatch for custom mounts; explicit `base_robot` form
  for extra init params.
- **Reuses existing composition:** builds entirely on `URDFCfg`/`URDFAssemblyManager`/
  `RobotCfg`/`merge_robot_cfg` — no new physics or assembly code.
- **Future Franka is one line:** register `("franka", FrankaRobotCfg, {})`;
  the engine reads its `"arm"` convention for free, assuming it follows the
  same single-arm pattern as `URRobotCfg`.
- **No single-arm edits:** `ur_robot.py` is untouched.
