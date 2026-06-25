# Robot Definition Refactor + Customizability + `add-robot` Skill

**Date:** 2026-06-25
**Status:** Approved (design)
**Scope:** Unify the two existing robot implementations (`CobotMagicCfg`, `DexforceW1Cfg`) into one shared protocol, fix accumulated bugs, update the `add_robot.rst` docs (and create the missing full tutorial), and add an `add-robot` skill that encodes the final protocol.

---

## 1. Goal

EmbodiChain has two robot configs — `CobotMagicCfg` (single-file) and `DexforceW1Cfg` (package) — that follow **different** construction patterns with no enforced interface, and carry several confirmed bugs. The full `docs/source/tutorial/add_robot.rst` tutorial does not exist (the quick-reference guide's `:doc:`/tutorial/add_robot`` link is broken), and no `add-robot` skill exists.

This refactor:

1. Lifts a shared protocol into the `RobotCfg` base: a single `_build_defaults` hook, a `from_dict` template, and generic serialization (`to_dict`/`to_string`/`save_to_file`).
2. Conforms both existing robots to that protocol and fixes all confirmed bugs.
3. Updates `guides/add_robot.rst` and creates `tutorial/add_robot.rst`.
4. Adds an `add-robot` skill (canonical + Claude + Copilot adapters) that scaffolds new robots following the protocol, with single-file and package-with-variants paths.

## 2. Scope Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Refactor target | Unify **both** robots into one shared protocol |
| 2 | Where the protocol lives | **Hybrid** — lift `_build_defaults` hook + `from_dict` template + serialization into `RobotCfg`; keep variant/version logic in subclasses |
| 3 | Skill scope | **Full scaffold + variant scaffold path** (mirrors `add-task-env` completeness, plus variant/version path) |
| 4 | Docs scope | **Update guide + create missing full tutorial** (resolves broken `:doc:` link) |

---

## 3. Current State (as explored)

### 3.1 Inheritance chain

```
ObjectBaseCfg      (embodichain/lab/sim/cfg.py:736)
  └─ ArticulationCfg  (cfg.py:1368)
       └─ RobotCfg    (cfg.py:1478)
            ├─ CobotMagicCfg   (embodichain/lab/sim/robots/cobotmagic.py:41)
            └─ DexforceW1Cfg   (embodichain/lab/sim/robots/dexforce_w1/cfg.py:51)
```

### 3.2 The two robots diverge on construction

| | `CobotMagicCfg` (single-file, ~218 lines) | `DexforceW1Cfg` (package: types/params/utils/cfg, ~1400 lines) |
|---|---|---|
| `from_dict` | one builder `_build_default_cfgs()` → `merge_robot_cfg` | three builders (`_build_default_cfg`, `_build_default_physics_cfgs`, `_build_default_solver_cfg`) → `merge_robot_cfg` |
| Solver | `OPWSolverCfg` (6-DOF) | `SRSSolverCfg` (7-DOF) |
| Versioning / variants | none | `version`, `arm_kind`, `with_default_eef`, hand brands |
| Serialization | none | `to_dict` / `to_string` / `save_to_file` |
| `build_pk_serial_chain` | yes | yes |

### 3.3 Confirmed bugs / inconsistencies

(found by reading the code, with file:line refs)

1. **`solver_cfg` set twice in `DexforceW1Cfg.from_dict`** (cfg.py:59) — `_build_default_cfg` → `build_dexforce_w1_cfg` sets a `PytorchSolver`, then `_build_default_solver_cfg` overwrites it with `SRSSolverCfg`. The first assignment is dead code.
2. **`build_pk_serial_chain` uses a different URDF than the sim URDF** in *both* robots — CobotMagic sims with `CobotMagicWithGripperV100.urdf` but builds the PK chain from `CobotMagicNoGripper.urdf` (cobotmagic.py:172); DexforceW1 sims from assembled components but builds the PK chain from a separate full-body URDF (cfg.py:350/352). Nothing checks the two agree.
3. **Lowercase `all` instead of `__all__`** in `dexforce_w1/types.py:19` and `utils.py:31` → `from module import *` is silently broken.
4. **`DexforceW1HandBrand` excluded** from `types.py`'s `all` (defined types.py:63).
5. **`HandManager.get_config` returns a 1-tuple** `(f"{prefix.lower()}_finger2_link",)` instead of a string for `root_link_name` (utils.py:222).
6. **Phantom `inst.validate()` call** in `W1ArmKineParams.from_dict` (params.py:256) — no `validate` method exists.
7. **`_build_default_cfgs` is `@staticmethod` but called via a throwaway instance** `cls()._build_default_cfgs()` (cobotmagic.py:50).
8. **No `__all__`** in `embodichain/lab/sim/robots/__init__.py`.
9. **Broken `:doc:`/tutorial/add_robot`` link** in `guides/add_robot.rst` — the target file does not exist.
10. **No `add-robot` skill** anywhere (canonical / `.claude` / `.github`). The `add-solver` skill is anomalously missing its `agents/openai.yaml` and Copilot adapter (related cleanup, flagged but out of scope for this skill's own files).

---

## 4. Design

### 4.1 Section 1 — Refactored `RobotCfg` base protocol

Lift into `RobotCfg` (in `embodichain/lab/sim/cfg.py`):

1. **`_build_defaults(self, init_dict: dict | None = None) -> None`** — the single hook subclasses override. Reads variant fields from `init_dict`, sets them on `self`, then populates `urdf_cfg`/`control_parts`/`solver_cfg`/`drive_pros`/`attrs`. Base provides a no-op default.
2. **`from_dict` template idiom** — each subclass keeps its `from_dict` but reduces it to the same 3-line template:
   ```python
   cfg = cls()
   cfg._build_defaults(init_dict)
   return merge_robot_cfg(cfg, init_dict)
   ```
   **Do NOT delete the subclass `from_dict` or move this template into the base `RobotCfg.from_dict`.** `merge_robot_cfg` (cfg_utils.py:86) internally calls `RobotCfg.from_dict(override_cfg_dict)` as its generic dict→cfg parser for the `else` branch. If the base `from_dict` became the `_build_defaults` → `merge_robot_cfg` template, it would **infinite-recurse** (`from_dict` → `merge_robot_cfg` → `RobotCfg.from_dict` → `merge_robot_cfg` → …). Therefore the base `RobotCfg.from_dict` **stays as the generic dict→cfg constructor** (handles `URDFCfg.from_dict`, `get_data_path` for `fpath`, dynamic `class_type` import for solvers), and the template lives in each subclass `from_dict`. The DRY win is the single `_build_defaults` hook replacing the divergent builders.
3. **Serialization** — lift `to_dict`/`to_string`/`save_to_file` (currently only on `DexforceW1Cfg`) into `RobotCfg`, made generic: enums → `.value`, numpy → `.tolist()`, nested configclass recursion, cycle-safe. Round-trip guarantee: `RobotCfg.from_dict(cfg.to_dict())` reproduces the cfg.

**Hook shape chosen: `_build_defaults(self, init_dict=None) -> None`** (instance method). Rationale: cleanest OOP shape; variant fields read naturally from `self`; trivial `from_dict`; DexforceW1 already nearly follows it (it pops `version`/`arm_kind`/`with_default_eef` in its current `from_dict`). Rejected: static flat-dict return (variant params thread awkwardly through `**params`); per-concern hooks (`_build_default_urdf`/`_solver`/`_physics`/`_control_parts` — over-engineered for two robots).

**Non-negotiable cleanups folded in:** drop the dead double-set of `solver_cfg`; fix `_build_default_cfgs` being `@staticmethod` called via throwaway `cls()`; standardize `__all__` across `types.py`/`utils.py`/`robots/__init__.py` and include `DexforceW1HandBrand`; fix the `root_link_name` 1-tuple; remove the phantom `inst.validate()` call.

### 4.2 Section 2 — Conformance changes to both robots + the `build_pk_serial_chain` URDF fix

**DexforceW1** (larger consolidation):
- Replace the body of `from_dict` (cfg.py:59) with the 3-line template (`cls()` → `_build_defaults` → `merge_robot_cfg`). Do **not** delete it (see §4.1 recursion note).
- Collapse the three builders into one `_build_defaults(self, init_dict=None)`: reads `version`/`arm_kind`/`with_default_eef` from `init_dict`, sets them on `self`, then populates `urdf_cfg`/`control_parts`/`solver_cfg`/`drive_pros`/`attrs`.
- This **removes the dead double-set of `solver_cfg`** (bug #1). In `utils.build_dexforce_w1_cfg`, stop setting `solver_cfg` — the cfg class owns it now.
- Delete `to_dict`/`to_string`/`save_to_file` — they lift into the base. **Verify** the generic base `to_dict` covers DexforceW1's cases (enums, numpy, nested configclass, cycle-safety); extend the base if not.
- Fix `__all__` in `types.py`/`utils.py`; add `DexforceW1HandBrand` to `types.__all__` (bugs #3, #4).
- Fix `HandManager.get_config` returning `(name,)` → `name` (bug #5).
- Remove the phantom `inst.validate()` call in `W1ArmKineParams.from_dict` (bug #6).

**CobotMagic** (smaller, gains symmetry):
- Convert `_build_default_cfgs` (flat-dict, `@staticmethod`, throwaway `cls()`) into `_build_defaults(self, init_dict=None)` setting fields on `self` (bug #7).
- Reduce the `from_dict` override to the 3-line template (`cls()` → `_build_defaults` → `merge_robot_cfg`). Do **not** delete it (see §4.1 recursion note).
- Gains `to_dict`/`to_string`/`save_to_file` from the base.
- Add `__all__ = ["CobotMagicCfg"]`.

**The `build_pk_serial_chain` URDF-mismatch fix** (bug #2 — the real correctness bug):

Chosen approach: **standardize on one explicit PK-URDF source per robot + a drift guard**, rather than forcing the assembled sim URDF. Each robot exposes `_pk_urdf_path` (or `_build_pk_urdf_path()`), documented as *the* URDF for FK/IK whose root_link→end_link kinematics must match the sim URDF's arm. `build_pk_serial_chain` reads from that one source (no more hardcoded path literals in the method body).

**Drift guard** (test-time, not runtime): in the `__main__` smoke test and the add-robot test stub, assert that the PK chain's joint names between root and end link are consistent with the matching `control_parts` entry. Catches silent drift without coupling PK to the assembled URDF.

*Rejected alternative:* derive the PK chain from `urdf_cfg.assemble_urdf()` (true single source of truth). Cleaner in theory, but risks breaking `pytorch_kinematics` parsing of the merged/sensor-laden URDF (almost certainly why both robots hand-curate an arm-only URDF today) and needs empirical validation. Flagged for the implementer if they want to pursue it; default is the explicit-source + drift-guard approach.

**Verification flag for the spec:** the new base `from_dict` template (`_build_defaults` → `merge_robot_cfg`) must cover everything the old base `from_dict` (cfg.py:1513) did — `URDFCfg.from_dict` for dict-form `urdf_cfg`, `get_data_path` for `fpath`, dynamic `class_type` import for solvers. Both subclasses already route through `merge_robot_cfg` successfully, so it's expected to carry most of this, but the implementation plan must include a verification step before deleting the old base logic.

### 4.3 Section 3 — The `add_robot.rst` documentation

**`docs/source/guides/add_robot.rst`** (update existing quick-reference):
- **Checklist** — rewrite the 9 items to match the new protocol. Step 2 becomes "override `_build_defaults(self, init_dict=None)`" (not `from_dict` + `_build_default_cfgs` — both now in the base); add a serialization step noting `to_dict`/`save_to_file` come free from the base.
- **Approaches** — keep single-file vs package; reframe package as *required when the robot has variants* and document the `types.py` (enums) + variant-aware `_build_defaults` pattern. Single-file stays for variant-less robots.
- **Key Parameters table** — match reality: add variant-fields row (optional subclass fields), add `build_pk_serial_chain` + `_pk_urdf_path` rows; drop the stale step-6 phrasing implying `build_pk_serial_chain` is separate from the cfg.
- **Add a Common Mistakes table** mirroring the skill's.
- **Fix the broken `:doc:`/tutorial/add_robot`` link** — resolves once the tutorial exists.

**`docs/source/tutorial/add_robot.rst`** (create the missing full tutorial — resolves the broken cross-reference):
1. Overview — what a robot config is, the inheritance chain (one line each), the protocol's two enforced hooks (`_build_defaults`, `build_pk_serial_chain`) + free serialization.
2. **Path A — Single-file robot** (variant-less, CobotMagic-shaped): full worked example — `@configclass` subclass, `_build_defaults`, `build_pk_serial_chain`, `__main__` smoke test. One complete, runnable file.
3. **Path B — Package robot with variants** (DexforceW1-shaped): `types.py` enums → variant-aware `_build_defaults` reading `version`/`arm_kind` from `init_dict` → `build_pk_serial_chain`. Calls out that `params.py`/`utils.py` (kinematic params, URDF assembly builders) are optional helpers, not protocol-required.
4. **Registering the robot** — the `robots/__init__.py` pattern (with `__all__`).
5. **Testing** — the `__main__` block + drift-guard assertion + `preview-asset` CLI.
6. **Serialization** — `to_dict`/`save_to_file`/`from_dict` round-trip, when to use it (saving tuned configs, snapshots).
- **Cross-links:** `:doc:`/guides/add_robot``, `:doc:`/tutorial/robot``, `:doc:`/overview/sim/solvers/index``, `:doc:`/resources/robot/index``. **See Also** updated to match the guide's.

**Consistency fix:** the guide, tutorial, add-robot skill (Section 4), and `agent_context/topics/robot-system/robot-system.md` (whose "Adding a New Robot" 8-step checklist is slightly stale) must describe the protocol with **identical** hook names and Key Parameters / Common Mistakes tables. Aligning `robot-system.md` is part of the docs step.

### 4.4 Section 4 — The `add-robot` skill

Matches existing skill patterns: frontmatter `name`+`description`, then `## When to Use` → domain context → `## Steps` → `## Common Mistakes` table → `## Quick Reference` table; adapters are thin routing pointers; each canonical skill carries an `agents/openai.yaml`; Copilot adapter + `instructions.md` index entry.

**Canonical skill — `.agents/skills/add-robot/SKILL.md`:**
- **Frontmatter:** `name: add-robot`, `description: Use when adding a new robot to EmbodiChain — scaffolds a RobotCfg subclass (single-file or package layout) with the _build_defaults hook, build_pk_serial_chain, registration, docs page, and test stub.`
- **`## When to Use`** — adding a new robot; adding a variant to an existing robot; scaffolding a `RobotCfg` subclass.
- **Domain context sections:**
  - `## The RobotCfg Protocol` — the two enforced hooks, free serialization, the `merge_robot_cfg` flow.
  - `## Two Layouts` — single-file (variant-less) vs package (variants); small decision table.
  - `## The Contract (read first)` — the exact field set a cfg must populate (`uid`, `urdf_cfg`/`fpath`, `control_parts`, `solver_cfg`, `drive_pros`, `attrs`) and the drift-guard requirement on `_pk_urdf_path`.
- **`## Steps`** (8):
  1. Pick layout (single-file vs package w/ variants) using the decision table.
  2. Create the cfg file(s); subclass `RobotCfg`; declare variant fields if any.
  3. Implement `_build_defaults(self, init_dict=None)` — set variant fields from `init_dict`, then populate `urdf_cfg`/`control_parts`/`solver_cfg`/`drive_pros`/`attrs`. (Embedded code template, single-file + variant variants.)
  4. Implement `build_pk_serial_chain` reading from `_pk_urdf_path`.
  5. Add `__all__` + register in `embodichain/lab/sim/robots/__init__.py`.
  6. Create `docs/source/resources/robot/<name>.md` + update `resources/robot/index.rst`.
  7. Add a test stub (`__main__` smoke test + drift-guard assertion) — defer to `/add-test` for full test scaffolding, but the skill provides the drift-guard snippet.
  8. Verify: `preview-asset` CLI + round-trip `from_dict(to_dict())`.
- **`## Common Mistakes`** table — the bugs we're cleaning up, as preventions: lowercase `all` vs `__all__`; `solver_cfg` set in multiple places; PK URDF drifting from sim URDF (no drift guard); `from_dict` reimplemented instead of using the base template; `root_link_name` as a tuple; calling nonexistent `validate()`.
- **`## Quick Reference`** table — Key Parameters (`uid`, `urdf_cfg`, `control_parts`, `solver_cfg`, `drive_pros`, `attrs`, variant fields, `_pk_urdf_path`) + file locations.
- **Code templates** embedded directly in `SKILL.md` as fenced blocks (no `templates/` dir — matches every existing skill).

**`agents/openai.yaml`** — standard 3 lines (`name`, `canonical_skill: .agents/skills/add-robot/SKILL.md`, `project: EmbodiChain`). Also **flags the anomaly** that `add-solver` is missing its `openai.yaml` — out of scope for this skill's own files, but noted as a related cleanup the implementer can pick up.

**Claude adapter — `.claude/skills/add-robot/SKILL.md`:** thin, ~20 lines, matching `add-task-env`'s minimal form: frontmatter, `Canonical source: .agents/skills/add-robot/`, `## When to use`, `## Start here` (2 numbered items), one summary paragraph.

**Copilot adapter — `.github/copilot/add-robot.md`:** 2–4 sentences, `# Add Robot for GitHub Copilot`, no frontmatter.

**`instructions.md` index — `.github/copilot/instructions.md`:** add `- Add robots: .github/copilot/add-robot.md` to the bullet list.

**Alignment:** the skill's Key Parameters and Common Mistakes tables must be **content-identical** to the guide's (Section 3) and `robot-system.md`'s.

### 4.5 Section 5 — Testing, error handling, migration, rollout

**Testing:**
- **Base `RobotCfg` additions** (highest risk — core `cfg.py`):
  - Generic `to_dict`/`from_dict` round-trip unit test against a synthetic cfg covering every serialization case: enums, numpy arrays, nested `URDFCfg`/`SolverCfg`/`JointDrivePropertiesCfg`, regex-dict fields (`stiffness={"LEFT_JOINT[1-6]": 7e4}`), and a cycle (a field referencing a parent). Assert `from_dict(cfg.to_dict())` reproduces the cfg field-for-field.
  - New base `from_dict` template coverage test asserting it does what the old base `from_dict` did: dict-form `urdf_cfg` → `URDFCfg.from_dict`, `fpath` → `get_data_path`, dynamic `class_type` import for solvers (the Section 2 verification step).
- **Existing robots** (regression safety):
  - `tests/sim/objects/test_robot.py` already exercises `DexforceW1Cfg` (line 24, 62) — extend it: `_build_defaults` produces the same `urdf_cfg`/`control_parts`/`solver_cfg`/`drive_pros` as before; `solver_cfg` set exactly once (dead double-set gone); `to_dict` round-trips.
  - Add a parallel `CobotMagicCfg` test (none exists today — only the `__main__` block). Same assertions + round-trip.
  - **Drift-guard test** for both robots: assert `build_pk_serial_chain`'s root→end joint names are consistent with the matching `control_parts` entry.
- **Skill smoke test:** the skill's scaffolded `__main__` block runs `from_dict` + `to_dict` round-trip + drift guard (mirrors `cobotmagic.py:207`).

**Error handling:**
- No new validation layer. If a subclass forgets a required field (`uid`, `urdf_cfg`/`fpath`, `control_parts`), the existing downstream `sim.add_robot` path already raises (e.g., missing `urdf_cfg` → `cfg.urdf_cfg.assemble_urdf()` → `AttributeError`). The skill's Common Mistakes table + the drift-guard test are the prevention, not runtime checks. No abstract-method enforcement — `build_pk_serial_chain` already returns `{}` in the base as its own signal. YAGNI.
- The drift guard is **test-time**, not runtime — keeps the hot path clean.

**Migration & compatibility:**
- **`from_dict` signature unchanged externally** — `CobotMagicCfg.from_dict(init_dict)` and `DexforceW1Cfg.from_dict(init_dict)` keep the same call signature and return type. The ~10 call sites (examples + `gym_utils.py` + tests) don't change. Key compatibility anchor.
- **`_build_default_cfgs` / `_build_default_cfg` / `_build_default_physics_cfgs` / `_build_default_solver_cfg` removed** — all private (underscore-prefixed), internal to each robot; grep confirms no external callers. Safe to rename/consolidate.
- **`to_dict`/`save_to_file` move from DexforceW1 to the base** — `DexforceW1Cfg().to_dict()` keeps working (now inherited); CobotMagic gains them. No breakage.
- **`robots/__init__.py`:** add `__all__ = ["DexforceW1Cfg", "CobotMagicCfg"]`. The `from .dexforce_w1 import *` already works once `types.py`/`utils.py`/`cfg.py` get proper `__all__` (today the lowercase `all` means `import *` pulls nothing from `types`/`utils` — a latent bug the refactor fixes).
- **Order of work** (implementation plan will sequence): (1) lift hooks + serialization into `RobotCfg` + base tests → (2) conform DexforceW1 (delete overrides, collapse builders, fix bugs) + tests → (3) conform CobotMagic + tests → (4) `build_pk_serial_chain` drift guard on both → (5) docs (guide + tutorial + `robot-system.md`) → (6) add-robot skill + adapters. Each step leaves the tree green.

**Rollout:**
- Single PR, commit-per-step matching the order above. `black==26.3.1` before commit; `/pre-commit-check` skill before PR; `/pr` skill for the PR.
- **No `agent_context` change beyond `robot-system.md`** alignment in step 5.

---

## 5. Files Touched

### Code
- `embodichain/lab/sim/cfg.py` — `RobotCfg`: add `_build_defaults`, `from_dict` template, `to_dict`/`to_string`/`save_to_file`.
- `embodichain/lab/sim/robots/cobotmagic.py` — conform to base; fix bugs #7; add `__all__`.
- `embodichain/lab/sim/robots/dexforce_w1/cfg.py` — collapse builders into `_build_defaults`; delete `from_dict` + serialization overrides; fix bug #1.
- `embodichain/lab/sim/robots/dexforce_w1/utils.py` — stop setting `solver_cfg` in `build_dexforce_w1_cfg`; fix `__all__` (bug #3); fix `root_link_name` tuple (bug #5).
- `embodichain/lab/sim/robots/dexforce_w1/types.py` — fix `__all__` (bug #3); add `DexforceW1HandBrand` (bug #4).
- `embodichain/lab/sim/robots/dexforce_w1/params.py` — remove phantom `validate()` call (bug #6).
- `embodichain/lab/sim/robots/__init__.py` — add `__all__` (bug #8).
- Both robots: `_pk_urdf_path` + `build_pk_serial_chain` reading from it (bug #2).

### Tests
- `tests/sim/objects/test_robot.py` — extend DexforceW1 assertions; add CobotMagic test; add round-trip + drift-guard tests; base `RobotCfg` round-trip + `from_dict` coverage tests.

### Docs
- `docs/source/guides/add_robot.rst` — rewrite checklist/params, add Common Mistakes, fix link.
- `docs/source/tutorial/add_robot.rst` — **create** (Path A + Path B + register + test + serialize).
- `agent_context/topics/robot-system/robot-system.md` — align "Adding a New Robot" checklist + tables.

### Skill
- `.agents/skills/add-robot/SKILL.md` — **create** (canonical).
- `.agents/skills/add-robot/agents/openai.yaml` — **create** (3 lines).
- `.claude/skills/add-robot/SKILL.md` — **create** (thin adapter).
- `.github/copilot/add-robot.md` — **create** (thin adapter).
- `.github/copilot/instructions.md` — add index entry.

---

## 6. Out of Scope

- Deriving the PK chain from `urdf_cfg.assemble_urdf()` (flagged alternative; default is explicit-source + drift guard).
- Adding the missing `agents/openai.yaml` and Copilot adapter for `add-solver` (flagged anomaly; related cleanup, not part of this skill's files).
- Any new runtime validation layer (YAGNI — downstream `sim.add_robot` already raises).
- Changes to `URDFCfg`, `JointDrivePropertiesCfg`, or `SolverCfg`/`OPWSolverCfg`/`SRSSolverCfg` (unchanged).
- Changes to the physics backend (the in-progress Newton refactor on `feature/newton-physics-backend` is independent).
