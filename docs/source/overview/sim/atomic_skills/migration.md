# Atomic Skill naming migration

Atomic Skill is the canonical name for a registered, typed robot capability.
Its implementation can contain multiple phases, replans and retries. Completion
still follows the configured projected or verified effect policy; the name does
not imply transaction rollback or guaranteed physical success.

Task Programs orchestrate Semantic Calls, which lower to Atomic Skill invocations.
Gym/RL actions and controller commands retain their existing terminology.

## Python imports and types

Use `embodichain.lab.sim.atomic_skills` and its submodules. The former
`embodichain.lab.sim.atomic_actions` package emits a `DeprecationWarning` and
reuses the canonical module and class objects, including primitive submodules.
It does not create a separate registry. Legacy imports remain available for the
0.x migration window; removal requires a separately announced breaking release.

| Former name | Canonical name |
| --- | --- |
| `AtomicAction` | `AtomicSkill` |
| `AtomicActionEngine` | `AtomicSkillEngine` |
| `ActionInvocation` | `SkillInvocation` |
| `ResolvedActionRequest` | `ResolvedSkillRequest` |
| `ActionPlan` | `SkillPlan` |
| `ActionOptions` | `SkillOptions` |
| `ActionBinding` | `SkillBinding` |
| `ActionControlOverrides` | `SkillControlOverrides` |
| `ActionPlanningServices` | `SkillPlanningServices` |
| `ActionPlanTemplateAdapter` | `SkillPlanTemplateAdapter` |
| `ObjectActionGoal` | `ObjectSkillGoal` |
| `BUILTIN_ACTION_TYPES` | `BUILTIN_SKILL_TYPES` |
| `create_simulation_atomic_action_engine` | `create_simulation_atomic_skill_engine` |

New code uses the canonical exports. Old symbol aliases are supplied by the
deprecated import package. Custom subclasses continue to implement `_plan()`;
goal, binding, tracking, recovery and verification behavior is unchanged.

## Fields and catalogs

Python keyword arguments and attributes on the canonical contracts use:

| Former name | Canonical name |
| --- | --- |
| `action_plans` | `skill_plans` |
| `action_waypoint_offset` | `skill_waypoint_offset` |
| `action_index` | `skill_index` |
| `action_binding` | `skill_binding` |
| `max_action_retries` | `max_skill_retries` |
| `action_timeout` | `skill_timeout` |
| `action_retry_counts` | `skill_retry_counts` |
| `action_option_templates` / `action_option_template` | `skill_option_templates` / `skill_option_template` |

Update Python construction and attribute access to these names, including when
using an old class import alias. The alias preserves type identity, not old
constructor keyword signatures.

`engine.skill_implementations` exposes all installed implementations.
`engine.skill_catalog` exposes agent-visible descriptors with explicit binding
contracts. The former `engine.actions` and `engine.skills` properties remain
compatibility aliases for those respective views.

Configured Task Program policy presets normalize the old `action_options` key
to `skill_options`; specifying both is rejected. Recovery policies accept old `max_action_retries` and
`action_timeout` keys and normalize them to their canonical names. Supplying both
spellings of the same field is rejected. Packaged policies use the new keys.
Trajectory expansion likewise normalizes `source.kind: atomic_action` to
`atomic_skill`; the generic expansion `unit_scope: action` remains unchanged.

## Commands, examples and developer skills

- Run examples with `embodichain run-example sim/atomic_skill --skill place` or
  `embodichain run-atomic-skill --skill place`. The former `run-atomic-action`
  command and `run-example sim/atomic_action/...` selectors forward to these
  same examples.
- Profile with `embodichain benchmark atomic-skill-profile --skill pick_up`.
  `atomic-action-profile` and its `--action` option remain aliases.
- Example source files live in `examples/sim/atomic_skill/`; tutorials live in
  `scripts/tutorials/atomic_skill/`; profiling scripts live in
  `scripts/benchmark/atomic_skill/`. Update direct filesystem paths and imports
  of those example/script modules. The CLI aliases do not recreate old files.
- New default example outputs use `/tmp/atomic-skill/<skill>`.
- The extension workflow is `/add-atomic-skill`. Stable project-context topic IDs
  and published `_static/atomic_actions/` media URLs remain valid.

## Events and benchmark records

Execution event names and values use `SKILL_*` / `skill_*` for planned, planning
failed, timeout, retry and completed events. Old `ACTION_*` enum member names are
aliases, and constructing `ExecutionEventKind` from the corresponding old
`action_*` string resolves to the canonical event. New traces emit `skill_*`.

Task Program result metadata uses `max_skill_retries`, `skill_timeout` and
`skill_retries`. Consumers of historical result dictionaries should accept the
old keys when reading previous runs; this migration does not rewrite stored
datasets or reports.

Benchmark planner adapters declare the `atomic_skill` capability. The compile
timing scope is `atomic_skill_compile`; readers comparing historical runs should
normalize the former `atomic_action_compile` spelling. `scenario: atomic_skill`,
skill IDs, measurements and physical-success criteria are unchanged.

The historical `sim/atom_actions.py` and `utility/atom_action_utils.py` interfaces
are separate legacy implementations. Their functional migration is outside this
naming change.
