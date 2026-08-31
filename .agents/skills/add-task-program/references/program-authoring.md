# Program Authoring

Author task intent independently of robots, simulators, and live providers.

## Choose serialized or Python form

Use YAML/JSON when the program is a deployable task asset or should be authored
by an agent/MLLM. Use Python `TaskProgramCfg` when the caller explicitly wants
programmatic composition, reuse in code, or a typed tutorial/test.

Serialized programs belong at:

```text
embodichain_tasks/configs/tasks/<category_path>/<task_name>/task_program/program.yaml
```

The public Python API is `embodichain.lab.task_program`. A provider-independent
Python example is:

```text
scripts/tutorials/task_program/build_and_compile.py
```

## Program contract

A program owns:

- one stable `program_id`;
- optional named targets;
- one bounded program tree.

Supported node families are sequence, repeat, segment, invoke, and parallel
with an owned barrier. Built-in call configs are Pick, Place, and HandOver.
`RegisteredSemanticCallCfg` is the extension payload for an allowlisted call.

Do not serialize the trusted integration selection. It is injected from the
runnable deployment after component composition.

## Authoring sequence

1. Express the minimum ordered task outcome as Semantic Calls.
2. Add explicit segments where post-policies, validators, instructions, or
   dataset boundaries matter.
3. Add named/cyclic targets only when multiple calls reuse or advance them.
4. Use bounded repeats instead of generated duplicate steps.
5. Add parallel branches only when resource claims, runtime targets, control
   grids, symbolic writes, and physical safety can be proven compatible.
6. Add validators for dataset/task acceptance; do not confuse them with
   per-call effect monitors.

Use exact logical scene IDs from the planned integration, not native simulator
UIDs unless the mapping intentionally uses the same string.

## Safety and limits

The strict decoder rejects unknown fields, duplicate keys, non-finite values,
invalid exact types, unresolved references, excessive depth/nodes/repeats, and
executable registered payloads. Do not bypass it by constructing mutable
internal objects.

Nested parallel blocks are unsupported. Parallel resource disjointness alone
is not physical collision-safety evidence.

## Validation

For program-only work, run focused language/compiler tests and load the file
with `load_task_program()`. Full deployment validation additionally needs the
trusted integration selection and catalog; use the deployment inspector after
the other components exist.
