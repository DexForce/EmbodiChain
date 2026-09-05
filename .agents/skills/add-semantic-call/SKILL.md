---
name: add-semantic-call
description: Add or expose a high-level EmbodiChain Task Program Semantic Call, including registered call descriptors/lowerers, configured runtime-service decoder support, or a deliberately promoted built-in call. Use when a Task Program call is unknown or must lower to an Atomic Skill.
---

# Add Semantic Call

Expose a reusable high-level instruction to Task Program while preserving the
boundary between declarative task intent and executable Atomic Skills.

## Route the request

Choose the narrowest mode:

| Situation | Action |
|---|---|
| Existing catalog already discovers the requested call | Change only the program/integration config; use `$add-task-program` |
| Existing Atomic Skill should be exposed under a task/integration-owned call ID | Add a registered Semantic Call extension |
| Required Atomic Skill does not exist | Invoke `$add-atomic-action` first |
| The concept must become a stable universal language primitive like Pick/Place/HandOver | Add a built-in Semantic Call only when explicitly justified |

Prefer a registered extension for new provider/task families. Do not add a new
`SemanticCallSpec` subclass for an extension; serialized extensions use
`RegisteredSemanticCallCfg` and runtime values use exact
`RegisteredSemanticCall`.

## Load current context

Read `agent_context/MAP.yaml` and resolve `task-programs` plus
`atomic-actions`. Verify the current source of truth:

| Concern | Path |
|---|---|
| Call values/catalog | `embodichain/lab/task_program/semantics/calls.py` |
| Language config/decoder | `embodichain/lab/task_program/language/schema.py`, `decoder.py` |
| Static linking/lowering | `embodichain/lab/task_program/compiler/lowering.py` |
| Registered extension declarations | `embodichain/lab/task_program/integrations/extensions.py` |
| Configured integration decoder | `embodichain/lab/task_program/integrations/configured.py` |
| Allowlisted simulation factories | `embodichain/lab/task_program/integrations/_configured_services.py` |
| Runtime assembly | `embodichain/lab/task_program/integrations/simulation/` |

Inspect the target Atomic Skill's exact `SkillDescriptor`, goal, options, and
binding contract before designing the Semantic Call.

## Registered Semantic Call mode

### 1. Define stable declarative identity

Choose a lowercase dotted call ID such as `simulation.articulation_link_slide`.
Define the smallest JSON-compatible argument mapping needed to select semantic
entities or configured values. Arguments contain no tensors, callables, import
paths, live objects, or motion generators.

Program form:

```yaml
kind: registered
call_id: vendor.my_call
arguments:
  target: logical_scene_entity
```

### 2. Declare descriptor and lowerer coverage

The integration catalog must include one exact `SemanticCallDescriptor` whose:

- `call_id` matches the serialized call;
- `spec_type` is `RegisteredSemanticCall`;
- target descriptor is the intended Atomic Skill descriptor.

It must also own exactly one fingerprinted
`RegisteredSemanticLowererFactory` for the same call ID and revision. The
factory creates a fresh live lowerer for each adapter assembly.

### 3. Keep lowering typed and narrow

The lowerer:

- strictly validates canonical arguments;
- resolves only declared semantic references/live services;
- produces the target Atomic Skill's typed goal;
- consumes action options from the bound profile/preset;
- declares look-ahead effects/targets when required by static analysis;
- does not step the simulator, emit controller commands, or contain a
  task-local motion generator.

Lowerers cannot replace the catalog-owned descriptor, resource binding,
effects, or options contract.

### 4. Expose configured YAML only through an allowlist

If `integration.yaml.runtime_services.registered_semantic_lowerers` must
instantiate the new provider family, add an explicit closed decoder branch and
typed factory in the configured integration modules. Never accept a generic
`class_type`, dotted import, or arbitrary kwargs escape hatch.

Add a task integration entry only after the core decoder recognizes its
`kind`.

## Built-in Semantic Call mode

Promote a call to the built-in language only when its semantics, arguments,
resource contract, effects, and Atomic Skill target are stable across providers
and tasks. Update the complete vertical slice:

1. language schema and strict decoder;
2. compiler conversion;
3. immutable semantic call and built-in catalog descriptor;
4. static linking/lowering and effect analysis;
5. public exports;
6. configured integration options/monitors if applicable;
7. focused docs and tests.

This is a language change, not a task integration shortcut. Preserve strict
unknown-field rejection and exact types.

## Validation

Registered-call coverage should prove:

- valid and invalid serialized argument shapes;
- call discovery and duplicate-ID rejection;
- exact descriptor-to-Atomic-Skill match;
- exactly one lowerer factory and stable fingerprint declaration;
- fresh lowerer construction;
- typed goal/options output;
- look-ahead/effect behavior;
- compiler preflight with missing/duplicate lowerers rejected;
- configured deployment decoding when a new service `kind` was added.

Use focused tests in:

```text
tests/lab/task_program/semantics/test_calls.py
tests/lab/task_program/test_decoder.py
tests/lab/task_program/test_semantic_compiler.py
tests/gym/envs/task_program/test_catalog.py
tests/gym/envs/task_program/test_configured_integration.py
tests/gym/envs/task_program/test_task_vertical_slices.py
```

Run `$add-task-program`'s inspector against every changed official deployment.
Live provider behavior still needs a real environment qualification.
