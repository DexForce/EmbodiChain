# Validation and Repair

Validate from inert configuration toward live execution. Report the highest
completed level and do not imply that a lower level proves physical success.

## Read-only static inspector

```bash
python .agents/skills/add-task-program/scripts/inspect_deployment.py \
  <task config>/task.<embodiment>.yaml
```

Pass multiple deployments to check embodiment variants in one run. Add
`--json` for machine-readable output.

The inspector:

1. loads the runnable config;
2. resolves environment and embodiment components;
3. composes the integration and execution policy;
4. checks semantic scene roots against native physical UIDs;
5. strictly loads the bound program;
6. runs catalog preflight and static Semantic Call lowering.

It never constructs a live simulator or registers a Gym environment.

## Focused test surfaces

```bash
pytest -q tests/lab/task_program
pytest -q tests/gym/envs/task_program/test_configured_integration.py
pytest -q tests/gym/envs/task_program/test_task_vertical_slices.py
pytest -q tests/gym/envs/task_program/test_task_hand_over.py
pytest -q tests/test_task_program_package_data.py
```

Choose only tests relevant to the changed boundary. Add
`tests/gym/envs/task_program/` coverage for new configured service/provider
families.

## Repair order

Fix the earliest failing ownership layer:

1. syntax/closed fields;
2. missing component path;
3. duplicate component-owned inline fields;
4. absent physical `simulation_uid`;
5. scene/embodiment/policy contract mismatch;
6. missing logical resource, endpoint, capability, preset, or service;
7. unknown Semantic Call or missing registered lowerer;
8. compiler/preflight workflow conflict;
9. live grounding/execution/evidence failure.

Do not compensate for an earlier-layer failure with task-local runtime code.

## Physical qualification

After static validation, run the actual deployment with controlled
seeds/randomization. Check:

- grounding and planning success;
- controller-ready action shape;
- effect assurance and recovery;
- post-policies and validators;
- final row-local acceptance;
- persisted completion metadata.

Projected execution proves the intended command trajectory completed, not that
the physical task outcome was measured.
