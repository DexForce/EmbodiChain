---
name: add-robot
description: Claude adapter for the canonical EmbodiChain add-robot skill.
---

# Add Robot - Claude Adapter

Canonical source: `.agents/skills/add-robot/`

## When to use

- Adding a new robot to EmbodiChain.
- Adding a variant to an existing robot.
- Scaffolding a `RobotCfg` subclass.

## Start here

1. Use this adapter when adding or extending a robot config.
2. Then follow `.agents/skills/add-robot/SKILL.md`.

The canonical skill covers the `RobotCfg` protocol (`_build_defaults` hook +
`build_pk_serial_chain` + inherited serialization), the single-file vs package
layouts, the 9-step scaffold, common mistakes, and a quick reference.
