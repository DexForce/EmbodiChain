---
name: add-solver
description: Claude adapter for the canonical EmbodiChain add-solver skill.
---

# Add Solver - Claude Adapter

Canonical source: `.agents/skills/add-solver/`

## When to use

- adding a new kinematic (IK/FK) solver
- scaffolding the solver module plus its docs, unit test, and benchmark entry together
- a new robot family needs a closed-form / numerical / Warp-kernel IK backend

## Start here

1. Use this adapter when the task asks for a new kinematic solver.
2. Then follow `.agents/skills/add-solver/SKILL.md`.

The full procedure — solver module, (optional) Warp kernel, Sphinx docs page,
unit test, benchmark entry, registration in `__init__.py` and the docs toctree,
plus `black` and verification under `conda activate embodichain` — is defined in
the canonical skill.
