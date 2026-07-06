# Task Environment Decoupling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract all task envs from `embodichain` into `embodichain_tasks` package, add entry_points discovery, init hooks, and unified launch.

**Architecture:** IsaacLab-style separation — `embodichain_tasks/` is a separate pip-installable package with its own `pyproject.toml`. Core `embodichain` discovers tasks via `importlib.metadata.entry_points(group="embodichain.tasks")`.

**Tech Stack:** Python 3.10+, setuptools, gymnasium, importlib.metadata

## Global Constraints

- All existing env IDs unchanged
- JSON/YAML config format unchanged
- `@register_env` API unchanged
- `gym.make()` workflow unchanged
- RoboSynChallenge NOT touched this iteration

---
