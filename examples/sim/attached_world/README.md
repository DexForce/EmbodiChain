# Attach to an external World

The attachment fixture exercises a single existing Arena with two robots, one moving cube and a static platform. It uses the installed DexSim wheel and EmbodiChain's UR5 asset and UR5 + DH PGI 140-80 task configuration. Run from the repository root in a Python 3.11 simulator environment. A headless renderer may require Xvfb on Linux.

```bash
xvfb-run -a python -m pytest tests/mcp/test_attached_world.py --run-gpu -v
```

Each case snapshots the host inventory, joint measurements and drive targets, cube state and topology before binding B. It also exercises a partially rejected attachment, a live host edit, planner lookup, topology invalidation, repeated detach and deferred cleanup. The fixture submits scene operations through DexSim's existing dispatcher.
