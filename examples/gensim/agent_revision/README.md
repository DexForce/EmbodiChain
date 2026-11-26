# Bounded UR5 program revision on DexUni

The authoring server starts before any World exists. Each native execution
uses the official single-arena UR5 DexUni deployment in a fresh Gym worker.
The agent can change the horizontal drop offset. The measured lift, release
and placement criteria, seed, two-attempt limit and 30-minute wall budget
stay frozen. A control-boundary cancellation or budget stop preserves its
job receipt; a worker that disappears without a terminal receipt is unknown.
Inspect or resubmit the original ID to reconcile an attempt.

Use Linux, a CUDA GPU, CPython 3.11, the qualified packages and the official
robot assets. Start a new private output directory for each independent
agent pilot. The GenSim and runtime records live outside the Agent workspace.

```bash
node examples/gensim/agent_revision/dexpilot.mjs /path/to/env/bin/python /tmp/dsh-ur5 /path/to/dexpilot/src/index.ts
python -m examples.gensim.agent_revision.codex_pilot --output /tmp/codex-ur5 --approve-experiment-tools
```

The second command requires explicit operator authorization to preapprove the
six named GenSim tools for this invocation. The Codex shell remains read-only.
The server pins the Codex experiment to one session and one budget.
Each completed attempt records its bundle, World session, actual physics
readbacks, measured horizontal error and lift, elapsed time and trajectory
digest. The cancellation request is asynchronous: poll the original job for
the acknowledged boundary. Keep the failed revision and trajectory alongside
the accepted attempt. Use the Agent's model/turn journal for usage accounting.
