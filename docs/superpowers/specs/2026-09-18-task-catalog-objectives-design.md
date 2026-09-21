# Task catalog and independent physical objectives

Approved scope: discussion of issues #106 and #655, followed by the user's instruction to proceed with subagents on 2026-09-18.

## Deliverable

Users can discover a logical task, inspect its concrete deployments, run an existing deployment, and distinguish physical objective results from expert execution/acceptance. The first measured vertical slice is repeated_pick_place. Existing task layouts, Gym IDs, registration, and expert completion authority remain compatible.

## Catalog

Add optional task-local catalog.yaml with task_key, title, summary, tags, default_deployment, and named deployments referencing relative runnable configurations. No top-level id: metadata must not register as an environment. Task identity is package plus stable key; categories derive from ownership paths. Resolve metadata and runnable references through one internal CLI catalog model reused by list-task, show-task and a static HTML gallery export. Preserve tasks without metadata, configuration-only Task Programs, JSON/YAML equivalents and lightweight RL tasks. Infer RL support for a concrete deployment from the trainer.gym_config association or explicit registration support, never merely the presence of an agents directory. Default list output remains compatible. Source/preview links must be real and validated; absent preview/qualification is explicitly absent, never invented. Gallery rendering escapes authored text and requires no external service or frontend build.

## Objective

An optional deployment-level objective component is decoded strictly and independently of Task Program integration. Start with an ordered placement objective, not a general predicate language. It binds one rigid object, ordered axis-aligned goal regions in environment-local coordinates, a sustained-duration threshold and physical stability limits. Advancement occurs once per control step, with isolated tensors per environment. Snapshot queries do not advance progress. Partial reset clears only selected rows. Milestone history and current/final truth remain distinguishable. Invalid shapes, nonfinite thresholds, unknown fields and missing physical UIDs fail before rollout.

This objective observes the world; it never mutates program state, invokes segment validators, changes authoritative program completion, adds early termination, or controls dataset persistence. Unconfigured objectives preserve existing behavior. Physical success is opt-in and published under a separate info/metadata namespace. Projected effects are not measured evidence. Use actual object poses and velocities, not an end-effector release pose, to assess stable placement. A sustained stable region does not prove gripper detachment unless measured separately; name and document the predicate honestly.

## Vertical slice and recording

Attach the objective to one explicit repeated_pick_place benchmark deployment, leaving existing examples compatible. Evaluate expert execution and dynamic action replay with the same objective. Preserve recorded control mode, joint order and cadence; incompatible trajectories fail early. Kinematic playback is not dynamic qualification. Add baseline and a small bounded initial-position perturbation using existing EventManager randomization. Record actual post-reset poses, seeds, fully resolved config/component snapshots, source revision/dirty status, deployment, action source and outcome provenance in a run artifact. Do not claim bitwise determinism across devices/backends or reset batching.

Reports distinguish execution outcome, physical outcome, demonstration acceptance and persistence status; unavailable and not-applicable are not false. Reuse existing expert bridge, replay wrapper and recording hooks; do not flatten Task Program to a stateless policy. A local sample run must produce truthful success or failure, not fabricate success to qualify the gallery.

## Boundaries

No new Task class hierarchy, general experiment scheduler, broad action-source adapter framework, relational placement, scene generation, full predicate DSL, or all-backend qualification. Keep file-owned physics. Keep external task entry points and bundled official distribution. Do not touch the original workspace's untracked media. Work occurs in the isolated codex/task-catalog-objectives worktree.

## Validation

Catalog: legacy listing, strict catalog references, conflicting identities, deployment capability association, lightweight entries, HTML escaping, installed package data. Objective: sustained/ordered/final-state semantics, query purity, partial resets, malformed declarations, info snapshot isolation and no changes to expert acceptance. Replay: existing raw actions and expert controller layouts/cadence. Vertical slice: one small headless real simulation run where supported, artifact contents, repeatability of authored initial conditions; report environmental failures explicitly. Run black==26.3.1, focused regressions, API coverage, package build and affected context review.
