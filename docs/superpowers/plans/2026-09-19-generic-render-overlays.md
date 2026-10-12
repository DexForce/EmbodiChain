# Generic render overlays implementation plan

> For agentic workers: use subagent-driven-development for the independent native and consumer changes.

**Goal:** Replace the dedicated DexSim debug-mesh factory and object subtype with existing mesh/render/material abstractions, and adapt EmbodiChain without changing marker behavior.

**Architecture:** Arena remains the native resource owner. RenderBody owns creation-time render routing and explicitly owned material retirement. Spawn descriptors and the existing render-actor construction path expose ordinary versus overlay rendering. EmbodiChain retains marker geometry, environment selection and attachment; it consumes the general Spawn entry point.

**Tech stack:** C++17, pybind11, Python, NumPy, DexSim/EmbodiChain tests.

**Spec / authority:** The user's 2026-09-19 instruction explicitly approves the preceding recommendation to extend generic render properties, generalize resource lifetime and adapt EmbodiChain. No repeated approval is required.

## Decision

This is an architectural refinement of the existing unmerged feature. Compared options: retain the specialized factory (least churn but parallel ownership); extend existing render components/descriptors (selected by the user); introduce a separate rendering/marker manager (unnecessary new ownership). Preserve existing queues, threads, defaults and physics boundaries. True native batch/instancing remains the separate DexSim #227 requirement, based on existing Scene environment/object ownership.

## Contract

- Remove `Arena.create_debug_mesh`, its C++ implementation, and `DebugMeshObject`.
- Reuse `create_actor`, RenderBody geometry/material operations and Arena removal.
- Configure overlay routing, shadow and picking before the first GPU build.
- Material ownership is generic, per-body and opt-in. Deleting/replacing resources must not invalidate explicitly shared or externally retained material instances.
- Ordinary mesh behavior remains the default. Unsupported rendering combinations fail explicitly.
- Extend RenderDesc and MaterialDesc where necessary; compose through the existing Spawn render-actor implementation. A public render-only Spawn factory may use an already existing Arena before physics preparation; it must not introduce another resource registry or finalize/rebuild physics.
- Keep RGBA, sensor-image exclusion by default, explicit overlay inclusion, world/env coordinates, selected updates and attach/detach behavior.
- No claim that Python batch semantics are native bulk calls or GPU instancing.

## Tasks

- [ ] Native component/binding changes and focused renderer/ownership tests; remove specialized factory and cleanup.
- [ ] Spawn descriptor/factory composition, validation before allocation, transactional cleanup and focused tests.
- [ ] EmbodiChain native adapter and capability detection migration; preserve rollback, resource reuse, batch and attachment tests.
- [ ] Migrate native rendering/lifetime integration tests; update API/context/PR descriptions and the batch issue as appropriate.
- [ ] Independent review, focused CPU/native checks, required formatting and API gates; update existing MR !1423 and PR #654, then record exact current-HEAD CI results.

## Worktree and delivery constraints

Use only `/home/dex/workspace/sources/dexsim-markers` and `/home/dex/workspace/sources/EmbodiChain-markers`. Other worktrees contain concurrent user changes. Preserve both feature branches. No merge or release. The previously identified Spawn COM/inertia incompatibility is tracked independently and must remain visible if it blocks real-asset qualification.
