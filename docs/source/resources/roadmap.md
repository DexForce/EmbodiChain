# Roadmap

EmbodiChain is in alpha and under active development. This roadmap summarizes
the main areas we are improving and the capabilities planned for upcoming
releases.

The roadmap is organized by product area so new work can be added without
changing the whole page. Each item should be short, user-facing, and grouped
under the area it improves.

## Status Legend

| Marker | Status | Meaning |
| --- | --- | --- |
| 🚧 | In progress | Work is actively being designed, implemented, or validated. |
| 📌 | Planned | Work is on the project roadmap but not yet released. |
| 🔬 | Research | Work is exploratory and may change as the technical approach matures. |

## Simulation

### Rendering

| Status | Planned capability |
| --- | --- |
| 📌 | Add deformation-aware 3DGS visuals for deformable objects, with batched simulation and reset support. ([#1199](https://github.com/DexForce/EmbodiChain/issues/1199)) |

### Physics

| Status | Planned capability |
| --- | --- |
| 🔬 | Advance high-accuracy simulation, differentiable dynamics, and neural physical models for end-to-end AI integration. |

### Sensors

| Status | Planned capability |
| --- | --- |
| 🚧 | Support third-party sensor packages with public lifecycle and sampling contracts, and add reference sensors such as encoders and IMUs. ([#835](https://github.com/DexForce/EmbodiChain/issues/835), [#1174](https://github.com/DexForce/EmbodiChain/issues/1174)) |

## Data Pipeline

| Status | Planned capability |
| --- | --- |
| 🚧 | Enable agent-driven GenSim workflows for inspection, validation, execution, observation, and revision within a fixed budget. ([#1198](https://github.com/DexForce/EmbodiChain/issues/1198), [#1122](https://github.com/DexForce/EmbodiChain/issues/1122), [#716](https://github.com/DexForce/EmbodiChain/issues/716)) |

## Models and Training Infrastructure

| Status | Planned capability |
| --- | --- |
| 🚧 | Provide configuration-driven world-model and APG policy training, with complete recovery, standard policy export, and independent physical evaluation. ([#1176](https://github.com/DexForce/EmbodiChain/issues/1176)) |

## Extending This Roadmap

When adding roadmap items:

- Add the item under the closest existing area before creating a new section.
- Use one row per user-facing capability.
- Keep status markers limited to the status legend above unless the legend is
  updated at the same time.
- Prefer concrete outcomes over implementation details.

New sections should follow this template:

```md
## Area Name

| Status | Planned capability |
| --- | --- |
| 📌 | Describe the capability and the user-facing outcome. |
```
