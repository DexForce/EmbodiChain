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
| 🚧 | Support a more efficient real-time denoiser. |
| 🔬 | Add 3DGS support for rendering and data generation. |

### Physics

| Status | Planned capability |
| --- | --- |
| 🔬 | Develop a next-generation physics backend with high-accuracy simulation, differentiable dynamics, and neural physical models for end-to-end AI integration. |

### Sensors

| Status | Planned capability |
| --- | --- |
| 📌 | Add more physical sensor models, such as force sensors, with runnable examples. |

### Motion Generation

| Status | Planned capability |
| --- | --- |
| 📌 | Add more advanced motion generation methods with examples. |

### Robot Integration

| Status | Planned capability |
| --- | --- |
| 📌 | Add support for more robot models, including LeRobot and Unitree H1/G1. |

## Data Pipeline

| Status | Planned capability |
| --- | --- |
| 📌 | Release a Real2Sim pipeline for automatic data generation and scaling from real-world seeding priors. |
| 📌 | Release an agentic skill generation framework for automated expert trajectory generation. |
| 📌 | Release a sim-ready asset and scene-layout generation framework for fast environment prototyping. |

## Models and Training Infrastructure

| Status | Planned capability |
| --- | --- |
| 📌 | Release a modular VLA framework for fast prototyping and training of embodied agents. |

## Embodied Tasks

| Status | Planned capability |
| --- | --- |
| 📌 | Add more benchmark tasks for EmbodiChain. |
| 📌 | Add more tasks with reinforcement learning support. |
| 📌 | Add manipulation tasks that demonstrate the data generation pipeline. |

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
