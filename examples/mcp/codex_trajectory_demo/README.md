# Codex-driven Franka trajectory demo

For the conceptual walkthrough, see the [EmbodiChain MCP tutorial](https://dexforce.github.io/EmbodiChain/main/guides/mcp.html).

This demo connects Codex to the EmbodiChain MCP server. Codex creates a
headless simulation World, loads an integrated Franka Panda configuration,
generates a bounded joint trajectory, validates it against the robot's joint
limits, and executes the complete trajectory through one coarse-grained MCP
call.

The high-frequency loop stays inside the SimulationManager adapter. Codex
chooses the tools and waypoints; it does not send one MCP request per physics
step.

## Install and register the server

From the repository root:

```bash
python -m pip install -e '.[mcp-demo]'
codex mcp add embodichain -- python -m embodichain mcp
codex mcp list
```

Start a new Codex session in this repository and ask:

```text
Use the EmbodiChain MCP server to run the Franka trajectory demo.

1. Read embodichain://demos/codex-trajectory/scene.
2. Create a world with backend default and seed 7.
3. Load the returned scene.
4. Read the current FrankaPanda qpos from the world state; use all nine joints in the trajectory.
5. Use the waypoints in examples/mcp/codex_trajectory_demo/trajectory.json.
6. Call generate_robot_trajectory with those waypoints and samples_per_segment=24.
7. Report the trajectory_id, number of samples, validation result, and resource URI.
8. Call list_cameras and confirm trajectory_camera is an offscreen RGB/depth camera.
9. Report the trajectory_id, number of samples, validation result, and resource URI.
10. Call record_trajectory with camera_id=trajectory_camera,
    output_path=outputs/mcp/codex_franka.mp4, fps=30, include_depth=true.
11. Read the world state again and summarize the final qpos and video artifact.
12. Destroy the world after the result is reported.
```

The built-in scene resource and the checked-in JSON files describe the same
robot. The resource is useful when the MCP client is isolated from the local
filesystem; the JSON files make the waypoints easy to edit and review.

## What Codex is doing

The tool loop is:

```text
read scene resource
  -> create_world
  -> load_task_or_scene
  -> get_world_state
  -> generate_robot_trajectory
  -> validate joint limits
  -> list_cameras
  -> record_trajectory (RGB MP4 + optional depth NPZ)
  -> get_world_state
  -> destroy_world
```

`generate_robot_trajectory` creates a trajectory handle and stores the samples
under `embodichain://trajectories/{trajectory_id}`. The first implementation
uses deterministic joint interpolation for this free-space example. The
validation checks finite values and configured joint limits; it reports that a
generic collision query is not available for this adapter.

`record_trajectory` is one MCP call. The backend applies samples, renders an
offscreen CameraGroup, and writes the video internally, so Codex remains at
task-level control granularity.

## Local protocol smoke test

The MCP server can also be inspected without an LLM:

```bash
python -m pytest -q tests/lab/mcp/test_service.py
python examples/mcp/codex_trajectory_demo/run_demo.py --record
```

The test covers tool discovery, resource templates, trajectory handles, and
the deterministic in-memory backend. A real headless Franka smoke run requires
the cached Franka asset and the simulator dependencies installed by the main
EmbodiChain environment. `--record` writes `outputs/mcp/codex_franka.mp4` and
`outputs/mcp/codex_franka.depth.npz`.
