# First Run

Follow {doc}`install` and {ref}`verify the installation <verify-install>`.
Choose a scene tutorial if you have a source checkout, or preview an official
task directly from the installed package.

(first-run-scene)=
## Create your first scene

From the repository root, run the scene tutorial.

```bash
python scripts/tutorials/sim/create_scene.py --headless
```

The terminal prints simulation step and FPS logs once the scene is running.
Press `Ctrl+C` to stop. Omit `--headless` to open the DexSim viewer on a machine
with a display, or add `--device cuda` to the headless command to select the GPU.

See {doc}`/tutorial/create_scene` for the code walkthrough.

### Watch with WebRTC

WebRTC streams DexSim's rendered view to your browser and shows the simulation's
startup information. This read-only viewer requires a WebRTC-enabled DexSim
build and a Linux host with an NVIDIA GPU that supports NVENC.

```bash
python scripts/tutorials/sim/create_scene.py --web --max_steps 100000
```

Open the address printed in the terminal, `http://127.0.0.1:8765/` by default.
For remote connections, see the [WebRTC networking and media requirements](https://github.com/DexForce/EmbodiChain/blob/main/embodichain/lab/visualization/README.md#dexsim-boundaries).

### Inspect with Viser

Run `python scripts/tutorials/sim/create_scene.py --viser` and open the endpoint
printed in the terminal. If the simulator runs on a remote host, forward its
loopback port. For the default port, run

```bash
ssh -N -L 8080:127.0.0.1:8080 user@worker-host
```

See {doc}`/overview/sim/visualization/viser` for browser setup details.

(first-run-task)=
## Preview an official task

Run the Franka pick-and-place deployment from the installed package.

```bash
embodichain run-task \
  --gym_config embodichain/extensions/tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml \
  --preview
```

The preview opens the initialized scene so you can inspect the Franka robot
and cube. Add `--web` to the command to watch the task in the WebRTC viewer.
See {doc}`/guides/run_task` for display and preview controls, or
{doc}`/resources/assets` for asset setup. To run and edit the program, follow
{doc}`/overview/task_program/quickstart`.
