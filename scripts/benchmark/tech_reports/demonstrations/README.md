# End-to-end demonstrations (D-series)

Directory scaffold for [issue #679](https://github.com/DexForce/EmbodiChain/issues/679).
The planned D-03 workflow assembles existing results, stage evidence and video
artifacts; it does not establish new performance measurements by itself.

- `experiments/`: demonstration definitions, evidence selection and assembly.
- `platforms/`: platform-specific evidence import/recording adapters when needed.
- `../assets/demonstrations/`: small input definitions and evidence references.

Preserve frozen run IDs, configurations, asset hashes, stage outcomes and
unedited recordings. Put generated indexes, reports and media in
`outputs/benchmarks/`. Pure assembly/report modules must work from frozen
artifacts without importing a simulator or initializing a GPU. RL/IL training,
generalization and sim-to-real claims remain outside this initial scaffold.

See the [benchmark layout](../README.md#technical-report-comparison-layout).

## Franka rope wrap

Run from the repository root with the EmbodiChain and Isaac Lab Newton Python
environments. Use the resolved `Fr3RopePadsV1.urdf` with its adjacent meshes
for the Isaac Lab run. The EmbodiChain task resolves the same robot asset.

```bash
platforms=scripts/benchmark/tech_reports/demonstrations/platforms
run_dir=outputs/benchmarks/franka-rope-wrap

"$EMBODICHAIN_PYTHON" "$platforms/embodichain_franka_rope_wrap.py" \
  --record-dir "$run_dir/recording"

recording="$run_dir/recording/traj_env0_000000.pt"
"$EMBODICHAIN_PYTHON" "$platforms/replay_embodichain_franka_rope_wrap.py" \
  --recording "$recording" --output "$run_dir/embodichain"
"$ISAACLAB_PYTHON" "$platforms/isaaclab_franka_rope_wrap.py" \
  --recording "$recording" --robot-urdf "$FRANKA_URDF" --output "$run_dir/isaaclab"
```

Omit `--record-dir` to run the YAML Task Program without recording. Both
replays save fresh cable poses in `trajectory.npz` and report the final winding
angle and maximum rope-tip displacement relative to the start of the last two
seconds in `result.json`. The reports include hashes for the recording and
scene inputs.

## Franka RJ45 insertion and withdrawal

Run from the repository root with a CUDA GPU. Use the EmbodiChain Python
environment with DexSim 0.6.0 and the Isaac Lab Newton environment with
Isaac Lab 17.0.2, isaaclab-newton 5.4.1, Newton 1.5.2, Warp 1.16.0,
TensorDict and PyYAML.

The source task resolves the calibrated Franka, plug and socket through the
EmbodiChain asset cache. The `Rj45Fixture` ID resolves to the
`Rj45FixtureV1` cache entry. Use these commands to prepare both assets before
running offline. [Pinned RJ45 fixture archive](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/dcb448da52ea3df5df5a249e9ff977acafead0e4/task_assets/Rj45FixtureV1.zip)

```bash
"$EMBODICHAIN_PYTHON" -m embodichain data download --name FrankaRj45
"$EMBODICHAIN_PYTHON" -m embodichain data download --name Rj45Fixture
```

Set `EMBODICHAIN_DATA_ROOT` before downloading, recording and replaying in
EmbodiChain when using a custom cache. Isaac Lab reads the prepared URDF paths
and hashes from the asset manifest. Keep the adjacent mesh directories in
place.

```bash
platforms=scripts/benchmark/tech_reports/demonstrations/platforms
run_dir=outputs/benchmarks/franka-rj45-cycle

"$EMBODICHAIN_PYTHON" "$platforms/embodichain_franka_rj45_cycle.py" \
  --record-dir "$run_dir/recording" --asset-manifest "$run_dir/assets.json"

recording="$run_dir/recording/traj_env0_000000.pt"
"$EMBODICHAIN_PYTHON" "$platforms/replay_embodichain_franka_rj45_cycle.py" \
  --recording "$recording" --output "$run_dir/embodichain"
"$ISAACLAB_PYTHON" "$platforms/isaaclab_franka_rj45_cycle.py" \
  --recording "$recording" --asset-manifest "$run_dir/assets.json" \
  --output "$run_dir/isaaclab"
```

Omit `--record-dir` to run the YAML Task Program without saving a recording.
The integration selects a packaged nominal trajectory and declares the
grasp-observation windows and object-frame offsets. The source program applies
feedback at those windows during execution. The recording stores the actual
position and velocity targets for all nine
Franka joints across the accepted 40-second episode. Both replays use this
recording and freshly solve the calibrated scene at 60 Hz with 32 physics
substeps per control. The Isaac Lab run uses native Newton VBD for the whole
scene. EmbodiChain uses the configured Newton/DexUni coupling.

Each replay writes measured robot, cable and plug states with target readbacks
to `trajectory.npz` and its own frame measurements to `metrics.json`.
`result.json` records input hashes and the original 17 gate results, including
seating, latch contact, cable attachment and final return. A completed replay
can report failed acceptance gates. The EmbodiChain result also reports the
largest differences from the recorded state after its fresh pre-roll.
