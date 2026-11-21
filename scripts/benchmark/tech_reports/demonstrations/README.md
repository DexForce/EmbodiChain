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
