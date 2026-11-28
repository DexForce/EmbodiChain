# DexUni rigid ownership qualification

Run from the EmbodiChain root with the DexSim candidate and CUDA available. Each process creates and closes its own world. Use one GPU lane for comparable synchronized wall times. JSON, command tapes and the summary are written under `outputs/dexuni/`.

| Mode | Requested scene |
|---|---|
| `auto` | DexUni with no ownership or coupling override |
| `all_mj` | DexUni with all finalized joints assigned to MuJoCo |
| `one_way`, `two_way` | Explicit DexUni mixed partition and reaction policy |
| `mj` | Direct MJ Warp with the deployed MuJoCo options in manipulation tasks |

In the pile benchmark `mj` selects DexUni's internal pure MuJoCo branch, and `vbd` selects empty MuJoCo articulation ownership. The output records the resolved backend.

## Physical measurements

The load benchmark runs 12 simultaneous combinations of payload mass (32, 63, 126 g), payload friction (0.35, 1) and smooth command ramp duration (12, 36 control frames). Each of three reset episodes runs 450 frames at 60 Hz, with ten native substeps per frame. A gravity-compensated, vertically compliant platform weighs each box. The static spring load is compared with the independently known weight. Following closure the gantry rises 80 mm, transports the box 80 mm horizontally, then releases it over the platform. Compare weight error, closure error, transport slip, retention and the final supported placement. The timed path includes device-resident commands and completed manager steps, with synchronization every 15 frames and state readback outside timing.

The native expert benchmark runs the configured UR5 pick/place, two-block stacking and drawer-storage experts including their validators. Pick/place and drawer receive seeded ±6 mm XY and ±3° yaw variation. Stacking retains its authored randomization. Payload mass cycles over 0.75, 1 and 1.25 times nominal. Episode time includes planning and validation. Early termination changes the amount of work.

The paired benchmark uses the first successful explicit full-MJ expert trajectory from `tasks`. Every mode receives the same tape, initial pose perturbations and mass factors. Row zero is centered on the source pose. Replays run the whole trajectory, so a failed grasp cannot inflate throughput by ending the measurement early. The stage checks measure lift, final position and tilt for pick/place and stacking, or rail travel, release and containment for drawer storage. The task-expert results separately retain the native segment, stability and planning decisions.

## Commands

```bash
python -m scripts.benchmark.robotics.dexuni.load auto
python -m scripts.benchmark.robotics.dexuni.tasks pick all_mj 0 1 2
python -m scripts.benchmark.robotics.dexuni.paired pick auto --size 30 --repeats 1
python -m scripts.benchmark.robotics.dexuni.piles mj 64 1
python -m scripts.benchmark.robotics.dexuni.piles two_way 16 1 --push
python -m scripts.benchmark.robotics.dexuni.summary
```

Repeat across modes and tasks, keeping the same tape within each task. For the pile pilot vary per-world size and number of worlds separately. Piles use a fixed 19 seed, 50 mm cubes stacked four high and four native substeps per 10 ms update. After 130 settling steps the measured window contains sixteen synchronized blocks of ten updates. Inspect late drift, tilt, floor penetration and vertical overlap of nominally adjacent layers before interpreting throughput. Vertical overlap is conservative for tilted boxes. The optional blade advances during the measurement window and can intentionally disrupt the stack. The final contact snapshot is outside timing. NVML memory is the maximum sampled process residency over 16 additional, synchronized, warmed steps.

To run the complete task matrix on the same GPU, first run all three expert seeds with `all_mj` to produce a qualified reference. The paired run uses 30 distinct starting rows per task and solver. The expert and replay denominators measure different policies.

```bash
for task in pick stack drawer; do
    for mode in all_mj auto one_way two_way mj; do
        python -m scripts.benchmark.robotics.dexuni.tasks "$task" "$mode" 0 1 2
        python -m scripts.benchmark.robotics.dexuni.paired "$task" "$mode" --size 30 --repeats 1
    done
done
for mode in auto all_mj one_way two_way mj; do
    python -m scripts.benchmark.robotics.dexuni.load "$mode"
    for mu in 0.35 1; do
        python -m scripts.benchmark.robotics.dexuni.load "$mode" --material-friction "$mu"
    done
done
```

The optional common material case authors the same friction on both fingers, platform and payload to avoid a larger finger coefficient masking the payload change under MuJoCo's mixing rule. Each such run has six rows. For P1, use `piles vbd 16 1 --iterations 20` to repeat with more AVBD iterations at the same quality gates.

For selected reset measurements use `load auto --lifecycle-graph 1` and repeat with `0` and the other modes. Five measured resets select alternating rows and verify joint/box restoration and isolation. For large pure-rigid worlds `piles mj 256 1 --cg-sparse` selects MuJoCo CG and a sparse Jacobian while leaving the time integration and geometry identical.
