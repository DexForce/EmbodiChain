# Grouped PushCube deadline repair

The [original grouped study](results/GROUPED_CORE.md) exposed two different failure modes in seed 147105. The two successful development panels returned at 169.6 and 299.1 seconds. Saving their checkpoints took until 80.9 and 196.9 seconds from CLI entry, respectively. Development execution and reporting therefore accounted for 191.0 of the 299.1 seconds. Starting the clock outside Xvfb would have put the selected result at about 300.4 seconds.

The selected policy completed 94 of its 100 reserved attempts. Each of its six failures timed out after 300 controls with the cube on the tabletop. Five pushed it forward but ended outside the 10 cm goal radius. One advanced the cube by less than 1 mm in X. All six started in the upper strip of the square reset distribution (Y above −0.03 m), near its left or right boundary (X below −0.565 m or above −0.45 m). Case inspection identified this region after evaluation. Counts measured there describe the old failure; fresh scenes verify the repair.

## Producer, policy and clock

- The 10-lane training group now includes nine jittered positions spanning the corners, edges and center of the original ±10 cm square, plus one unrestricted uniform draw. Each group shuffles the assignment to arena IDs. The producer restores the original event configuration after its resets. All evaluation scenes retain their original uniform reset. Per-lane initial cube XY, seed, stratum, attempt outcome and length are logged.
- ACT predicts a 30-step chunk. Every three control steps the deployed policy queries a fresh image and uses the native temporal ensemble on three-command blocks. The decay is scaled by three so predictions for the same absolute control time have the corresponding temporal weights. The controller applies every action at 30 Hz and checks termination on every physics control step. During development, the simulator can omit sensor capture for an observation the policy will discard. The reserved evaluations keep the original camera capture at every step in independent single arenas. Batch/ensemble state resets at episode boundaries.
- The training owner services refill between optimizer updates. An outstanding native fill can complete during evaluation. The repair schedules further requests after development returns, avoiding repeated producer render work during a long development panel. All admission, release and 192-draw checks still go through the same core `OnlineDataEngine`.
- The cold launcher records monotonic time before exec of Xvfb and the heavy training process. The public deadline remains 300 seconds. The selection cutoff is 295 seconds to leave time for publication. If the previous scheduled panel is at least 40/50 and the current panel reaches 50/50, the owner locks the best eligible trained checkpoint immediately. Otherwise the timer selects the best eligible policy at the cutoff. The same two consecutive 40/50 minimum, independent 95/100 threshold and five-seed denominator apply.

## Reproduction

Use a fresh `ACT_ROOT`, a matching expert qualification, the runtime described in [the grouped example](GROUP_PUBLICATION.md), and one physical GPU per run. Each arm uses a single GPU throughout. The five seed blocks can share the host on separate cards.

```bash
export CUDA_VISIBLE_DEVICES=4 EMBODICHAIN_SIM_EXIT_PROCESS=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/extensions:$PWD/extensions/embodichain_tasks"
export PROTOCOL="$PWD/examples/learning/act/grouped_repair_protocol.json"
for seed in 167101 167102 167103 167104 167105; do
    python -m examples.learning.act.timed_grouped --seed "$seed" --protocol "$PROTOCOL" \
        --qualification "$QUALIFICATION" --output "$ACT_ROOT/$seed"
done
python -m examples.learning.act.audit_grouped --root "$ACT_ROOT" --protocol "$PROTOCOL"
```

The audit checks returned development results and the lock receipt, recomputes physical outcomes, verifies native checkpoint fingerprints and checks each release ledger. All five runs must submit and pass. Post-selection test time is recorded separately from the training budget.

## Timing follow-up

The 224-pixel, three-step query variant produced two late or ineligible development pairs in the first five-seed repair study. The follow-up keeps native 224×224 acquisition and computes colour landmarks before resizing. Area pooling supplies a 56×56 image to the ResNet while the measured joint positions, landmarks, transformer, release budget and evaluator task remain the same. Query blocks contain six consecutive native control targets and overlap on their original time grid. Full-rate RGB capture is retained in the reserved simulator. A second receipt hashes the decision file after its atomic write and records elapsed time at that point.

The [follow-up protocol](grouped_spatial_protocol.json) specifies fresh seeds 187101–187105, development scenes 19310000–19310049 and reserved scenes 19410000–19410099. It uses scheduled updates 1400, 2050, 2450, 3000 and 4000. Use the same launcher and audit commands above with this protocol and a fresh output root.

## Elapsed control input

For the final cohort the grouped owner reads the anchor's native ODS frame index,
clips it to the median accepted demonstration length measured in the first
publication and normalizes the resulting fraction with first-publication statistics.
At deployment the policy increments a counter on every issued control, including
cached commands, and resets it with the ensemble. The input uses only controller
history. Sensors, action cadence and the task's termination conditions are identical
to those in the spatial follow-up. In the separate engineering comparison on
scenes 19810000–19810049 with full-rate camera capture, the spatial policy from
187103 at step 2050 scored 44/50. Elapsed-input training with the same learner and
producer seeds scored 50/50 at steps 1500, 2000 and 2500.

Use the [elapsed-input protocol](grouped_elapsed_protocol.json) with training
seeds 207101–207105 and a fresh output root. Its development and reserved panels
begin at 21310000 and 21410000. The checkpoint schedule is 1000, 1500, 2000,
2500 and 4000. The recipe and checkpoint selection rule are frozen before
launching these five single-GPU jobs.
