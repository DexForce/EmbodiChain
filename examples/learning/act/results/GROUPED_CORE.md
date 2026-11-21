# Grouped ODS core: single-card PushCube acceptance

**FAIL**. 4/5 seeds published a qualifying checkpoint by 300 seconds and scored at least 95/100 on the separate physical simulation panel. The model received one current camera frame and joint state plus a padded 30-action training target. The producer, optimizer and development evaluation shared one GPU per run. [Protocol](../grouped_protocol.json) · [Implementation and reproduction](../GROUP_PUBLICATION.md) · [Evidence JSON](grouped_core.json)

| Seed | Selected step | Development result at s | Reserved successes / 100 | Replacement permissions (episodes) | Smallest credit at release |
| --- | ---: | ---: | ---: | ---: | ---: |
| 147101 | 1850 | 274.3 | 100 | 80 | 192 |
| 147102 | 1500 | 220.8 | 99 | 60 | 192 |
| 147103 | 1500 | 222.6 | 99 | 70 | 192 |
| 147104 | 1850 | 289.6 | 98 | 80 | 192 |
| 147105 | 1500 | 299.1 | 94 | 50 | 192 |

Across the five runs, 340 occupied-episode replacement authorizations were audited. Each trajectory received at least 192 committed optimizer draws before the next group could write its slot. Empty first-use groups and final resident rows do not count as replacements. The shared read area held 20 episodes, replenished in groups of ten. The current image and future action window were copied together under the producer lock. A full historical CPU Replay was absent.

Each chosen policy was fixed by the best eligible pair of consecutive 40/50 development results. Returned timestamps include checkpoint saving and synchronous physical development evaluation. Independent verification followed the deadline. Five learner seeds shared the reserved reset seeds; repeated scene panels are dependent. Concurrent seed blocks used distinct RTX PRO 5000 72GB cards on a shared host. Python 3.12, PyTorch 2.11, DexSim 0.6.0 and LeRobot 0.6.1 were used. A process-local metadata overlay restored the checkout’s entry points for this host’s editable installation.

The protocol clock starts at entry to the training CLI, before its heavy training imports. The shell/Xvfb launcher and earlier Python module import add measurable latency. A separate filesystem-clock alignment from log creation to the selected dev report measured 147101: 275.87 s, 147102: 222.31 s, 147103: 223.97 s, 147104: 290.99 s, 147105: 300.43 s. It is a launch-overhead diagnostic with a different time origin.

Seed 147105 missed the reserved target by one scene. Its six failures all reached the 300-step limit, with final XY errors of 10.6–19.9 cm against the 10 cm target. The earlier pre-commit integration trial scored 3/5 on the same seeds; this is repeatability evidence, not an additional independent cohort.

## Prior double-layer measurements

The previous isolated group experiment recorded 5/5 for both arms. Median locked result time was 276.2 s with direct sampling and 238.7 s with historical Replay. Sampled training process-tree PSS was 11.23 and 15.32 GiB, respectively. These are archived measurements from different seeds. The core replication above ran only the single-layer arm.
