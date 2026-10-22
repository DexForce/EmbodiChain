# Bounded CPU AV1 encoder-thread qualification

12 complete runs: one fixed 479-frame PickPlace episode per vision; 256×256; 1/2/4 encoder threads; two repetitions with the order reversed in round 2. CPU affinity 16–23 (8 cores), Torch/OMP/MKL/OpenBLAS 1 thread, no GPU. Inputs were decoded once from retained formal datasets; that setup and all verification are excluded from writer timing. All slower results are retained.

The experiment measures isolated recording service. It cannot predict complete ODS pipeline time or contention with a learner. Raw inputs for this experiment are the decoded historical videos, so quality is measured against those fixed inputs after one additional AV1 encoding. No codec, CRF, GOP, resolution or frame-count change.

## Time and memory

Seconds: mean [min, max], n=2. `add_frame` includes frame assembly and synchronous image persistence; `save_episode` includes statistics, metadata/numeric commit and AV1 encoding. Nested stages must not be added to the parent total.

| Vision | Encoder threads | Writer start to joined | Validation/provenance | add_frame | save_episode | References/manifest | SDK create | SDK finalize |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single | 1 | 3.613217 [3.593743, 3.632691] | 0.000144 [0.000075, 0.000213] | 1.175774 [1.166152, 1.185396] | 2.424662 [2.415497, 2.433826] | 0.001168 [0.001164, 0.001173] | 0.003973 [0.003424, 0.004521] | 0.004107 [0.004064, 0.004150] |
| single | 2 | 2.957492 [2.939165, 2.975819] | 0.000075 [0.000075, 0.000076] | 1.176361 [1.169745, 1.182977] | 1.769053 [1.757641, 1.780464] | 0.001027 [0.000991, 0.001063] | 0.003562 [0.003447, 0.003678] | 0.003938 [0.003893, 0.003984] |
| single | 4 | 3.007533 [2.991572, 3.023494] | 0.000075 [0.000074, 0.000076] | 1.184543 [1.184214, 1.184872] | 1.810890 [1.795368, 1.826413] | 0.001020 [0.001012, 0.001029] | 0.003521 [0.003513, 0.003529] | 0.004067 [0.003936, 0.004198] |
| dual | 1 | 6.552071 [6.457571, 6.646570] | 0.000073 [0.000073, 0.000073] | 2.249229 [2.243969, 2.254488] | 4.289685 [4.189604, 4.389765] | 0.001625 [0.001570, 0.001680] | 0.003501 [0.003473, 0.003530] | 0.004495 [0.004388, 0.004602] |
| dual | 2 | 5.563184 [5.543997, 5.582372] | 0.000070 [0.000068, 0.000072] | 2.243655 [2.237617, 2.249693] | 3.306378 [3.281187, 3.331568] | 0.001489 [0.001450, 0.001529] | 0.003566 [0.003449, 0.003684] | 0.004566 [0.004560, 0.004573] |
| dual | 4 | 5.617567 [5.583845, 5.651290] | 0.000068 [0.000066, 0.000070] | 2.222345 [2.215296, 2.229393] | 3.381671 [3.355210, 3.408131] | 0.001567 [0.001544, 0.001589] | 0.003779 [0.003535, 0.004022] | 0.004573 [0.004540, 0.004606] |

Final corpus files total 28,710,708 bytes across all 12 runs. Temporary peak disk and process peak RSS were not sampled; no constant-memory or temporary-peak claim is made.

## Correctness and quality

Full official LeRobot Dataset / TorchCodec decoding checked 5,748 dataset frames and 8,622 camera frames, with exact numeric state/action comparisons on every frame. For each vision/camera, the decoded uint8 image hash is identical across all six thread/repetition combinations. All writers finalized and joined.

| Vision / camera | Decoded camera frames | MAE (uint8) | PSNR (dB) |
|---|---:|---:|---:|
| single / cam_high | 2874 | 1.437953 | 42.854704 |
| dual / cam_high | 2874 | 1.439571 | 42.849225 |
| dual / cam_wrist | 2874 | 1.408247 | 43.077996 |

## Comparison

| Vision | 2 threads vs 1: reduction in mean writer time | 4 threads vs 2: additional change |
|---|---:|---:|
| single | 18.15% | 1.69% slower |
| dual | 15.09% | 0.98% slower |

Two encoder threads are a supported candidate for the same eight-core, 256×256 recording workload: both repetitions improve over one, while four do not improve over two. Keep the compatibility default at one until an actual GPU training pipeline comparison verifies the contention tradeoff. This limited experiment does not establish a universal optimal thread count.

Raw outcomes, individual stage intervals, source hashes and input hashes: [profile.json](profile.json). All original corpora and event logs remain alongside this report. Temporary qualification script: `/tmp/ods-recorder-profile-20261023/thread_sweep.py`.
