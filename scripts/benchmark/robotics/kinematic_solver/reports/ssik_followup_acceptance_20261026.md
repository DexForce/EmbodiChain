# SSIK follow-up acceptance

RTX 5090 / Ryzen 9 9950X, Torch2.7.1+cu128, Warp1.17.0, driver595.58.03. Returned joints are float32; internal geometry and coefficient arithmetic are float64. Timed input dtypes are recorded per scope (fixed-size matrix: float32). Accepted outputs independently satisfy original URDF/PK TCP FK at1e-5 m/rad, effective limits and the hard step bound where configured.

[Complete measurements, input/source hashes, raw-artifact indices, offline recipes and counterexamples](ssik_followup_acceptance_20261026.json). [Previous delivery](ssik_cuda_delivery_20261026.md) remains unchanged. P1 Aloha Mini and both Marvin variants have completed all mandatory accuracy, continuity, latency, bulk and resource gates. Their arm defaults now use validated SSIK Warp; original Pytorch restoration and explicit caller overrides remain available. ARX5 current numerical-hybrid control, Robot and memory gates pass; it remains an explicit opt-in with a historical original-Pytorch bulk reference. Rizon4S current conditioned CUDA gates pass; its default remains Pytorch for CPU compatibility.

Final merged-source CPU fast lane:9,981 passed,31 skipped,0 failures; optional-SSIK-blocked installed-wheel checks pass after the latest mainline model-package merge. The previous hosted full CI passed10,814 tests with35 skips on5acde0417, including actual GPU/simulation lanes and the fresh-process CUDA case. That result does not certify the P5/default or merged head; current-head CI status is recorded separately inJSON.

## Time & Memory

| Model | Backend | Scope | N | Calls | p50_ms | p95_ms | p99_ms | max_ms | Scratch_MiB | Sustained_queries_s | Qualification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aloha_left | warp/cpu | streaming_control | 1 | 10000 | 0.139622 | 0.146596 | 0.164911 | 0.591763 | — | — | final monitored source |
| aloha_right | warp/cpu | streaming_control | 1 | 10000 | 0.138461 | 0.146115 | 0.205928 | 0.72332 | — | — | final monitored source |
| aloha | warp/cpu | paired_streaming_control | 2 | 10000 | 0.280838 | 0.290256 | 0.301819 | 0.877881 | — | — | final monitored source |
| marvin_bare_left | warp/cpu | streaming_control | 1 | 10000 | 0.332174 | 0.345401 | 0.457911 | 0.859948 | — | — | final monitored source |
| marvin_bare_right | warp/cpu | streaming_control | 1 | 10000 | 0.331404 | 0.342805 | 0.478651 | 0.95144 | — | — | final monitored source |
| marvin_bare | warp/cpu | paired_streaming_control | 2 | 10000 | 0.663548 | 0.688837 | 0.809053 | 1.34885 | — | — | final monitored source |
| marvin_gripper_left | warp/cpu | streaming_control | 1 | 10000 | 0.338787 | 0.352985 | 0.385406 | 1.06101 | — | — | final monitored source |
| marvin_gripper_right | warp/cpu | streaming_control | 1 | 10000 | 0.339068 | 0.352663 | 0.5314 | 1.05996 | — | — | final monitored source |
| marvin_gripper | warp/cpu | paired_streaming_control | 2 | 10000 | 0.679299 | 0.705241 | 0.877551 | 1.49336 | — | — | final monitored source |
| aloha_left | warp/cuda | streaming_control | 1 | 10000 | 0.486916 | 0.741358 | 0.914122 | 1.61089 | — | — | final monitored source |
| aloha_right | warp/cuda | streaming_control | 1 | 10000 | 0.469037 | 0.551835 | 0.725766 | 1.03532 | — | — | final monitored source |
| aloha | warp/cuda | paired_streaming_control | 2 | 10000 | 0.810887 | 0.951256 | 1.12347 | 2.57976 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | streaming_control | 1 | 10000 | 2.0625 | 2.31363 | 2.43137 | 2.96206 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | streaming_control | 1 | 10000 | 2.06548 | 2.31437 | 2.43185 | 2.73348 | — | — | final monitored source |
| marvin_bare | warp/cuda | paired_streaming_control | 2 | 10000 | 3.91123 | 4.3196 | 4.39056 | 4.67462 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | streaming_control | 1 | 10000 | 2.07373 | 2.31416 | 2.42607 | 2.72948 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | streaming_control | 1 | 10000 | 2.07218 | 2.3191 | 2.43848 | 2.75474 | — | — | final monitored source |
| marvin_gripper | warp/cuda | paired_streaming_control | 2 | 10000 | 3.90997 | 4.33085 | 4.40109 | 4.92492 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload resident | 1 | 10000 | 0.498016 | 0.573552 | 0.766959 | 1.14362 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload transfer | 1 | 10000 | 0.538403 | 0.682811 | 0.971195 | 1.25109 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 2597.99 | final monitored source |
| aloha_left | warp/cuda | fixed workload resident | 8 | 10000 | 0.508696 | 0.598661 | 0.769954 | 0.989701 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload transfer | 8 | 10000 | 0.54728 | 0.676605 | 0.987968 | 1.25665 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 21576.6 | final monitored source |
| aloha_left | warp/cuda | fixed workload resident | 100 | 30 | 0.590954 | 0.771329 | 0.854132 | 0.879381 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload transfer | 100 | 30 | 0.623916 | 0.840392 | 1.06422 | 1.11769 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 242748 | final monitored source |
| aloha_left | warp/cuda | fixed workload resident | 1000 | 30 | 1.41144 | 1.56832 | 1.68111 | 1.68239 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload transfer | 1000 | 30 | 1.46833 | 1.64724 | 1.89689 | 1.98216 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 795534 | final monitored source |
| aloha_left | warp/cuda | fixed workload resident | 10000 | 30 | 12.4527 | 13.2064 | 13.3171 | 13.3476 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload transfer | 10000 | 30 | 12.6746 | 13.3884 | 13.6151 | 13.6899 | — | — | final monitored source |
| aloha_left | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 805613 | final monitored source |
| aloha_right | warp/cuda | fixed workload resident | 1 | 10000 | 0.498748 | 0.600017 | 0.757733 | 1.1607 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload transfer | 1 | 10000 | 0.534917 | 0.634291 | 0.975212 | 1.24362 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 2790.06 | final monitored source |
| aloha_right | warp/cuda | fixed workload resident | 8 | 10000 | 0.509679 | 0.593149 | 0.781292 | 1.29475 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload transfer | 8 | 10000 | 0.549014 | 0.644016 | 0.984333 | 1.3714 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 22150 | final monitored source |
| aloha_right | warp/cuda | fixed workload resident | 100 | 30 | 0.583484 | 0.631268 | 0.800648 | 0.862068 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload transfer | 100 | 30 | 0.630378 | 0.767272 | 0.982286 | 1.06884 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 253846 | final monitored source |
| aloha_right | warp/cuda | fixed workload resident | 1000 | 30 | 1.41361 | 1.42689 | 1.60487 | 1.6767 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload transfer | 1000 | 30 | 1.48433 | 1.75172 | 1.75328 | 1.75339 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 798876 | final monitored source |
| aloha_right | warp/cuda | fixed workload resident | 10000 | 30 | 12.8657 | 13.2251 | 13.3701 | 13.401 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload transfer | 10000 | 30 | 12.4638 | 13.2342 | 13.3226 | 13.3361 | — | — | final monitored source |
| aloha_right | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 813465 | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload resident | 1 | 10000 | 2.86922 | 3.28233 | 3.36698 | 3.87881 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload transfer | 1 | 10000 | 2.90542 | 3.33045 | 3.48628 | 4.11346 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 365.626 | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload resident | 8 | 10000 | 3.0092 | 3.42014 | 3.49157 | 3.98063 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload transfer | 8 | 10000 | 3.04342 | 3.46891 | 3.6365 | 4.25448 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 2807.29 | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload resident | 100 | 30 | 3.41299 | 3.83761 | 3.90608 | 3.93255 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload transfer | 100 | 30 | 3.45475 | 3.8898 | 4.01065 | 4.05584 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 30647.6 | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload resident | 1000 | 30 | 11.2444 | 11.7946 | 11.8166 | 11.8205 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload transfer | 1000 | 30 | 11.1164 | 11.5855 | 11.6454 | 11.6693 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 93200.9 | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload resident | 10000 | 30 | 108.638 | 111.852 | 112.093 | 112.158 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload transfer | 10000 | 30 | 108.981 | 112.149 | 113.265 | 113.67 | — | — | final monitored source |
| marvin_bare_left | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 91980.4 | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload resident | 1 | 10000 | 2.85279 | 3.25998 | 3.3117 | 3.86473 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload transfer | 1 | 10000 | 2.89453 | 3.31428 | 3.45759 | 3.80671 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 357.852 | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload resident | 8 | 10000 | 3.00152 | 3.41079 | 3.48504 | 4.60843 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload transfer | 8 | 10000 | 3.02828 | 3.45152 | 3.62001 | 4.0144 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 2832.67 | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload resident | 100 | 30 | 3.40743 | 3.84982 | 3.87448 | 3.8791 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload transfer | 100 | 30 | 3.50403 | 3.91041 | 3.91655 | 3.91679 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 29845.4 | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload resident | 1000 | 30 | 11.2184 | 11.5646 | 11.5807 | 11.5865 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload transfer | 1000 | 30 | 10.856 | 11.4756 | 11.6338 | 11.6553 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 93524.8 | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload resident | 10000 | 30 | 108.245 | 112.313 | 112.783 | 112.824 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload transfer | 10000 | 30 | 108.901 | 111.883 | 112.096 | 112.133 | — | — | final monitored source |
| marvin_bare_right | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 92653.9 | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload resident | 1 | 10000 | 2.86303 | 3.27225 | 3.36261 | 4.27868 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload transfer | 1 | 10000 | 2.90131 | 3.32566 | 3.47821 | 4.31932 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 368.115 | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload resident | 8 | 10000 | 2.99921 | 3.41542 | 3.48985 | 4.0969 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload transfer | 8 | 10000 | 3.03526 | 3.46299 | 3.61185 | 4.3087 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 2805.98 | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload resident | 100 | 30 | 3.41464 | 3.87566 | 3.92253 | 3.9272 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload transfer | 100 | 30 | 3.5015 | 3.96273 | 4.1691 | 4.24044 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 29653 | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload resident | 1000 | 30 | 11.1987 | 11.6573 | 11.7795 | 11.802 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload transfer | 1000 | 30 | 10.9266 | 11.5097 | 11.9307 | 12.0979 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 93476 | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload resident | 10000 | 30 | 108.913 | 111.564 | 111.981 | 112.087 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload transfer | 10000 | 30 | 108.768 | 112.939 | 113.503 | 113.592 | — | — | final monitored source |
| marvin_gripper_left | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 92513.2 | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload resident | 1 | 10000 | 2.85889 | 3.27278 | 3.37437 | 4.00952 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload transfer | 1 | 10000 | 2.90162 | 3.32173 | 3.43624 | 3.72285 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 367.896 | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload resident | 8 | 10000 | 2.99396 | 3.40096 | 3.4529 | 3.68617 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload transfer | 8 | 10000 | 3.04427 | 3.47178 | 3.61258 | 4.01252 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 2790.12 | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload resident | 100 | 30 | 3.40379 | 3.7988 | 3.88897 | 3.89922 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload transfer | 100 | 30 | 3.45486 | 3.91301 | 4.00385 | 4.03987 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 30019 | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload resident | 1000 | 30 | 11.1464 | 11.6812 | 11.8073 | 11.8406 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload transfer | 1000 | 30 | 11.0864 | 11.832 | 11.9231 | 11.9294 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 93674.2 | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload resident | 10000 | 30 | 108.796 | 112.061 | 112.673 | 112.729 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload transfer | 10000 | 30 | 108.709 | 112.64 | 112.938 | 113.022 | — | — | final monitored source |
| marvin_gripper_right | warp/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 92627.2 | final monitored source |
| aloha_left | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 438.241 | 446.001 | 447.42 | 447.634 | — | — | final monitored same-cohort source |
| aloha_left | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 5032.98 | 5054.45 | 5080.33 | 5090.55 | — | — | final monitored same-cohort source |
| aloha_right | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 448.602 | 455.993 | 456.691 | 456.898 | — | — | final monitored same-cohort source |
| aloha_right | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 5022.07 | 5053.03 | 5067 | 5071.82 | — | — | final monitored same-cohort source |
| marvin_bare_left | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 373.733 | 377.489 | 378.847 | 379.353 | — | — | final monitored same-cohort source |
| marvin_bare_left | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 5229.63 | 5250.48 | 5264.94 | 5270.65 | — | — | final monitored same-cohort source |
| marvin_bare_right | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 373.564 | 379.987 | 380.751 | 380.93 | — | — | final monitored same-cohort source |
| marvin_bare_right | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 5232.67 | 5255.41 | 5265.41 | 5269.27 | — | — | final monitored same-cohort source |
| marvin_gripper_left | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 470.276 | 477.877 | 479.1 | 479.289 | — | — | final monitored same-cohort source |
| marvin_gripper_left | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 6440.13 | 6451.28 | 6458.9 | 6460.66 | — | — | final monitored same-cohort source |
| marvin_gripper_right | Pytorch_original/cuda | resident original5e-4 config | 1000 | 30 | 434.324 | 439.749 | 442.215 | 443.027 | — | — | final monitored same-cohort source |
| marvin_gripper_right | Pytorch_original/cuda | resident original5e-4 config | 10000 | 30 | 6460.67 | 6473.51 | 6499.53 | 6510.05 | — | — | final monitored same-cohort source |
| aloha_left | warp/cuda | fresh driver+allocator: uniform | 10000 | — | — | — | — | — | 34.25 | — | final monitored source; own fresh PID |
| marvin_bare_left | warp/cuda | fresh driver+allocator: uniform | 10000 | — | — | — | — | — | 80 | — | final monitored source; own fresh PID |
| marvin_bare_left | warp/cuda | fresh driver+allocator: seed_local | 10000 | — | — | — | — | — | 80 | — | final monitored source; own fresh PID |
| aloha_left | warp/cpu | robot_streaming_control | 1 | 10000 | 1.69285 | 1.80209 | 2.02254 | 4.17687 | — | — | final monitored source; native NoRender |
| aloha_right | warp/cpu | robot_streaming_control | 1 | 10000 | 1.61742 | 1.80841 | 2.21262 | 3.20781 | — | — | final monitored source; native NoRender |
| marvin_bare_left | warp/cpu | robot_streaming_control | 1 | 10000 | 1.37349 | 1.48608 | 1.58834 | 3.16776 | — | — | final monitored source; native NoRender |
| marvin_bare_right | warp/cpu | robot_streaming_control | 1 | 10000 | 1.46221 | 1.54571 | 1.67529 | 5.1328 | — | — | final monitored source; native NoRender |
| marvin_gripper_left | warp/cpu | robot_streaming_control | 1 | 10000 | 1.8451 | 2.02357 | 2.38834 | 3.5994 | — | — | final monitored source; native NoRender |
| marvin_gripper_right | warp/cpu | robot_streaming_control | 1 | 10000 | 1.81769 | 1.94766 | 2.18126 | 4.50987 | — | — | final monitored source; native NoRender |
| aloha_pair | warp/cpu | robot_paired_streaming_control | 2 | 10000 | 3.24179 | 3.51027 | 3.86993 | 6.49772 | — | — | final monitored source; native NoRender |
| marvin_bare_pair | warp/cpu | robot_paired_streaming_control | 2 | 10000 | 2.77982 | 2.98977 | 3.19427 | 4.75932 | — | — | final monitored source; native NoRender |
| marvin_gripper_pair | warp/cpu | robot_paired_streaming_control | 2 | 10000 | 3.84242 | 3.99242 | 4.08641 | 6.19878 | — | — | final monitored source; native NoRender |
| aloha_left | warp/cuda | robot_streaming_control | 1 | 10000 | 0.770129 | 0.929891 | 1.16232 | 2.00631 | — | — | final monitored source; native NoRender |
| aloha_right | warp/cuda | robot_streaming_control | 1 | 10000 | 0.79935 | 0.952927 | 1.20152 | 1.6467 | — | — | final monitored source; native NoRender |
| marvin_bare_left | warp/cuda | robot_streaming_control | 1 | 10000 | 2.36737 | 2.78754 | 2.97233 | 3.87057 | — | — | final monitored source; native NoRender |
| marvin_bare_right | warp/cuda | robot_streaming_control | 1 | 10000 | 2.35444 | 2.77498 | 2.93903 | 3.54833 | — | — | final monitored source; native NoRender |
| marvin_gripper_left | warp/cuda | robot_streaming_control | 1 | 10000 | 2.40172 | 2.80435 | 2.93924 | 3.27423 | — | — | final monitored source; native NoRender |
| marvin_gripper_right | warp/cuda | robot_streaming_control | 1 | 10000 | 2.36935 | 2.78826 | 2.9389 | 3.48121 | — | — | final monitored source; native NoRender |
| aloha_pair | warp/cuda | robot_paired_streaming_control | 2 | 10000 | 1.54402 | 1.81683 | 2.14533 | 2.56788 | — | — | final monitored source; native NoRender |
| marvin_bare_pair | warp/cuda | robot_paired_streaming_control | 2 | 10000 | 4.72888 | 5.25622 | 5.40338 | 6.34117 | — | — | final monitored source; native NoRender |
| marvin_gripper_pair | warp/cuda | robot_paired_streaming_control | 2 | 10000 | 4.75514 | 5.28033 | 5.43258 | 6.07891 | — | — | final monitored source; native NoRender |
| arx5 | near_parallel/cpu | streaming_control | 1 | 10000 | 0.22765 | 0.252189 | 0.291553 | 0.720803 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cpu | robot_streaming_control | 1 | 10000 | 0.917244 | 1 | 1.06612 | 2.28374 | — | — | final monitored numerical-hybrid source; native NoRender |
| arx5 | near_parallel/cuda | streaming_control | 1 | 10000 | 0.919538 | 2.00202 | 2.54441 | 3.85748 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | robot_streaming_control | 1 | 10000 | 1.26491 | 2.47085 | 2.87536 | 4.23558 | — | — | final monitored numerical-hybrid source; native NoRender |
| arx5 | near_parallel/cuda | fixed workload resident | 1 | 10000 | 1.03068 | 1.15569 | 1.2887 | 2.11238 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload transfer | 1 | 10000 | 1.06192 | 1.28709 | 1.67132 | 1.80471 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload sustained | 1 | 30 | — | — | — | — | — | 1177.98 | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload resident | 8 | 10000 | 1.95772 | 2.18232 | 2.26039 | 2.63071 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload transfer | 8 | 10000 | 1.99589 | 2.44605 | 2.64248 | 2.90285 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload sustained | 8 | 30 | — | — | — | — | — | 4547.6 | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload resident | 100 | 30 | 3.689 | 4.10468 | 4.12531 | 4.13331 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload transfer | 100 | 30 | 3.75604 | 4.25113 | 4.35589 | 4.384 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload sustained | 100 | 30 | — | — | — | — | — | 28179.5 | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload resident | 1000 | 30 | 8.07036 | 8.84545 | 8.91036 | 8.93604 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload transfer | 1000 | 30 | 8.17259 | 8.92594 | 8.94098 | 8.94492 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 125170 | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload resident | 10000 | 30 | 83.7737 | 86.4196 | 87.186 | 87.3236 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload transfer | 10000 | 30 | 84.0086 | 86.1636 | 86.4716 | 86.5768 | — | — | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 119890 | final monitored numerical-hybrid source |
| arx5 | near_parallel/cuda | fresh driver+allocator; outputs included | 10000 | — | — | — | — | — | 34 | — | final monitored numerical-hybrid source; own fresh PID |
| arx5 | near_parallel/cpu (unmonitored history) | streaming_control | 1 | 10000 | 0.231647 | 0.250464 | 0.263859 | 0.792075 | — | — | historical performance |
| arx5 | near_parallel/cpu (unmonitored history) | resident | 1000 | 30 | 31.0031 | 31.2307 | 31.3238 | 31.3589 | — | — | historical performance |
| arx5 | near_parallel/cpu (unmonitored history) | resident | 10000 | 30 | 313.404 | 316.53 | 317.381 | 317.655 | — | — | historical performance |
| arx5 | near_parallel/cuda (unmonitored history) | streaming_control | 1 | 10000 | 0.91727 | 2.04117 | 2.67467 | 5.52735 | — | — | historical performance |
| arx5 | near_parallel/cuda (unmonitored history) | resident | 1000 | 30 | 8.42072 | 8.61565 | 8.8681 | 8.96775 | — | — | historical performance |
| arx5 | near_parallel/cuda (unmonitored history) | resident | 10000 | 30 | 85.3772 | 86.1101 | 86.304 | 86.3374 | — | — | historical performance |
| rizon4s | cuda_rr/seed_local conditioned | streaming_control | 1 | 10000 | 2.60823 | 2.983 | 3.15858 | 3.75298 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload resident | 1 | 10000 | 3.01722 | 3.43527 | 3.53882 | 4.15463 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload resident | 8 | 10000 | 3.69167 | 4.12221 | 4.23078 | 4.74425 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload resident | 100 | 30 | 15.4639 | 16.0967 | 16.2115 | 16.2446 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload resident | 1000 | 30 | 128.898 | 130.624 | 131.013 | 131.127 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload resident | 10000 | 30 | 1257.2 | 1283.07 | 1288.51 | 1288.94 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload transfer | 1 | 10000 | 3.06787 | 3.50275 | 3.65553 | 4.18797 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload transfer | 8 | 10000 | 3.74105 | 4.21067 | 4.37254 | 4.92867 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload transfer | 100 | 30 | 15.4511 | 16.2117 | 16.5873 | 16.7324 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload transfer | 1000 | 30 | 128.794 | 131.039 | 131.861 | 132.014 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload transfer | 10000 | 30 | 1256.55 | 1267.1 | 1267.18 | 1267.21 | — | — | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload sustained | 1 | 30 | — | — | — | — | — | 354.45 | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload sustained | 8 | 30 | — | — | — | — | — | 2265.99 | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload sustained | 100 | 30 | — | — | — | — | — | 6539.17 | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload sustained | 1000 | 30 | — | — | — | — | — | 7792.24 | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fixed workload sustained | 10000 | 30 | — | — | — | — | — | 7949.65 | final monitored conditioned source |
| rizon4s | cuda_rr/seed_local conditioned | fresh driver+allocator minus output | 10000 | — | — | — | — | — | 73.7234 | — | final monitored conditioned source; own fresh PID |
| rizon4s | cuda_rr/seed_local conditioned | robot_streaming_control | 1 | 10000 | 2.92712 | 3.36294 | 3.53198 | 4.22495 | — | — | final monitored conditioned source; native NoRender |
| rizon4s | ssik_pytorch_original_cuda | resident same13-array fixture | 1000 | 30 | 430.093 | 434.737 | 437.404 | 438.432 | — | — | final monitored baseline; matched physical model and cohort |
| rizon4s | ssik_pytorch_original_cuda | resident same13-array fixture | 10000 | 30 | 4996.13 | 5019.87 | 5021.73 | 5022.12 | — | — | final monitored baseline; matched physical model and cohort |
| rizon4s | ssik_pytorch_matched_cuda | resident same13-array fixture | 1000 | 30 | 431.761 | 441.43 | 446.743 | 448.053 | — | — | final monitored baseline; matched physical model and cohort |
| rizon4s | ssik_pytorch_matched_cuda | resident same13-array fixture | 10000 | 30 | 4992.25 | 5015.34 | 5017.28 | 5017.43 | — | — | final monitored baseline; matched physical model and cohort |
| rizon4s | cuda_rr/seed_local unconditioned history | streaming_control | 1 | 10000 | 2.58528 | 2.95723 | 3.09158 | 3.47497 | — | — | historical performance |
| rizon4s | cuda_rr/seed_local unconditioned history | resident | 1000 | 30 | 129.707 | 130.885 | 131.825 | 132.181 | — | — | historical performance |
| rizon4s | cuda_rr/seed_local unconditioned history | resident | 10000 | 30 | 1271.58 | 1500.57 | 1540.51 | 1541.89 | — | — | historical performance |
| rizon4s | cuda_rr/seed_local unconditioned history | unbounded fixed-pose diagnostic | 1 | 10000 | 3.03397 | 5.65198 | 7.71473 | 8.16781 | — | — | historical performance |
| rizon4s | cuda_rr/seed_local unconditioned history | fresh driver+allocator minus output | 10000 | — | — | — | — | — | 73.7234 | — | historical performance |
| arx5 | warp/cpu unmonitored history | robot_streaming_control | 1 | 10000 | 0.929207 | 1.06289 | 1.24142 | 2.394 | — | — | historical performance |
| arx5 | warp/cuda unmonitored history | robot_streaming_control | 1 | 10000 | 1.95876 | 10.0683 | 10.8549 | 15.1313 | — | — | historical performance |
| flexiv_rizon4s | cuda_rr/cuda unmonitored history | robot_streaming_control | 1 | 10000 | 3.63824 | 13.7908 | 14.4285 | 15.0386 | — | — | historical performance |

Streaming measurements use10,000 changing-target calls, previous-result seeds and0.05rad hard steps. Fixed-pose diagnostics, resident bulk, transfer, initialization/JIT and Robot-interface timings are distinct. Bulk rows have30 completed calls. Fresh driver observations include cold module/stack and allocator effects, rather than treating Torch-only counters as all CUDA memory. Full per-size/scoped data and measurement repetition counts are inJSON.

Actual Robot measurements create an engine-backed SimulationManager and Robot, query its live root state, convert arena-frame goals, and call the configured solver. Paired rows execute both calls through one Robot. Setup, physical stepping and external-device transport are excluded. Each arm is independently checked for world FK, effective limits, hard steps and failure hold; coverage must not regress from matching standalone streaming calls.

All18 final P1 actual Robot groups (six singles and three pairs on each device) explicitly use native no-render with real native state retrieval,10,000 changing-target calls and independent FK/limit/step/failure-hold checks. Their source-qualified latency and coverage rows above supersede the archived unmonitored long windows. Headless alone selected Hybrid in the initial experiment: Aloha/Marvin/Rizon single p99 reached12.16/13.90/16.52ms and Marvin paired25.18ms; these failures are preserved inJSON. A matched3,000-call component diagnostic reduced Marvin p99 from13.904 to3.029ms when selecting no-render. This is diagnostic evidence, not a10,000-call certificate or a claim about simultaneous rendering. Final P1 windows used a live one-second system NVML monitor with three empty compute samples before resumption;18 qualified phase records contain no foreign compute observations. External training and evaluation ended naturally; no unrelated process was paused. The overlapping gripper/matrix attempt and initialization-only harness error remain separate history, never mixed into final percentiles. Robot single/batch frame conversion also uses the existing gradient-preserving rigid inverse, with a red/green CUDA no-host-synchronization regression.

## Success & Other Metrics

| Model | Device | Scope | Wide | Boundary | Singular | Unreachable_false | Path_20000 | Control_10000 | Reported_coverage | False_successes | Step_violations | Failed_hold_violations | Max_position_m | Max_rotation_rad | Max_step_rad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aloha_left | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 1 | — | — | 0 | 0 | — | 1.8595e-07 | 1.16345e-06 | 0.00133796 |
| aloha_right | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 1 | — | — | 0 | 0 | — | 2.07156e-07 | 7.30715e-07 | 0.00131807 |
| marvin_bare_left | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.98345 | — | — | 0 | 0 | — | 9.02737e-06 | 1.45419e-06 | 0.0128771 |
| marvin_bare_right | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.97185 | — | — | 0 | 0 | — | 8.04953e-06 | 1.35085e-06 | 0.0350286 |
| marvin_gripper_left | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.98345 | — | — | 0 | 0 | — | 9.02542e-06 | 1.29425e-06 | 0.012731 |
| marvin_gripper_right | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.97185 | — | — | 0 | 0 | — | 8.04772e-06 | 1.37078e-06 | 0.0350729 |
| arx5 | cpu | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.95225 | — | — | 0 | — | — | 9.51341e-06 | 9.3719e-06 | 0.0116053 |
| aloha_left | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 1 | — | — | 0 | 0 | — | 1.8595e-07 | 1.16345e-06 | 0.00133796 |
| aloha_right | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 1 | — | — | 0 | 0 | — | 2.07156e-07 | 7.30715e-07 | 0.00131807 |
| marvin_bare_left | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.98345 | — | — | 0 | 0 | — | 9.02527e-06 | 1.45419e-06 | 0.0128772 |
| marvin_bare_right | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.97185 | — | — | 0 | 0 | — | 8.04965e-06 | 1.35085e-06 | 0.0350287 |
| marvin_gripper_left | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.98345 | — | — | 0 | 0 | — | 9.02785e-06 | 1.29425e-06 | 0.0127311 |
| marvin_gripper_right | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.97185 | — | — | 0 | 0 | — | 8.04768e-06 | 1.37078e-06 | 0.0350729 |
| arx5 | cuda | reachable static + 20k paths | 1 | 1 | 1 | 0 | 0.95225 | — | — | 0 | — | — | 9.51341e-06 | 9.3719e-06 | 0.0116053 |
| rizon4s | cuda conditioned | reachable static + 20k paths | 0.9999 | 1 | 1 | 0 | 1 | — | — | 0 | 0 | — | 9.78324e-07 | 3.71931e-06 | 0.02 |
| aloha_left | cpu | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.28969e-07 | 2.33834e-07 | 0.00121601 |
| aloha_right | cpu | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.47914e-07 | 2.16236e-07 | 0.00121561 |
| marvin_bare_left | cpu | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.0265e-06 | 1.43271e-06 | 0.012881 |
| marvin_bare_right | cpu | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 7.82369e-07 | 1.2686e-06 | 0.00151515 |
| marvin_gripper_left | cpu | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.03071e-06 | 1.36846e-06 | 0.0127336 |
| marvin_gripper_right | cpu | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.04172e-06 | 1.37078e-06 | 0.00151515 |
| aloha_pair | cpu | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.47914e-07 | 2.33834e-07 | 0.00121601 |
| marvin_bare_pair | cpu | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.02684e-06 | 1.47544e-06 | 0.0132623 |
| marvin_gripper_pair | cpu | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.03277e-06 | 1.2319e-06 | 0.0134375 |
| aloha_left | cuda | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.38728e-07 | 2.39507e-07 | 0.00121601 |
| aloha_right | cuda | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.73855e-07 | 2.23456e-07 | 0.00121558 |
| marvin_bare_left | cuda | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.02735e-06 | 1.42107e-06 | 0.0128573 |
| marvin_bare_right | cuda | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 8.58745e-07 | 1.4301e-06 | 0.00151503 |
| marvin_gripper_left | cuda | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.02878e-06 | 1.34174e-06 | 0.0127211 |
| marvin_gripper_right | cuda | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.24165e-06 | 1.62561e-06 | 0.00151503 |
| aloha_pair | cuda | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 1.73855e-07 | 2.39507e-07 | 0.00121601 |
| marvin_bare_pair | cuda | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.01592e-06 | 1.19e-06 | 0.0136603 |
| marvin_gripper_pair | cuda | Robot10k final monitored | — | — | — | — | — | 0.9669 | 0.9669 | 0 | 0 | 0 | 9.02596e-06 | 1.51295e-06 | 0.0127087 |
| arx5 | cpu | Standalone10k final monitored hybrid | — | — | — | — | — | 0.9045 | 0.9045 | 0 | 0 | 0 | 9.51622e-06 | 9.36804e-06 | 0.00695911 |
| arx5 | cpu | Robot10k final monitored hybrid | — | — | — | — | — | 0.9045 | 0.9045 | 0 | 0 | 0 | 9.86416e-06 | 8.3391e-06 | 0.00695911 |
| arx5 | cuda | Standalone10k final monitored hybrid | — | — | — | — | — | 0.9045 | 0.9045 | 0 | 0 | 0 | 9.51622e-06 | 9.36804e-06 | 0.00695911 |
| arx5 | cuda | Robot10k final monitored hybrid | — | — | — | — | — | 0.9045 | 0.9045 | 0 | 0 | 0 | 9.86416e-06 | 8.3391e-06 | 0.00695911 |
| rizon4s | cuda conditioned | Robot10k final monitored | — | — | — | — | — | 1 | 1 | 0 | 0 | 0 | 4.19781e-06 | 9.2017e-06 | 0.0171718 |
| aloha_left | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 1 | 1000 | — | — | — | — | — |
| aloha_left | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9992 | 9992 | — | — | — | — | — |
| aloha_right | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 0.998 | 998 | — | — | — | — | — |
| aloha_right | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9996 | 9996 | — | — | — | — | — |
| marvin_bare_left | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 0.997 | 997 | — | — | — | — | — |
| marvin_bare_left | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9949 | 9949 | — | — | — | — | — |
| marvin_bare_right | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 0.997 | 997 | — | — | — | — | — |
| marvin_bare_right | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9949 | 9949 | — | — | — | — | — |
| marvin_gripper_left | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 0.996 | 996 | — | — | — | — | — |
| marvin_gripper_left | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9949 | 9949 | — | — | — | — | — |
| marvin_gripper_right | Pytorch_original/cuda | wide prefixN1000; original5e-4 config | 0 | — | — | — | — | — | 0.997 | 997 | — | — | — | — | — |
| marvin_gripper_right | Pytorch_original/cuda | wide prefixN10000; original5e-4 config | 0 | — | — | — | — | — | 0.9949 | 9949 | — | — | — | — | — |

Each model/device uses10,000 wide,1,000 boundary,200 constructed singular/near-singular,500 clearly unreachable targets and20 trajectories×1,000frames. Rates are fractions. Six strict-model CPU/CUDA masks agree at all190,200 targets/frames. Joint values need not agree at singularities when both represent valid poses. Non-singular fixed-swivel branches and complete current candidate slots have separate comparisons;67 CPU and35 CUDA focused cases passed, including the narrow nonorthogonal shoulder with an invalid first elbow. Marvin path losses are confined to the certified outside-shell line/circle intervals (331 left and563 right over20,000frames), with no unexplained reachable loss.

ARX5 retains all9 candidates and at most12 bounded correction iterations. Actual-TCP normalized block balancing recovers the old false-negative frames43/44. Its one incomplete line has a conservative original-model branch-component certificate ruling out a complete0.05rad-step path from that seed; this does not declare each failed frame unreachable. The near-singular general6R branch limitation is separately recorded. This mode is explicitly a numerical hybrid. Its new15-case CUDA matrix and six source/isolation phase records retain their raw vector and input hashes; no new quiet original-Pytorch baseline was measured, and the default remains Pytorch.

Rizon4S uses16 global and9 local locks, all600 algebraic root slots. Tangent calculations select only lock angles; they never return a numerical Cartesian IK solution. The one static miss is a finite-lock-sampling miss verified by independent CPUQZ at its known reachable source lock. Internal coefficient preparation now conditions small valid-pose SO(3) roundoff; final acceptance still checks the original target. The previous two direct continuous failures and one extra world-frame failure are recovered. No solver fallback is enabled. On the byte-identical standard fixture, matched Pytorch CPU is99.98% wide/98.6% trajectory; conditioned CUDA RR is99.99%/100%. Original Pytorch uses5e-4 internal stopping tolerances; its strict external1e-5 results are not described as a new bug.

## Leaderboard

| Model | Mode | Precision | Continuity | Solver_control | Scratch | Bulk_original | Default |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Aloha Mini | strict6R | 100% reachable static; 0 false | 100% | CPU/CUDA single+pair and actual Robot PASS | 34.25 MiB final fresh PID | min310.492x same-cohort | SSIK Warp; P5 validated |
| Marvin bare/gripper | strict SRS seed_local | 100% reachable static; 0 false | L98.345% / R97.185%; CPU/CUDA masks agree | CPU/CUDA single+pair and actual Robot PASS | 80 MiB final fresh PID | min33.2374x same-cohort | SSIK Warp; P5 validated |
| ARX5 | near_parallel numerical hybrid | 100% reachable static; 0 false | 95.225%; remaining line classified | CPU/CUDA control+Robot PASS; each9045/10000 | 34 MiB final fresh PID; outputs included | observed56.726x/66.582x; historical baseline | Pytorch; explicit opt-in |
| Rizon4S | parameterized CUDA RR seed_local | 99.99% wide conditioned | 100%; actual Robot10k100% | CUDA control3.159ms / Robot3.532ms PASS | 73.7234 MiB final fresh PID | 3.3367x/3.9740x same13-array cohort | Pytorch; CUDA-only opt-in |

These are model-specific gates, not cross-robot algorithm rankings. P1 has12 fresh original-default Pytorch CUDA resident bulk comparisons (all six arms at1,000/10,000,30 completed calls each). All asset/model/root/end/joint/TCP/limit, full-cohort and actual float32 target/seed tensor fingerprints match; the minimum median completed-call ratio is33.2374×. The original baseline retains5e-4 internal stopping tolerances and is independently judged at1e-5; its large strict-validation rejection counts are a tolerance-policy difference, not a new defect. P1 quality-matched Pytorch bulk remains UNMEASURED and is not substituted into the original-default gate. ARX5 current monitored resident medians at1,000/10,000 are8.07036/83.77374ms. Comparing to the byte-identical historical original-Pytorch cohort gives observed ratios56.7264×/66.5816×; that baseline was unmonitored and is not relabelled as a current fully monitored comparison. Rizon4S current conditioned medians at1,000/10,000 are128.898/1257.203ms versus430.093/4996.131ms original Pytorch (3.33670×/3.97400×) and431.761/4992.253ms matched Pytorch (3.34965×/3.97092×). All13 saved fixture arrays are bitwise identical across these runs. The older20261025/unconditioned ratios remain history and are not substituted into the current standard2531 comparison.

Rizon unconditioned-source timing (bounded3.092ms, unbounded7.715ms) is historical. Current conditioned10,000-call control p99 is3.158577ms; unbounded repeated-pose N=1 p99 is3.538823ms, a separate scope; actual Robot p99 is3.531977ms with100% validated coverage and zero FK/step/failure-hold violations. Fresh driver-inclusive scratch is73.723434MiB. Its five-size resident/transfer/sustained matrix retains15 rows with SHA-bound raw vectors, plus800 one-second root GPU/device observations with no foreign compute. Root coordinated CPU quiet and recorded thread settings/affinity; no continuous CPU profiler was used or claimed. All six current ARX stages are qualified with no foreign observations and immutable sources. Standalone CPU/CUDA p99 is0.291553/2.544410ms and actual Robot p99 is1.066118/2.875361ms; all four10,000-call streams validate9045/10000 with zero false/step/failure-hold violations. The first10 whole paths are a different scope from its full20,000-frame95.225% quality set. Fresh NVML and memGetInfo growth is34MiB, conservatively including final outputs; old2MiB driver and8.49MiB attributed counters are history only. P1 now retains the complete90-case CUDA resident/transfer/sustained matrix, final CPU/CUDA10,000-call singles/pairs, all18 actual Robot groups and fresh driver-inclusive memory probes. Fresh P1 scratch is34.25MiB for6R and80MiB for SRS, including driver/allocator growth above model/input initialization. P1 CPU-reference full timing, P1 quality-matched bulk and the full large CPU fixed-size matrix are optional context still marked UNMEASURED; they are not invented blockers or fake passes for the original-default throughput requirement. CI and P5 validation are recorded separately. CUDA-only RR cannot replace a CPU-compatible default. P5 validated the Aloha arm and Marvin bare/gripper default switches; torso, TCP, joint ordering and explicit overrides are preserved. The validation receipt is a pre-commit snapshot; root subsequently committed and pushed4748f3de969b940e875bfcd6a609b128d463fd39 and passed a fresh strict Sphinx build. This does not claim CI success for that head. Explicit Pytorch restoration stays available. These measurements are not a hard real-time guarantee.

## Reproduction

Use the maintained unified entry point with full workloads and matching assets:

```bash
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark --solvers ssik --ssik-srs-search seed_local --ssik-robot-e2e
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark --solvers ssik --ssik-models arx5 --ssik-geometry-mode near_parallel --ssik-robot-e2e
python -m scripts.benchmark.robotics.kinematic_solver.run_benchmark --solvers ssik --ssik-models flexiv_rizon4s --ssik-devices cuda --ssik-cuda-rr --ssik-rr-search seed_local --ssik-robot-e2e
```

Install optional pinned CPU reference only through requirements/ssik-reference.txt. Rebuild parameterized RR offline with prepare_ssik_rr_rizon4.py --parameterized --urdf PATH --ssik-source PINNED_SOURCE. The manifest retains14 exact symbolic identities,2,048 independent PK certificates and source/data hashes. Large raw archives are kept under ignored outputs/benchmarks/ssik_followup_acceptance_20261026; JSON embeds measurement recipes as validation evidence, not alternate production entry points. The raw-archive receipt binds an immutable index checkpoint; historical missing entries remain explicit, without a mutable report/index hash cycle.
