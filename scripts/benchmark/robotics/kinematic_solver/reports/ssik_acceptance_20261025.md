# ssik_acceptance Benchmark Report

Generated at: 2026-10-25T19:48:01

Aloha Mini arms pass the measured first-phase numeric, control and memory budgets and use SSIK by default. Tianji Marvin retains Pytorch because full-path completion trails the matched-quality baseline.

The primary report includes four arm chains on CPU/CUDA, both Marvin tool variants for quality, 10,000-frame single/pair control certificates, 30-repeat bulk profiles, and source/asset provenance. The CPU reference backend is available, with full-size reference acceptance still unmeasured.

Hardware and execution environment: AMD Ryzen 9 9950X, NVIDIA GeForce RTX 5090; Python 3.11.14, Torch 2.7.1+cu128, CUDA runtime 12.8, Warp 1.17.0, pytorch-kinematics 0.10.0, NumPy 2.2.6, SciPy 1.17.1. Each of the 18 recorded run metadata files reports one Torch intra-op thread, 32 inter-op threads, and unset OMP_NUM_THREADS/MKL_NUM_THREADS/OPENBLAS_NUM_THREADS; this is a per-run provenance check, not an assumed machine-wide setting. CPU baseline quality jobs could run independently; final latency certificates ran with other benchmark jobs idle.

Tracked compact evidence and complete source/model fingerprints are in [ssik_acceptance_20261025.json](ssik_acceptance_20261025.json). Large target/joint arrays, raw timing samples and full execution logs remain in the ignored [raw evidence directory](../../../../../outputs/benchmarks/ssik_acceptance_20261025/).

## Time & Memory

| model | impl | scope | cost_time_ms | sample_size | p95_ms | p99_ms | max_ms | repeats | cpu_rss_delta_mb | torch_peak_increment_mb | warp_pool_delta_mb | warp_pool_peak_mb | solver_buffer_mb | attributed_scratch_mb | scratch_gate | queries_per_second | sustained_queries_per_second | latency_gate | failure_count | reported_success_rate | false_successes | validated_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aloha_left | ssik_warp_cpu | initialization | 34.912841 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | first_call_including_jit | 7.903662 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | resident | 0.171765 | 1 | 0.218105 | 0.246130 | 0.802829 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.109657 | PASS | 5821.908294 | 5981.531475 |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | resident | 0.196421 | 8 | 0.208595 | 0.268419 | 0.792499 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.110832 | PASS | 40728.855734 | 42655.640521 |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | resident | 0.488700 | 100 | 0.512460 | 0.517306 | 0.518070 | 30 | 0.003906 | 0.000000 | unavailable | unavailable | 0.109375 | 0.126274 | PASS | 204624.710706 | 212008.291899 |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | resident | 3.833567 | 1000 | 3.914336 | 3.924257 | 3.926137 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.277336 | PASS | 260853.631089 | 265060.157316 |  |  |  |  |  |
| aloha_left | ssik_warp_cpu | resident | 38.050296 | 10000 | 41.120505 | 43.008020 | 43.626315 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 1.787956 | PASS | 262810.044771 | 259561.104136 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | initialization | 99.798284 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | first_call_including_jit | 30161.434820 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | resident | 0.486561 | 1 | 0.557400 | 0.661312 | 0.836193 | 10000 | 0.074219 | 0.002930 | 0.000000 | 0.000001 | 0.109375 | 0.112281 | PASS | 2055.240842 | 2869.085917 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | host_transfer_inclusive | 0.515025 | 1 | 0.694808 | 0.894966 | 4.165120 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | resident | 0.601578 | 8 | 0.770388 | 1.061441 | 2.747658 | 10000 | 0.015625 | 0.003418 | 0.000000 | 0.000001 | 0.109375 | 0.112602 | PASS | 13298.360379 | 17073.107345 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | host_transfer_inclusive | 0.631755 | 8 | 0.853306 | 1.074816 | 4.534609 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | resident | 0.872867 | 100 | 0.927650 | 0.979904 | 0.999130 | 30 | 0.003906 | 0.020996 | 0.000000 | 0.000001 | 0.109375 | 0.127987 | PASS | 114565.043525 | 137307.839498 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | host_transfer_inclusive | 0.877791 | 100 | 0.970354 | 1.032500 | 1.049936 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | resident | 4.001951 | 1000 | 4.294883 | 4.312447 | 4.318801 | 30 | 0.000000 | 0.192871 | 0.000000 | 0.000001 | 0.109375 | 0.278404 | PASS | 249878.117079 | 253515.565841 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | host_transfer_inclusive | 4.071624 | 1000 | 4.513630 | 5.022956 | 5.212371 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | resident | 40.885756 | 10000 | 42.951692 | 43.108733 | 43.125780 | 30 | 0.000000 | 1.918457 | 0.000000 | 0.000001 | 0.109375 | 1.789413 | PASS | 244583.957158 | 257627.735943 |  |  |  |  |  |
| aloha_left | ssik_warp_cuda | host_transfer_inclusive | 41.202030 | 10000 | 43.382635 | 52.591935 | 56.299283 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | initialization | 14.259277 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | first_call_including_jit | 0.320085 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | resident | 0.149302 | 1 | 0.160276 | 0.219129 | 1.333472 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.109657 | PASS | 6697.836085 | 6957.057157 |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | resident | 0.172496 | 8 | 0.205208 | 0.283648 | 1.348842 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.110832 | PASS | 46377.866910 | 48760.744891 |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | resident | 0.434813 | 100 | 0.465344 | 0.496357 | 0.508662 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.126274 | PASS | 229983.938271 | 234419.100552 |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | resident | 3.514820 | 1000 | 3.776630 | 4.127107 | 4.212970 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 0.277336 | PASS | 284509.627594 | 280477.613935 |  |  |  |  |  |
| aloha_right | ssik_warp_cpu | resident | 34.816058 | 10000 | 36.602577 | 36.859664 | 36.953242 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 0.109375 | 1.787956 | PASS | 287223.782350 | 278756.334847 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | initialization | 18.349616 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | first_call_including_jit | 0.685717 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | resident | 0.492652 | 1 | 0.604855 | 0.718854 | 1.672354 | 10000 | 0.011719 | 0.002930 | 0.000000 | 0.000001 | 0.109375 | 0.112281 | PASS | 2029.830127 | 2808.797576 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | host_transfer_inclusive | 0.527934 | 1 | 0.809412 | 0.999001 | 4.407288 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | resident | 0.597761 | 8 | 0.762297 | 1.609937 | 1.879897 | 10000 | 0.011719 | 0.003418 | 0.000000 | 0.000001 | 0.109375 | 0.112602 | PASS | 13383.274973 | 16428.557568 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | host_transfer_inclusive | 0.634200 | 8 | 1.020856 | 1.424678 | 5.030246 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | resident | 0.874761 | 100 | 1.073591 | 1.089271 | 1.089341 | 30 | 0.000000 | 0.020996 | 0.000000 | 0.000001 | 0.109375 | 0.127987 | PASS | 114316.996552 | 124594.895618 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | host_transfer_inclusive | 1.027003 | 100 | 1.376587 | 1.396540 | 1.398826 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | resident | 4.398577 | 1000 | 4.767791 | 4.799363 | 4.801133 | 30 | 0.000000 | 0.192871 | 0.000000 | 0.000001 | 0.109375 | 0.278404 | PASS | 227346.251480 | 230069.030002 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | host_transfer_inclusive | 4.845763 | 1000 | 5.126501 | 5.220617 | 5.252828 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | resident | 43.196208 | 10000 | 43.832229 | 43.883145 | 43.900325 | 30 | 0.000000 | 1.918457 | 0.000000 | 0.000001 | 0.109375 | 1.789413 | PASS | 231501.801785 | 232160.316448 |  |  |  |  |  |
| aloha_right | ssik_warp_cuda | host_transfer_inclusive | 43.303572 | 10000 | 43.790239 | 43.799829 | 43.803112 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | initialization | 11.009229 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | first_call_including_jit | 0.787310 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | resident | 0.600817 | 1 | 0.629487 | 0.836699 | 1.691350 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.125309 | PASS | 1664.400222 | 1680.904287 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | resident | 2.775161 | 8 | 3.009872 | 3.498926 | 5.059592 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.126537 | PASS | 2882.715691 | 2923.414969 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | resident | 34.996997 | 100 | 35.684069 | 36.010727 | 36.092174 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.142681 | PASS | 2857.388013 | 2858.161936 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | resident | 362.531471 | 1000 | 366.729542 | 367.634235 | 367.905464 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.300610 | PASS | 2758.381216 | 2741.322935 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cpu | resident | 3661.255692 | 10000 | 3710.167780 | 4030.345345 | 4159.253516 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 17.879894 | PASS | 2731.303367 | 2731.985482 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | initialization | 16.770912 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | first_call_including_jit | 1.883742 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | resident | 1.728776 | 1 | 1.879120 | 2.082010 | 4.239378 | 10000 | 0.015625 | 0.002930 | 0.000000 | 0.000001 | 16.125000 | 16.127902 | PASS | 578.443930 | 618.211408 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | host_transfer_inclusive | 1.756973 | 1 | 2.027030 | 2.245443 | 5.232837 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | resident | 2.684296 | 8 | 3.009079 | 4.900494 | 5.453865 | 10000 | 0.019531 | 0.003418 | 0.000000 | 0.000001 | 16.125000 | 16.128197 | PASS | 2980.296746 | 3088.363292 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | host_transfer_inclusive | 2.736416 | 8 | 3.328440 | 7.326532 | 10.450023 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | resident | 4.183193 | 100 | 4.523239 | 4.555853 | 4.560232 | 30 | 0.000000 | 0.021973 | 0.000000 | 0.000001 | 16.125000 | 16.144207 | PASS | 23905.187220 | 25384.106393 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | host_transfer_inclusive | 4.261248 | 100 | 8.427199 | 8.541509 | 8.542436 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | resident | 52.314448 | 1000 | 63.232731 | 65.604132 | 66.226591 | 30 | 0.000000 | 0.204590 | 0.000000 | 0.000001 | 16.125000 | 16.301933 | PASS | 19115.178258 | 19378.924850 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | host_transfer_inclusive | 52.755306 | 1000 | 53.780396 | 53.856252 | 53.883413 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | resident | 506.793678 | 10000 | 529.374494 | 531.837038 | 532.229459 | 30 | 0.000000 | 2.032715 | 0.000000 | 0.000001 | 16.125000 | 17.881149 | PASS | 19731.895725 | 20610.124210 |  |  |  |  |  |
| marvin_bare_left | ssik_warp_cuda | host_transfer_inclusive | 478.379542 | 10000 | 517.421465 | 522.378004 | 524.110253 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | initialization | 10.826033 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | first_call_including_jit | 0.800396 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | resident | 0.599833 | 1 | 0.617702 | 0.759025 | 0.878125 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.125309 | PASS | 1667.132055 | 1697.085992 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | resident | 2.752233 | 8 | 2.883080 | 3.455774 | 5.132682 | 10000 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.126537 | PASS | 2906.730208 | 2876.246579 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | resident | 35.329832 | 100 | 42.233961 | 44.712510 | 45.697309 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.142681 | PASS | 2830.469200 | 2839.516408 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | resident | 364.240750 | 1000 | 370.617238 | 386.633761 | 392.607801 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 16.300610 | PASS | 2745.436913 | 2736.070090 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cpu | resident | 3652.540627 | 10000 | 3912.472858 | 4165.953974 | 4235.671325 | 30 | 0.000000 | 0.000000 | unavailable | unavailable | 16.125000 | 17.879894 | PASS | 2737.820334 | 2756.959069 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | initialization | 13.990776 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | first_call_including_jit | 1.889880 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | resident | 1.718961 | 1 | 1.865669 | 1.919630 | 4.938836 | 10000 | 0.015625 | 0.002930 | 0.000000 | 0.000001 | 16.125000 | 16.127902 | PASS | 581.746629 | 612.456344 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | host_transfer_inclusive | 1.837783 | 1 | 2.227500 | 2.431574 | 3.440145 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | resident | 3.017118 | 8 | 3.206784 | 3.310541 | 3.577804 | 10000 | 0.015625 | 0.003418 | 0.000000 | 0.000001 | 16.125000 | 16.128197 | PASS | 2651.536616 | 2805.287876 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | host_transfer_inclusive | 3.219703 | 8 | 3.484678 | 3.618741 | 4.183991 | 10000 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | resident | 4.380986 | 100 | 4.619078 | 4.771177 | 4.822562 | 30 | 0.003906 | 0.021973 | 0.000000 | 0.000001 | 16.125000 | 16.144207 | PASS | 22825.909481 | 22532.697231 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | host_transfer_inclusive | 4.719549 | 100 | 5.011196 | 5.025183 | 5.027493 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | resident | 54.600081 | 1000 | 55.002423 | 55.072906 | 55.099387 | 30 | 0.000000 | 0.204590 | 0.000000 | 0.000001 | 16.125000 | 16.301933 | PASS | 18314.991125 | 18383.839870 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | host_transfer_inclusive | 54.851456 | 1000 | 55.505322 | 55.888329 | 55.970404 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | resident | 478.282283 | 10000 | 540.800320 | 541.319282 | 541.484574 | 30 | 0.000000 | 2.671387 | 0.000000 | 0.000001 | 16.125000 | 18.519821 | PASS | 20908.154774 | 19409.466444 |  |  |  |  |  |
| marvin_bare_right | ssik_warp_cuda | host_transfer_inclusive | 477.733391 | 10000 | 506.551753 | 507.940730 | 508.420372 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_left | ssik_warp_cpu | initialization | 36.232229 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_left | ssik_warp_cpu | first_call_including_jit | 8.653623 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_left | ssik_warp_cuda | initialization | 96.578502 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_left | ssik_warp_cuda | first_call_including_jit | 30.850177 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_right | ssik_warp_cpu | initialization | 16.794158 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_right | ssik_warp_cpu | first_call_including_jit | 0.810545 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_right | ssik_warp_cuda | initialization | 23.024171 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_gripper_right | ssik_warp_cuda | first_call_including_jit | 1.907406 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cpu | initialization | 5234.985442 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cpu | first_call_including_jit | 341.552174 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_matched_cpu | initialization | 20.596049 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_matched_cpu | first_call_including_jit | 34.215334 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cpu | initialization | 3954.539915 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cpu | first_call_including_jit | 344.679214 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_matched_cpu | initialization | 19.674812 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_matched_cpu | first_call_including_jit | 33.822168 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cpu | initialization | 3119.623146 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cpu | first_call_including_jit | 327.464422 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | initialization | 13.154993 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | first_call_including_jit | 43.262275 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cpu | initialization | 2413.255288 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cpu | first_call_including_jit | 330.862667 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | initialization | 14.006137 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | first_call_including_jit | 44.619245 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | initialization | 1476.336452 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | first_call_including_jit | 1066.550570 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_matched_cuda | initialization | 19.396275 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_matched_cuda | first_call_including_jit | 95.755840 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cuda | initialization | 1492.109304 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cuda | first_call_including_jit | 1023.367327 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_matched_cuda | initialization | 24.665842 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_matched_cuda | first_call_including_jit | 103.208112 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cuda | initialization | 188.860766 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cuda | first_call_including_jit | 1503.390692 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | initialization | 13.401357 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | first_call_including_jit | 105.161130 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cuda | initialization | 13.819006 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cuda | first_call_including_jit | 89.178458 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | initialization | 16.079805 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | first_call_including_jit | 113.455947 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | resident | 462.278427 | 1000 | 470.227394 | 471.735566 | 472.258430 | 30 | 0.070312 | 131.677246 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 2163.198503 | 2183.223234 |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | host_transfer_inclusive | 457.419935 | 1000 | 682.839196 | 708.946298 | 715.628753 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | resident | 5056.226460 | 10000 | 6649.961122 | 6689.891223 | 6696.247345 | 30 | 0.011719 | 1312.278320 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1977.759517 | 1975.535086 |  |  |  |  |  |
| aloha_left | ssik_pytorch_original_cuda | host_transfer_inclusive | 5109.819414 | 10000 | 5769.010898 | 5992.718428 | 6070.968709 | 30 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cuda | resident | 463.124201 | 1000 | 598.833156 | 646.875311 | 662.764679 | 30 | 0.062500 | 131.677246 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 2159.247992 | UNMEASURED |  |  |  |  |  |
| aloha_right | ssik_pytorch_original_cuda | resident | 5112.925055 | 10000 | 6834.575606 | 6883.525704 | 6887.526686 | 30 | 0.066406 | 1312.278320 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1955.827612 | UNMEASURED |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cuda | resident | 374.440952 | 1000 | 382.080429 | 382.522402 | 382.684203 | 30 | 0.000000 | 140.628906 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 2670.648057 | UNMEASURED |  |  |  |  |  |
| marvin_bare_left | ssik_pytorch_original_cuda | resident | 5233.520239 | 10000 | 6172.087881 | 6352.149386 | 6363.088776 | 30 | 0.003906 | 1407.150879 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1910.759784 | UNMEASURED |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cuda | resident | 373.191603 | 1000 | 402.815626 | 404.479222 | 404.846851 | 30 | 0.000000 | 140.628906 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 2679.588694 | UNMEASURED |  |  |  |  |  |
| marvin_bare_right | ssik_pytorch_original_cuda | resident | 5276.089277 | 10000 | 5309.413987 | 5310.424493 | 5310.659353 | 30 | 0.003906 | 1407.150879 | 0.000000 | 0.000001 | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1895.343213 | UNMEASURED |  |  |  |  |  |
| aloha_left | ssik_reference_cpu | initialization | 25.114049 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_cpu | first_call_including_jit | 2.240338 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_cpu | resident | 1.244093 | 1 | 1.345229 | 1.354219 | 1.356466 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 803.798392 | 824.740238 |  |  |  |  |  |
| aloha_left | ssik_reference_cpu | resident | 5.836839 | 8 | 5.953051 | 5.963381 | 5.965964 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1370.604856 | 1430.992957 |  |  |  |  |  |
| aloha_right | ssik_reference_cpu | initialization | 16.577343 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_cpu | first_call_including_jit | 1.886538 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_cpu | resident | 1.238272 | 1 | 1.314665 | 1.321455 | 1.323153 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 807.577123 | 830.654196 |  |  |  |  |  |
| aloha_right | ssik_reference_cpu | resident | 6.479665 | 8 | 7.184510 | 7.247163 | 7.262826 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 1234.631740 | 1242.958566 |  |  |  |  |  |
| marvin_bare_left | ssik_reference_cpu | initialization | 13.464476 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_cpu | first_call_including_jit | 76.464420 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_cpu | resident | 74.839407 | 1 | 74.933995 | 74.942403 | 74.944505 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 13.361944 | 13.270050 |  |  |  |  |  |
| marvin_bare_left | ssik_reference_cpu | resident | 596.878468 | 8 | 973.633723 | 1007.123079 | 1015.495418 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 13.403063 | 13.176701 |  |  |  |  |  |
| marvin_bare_right | ssik_reference_cpu | initialization | 12.321645 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_cpu | first_call_including_jit | 75.462136 | 1 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_cpu | resident | 78.246603 | 1 | 79.016985 | 79.085463 | 79.102583 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 12.780108 | 12.632633 |  |  |  |  |  |
| marvin_bare_right | ssik_reference_cpu | resident | 619.541676 | 8 | 627.649765 | 628.370484 | 628.550664 | 3 | 0.000000 | 0.000000 | unavailable | unavailable | unavailable | unavailable | UNVALIDATED: allocation ownership not established for this backend | 12.912771 | 10.522559 |  |  |  |  |  |
| aloha_pair | ssik_reference_cpu | paired_resident | 2.392224 | 2 | 2.398555 | 2.399117 | 2.399258 | 3 |  |  |  |  |  |  |  |  |  | unvalidated: smoke |  |  |  |  |
| marvin_bare_pair | ssik_reference_cpu | paired_resident | 150.692216 | 2 | 171.324437 | 173.158412 | 173.616906 | 3 |  |  |  |  |  |  |  |  |  | unvalidated: smoke |  |  |  |  |
| aloha_left | ssik_warp_cpu | streaming_control | 0.136520 | 1 | 0.145728 | 0.202387 | 0.473823 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| aloha_right | ssik_warp_cpu | streaming_control | 0.137021 | 1 | 0.141499 | 0.158423 | 0.214298 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| marvin_bare_left | ssik_warp_cpu | streaming_control | 0.283811 | 1 | 0.402557 | 0.408199 | 0.689363 | 10000 |  |  |  |  |  |  |  |  |  |  | 337 | 0.966300 | 0 | 0.966300 |
| marvin_bare_right | ssik_warp_cpu | streaming_control | 0.285754 | 1 | 0.404811 | 0.416496 | 6.480575 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| aloha_pair | ssik_warp_cpu | paired_streaming_control | 0.272239 | 2 | 0.405083 | 0.411284 | 0.681298 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 |  |  |
| marvin_bare_pair | ssik_warp_cpu | paired_streaming_control | 0.567371 | 2 | 0.800324 | 0.807319 | 3.034881 | 10000 |  |  |  |  |  |  |  |  |  |  | 337 | 0.966300 |  |  |
| aloha_left | ssik_warp_cuda | streaming_control | 0.450378 | 1 | 0.570340 | 1.463107 | 2.526051 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| aloha_right | ssik_warp_cuda | streaming_control | 0.448093 | 1 | 0.558253 | 1.475061 | 1.792264 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| marvin_bare_left | ssik_warp_cuda | streaming_control | 0.515001 | 1 | 0.657324 | 1.535787 | 1.882194 | 10000 |  |  |  |  |  |  |  |  |  |  | 337 | 0.966300 | 0 | 0.966300 |
| marvin_bare_right | ssik_warp_cuda | streaming_control | 0.515682 | 1 | 0.654410 | 0.770959 | 4.394860 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 | 0 | 1.000000 |
| aloha_pair | ssik_warp_cuda | paired_streaming_control | 0.757663 | 2 | 0.965878 | 1.094246 | 2.006021 | 10000 |  |  |  |  |  |  |  |  |  |  | 0 | 1.000000 |  |  |
| marvin_bare_pair | ssik_warp_cuda | paired_streaming_control | 0.887150 | 2 | 1.100848 | 1.211928 | 2.524437 | 10000 |  |  |  |  |  |  |  |  |  |  | 337 | 0.966300 |  |  |
| aloha_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:wide | 1.145697 | 1 | 1.418407 | 2.279478 | 2.617542 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_default_subset_cpu | subset_upstream_core:wide | 0.529571 | 1 | 0.652939 | 0.703869 | 0.714972 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:boundary | 1.165975 | 1 | 1.604752 | 1.628381 | 1.634288 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_default_subset_cpu | subset_upstream_core:boundary | 0.520454 | 1 | 0.563083 | 0.580178 | 0.584452 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:singular | 0.804771 | 1 | 1.991884 | 2.123406 | 2.156286 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_reference_default_subset_cpu | subset_upstream_core:singular | 58.069388 | 1 | 64.390032 | 66.311177 | 66.791463 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:wide | 1.145973 | 1 | 1.282338 | 1.382049 | 1.402985 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_default_subset_cpu | subset_upstream_core:wide | 0.528165 | 1 | 0.564121 | 0.654293 | 0.692751 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:boundary | 1.155375 | 1 | 1.429377 | 1.500127 | 1.517815 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_default_subset_cpu | subset_upstream_core:boundary | 0.534876 | 1 | 0.582671 | 0.609944 | 0.616762 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:singular | 0.737604 | 1 | 1.916229 | 2.162638 | 2.224241 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_right | ssik_reference_default_subset_cpu | subset_upstream_core:singular | 57.680072 | 1 | 65.659855 | 65.961310 | 66.036674 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:wide | 75.210694 | 1 | 79.782859 | 85.298549 | 87.424263 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | subset_upstream_core:wide | 5.223410 | 1 | 5.523171 | 7.213488 | 7.972679 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:boundary | 78.578849 | 1 | 83.015333 | 83.137514 | 83.168059 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | subset_upstream_core:boundary | 5.392802 | 1 | 5.841972 | 5.899678 | 5.914104 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | subset_adapter_api:singular | 75.890717 | 1 | 79.859937 | 83.468652 | 84.370831 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | subset_upstream_core:singular | 5.390820 | 1 | 1464.647722 | 1807.753355 | 1893.529763 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:wide | 73.518670 | 1 | 79.281765 | 80.116496 | 80.117045 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | subset_upstream_core:wide | 5.053269 | 1 | 5.579615 | 5.611175 | 5.620650 | 32 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:boundary | 74.259078 | 1 | 77.112181 | 77.557887 | 77.669314 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | subset_upstream_core:boundary | 5.090297 | 1 | 5.318845 | 5.327231 | 5.329327 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | subset_adapter_api:singular | 69.951914 | 1 | 79.975941 | 82.872313 | 83.596406 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | subset_upstream_core:singular | 5.230804 | 1 | 1812.066274 | 1837.371517 | 1843.697828 | 16 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| aloha_left | ssik_warp_cpu_robot_facade_diagnostic | actual_robot_compute_ik_fixed_root_stub_device_resident_targets | 0.152381 | 1 | 0.168951 | 0.201117 | 0.608252 | 1000 |  |  |  |  |  |  |  |  |  |  |  | 1.000000 | 0 | 1.000000 |
| marvin_bare_left | ssik_warp_cpu_robot_facade_diagnostic | actual_robot_compute_ik_fixed_root_stub_device_resident_targets | 0.297269 | 1 | 0.306653 | 0.330690 | 0.388727 | 1000 |  |  |  |  |  |  |  |  |  |  |  | 1.000000 | 0 | 1.000000 |
| aloha_left | ssik_warp_cuda_robot_facade_diagnostic | actual_robot_compute_ik_fixed_root_stub_device_resident_targets | 0.534852 | 1 | 0.792178 | 1.049311 | 1.195196 | 1000 |  |  |  |  |  |  |  |  |  |  |  | 1.000000 | 0 | 1.000000 |
| marvin_bare_left | ssik_warp_cuda_robot_facade_diagnostic | actual_robot_compute_ik_fixed_root_stub_device_resident_targets | 0.597162 | 1 | 0.848133 | 1.049944 | 1.230952 | 1000 |  |  |  |  |  |  |  |  |  |  |  | 1.000000 | 0 | 1.000000 |

## Success & Other Metrics

| model | impl | scenario | sample_size | success_rate | reported_success_rate | false_successes | translation_err_mm | rotation_err_deg | max_position_m | max_rotation_rad | max_accepted_joint_step_rad |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aloha_left | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000027 | 0.000006 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000038 | 0.000011 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000029 | 0.000014 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_warp_cpu | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.000023 | 0.000003 | 0.000000 | 0.000000 | 0.001338 |
| aloha_left | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000027 | 0.000006 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000038 | 0.000011 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000029 | 0.000014 | 0.000000 | 0.000001 |  |
| aloha_left | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_warp_cuda | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.000023 | 0.000003 | 0.000000 | 0.000000 | 0.001338 |
| aloha_right | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000027 | 0.000006 | 0.000000 | 0.000001 |  |
| aloha_right | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000035 | 0.000009 | 0.000000 | 0.000001 |  |
| aloha_right | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000024 | 0.000010 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_warp_cpu | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.000023 | 0.000003 | 0.000000 | 0.000000 | 0.001318 |
| aloha_right | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000027 | 0.000006 | 0.000000 | 0.000001 |  |
| aloha_right | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000035 | 0.000009 | 0.000000 | 0.000001 |  |
| aloha_right | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000024 | 0.000010 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_warp_cuda | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.000023 | 0.000003 | 0.000000 | 0.000000 | 0.001318 |
| marvin_bare_left | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000042 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000040 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_warp_cpu | trajectory | 20000 | 0.964650 | 0.964650 | 0 | 0.000039 | 0.000002 | 0.000000 | 0.000000 | 0.042003 |
| marvin_bare_left | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000042 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000040 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_warp_cuda | trajectory | 20000 | 0.964650 | 0.964650 | 0 | 0.000039 | 0.000002 | 0.000000 | 0.000000 | 0.042003 |
| marvin_bare_right | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000043 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000037 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_warp_cpu | trajectory | 20000 | 0.953200 | 0.953200 | 0 | 0.000039 | 0.000002 | 0.000000 | 0.000000 | 0.048523 |
| marvin_bare_right | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000043 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000037 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_warp_cuda | trajectory | 20000 | 0.953200 | 0.953200 | 0 | 0.000039 | 0.000002 | 0.000000 | 0.000000 | 0.048523 |
| marvin_gripper_left | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000047 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_gripper_left | ssik_warp_cpu | trajectory | 20000 | 0.964700 | 0.964700 | 0 | 0.000046 | 0.000003 | 0.000000 | 0.000000 | 0.049554 |
| marvin_gripper_left | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000047 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_left | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_gripper_left | ssik_warp_cuda | trajectory | 20000 | 0.964700 | 0.964700 | 0 | 0.000046 | 0.000003 | 0.000000 | 0.000000 | 0.049554 |
| marvin_gripper_right | ssik_warp_cpu | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000047 | 0.000006 | 0.000000 | 0.000001 |  |
| marvin_gripper_right | ssik_warp_cpu | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_right | ssik_warp_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_right | ssik_warp_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_gripper_right | ssik_warp_cpu | trajectory | 20000 | 0.953200 | 0.953200 | 0 | 0.000041 | 0.000003 | 0.000000 | 0.000000 | 0.048523 |
| marvin_gripper_right | ssik_warp_cuda | wide | 10000 | 1.000000 | 1.000000 | 0 | 0.000047 | 0.000006 | 0.000000 | 0.000001 |  |
| marvin_gripper_right | ssik_warp_cuda | boundary | 1000 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_right | ssik_warp_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.000041 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_gripper_right | ssik_warp_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_gripper_right | ssik_warp_cuda | trajectory | 20000 | 0.953200 | 0.953200 | 0 | 0.000041 | 0.000003 | 0.000000 | 0.000000 | 0.048523 |
| aloha_left | ssik_pytorch_original_cpu | wide | 10000 | 0.000000 | 0.999400 | 9994 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cpu | boundary | 1000 | 0.000000 | 0.986000 | 986 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cpu | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_matched_cpu | wide | 10000 | 0.999500 | 0.999500 | 0 | 0.003178 | 0.000134 | 0.000010 | 0.000005 |  |
| aloha_left | ssik_pytorch_matched_cpu | boundary | 1000 | 0.996000 | 0.996000 | 0 | 0.003225 | 0.000131 | 0.000008 | 0.000010 |  |
| aloha_left | ssik_pytorch_matched_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.002389 | 0.000164 | 0.000006 | 0.000004 |  |
| aloha_left | ssik_pytorch_matched_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cpu | trajectory | 20000 | 0.011050 | 1.000000 | 19779 | 0.004171 | 0.000029 | 0.000010 | 0.000009 | 0.001361 |
| aloha_left | ssik_pytorch_matched_cpu | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.002962 | 0.000085 | 0.000004 | 0.000004 | 0.001338 |
| aloha_right | ssik_pytorch_original_cpu | wide | 10000 | 0.000000 | 0.999200 | 9992 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cpu | boundary | 1000 | 0.000000 | 0.992000 | 992 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cpu | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_matched_cpu | wide | 10000 | 0.999500 | 0.999500 | 0 | 0.003166 | 0.000134 | 0.000010 | 0.000004 |  |
| aloha_right | ssik_pytorch_matched_cpu | boundary | 1000 | 0.992000 | 0.993000 | 1 | 0.003186 | 0.000132 | 0.000009 | 0.000005 |  |
| aloha_right | ssik_pytorch_matched_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.004341 | 0.000141 | 0.000006 | 0.000004 |  |
| aloha_right | ssik_pytorch_matched_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cpu | trajectory | 20000 | 0.011050 | 1.000000 | 19779 | 0.004172 | 0.000029 | 0.000010 | 0.000009 | 0.001341 |
| aloha_right | ssik_pytorch_matched_cpu | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.002984 | 0.000083 | 0.000004 | 0.000004 | 0.001318 |
| marvin_bare_left | ssik_pytorch_original_cpu | wide | 10000 | 0.000000 | 0.995800 | 9958 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cpu | boundary | 1000 | 0.000000 | 0.988000 | 988 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cpu | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | wide | 10000 | 0.995800 | 0.995800 | 0 | 0.002837 | 0.000137 | 0.000009 | 0.000006 |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | boundary | 1000 | 0.986000 | 0.986000 | 0 | 0.002879 | 0.000131 | 0.000006 | 0.000004 |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.003819 | 0.000126 | 0.000006 | 0.000004 |  |
| marvin_bare_left | ssik_pytorch_matched_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cpu | trajectory | 20000 | 0.008350 | 0.980150 | 19436 | 0.004256 | 0.000030 | 0.000010 | 0.000010 | 0.008277 |
| marvin_bare_left | ssik_pytorch_matched_cpu | trajectory | 20000 | 0.967850 | 0.967850 | 0 | 0.002788 | 0.000076 | 0.000010 | 0.000004 | 0.008841 |
| marvin_bare_right | ssik_pytorch_original_cpu | wide | 10000 | 0.000000 | 0.995800 | 9958 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cpu | boundary | 1000 | 0.000000 | 0.988000 | 988 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cpu | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | wide | 10000 | 0.995800 | 0.995800 | 0 | 0.002837 | 0.000137 | 0.000009 | 0.000006 |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | boundary | 1000 | 0.986000 | 0.986000 | 0 | 0.002879 | 0.000131 | 0.000006 | 0.000004 |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | singular | 200 | 1.000000 | 1.000000 | 0 | 0.003808 | 0.000117 | 0.000006 | 0.000004 |  |
| marvin_bare_right | ssik_pytorch_matched_cpu | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cpu | trajectory | 20000 | 0.008350 | 0.960150 | 19036 | 0.004262 | 0.000030 | 0.000010 | 0.000009 | 0.016908 |
| marvin_bare_right | ssik_pytorch_matched_cpu | trajectory | 20000 | 0.956250 | 0.956250 | 0 | 0.002790 | 0.000073 | 0.000010 | 0.000004 | 0.022289 |
| aloha_left | ssik_pytorch_original_cuda | wide | 10000 | 0.000000 | 0.999600 | 9996 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cuda | boundary | 1000 | 0.000000 | 0.985000 | 985 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cuda | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_matched_cuda | wide | 10000 | 0.999400 | 0.999400 | 0 | 0.003168 | 0.000135 | 0.000010 | 0.000004 |  |
| aloha_left | ssik_pytorch_matched_cuda | boundary | 1000 | 0.995000 | 0.995000 | 0 | 0.003222 | 0.000128 | 0.000010 | 0.000009 |  |
| aloha_left | ssik_pytorch_matched_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.004589 | 0.000187 | 0.000009 | 0.000004 |  |
| aloha_left | ssik_pytorch_matched_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cuda | wide | 10000 | 0.000000 | 0.999400 | 9994 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cuda | boundary | 1000 | 0.000000 | 0.987000 | 987 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cuda | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_original_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_pytorch_matched_cuda | wide | 10000 | 0.999600 | 0.999600 | 0 | 0.003161 | 0.000135 | 0.000010 | 0.000007 |  |
| aloha_right | ssik_pytorch_matched_cuda | boundary | 1000 | 0.993000 | 0.993000 | 0 | 0.003186 | 0.000131 | 0.000010 | 0.000010 |  |
| aloha_right | ssik_pytorch_matched_cuda | singular | 200 | 0.755000 | 0.755000 | 0 | 0.003537 | 0.000185 | 0.000009 | 0.000004 |  |
| aloha_right | ssik_pytorch_matched_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cuda | wide | 10000 | 0.000000 | 0.993900 | 9939 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cuda | boundary | 1000 | 0.000000 | 0.981000 | 981 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cuda | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_original_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | wide | 10000 | 0.993900 | 0.993900 | 0 | 0.002844 | 0.000138 | 0.000010 | 0.000004 |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | boundary | 1000 | 0.981000 | 0.981000 | 0 | 0.002895 | 0.000133 | 0.000006 | 0.000004 |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.003703 | 0.000138 | 0.000006 | 0.000004 |  |
| marvin_bare_left | ssik_pytorch_matched_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cuda | wide | 10000 | 0.000000 | 0.993900 | 9939 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cuda | boundary | 1000 | 0.000000 | 0.981000 | 981 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cuda | singular | 200 | 0.000000 | 1.000000 | 200 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_original_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | wide | 10000 | 0.993900 | 0.993900 | 0 | 0.002843 | 0.000138 | 0.000010 | 0.000004 |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | boundary | 1000 | 0.981000 | 0.981000 | 0 | 0.002894 | 0.000133 | 0.000006 | 0.000004 |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | singular | 200 | 1.000000 | 1.000000 | 0 | 0.003748 | 0.000132 | 0.000006 | 0.000004 |  |
| marvin_bare_right | ssik_pytorch_matched_cuda | unreachable | 500 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_pytorch_original_cuda | trajectory | 20000 | 0.010650 | 1.000000 | 19787 | 0.004164 | 0.000030 | 0.000010 | 0.000009 | 0.001348 |
| aloha_left | ssik_pytorch_matched_cuda | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.003120 | 0.000085 | 0.000004 | 0.000004 | 0.001338 |
| aloha_right | ssik_pytorch_original_cuda | trajectory | 20000 | 0.010650 | 1.000000 | 19787 | 0.004168 | 0.000030 | 0.000010 | 0.000009 | 0.001383 |
| aloha_right | ssik_pytorch_matched_cuda | trajectory | 20000 | 1.000000 | 1.000000 | 0 | 0.003105 | 0.000095 | 0.000004 | 0.000004 | 0.001320 |
| marvin_bare_left | ssik_pytorch_original_cuda | trajectory | 20000 | 0.008350 | 0.980150 | 19436 | 0.004262 | 0.000030 | 0.000010 | 0.000009 | 0.008289 |
| marvin_bare_left | ssik_pytorch_matched_cuda | trajectory | 20000 | 0.967850 | 0.967850 | 0 | 0.002807 | 0.000075 | 0.000010 | 0.000004 | 0.009008 |
| marvin_bare_right | ssik_pytorch_original_cuda | trajectory | 20000 | 0.008350 | 0.960150 | 19036 | 0.004259 | 0.000030 | 0.000010 | 0.000009 | 0.016923 |
| marvin_bare_right | ssik_pytorch_matched_cuda | trajectory | 20000 | 0.956250 | 0.956250 | 0 | 0.002789 | 0.000073 | 0.000010 | 0.000004 | 0.022290 |
| aloha_left | ssik_reference_cpu | wide | 32 | 1.000000 | 1.000000 | 0 | 0.000024 | 0.000004 | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_cpu | boundary | 12 | 1.000000 | 1.000000 | 0 | 0.000030 | 0.000006 | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_cpu | singular | 8 | 0.750000 | 0.750000 | 0 | 0.000014 | 0.000004 | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_cpu | unreachable | 8 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_left | ssik_reference_cpu | trajectory | 48 | 0.729167 | 0.729167 | 0 | 0.000024 | 0.000003 | 0.000000 | 0.000000 | 0.040792 |
| aloha_right | ssik_reference_cpu | wide | 32 | 1.000000 | 1.000000 | 0 | 0.000022 | 0.000005 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_cpu | boundary | 12 | 1.000000 | 1.000000 | 0 | 0.000034 | 0.000007 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_cpu | singular | 8 | 0.750000 | 0.750000 | 0 | 0.000016 | 0.000006 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_cpu | unreachable | 8 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| aloha_right | ssik_reference_cpu | trajectory | 48 | 0.729167 | 0.729167 | 0 | 0.000019 | 0.000003 | 0.000000 | 0.000000 | 0.034140 |
| marvin_bare_left | ssik_reference_cpu | wide | 32 | 1.000000 | 1.000000 | 0 | 0.000048 | 0.000005 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_cpu | boundary | 12 | 1.000000 | 1.000000 | 0 | 0.000033 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_cpu | singular | 8 | 0.750000 | 0.750000 | 0 | 0.000045 | 0.000004 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_cpu | unreachable | 8 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_left | ssik_reference_cpu | trajectory | 48 | 0.708333 | 0.708333 | 0 | 0.000045 | 0.000002 | 0.000000 | 0.000000 | 0.038440 |
| marvin_bare_right | ssik_reference_cpu | wide | 32 | 1.000000 | 1.000000 | 0 | 0.000044 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_cpu | boundary | 12 | 1.000000 | 1.000000 | 0 | 0.000036 | 0.000006 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_cpu | singular | 8 | 0.375000 | 0.375000 | 0 | 0.000043 | 0.000008 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_cpu | unreachable | 8 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 | None | None |  |
| marvin_bare_right | ssik_reference_cpu | trajectory | 48 | 0.708333 | 0.708333 | 0 | 0.000029 | 0.000002 | 0.000000 | 0.000000 | 0.038440 |
| aloha_left | ssik_warp_cpu | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000020 | 0.000003 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_warp_cpu | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000020 | 0.000003 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cpu | streaming_control | 10000 | 0.966300 | 0.966300 | 0 | 0.000013 | 0.000002 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cpu | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000012 | 0.000002 | 0.000000 | 0.000000 |  |
| aloha_left | ssik_warp_cuda | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000020 | 0.000003 | 0.000000 | 0.000000 |  |
| aloha_right | ssik_warp_cuda | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000020 | 0.000003 | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_warp_cuda | streaming_control | 10000 | 0.966300 | 0.966300 | 0 | 0.000013 | 0.000002 | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_warp_cuda | streaming_control | 10000 | 1.000000 | 1.000000 | 0 | 0.000012 | 0.000002 | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_adapter_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_default_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_adapter_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_default_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_adapter_subset_cpu | singular_reference_subset | 16 | 0.437500 | 0.437500 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_left | ssik_reference_default_subset_cpu | singular_reference_subset | 16 | 0.437500 | 0.437500 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_adapter_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_default_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_adapter_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_default_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_adapter_subset_cpu | singular_reference_subset | 16 | 0.437500 | 0.437500 | 0 |  |  | 0.000000 | 0.000000 |  |
| aloha_right | ssik_reference_default_subset_cpu | singular_reference_subset | 16 | 0.437500 | 0.437500 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_adapter_subset_cpu | singular_reference_subset | 16 | 0.687500 | 0.687500 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_left | ssik_reference_default_subset_cpu | singular_reference_subset | 16 | 0.750000 | 0.750000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | wide_reference_subset | 32 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | boundary_reference_subset | 16 | 1.000000 | 1.000000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_adapter_subset_cpu | singular_reference_subset | 16 | 0.625000 | 0.625000 | 0 |  |  | 0.000000 | 0.000000 |  |
| marvin_bare_right | ssik_reference_default_subset_cpu | singular_reference_subset | 16 | 0.875000 | 0.875000 | 0 |  |  | 0.000000 | 0.000000 |  |

## Leaderboard

| rank | algorithm | overall_success_rate | avg_translation_err_mm | avg_rotation_err_deg |
| --- | --- | --- | --- | --- |
| 1 | ssik_warp_cpu:aloha_left | 100.00% | 0.000025 | 0.000004 |
| 2 | ssik_warp_cuda:aloha_left | 100.00% | 0.000025 | 0.000004 |
| 3 | ssik_warp_cpu:aloha_right | 100.00% | 0.000025 | 0.000004 |
| 4 | ssik_warp_cuda:aloha_right | 100.00% | 0.000025 | 0.000004 |
| 5 | ssik_pytorch_matched_cpu:aloha_left | 99.97% | 0.003036 | 0.000103 |
| 6 | ssik_pytorch_matched_cuda:aloha_left | 99.96% | 0.003148 | 0.000103 |
| 7 | ssik_pytorch_matched_cpu:aloha_right | 99.96% | 0.003058 | 0.000101 |
| 8 | ssik_pytorch_matched_cuda:aloha_right | 99.81% | 0.003128 | 0.000110 |
| 9 | ssik_pytorch_matched_cpu:marvin_bare_left | 97.76% | 0.002814 | 0.000098 |
| 10 | ssik_warp_cpu:marvin_gripper_left | 97.74% | 0.000046 | 0.000004 |
| 11 | ssik_warp_cuda:marvin_gripper_left | 97.74% | 0.000046 | 0.000004 |
| 12 | ssik_warp_cpu:marvin_bare_left | 97.73% | 0.000040 | 0.000003 |
| 13 | ssik_warp_cuda:marvin_bare_left | 97.73% | 0.000040 | 0.000003 |
| 14 | ssik_pytorch_matched_cuda:marvin_bare_left | 97.68% | 0.002828 | 0.000098 |
| 15 | ssik_pytorch_matched_cpu:marvin_bare_right | 97.02% | 0.002815 | 0.000096 |
| 16 | ssik_warp_cpu:marvin_bare_right | 97.00% | 0.000040 | 0.000003 |
| 17 | ssik_warp_cuda:marvin_bare_right | 97.00% | 0.000040 | 0.000003 |
| 18 | ssik_warp_cpu:marvin_gripper_right | 97.00% | 0.000043 | 0.000004 |
| 19 | ssik_warp_cuda:marvin_gripper_right | 97.00% | 0.000043 | 0.000004 |
| 20 | ssik_pytorch_matched_cuda:marvin_bare_right | 96.94% | 0.002816 | 0.000096 |
| 21 | ssik_pytorch_original_cpu:aloha_left | 0.71% | 0.004171 | 0.000029 |
| 22 | ssik_pytorch_original_cpu:aloha_right | 0.71% | 0.004172 | 0.000029 |
| 23 | ssik_pytorch_original_cuda:aloha_left | 0.68% | 0.004164 | 0.000030 |
| 24 | ssik_pytorch_original_cuda:aloha_right | 0.68% | 0.004168 | 0.000030 |
| 25 | ssik_pytorch_original_cpu:marvin_bare_left | 0.54% | 0.004256 | 0.000030 |
| 26 | ssik_pytorch_original_cpu:marvin_bare_right | 0.54% | 0.004262 | 0.000030 |
| 27 | ssik_pytorch_original_cuda:marvin_bare_left | 0.54% | 0.004262 | 0.000030 |
| 28 | ssik_pytorch_original_cuda:marvin_bare_right | 0.54% | 0.004259 | 0.000030 |
| unranked | ssik_warp_cpu:aloha_pair | UNMEASURED | N/A | N/A |
| unranked | ssik_warp_cpu:marvin_bare_pair | UNMEASURED | N/A | N/A |
| unranked | ssik_warp_cuda:aloha_pair | UNMEASURED | N/A | N/A |
| unranked | ssik_warp_cuda:marvin_bare_pair | UNMEASURED | N/A | N/A |
| unranked | ssik_reference_adapter_subset_cpu:aloha_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_adapter_subset_cpu:aloha_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_adapter_subset_cpu:marvin_bare_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_adapter_subset_cpu:marvin_bare_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:aloha_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:aloha_pair | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:aloha_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:marvin_bare_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:marvin_bare_pair | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_cpu:marvin_bare_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_default_subset_cpu:aloha_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_default_subset_cpu:aloha_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_default_subset_cpu:marvin_bare_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_reference_default_subset_cpu:marvin_bare_right | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_warp_cpu_robot_facade_diagnostic:aloha_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_warp_cpu_robot_facade_diagnostic:marvin_bare_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_warp_cuda_robot_facade_diagnostic:aloha_left | DIAGNOSTIC (see sample counts) | N/A | N/A |
| unranked | ssik_warp_cuda_robot_facade_diagnostic:marvin_bare_left | DIAGNOSTIC (see sample counts) | N/A | N/A |

## Notes

- Persistent raw inputs, output masks/joints, timing samples, runtime metadata, source hashes and the exact standalone streaming certificate are in [raw evidence](../../../../../outputs/benchmarks/ssik_acceptance_20261025/).
- Acceptance scope: four primary arm chains (Aloha left/right and Marvin bare left/right), both CPU and CUDA. Marvin gripper variants receive full quality/integration coverage; duplicate large CPU timing was intentionally omitted because they use the same seven-revolute computation with a constant tool transform.
- Aloha Mini passes all standard quality sets, strict accepted-output FK/limit checks, changing-frame single/pair latency budgets, GPU batch speedup budgets, and the attributed scratch budget. Independent branch parity and Robot facade/configuration tests are separate project evidence. Aloha arm defaults now use SSIK; torso and explicit Pytorch overrides remain supported.
- Separate project validation: 76 SSIK CPU/CUDA tests passed (including real setter ordering), 35 BaseSolver tests passed after an equivalent annotation correction, 127 robot configuration tests passed (including typed Pytorch override/copy ownership), 34 Aloha/Marvin preset cases passed, six benchmark behavior tests passed, two Robot facade cases passed, fixed-swivel 7R parity covered 90 cases and 600 branches, and 6R parity covered 2,052 branches. Counts are separate suites, not a disjoint total. These are kinematics/facade checks, not a claim of live physical simulation.
- Documentation: the latest focused strict SSIK Sphinx build with sphinx_autodoc_typehints passed with zero warnings, and API coverage aligned 3,076 entries. Whole-site HTML was generated from an isolated clean Git snapshot, but the strict whole-site warning gate failed (exit 1, 808 warnings). Three inherited-annotation warnings affecting SSIK in that earlier full run were subsequently corrected and verified by the focused strict build; unrelated existing warnings remain. Both logs are preserved.
- Delivery checks: whole-tree black==26.10.1 left 1,540 files unchanged; git diff --check passed. The wheel built successfully and exact delivered solver/kernel/config/preset/license bytes were verified. Project context check and 42 context tests passed; affected review against origin/main revised IK/robot context while other broad matches retained their existing ownership/interfaces.
- Tianji Marvin keeps the Pytorch default. Its Warp public-batch trajectory completion is slightly below the quality-matched Pytorch baseline despite passing the latency, wide-pose coverage and memory budgets. No implicit numerical fallback is enabled.
- Original Pytorch uses the original 30 starts, 500 iterations and 5e-4 internal pose tolerances. Every algorithm is rechecked independently at 1e-5 m/rad. Thus original-mask successes rejected by the stricter benchmark are counted in false_successes; this is a precision-budget mismatch, not a claim that the original configured solver violates its own 5e-4 contract. Matched Pytorch changes only its internal tolerances to 1e-5.
- The independent oracle uses raw float64 PK transforms/URDF limits and allows 1e-7 rad around those raw limits solely for float32 output representation. Production acceptance enforces the BaseSolver effective float32 limits strictly. The 1e-5 m position and 1e-5 rad rotation thresholds are unchanged; no pose slack is added.
- Speedups compare resident completed-call latency against the original default baseline, at equivalent target/seed/limits. The quality-matched baseline is reported as essential correctness/continuity context; its batch throughput was not measured and no speedup against it is claimed.
- Fixed repeated-target timings are diagnostics. The 5 ms/8 ms control gates below use the separate 10,000 changing-frame certificate, with prior returned joints as seed, per-path resets only, .05 rad bound, and warmup/input preparation/output reads/FK outside the timer. These desktop measurements are statistical targets, not hard real-time guarantees.
- Streaming single-arm returns were independently rechecked by raw float64 PK FK. Paired certificate rates are the conjunction of reported masks; each underlying solver still applies its original-model TCP FK gate. Future unified benchmark runs independently recheck paired outputs too.
- Public batch trajectories use the full 1,000-frame path with one fixed initial seed, matching Robot batch semantics. Earlier per-32-frame refresh results and pre-repair Warp measurements are superseded and excluded. Their baseline trajectory rows remain preserved under lineage.json for audit.
- Reference rows are explicitly smoke-sized diagnostic evidence from pinned revision 4109147f37a7ca3e2516c83853aca583612783f3, not a 10,000-target reference acceptance or performance certificate. Missing reference large-sample budgets remain unmeasured.
- The additional pinned-reference policy comparison uses the same first 32 wide, 16 boundary and 16 singular targets per arm (256 fixed targets, 512 policy queries). Rescue/refinement-disabled adapter and upstream defaults both solve all wide/boundary subsets. Singular adapter/default counts are Aloha L 7/7 of 16, Aloha R 7/7, Marvin L 11/12, Marvin R 10/14. The five additional upstream-default recoveries are flagged as numeric refinement/rescue. The remaining 9/9/4/2 per-arm omissions have no upstream returned candidate, despite known generating joints; they are classified omissions, not unreachable targets or a proven inner-root-loss diagnosis. Raw records retain candidate/filter outcomes and full model/source hashes.
- Reference policy comparison is not a rescue-only ablation: the adapter uses 128 SRS samples and no multi-winding output, while upstream defaults use their default search/winding/rescue policy. Adapter API timings include adaptation/filtering; upstream core timings exclude those steps, so their distinct scopes must not be divided into a speedup. No full reference coverage/performance gate is certified.
- The focused actual Robot.compute_ik diagnostic covers Aloha left and Marvin bare left on CPU/CUDA, 1,000 distinct frames each, a nonidentity root transform, .05 rad step bound and previous-returned seeds. Tensor adaptation, root inverse/bmm, real dispatch and backend completion are timed; only get_link_pose is stubbed to a fixed device-resident root. All 4,000 outputs pass independent world-frame FK/limits/steps. Its p99 values are diagnostic because 1,000 samples do not meet the formal 10,000-call gate; backend state retrieval and live simulation are excluded.
- Leaderboard entries are model-specific, weight only the full standard reachable/trajectory attempts, and exclude deliberate unreachable controls, separate streaming certificates, reference subsets and facade diagnostics. Every diagnostic algorithm is listed as unranked with its sample scope. Rankings across model classes do not establish unconditional algorithm superiority.
- Scratch accounting separates Torch peaks and Warp pool counters. SSIK candidates are Torch-owned zero-copy Warp views; retained candidate storage and transient conversions are attributed, excluding caller inputs, final outputs, model constants and persistent JIT/module code. Other backends are not certified by this scratch calculation.
- Preset import provenance was checked explicitly: the editable preset files used by numerical runs were byte-identical to the worktree measurement-start HEAD. Fresh root-worktree asset, frame, joint-order, limit, TCP and model fingerprints match all six frozen fixtures. Reproduction commands should set PYTHONPATH to this worktree and its extensions directory; [provenance.json](../../../../../outputs/benchmarks/ssik_acceptance_20261025/provenance.json) preserves execution and delivered source hashes separately.
- Historical W1 V030 data preserve 14 reachable pinned-reference failures with URDF hashes and generating joints. They were not remeasured or used to change its existing FEP default. General RR/HP GPU algorithms and approximate-geometry Warp paths remain outside this first phase.
- Fixed repeated-pose latency diagnostic aloha_left/cpu: PASS (p99 0.2461 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate aloha_left/cpu: PASS (p99 0.2024 ms; 0 reported failures).
- Reachable coverage and false-accept gate aloha_left/cpu: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison aloha_left/cpu: PASS.
- Quality-matched baseline coverage/continuity context aloha_left/cpu: PASS (public batch trajectory Warp 100.0000%, matched Pytorch 100.0000%).
- Scratch gate aloha_left/cpu: PASS.
- Fixed repeated-pose latency diagnostic aloha_left/cuda: PASS (p99 0.6613 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate aloha_left/cuda: PASS (p99 1.4631 ms; 0 reported failures).
- Reachable coverage and false-accept gate aloha_left/cuda: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison aloha_left/cuda: PASS.
- Quality-matched baseline coverage/continuity context aloha_left/cuda: PASS (public batch trajectory Warp 100.0000%, matched Pytorch 100.0000%).
- Throughput gate aloha_left/cuda n=1000: PASS (115.513x).
- Throughput gate aloha_left/cuda n=10000: PASS (123.667x).
- Scratch gate aloha_left/cuda: PASS.
- Fixed repeated-pose latency diagnostic aloha_right/cpu: PASS (p99 0.2191 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate aloha_right/cpu: PASS (p99 0.1584 ms; 0 reported failures).
- Reachable coverage and false-accept gate aloha_right/cpu: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison aloha_right/cpu: PASS.
- Quality-matched baseline coverage/continuity context aloha_right/cpu: PASS (public batch trajectory Warp 100.0000%, matched Pytorch 100.0000%).
- Scratch gate aloha_right/cpu: PASS.
- Fixed repeated-pose latency diagnostic aloha_right/cuda: PASS (p99 0.7189 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate aloha_right/cuda: PASS (p99 1.4751 ms; 0 reported failures).
- Reachable coverage and false-accept gate aloha_right/cuda: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison aloha_right/cuda: PASS.
- Quality-matched baseline coverage/continuity context aloha_right/cuda: PASS (public batch trajectory Warp 100.0000%, matched Pytorch 100.0000%).
- Throughput gate aloha_right/cuda n=1000: PASS (105.290x).
- Throughput gate aloha_right/cuda n=10000: PASS (118.365x).
- Scratch gate aloha_right/cuda: PASS.
- Fixed repeated-pose latency diagnostic marvin_bare_left/cpu: PASS (p99 0.8367 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate marvin_bare_left/cpu: PASS (p99 0.4082 ms; 337 reported failures).
- Reachable coverage and false-accept gate marvin_bare_left/cpu: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison marvin_bare_left/cpu: PASS.
- Quality-matched baseline coverage/continuity context marvin_bare_left/cpu: FAIL (public batch trajectory Warp 96.4650%, matched Pytorch 96.7850%).
- Scratch gate marvin_bare_left/cpu: PASS.
- Fixed repeated-pose latency diagnostic marvin_bare_left/cuda: PASS (p99 2.0820 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate marvin_bare_left/cuda: PASS (p99 1.5358 ms; 337 reported failures).
- Reachable coverage and false-accept gate marvin_bare_left/cuda: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison marvin_bare_left/cuda: PASS.
- Quality-matched baseline coverage/continuity context marvin_bare_left/cuda: FAIL (public batch trajectory Warp 96.4650%, matched Pytorch 96.7850%).
- Throughput gate marvin_bare_left/cuda n=1000: PASS (7.158x).
- Throughput gate marvin_bare_left/cuda n=10000: PASS (10.327x).
- Scratch gate marvin_bare_left/cuda: PASS.
- Fixed repeated-pose latency diagnostic marvin_bare_right/cpu: PASS (p99 0.7590 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate marvin_bare_right/cpu: PASS (p99 0.4165 ms; 0 reported failures).
- Reachable coverage and false-accept gate marvin_bare_right/cpu: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison marvin_bare_right/cpu: PASS.
- Quality-matched baseline coverage/continuity context marvin_bare_right/cpu: FAIL (public batch trajectory Warp 95.3200%, matched Pytorch 95.6250%).
- Scratch gate marvin_bare_right/cpu: PASS.
- Fixed repeated-pose latency diagnostic marvin_bare_right/cuda: PASS (p99 1.9196 ms); this does not certify continuous control.
- Single-arm continuous-control latency gate marvin_bare_right/cuda: PASS (p99 0.7710 ms; 0 reported failures).
- Reachable coverage and false-accept gate marvin_bare_right/cuda: PASS (wide 100.0000%; singular coverage 100.0000% recorded separately).
- Original-baseline coverage/continuity comparison marvin_bare_right/cuda: PASS.
- Quality-matched baseline coverage/continuity context marvin_bare_right/cuda: FAIL (public batch trajectory Warp 95.3200%, matched Pytorch 95.6250%).
- Throughput gate marvin_bare_right/cuda n=1000: PASS (6.835x).
- Throughput gate marvin_bare_right/cuda n=10000: PASS (11.031x).
- Scratch gate marvin_bare_right/cuda: PASS.
- Paired continuous-control latency gate aloha_pair/cpu: PASS (p99 0.4113 ms; 0 reported failures).
- Paired continuous-control latency gate aloha_pair/cuda: PASS (p99 1.0942 ms; 0 reported failures).
- Paired continuous-control latency gate marvin_bare_pair/cpu: PASS (p99 0.8073 ms; 337 reported failures).
- Paired continuous-control latency gate marvin_bare_pair/cuda: PASS (p99 1.2119 ms; 337 reported failures).

## Reproduction

Run from the repository root using an environment with Torch, Warp, PK and the project simulation import dependencies. Reference runs additionally require SSIK at the pinned revision. The interpreter below is the measured local environment; substitute its equivalent on another machine.

```bash
export PYTHONPATH="$PWD:$PWD/extensions"
PYTHON=/home/dex/miniconda3/envs/open/bin/python

# Public CLI smoke: all five quality scopes plus changing-frame control.
"$PYTHON" -m embodichain.cli.benchmark robotics-kinematic-solver --solvers ssik --ssik-smoke --ssik-models aloha_left --ssik-devices cpu --ssik-skip-reference --ssik-skip-baseline --ssik-batch-sizes 1 --ssik-artifact-dir /tmp/ssik-cli-smoke

# Full Warp primary quality, 1/8/100/1,000/10,000 profiles and control gates.
"$PYTHON" -m embodichain.cli.benchmark robotics-kinematic-solver --solvers ssik --ssik-models aloha_left aloha_right marvin_bare_left marvin_bare_right --ssik-devices cpu cuda --ssik-skip-reference --ssik-skip-baseline --ssik-artifact-dir /tmp/ssik-final-acceptance

# Full original and quality-matched baseline quality; no redundant Warp run.
"$PYTHON" -m embodichain.cli.benchmark robotics-kinematic-solver --solvers ssik --ssik-models aloha_left aloha_right marvin_bare_left marvin_bare_right --ssik-devices cpu cuda --ssik-skip-reference --ssik-skip-warp --ssik-quality-only --ssik-artifact-dir /tmp/ssik-baseline-quality

# Thirty resident GPU batch calls for the original-default speedup gate.
"$PYTHON" -m embodichain.cli.benchmark robotics-kinematic-solver --solvers ssik --ssik-models aloha_left aloha_right marvin_bare_left marvin_bare_right --ssik-devices cuda --ssik-skip-reference --ssik-skip-warp --ssik-timing-only --ssik-baseline-variants original --ssik-batch-sizes 1000 10000 --ssik-timing-scopes resident --ssik-artifact-dir /tmp/ssik-baseline-bulk
```

The compact JSON also embeds the exact three standalone diagnostic/certificate scripts, preserving their recorded SHA256. They can be extracted without relying on temporary files from the original machine:

```bash
"$PYTHON" - <<'PY'
import hashlib, json
from pathlib import Path
data = json.loads(Path("scripts/benchmark/robotics/kinematic_solver/reports/ssik_acceptance_20261025.json").read_text())
for name, script in data["reproduction_scripts"].items():
    content = script["source"].encode()
    assert hashlib.sha256(content).hexdigest() == script["sha256"]
    Path("/tmp", name).write_bytes(content)
PY
"$PYTHON" /tmp/ssik-streaming-certificate.py --fixtures /tmp/ssik-final-acceptance --output /tmp/ssik-streaming-recheck.json
"$PYTHON" /tmp/ssik-robot-facade-timing.py --fixtures /tmp/ssik-final-acceptance --output /tmp/ssik-facade-recheck.json
# The reference script uses the same /tmp/ssik-final-acceptance fixture path.
# SSIK must be installed at 4109147f37a7ca3e2516c83853aca583612783f3,
# or its pinned checkout src directory must be appended to PYTHONPATH.
"$PYTHON" /tmp/ssik-reference-policy-comparison.py
```

The final public CLI smoke completed successfully after streaming integration. Its original report is retained in `outputs/benchmarks/kinematic_solver_20261025_192542_937591.md`. Exact full-run ancestry and the superseded per-window trajectory exclusions are retained in `lineage.json` in the raw evidence directory. The full-site Sphinx log describes the earlier clean snapshot; the latest focused zero-warning log verifies the subsequent equivalent annotation correction.
