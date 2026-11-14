# AgiBot G2 IK qualification

Command: `CUDA_VISIBLE_DEVICES=2 python -m scripts.benchmark.robotics.kinematic_solver.agibot_g2 --repeats 3 --output qualification.md`

Run on a shared host; no exclusive CPU-core reservation. IK-only qualification uses the isolated environment described in PR #1135.

Python 3.11.14, Torch 2.7.0+cu128, Warp 1.17.0; CPU AMD EPYC 9K65 192-Core Processor, Torch threads=1.
CUDA: NVIDIA RTX PRO 5000 72GB Blackwell.
Target and numerical sampler RNG seed=42; float32 device-resident targets/seeds; two warmups per shape; 3 completed-call repeats, median and min/max wall latency; excludes setup and transfers.
Success requires solver validity, finite in-limit joints, translation error <=1e-5 m, and rotation Frobenius error <=sqrt(2)*1e-5. Failed targets remain in the denominator.
Memory uses a separate warmed call, retains one output, and reports RSS retained delta, not CPU transient peak. GPU statistics cover Torch allocations only, including total peak, and exclude Warp/native persistent scratch. These are not total device-memory measurements.
Legacy SRSSolver: not applicable without changing its shoulder/elbow reconstruction; no latency or success percentage is assigned to an unsupported configuration.

## Time & Memory

| algorithm | device | arm | seed | batch | cost_time_ms | min_ms | max_ms | cpu_delta_mb | gpu_delta_mb | peak_gpu_mb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pytorch | cpu | left_arm | midpoint | 1 | 42.002 | 35.523 | 42.895 | 0.0 | n/a | n/a |
| Pytorch | cpu | left_arm | nearby | 1 | 39.977 | 36.397 | 40.742 | 0.0 | n/a | n/a |
| Pytorch | cpu | left_arm | midpoint | 32 | 399.821 | 360.915 | 400.927 | 0.0 | n/a | n/a |
| Pytorch | cpu | left_arm | nearby | 32 | 337.84 | 324.747 | 342.807 | 0.0 | n/a | n/a |
| Pytorch | cpu | left_arm | midpoint | 128 | 1361.962 | 1332.954 | 1527.4 | 3.406 | n/a | n/a |
| Pytorch | cpu | left_arm | nearby | 128 | 1933.911 | 1528.915 | 2231.506 | 6.445 | n/a | n/a |
| SSIK | cpu | left_arm | midpoint | 1 | 0.687 | 0.683 | 0.7 | 0.0 | n/a | n/a |
| SSIK | cpu | left_arm | nearby | 1 | 0.646 | 0.64 | 0.657 | 0.0 | n/a | n/a |
| SSIK | cpu | left_arm | midpoint | 32 | 12.993 | 12.987 | 12.994 | 0.0 | n/a | n/a |
| SSIK | cpu | left_arm | nearby | 32 | 13.151 | 13.146 | 13.159 | 0.0 | n/a | n/a |
| SSIK | cpu | left_arm | midpoint | 128 | 52.125 | 52.038 | 52.132 | 0.031 | n/a | n/a |
| SSIK | cpu | left_arm | nearby | 128 | 52.83 | 52.773 | 53.299 | 0.0 | n/a | n/a |
| Pytorch | cpu | right_arm | midpoint | 1 | 47.741 | 43.876 | 53.357 | 0.0 | n/a | n/a |
| Pytorch | cpu | right_arm | nearby | 1 | 45.17 | 44.09 | 52.404 | 0.0 | n/a | n/a |
| Pytorch | cpu | right_arm | midpoint | 32 | 338.58 | 337.356 | 349.887 | 0.0 | n/a | n/a |
| Pytorch | cpu | right_arm | nearby | 32 | 336.115 | 330.446 | 374.287 | 0.0 | n/a | n/a |
| Pytorch | cpu | right_arm | midpoint | 128 | 1638.531 | 1400.625 | 1649.182 | 3.039 | n/a | n/a |
| Pytorch | cpu | right_arm | nearby | 128 | 1391.492 | 1307.077 | 1493.965 | 9.156 | n/a | n/a |
| SSIK | cpu | right_arm | midpoint | 1 | 0.645 | 0.636 | 0.657 | 0.0 | n/a | n/a |
| SSIK | cpu | right_arm | nearby | 1 | 0.634 | 0.631 | 0.656 | 0.0 | n/a | n/a |
| SSIK | cpu | right_arm | midpoint | 32 | 12.986 | 12.975 | 12.995 | 0.0 | n/a | n/a |
| SSIK | cpu | right_arm | nearby | 32 | 13.139 | 13.136 | 13.157 | 0.0 | n/a | n/a |
| SSIK | cpu | right_arm | midpoint | 128 | 51.999 | 51.94 | 52.005 | 0.031 | n/a | n/a |
| SSIK | cpu | right_arm | nearby | 128 | 52.693 | 52.648 | 52.792 | 0.0 | n/a | n/a |
| Pytorch | cuda | left_arm | midpoint | 1 | 119.582 | 110.787 | 140.879 | 0.0 | 0.0009765625 | 8.38623046875 |
| Pytorch | cuda | left_arm | nearby | 1 | 128.224 | 117.836 | 129.117 | 0.0 | 0.0009765625 | 8.3876953125 |
| Pytorch | cuda | left_arm | midpoint | 32 | 148.945 | 138.348 | 160.397 | 0.0 | 0.00146484375 | 13.4365234375 |
| Pytorch | cuda | left_arm | nearby | 32 | 147.636 | 139.883 | 165.61 | 0.0 | 0.00146484375 | 13.43701171875 |
| Pytorch | cuda | left_arm | midpoint | 128 | 167.433 | 142.57 | 171.042 | 0.0 | 0.00390625 | 29.25048828125 |
| Pytorch | cuda | left_arm | nearby | 128 | 167.389 | 158.526 | 189.451 | 0.012 | 0.00390625 | 29.25341796875 |
| SSIK | cuda | left_arm | midpoint | 1 | 3.218 | 3.2 | 3.259 | 0.0 | 0.0009765625 | 8.3154296875 |
| SSIK | cuda | left_arm | nearby | 1 | 3.163 | 3.158 | 3.182 | 0.027 | 0.0009765625 | 8.3154296875 |
| SSIK | cuda | left_arm | midpoint | 32 | 3.716 | 3.713 | 3.738 | 0.0 | 0.00146484375 | 11.22900390625 |
| SSIK | cuda | left_arm | nearby | 32 | 3.701 | 3.698 | 3.719 | 0.0 | 0.00146484375 | 11.2294921875 |
| SSIK | cuda | left_arm | midpoint | 128 | 4.329 | 4.321 | 4.349 | 0.0 | 0.00390625 | 20.26611328125 |
| SSIK | cuda | left_arm | nearby | 128 | 4.324 | 4.322 | 4.326 | 0.0 | 0.00390625 | 20.26904296875 |
| Pytorch | cuda | right_arm | midpoint | 1 | 119.1 | 116.715 | 134.99 | 0.012 | 0.0009765625 | 8.38818359375 |
| Pytorch | cuda | right_arm | nearby | 1 | 112.529 | 110.538 | 122.497 | 0.004 | 0.0009765625 | 8.38818359375 |
| Pytorch | cuda | right_arm | midpoint | 32 | 144.2 | 143.912 | 157.129 | 0.0 | 0.00146484375 | 13.43701171875 |
| Pytorch | cuda | right_arm | nearby | 32 | 136.347 | 134.34 | 140.001 | 0.008 | 0.00146484375 | 13.4375 |
| Pytorch | cuda | right_arm | midpoint | 128 | 169.355 | 150.275 | 172.91 | 0.016 | 0.00390625 | 29.2509765625 |
| Pytorch | cuda | right_arm | nearby | 128 | 154.198 | 148.111 | 161.998 | 0.0 | 0.00390625 | 29.25390625 |
| SSIK | cuda | right_arm | midpoint | 1 | 3.198 | 3.185 | 3.218 | 0.0 | 0.0009765625 | 8.31591796875 |
| SSIK | cuda | right_arm | nearby | 1 | 3.183 | 3.182 | 3.191 | 0.0 | 0.0009765625 | 8.31591796875 |
| SSIK | cuda | right_arm | midpoint | 32 | 3.753 | 3.744 | 3.757 | 0.0 | 0.00146484375 | 11.2294921875 |
| SSIK | cuda | right_arm | nearby | 32 | 3.728 | 3.722 | 3.729 | 0.0 | 0.00146484375 | 11.22998046875 |
| SSIK | cuda | right_arm | midpoint | 128 | 4.348 | 4.345 | 4.364 | 0.0 | 0.00390625 | 20.2666015625 |
| SSIK | cuda | right_arm | nearby | 128 | 4.36 | 4.335 | 4.393 | 0.0 | 0.00390625 | 20.26953125 |

## Success & Other Metrics

| algorithm | device | arm | seed | batch | successes | success_rate | max_position_m | max_rotation_frobenius |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pytorch | cpu | left_arm | midpoint | 1 | 1 | 1.0 | 3.3578983220650116e-06 | 1.8209050267614657e-06 |
| Pytorch | cpu | left_arm | nearby | 1 | 1 | 1.0 | 1.2101403399356059e-06 | 4.791024366568308e-06 |
| Pytorch | cpu | left_arm | midpoint | 32 | 32 | 1.0 | 3.877280960296048e-06 | 5.059834165876964e-06 |
| Pytorch | cpu | left_arm | nearby | 32 | 32 | 1.0 | 3.372019591552089e-06 | 5.095510914543411e-06 |
| Pytorch | cpu | left_arm | midpoint | 128 | 128 | 1.0 | 4.644319687940879e-06 | 5.5025361689331476e-06 |
| Pytorch | cpu | left_arm | nearby | 128 | 128 | 1.0 | 3.372019591552089e-06 | 5.418145065050339e-06 |
| SSIK | cpu | left_arm | midpoint | 1 | 1 | 1.0 | 8.97168703772877e-08 | 4.600092609052808e-07 |
| SSIK | cpu | left_arm | nearby | 1 | 1 | 1.0 | 1.2013700256829907e-07 | 5.489717977980035e-07 |
| SSIK | cpu | left_arm | midpoint | 32 | 32 | 1.0 | 1.3075715799004684e-07 | 6.906169573994703e-07 |
| SSIK | cpu | left_arm | nearby | 32 | 32 | 1.0 | 1.4057727071303816e-07 | 6.96858364790387e-07 |
| SSIK | cpu | left_arm | midpoint | 128 | 128 | 1.0 | 1.365714012990793e-07 | 1.0685545248634298e-06 |
| SSIK | cpu | left_arm | nearby | 128 | 128 | 1.0 | 1.611807078916172e-07 | 7.21745550436026e-07 |
| Pytorch | cpu | right_arm | midpoint | 1 | 1 | 1.0 | 1.7203743709615082e-06 | 4.790180355485063e-06 |
| Pytorch | cpu | right_arm | nearby | 1 | 1 | 1.0 | 1.3508376923709875e-06 | 4.828947567148134e-06 |
| Pytorch | cpu | right_arm | midpoint | 32 | 32 | 1.0 | 3.5127966384607134e-06 | 5.069680810265709e-06 |
| Pytorch | cpu | right_arm | nearby | 32 | 32 | 1.0 | 3.3089386306528468e-06 | 5.064641300123185e-06 |
| Pytorch | cpu | right_arm | midpoint | 128 | 127 | 0.9921875 | 4.619576429831795e-06 | 5.553441951633431e-06 |
| Pytorch | cpu | right_arm | nearby | 128 | 128 | 1.0 | 3.3612029710639035e-06 | 5.20678304383182e-06 |
| SSIK | cpu | right_arm | midpoint | 1 | 1 | 1.0 | 1.4901161193847656e-08 | 1.8925794620372471e-07 |
| SSIK | cpu | right_arm | nearby | 1 | 1 | 1.0 | 1.2013700256829907e-07 | 2.318076610663411e-07 |
| SSIK | cpu | right_arm | midpoint | 32 | 32 | 1.0 | 9.064022776783531e-08 | 3.090868574417982e-07 |
| SSIK | cpu | right_arm | nearby | 32 | 32 | 1.0 | 1.4600097131278744e-07 | 3.5388296737437486e-07 |
| SSIK | cpu | right_arm | midpoint | 128 | 128 | 1.0 | 9.064022776783531e-08 | 4.4304337620815204e-07 |
| SSIK | cpu | right_arm | nearby | 128 | 128 | 1.0 | 1.7885272995954438e-07 | 4.2252085563632136e-07 |
| Pytorch | cuda | left_arm | midpoint | 1 | 1 | 1.0 | 3.4424019759171642e-06 | 1.9202741441404214e-06 |
| Pytorch | cuda | left_arm | nearby | 1 | 1 | 1.0 | 1.3012945601076353e-06 | 4.953336429025512e-06 |
| Pytorch | cuda | left_arm | midpoint | 32 | 32 | 1.0 | 4.218160484015243e-06 | 5.819032139697811e-06 |
| Pytorch | cuda | left_arm | nearby | 32 | 32 | 1.0 | 3.330476147311856e-06 | 5.134482762514381e-06 |
| Pytorch | cuda | left_arm | midpoint | 128 | 128 | 1.0 | 4.805572643817868e-06 | 5.195485755393747e-06 |
| Pytorch | cuda | left_arm | nearby | 128 | 128 | 1.0 | 3.330476147311856e-06 | 5.3923263294564094e-06 |
| SSIK | cuda | left_arm | midpoint | 1 | 1 | 1.0 | 4.5320113883917656e-08 | 3.8471625884994864e-07 |
| SSIK | cuda | left_arm | nearby | 1 | 1 | 1.0 | 1.2287812012345967e-07 | 5.55155736492452e-07 |
| SSIK | cuda | left_arm | midpoint | 32 | 32 | 1.0 | 1.2287812012345967e-07 | 5.908667617404717e-07 |
| SSIK | cuda | left_arm | nearby | 32 | 32 | 1.0 | 1.2578027508425293e-07 | 6.172750772748259e-07 |
| SSIK | cuda | left_arm | midpoint | 128 | 128 | 1.0 | 1.497548112183722e-07 | 1.0645990187185816e-06 |
| SSIK | cuda | left_arm | nearby | 128 | 128 | 1.0 | 1.712013073529306e-07 | 7.653454758838052e-07 |
| Pytorch | cuda | right_arm | midpoint | 1 | 1 | 1.0 | 1.0693854619603371e-06 | 4.952310519001912e-06 |
| Pytorch | cuda | right_arm | nearby | 1 | 1 | 1.0 | 1.265283231077774e-06 | 4.773194632434752e-06 |
| Pytorch | cuda | right_arm | midpoint | 32 | 32 | 1.0 | 4.09563108405564e-06 | 4.895048277830938e-06 |
| Pytorch | cuda | right_arm | nearby | 32 | 32 | 1.0 | 3.320694531794288e-06 | 5.057643647887744e-06 |
| Pytorch | cuda | right_arm | midpoint | 128 | 128 | 1.0 | 4.745429123431677e-06 | 5.486724148795474e-06 |
| Pytorch | cuda | right_arm | nearby | 128 | 128 | 1.0 | 3.320694531794288e-06 | 5.246330601949012e-06 |
| SSIK | cuda | right_arm | midpoint | 1 | 1 | 1.0 | 2.2351741790771484e-08 | 2.1296152397098922e-07 |
| SSIK | cuda | right_arm | nearby | 1 | 1 | 1.0 | 7.48774056091861e-08 | 1.654297392406079e-07 |
| SSIK | cuda | right_arm | midpoint | 32 | 32 | 1.0 | 8.024520781191313e-08 | 3.4763166922857636e-07 |
| SSIK | cuda | right_arm | nearby | 32 | 32 | 1.0 | 1.115100829451876e-07 | 3.998401041371835e-07 |
| SSIK | cuda | right_arm | midpoint | 128 | 128 | 1.0 | 1.1920928955078125e-07 | 4.0432701098325197e-07 |
| SSIK | cuda | right_arm | nearby | 128 | 128 | 1.0 | 1.252412431540506e-07 | 3.998401041371835e-07 |

## Leaderboard

| algorithm | success_rate |
| --- | --- |
| SSIK | 1.0 |
| Pytorch | 0.9992236024844721 |
