# Observer-disabled contact sensor comparison

Reference main: `ae23b06e5acd60ca6a2c2de184bfc3992038cdb9`. The candidate sensor source SHA is recorded in each JSON result.

The benchmark runs the actual ContactSensor and ContactHistory implementations,
including Warp kernels, cached CUDA graphs and stream synchronization. A fixed
SDK contact buffer supplies four foot contacts per environment. Native physics,
rendering and policy inference are excluded: these results qualify the sensor
path, not end-to-end locomotion training FPS.

Hardware: AMD EPYC 9K65; NVIDIA RTX PRO 5000 72GB Blackwell (physical GPU 4).
Python 3.11.14, Torch 2.7.0+cu128, float32, one Torch CPU thread, seed 0.
Each control step contains four 2.5 ms physics samples. Capacity is 64 contacts
per environment. Twenty control steps warm each implementation; 21 alternating
A/B windows contain 100 control steps each. CUDA timing synchronizes both owning
Warp and Torch devices at window boundaries. No synchronization is inserted into
production code.

| Device | Environments | Main ms/control step | Candidate ms/control step | Difference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 1 | 1.031538 | 1.028674 | -0.28% |
| CPU | 64 | 1.273880 | 1.271506 | -0.19% |
| CUDA | 1 | 0.135354 | 0.135955 | +0.44% |
| CUDA | 256 | 0.135616 | 0.136251 | +0.47% |
| CUDA | 2048 | 0.136527 | 0.137135 | +0.44% |

All compared sensor buffers and history results match exactly. Each source is
fetched exactly 8,480 times per case in both implementations. Differences are
within 0.5% for this workload; no claim of literally zero Python instruction
cost or identical training throughput is made. The disabled path retains small
conditional checks for dynamic subscriptions, but creates no observer snapshots,
generators, callbacks or additional GPU work.

Raw samples and software/hardware metadata are in
[CPU results](results/contact_default_cpu.json) and
[CUDA results](results/contact_default_cuda.json).

Reproduce from a repository checkout with its environment active:

```bash
python -m scripts.benchmark.sensors.contact_default --reference-ref ae23b06e5acd60ca6a2c2de184bfc3992038cdb9 --device cpu --envs 1 64 --output /tmp/contact-cpu.json
CUDA_VISIBLE_DEVICES=4 python -m scripts.benchmark.sensors.contact_default --reference-ref ae23b06e5acd60ca6a2c2de184bfc3992038cdb9 --device cuda:0 --envs 1 256 2048 --output /tmp/contact-cuda.json
```

The reference class is loaded from the specified trusted Git revision. Both
implementations use the same unchanged history implementation and numerical
kernels. Select an available GPU when reproducing; its physical index is not part
of the algorithm. Small timing differences vary with host load and hardware.
