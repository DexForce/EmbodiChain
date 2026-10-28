# Recorded visual ACT qualification

These reports capture one local qualification run with the task identity,
dataset identities, checkpoints, seed panels and software listed in
[summary.json](summary.json). The detailed JSON files retain every scene and
checkpoint selection result; their absolute `/tmp` paths describe the machine
where the run was performed and are not portable dataset locations. Datasets,
checkpoints and videos are not bundled with the repository.
The run preceded the move to the repository's canonical `policy.act` import
path; existing checkpoints and training recovery were subsequently checked with
that path. The default action aggregation and model computations are unchanged.
The later image-column-only reader produced byte-identical RGB for 100 sampled
training frames compared with the SDK access used during the original runs.

| Check | Result |
| --- | --- |
| Expert qualification | 100/100 independent scenes |
| Recorded training / validation episodes | 100 / 20, disjoint seeds |
| Four-scene training reset diagnostic | 4/4 |
| Development selection | 15 checkpoints per seed, 50 scenes each |
| Reserved test, seeds 19001 / 19002 / 19003 | 100/100, 100/100, 98/100 |
| Fixed and untrained baselines | 0/100 each on the same test panel |
| RGB diagnostic, live / shuffled | 10/10 versus 1/10 on development scenes |

The test mean is 99.3%, above the example's specified acceptance thresholds.
The result applies to this simulator, camera, object, control rate and reset
distribution. It does not establish transfer to real hardware or different
objects. The 3 cm entry-accuracy metric was 0/100 for each trained seed because
ordinary episodes terminate at the 10 cm success boundary; it does not measure
whether a continued policy could push the cube more precisely. The shuffled
image diagnostic uses one checkpoint and ten scenes, not a test panel.

`benchmark.json` reports a separate batch-32 sampling/update and batch-1/32
inference measurement on the listed GPU, including process and allocator peak
memory. Training initialized the ResNet-18 from the torchvision ImageNet weights
whose hash appears in the summary. Three independent training processes used one
GPU each; their outputs can be reproduced sequentially on a single GPU. The
file timestamps indicate about 38.5 minutes from recipe creation through the
last checkpoint for each run, including loading and offline validation; this is
an approximate wall-time span, not isolated optimizer time.
The GPU used had 72 GB; the batch-32 Torch benchmark reserved about 1.16 GiB.
Additional 0.5-second `nvidia-smi` compute-process samples observed up to
1,558 MiB over a complete 30,000-update run and 5,872 MiB over one rendered
rollout, both with the Torch allocator capped at 20 GiB. The 4,000-update
checkpoint from that additional run succeeded in 10/10 development scenes.
An additional current-code run sampled NVML compute and graphics counters every
0.25 seconds: four recorded episodes and 100 test rollouts completed with the
same Torch cap. The largest process counter on the main GPU was about 5.7 GiB
for both collection and evaluation. Compute and graphics counters on
that GPU reported the same value and may overlap; a roughly 7 MiB graphics
context also appeared on another GPU. The repeated test panel succeeded
100/100. Sampling does not prove an absolute peak, and a physical 24 GB GPU was
not available for an end-to-end run. The hardware-budget check accepts these
capped and sampled measurements.
