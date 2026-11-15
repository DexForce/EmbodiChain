# Native qualification — 2026-11-15

**Metric depth qualification failed in the subsequent geometry audit.** The
earlier passes below establish physical task outcomes, RGB rendering and raw
stream integrity, not visible-surface depth correctness. The current Library
composite depth must not be used as geometric supervision or collision evidence.

Qualified on Linux with Python 3.12, NVIDIA RTX PRO 5000 72 GB Blackwell,
driver 580.126.20, CPU physics and native Hybrid rendering. Software:

- DexSim `0.6.0rc2`, engine commit `dbaca660471d5aaa7e750093e7fe1c207465c4af`.
- PyTorch `2.11.0+cu128`, NumPy `2.2.6`, Black `26.10.1`.
- USD environment support from #1157 at `364dfd0`.
- Original Library archive SHA256
  `5cc314fb7a8ff1f548aff19249643c902d5ea1f1d5f2180c1f03b490c7eedfe0`;
  original PLY SHA256
  `ce753ff95c75db926804cfdb414501932484fd25e8c60ff4f966b341a5be146d`.

## Complete task and recording

One `--episodes 10 --seed 42` invocation reused one World across all resets.
All ten attempts passed without episode retries, producing **4,780 actions and
4,790 RGB-D observations**, at 960x540 and 25 Hz simulation cadence.

| Measurement | Observed |
| --- | --- |
| Per-episode control steps / media frames | 478 / 479 |
| Measured lift | 0.15175–0.15181 m |
| Final position error | 0.00107–0.00144 m |
| Per-episode wall time | 17.10–19.49 s |
| Peak retained media frames | 65,318,400 bytes, below 64 MiB |
| Process peak RSS | 4,023,287,808 bytes |

Wall times include each episode's reset, execution, capture, persistence and
preview; they exclude World construction and the pre-publication audit. These observations are
scoped to this hardware, native build and asset, not a renderer speedup claim.
Raw NPZ media uses about 1.7 GB per episode. Native mesh RGB has the expected
single-sample noise with denoising disabled. Saved previews were inspected for
the Library background, orange cube, gripper motion and mixed-render occlusion.

The ten-episode run was moved to a different directory, then streamed through
the current verifier with no visible CUDA devices. All UUIDs, file/chunk hashes,
T+1 captures, calibration, timestamps and measured outcomes passed. Run UUID:
`ee745bf31fe84bc686946d09cbfd3ee2`. Its `run.json` records the exact input,
configuration and example-code hashes used during collection.

A separate final one-episode smoke also passed with `SDL_VIDEODRIVER` and
`XDG_RUNTIME_DIR` initially unset, exercising the entry point's default
headless setup: 478 steps, 479 frames and 0.00108 m final error.

## Failure and replay checks

- Removed NPZ chunk, altered artifact bytes and removed completion marker:
  each disposable recording copy was rejected with nonzero exit status.
- Removed the last media chunk and recomputed the referring artifact hashes:
  verification still rejected the incomplete frame stream.
- Inserted a non-finite terminal pose and recomputed all referring hashes:
  verification rejected invalid physical states.
- Injected a media-write exception during actual native execution: the executor
  attempted its ordinary safe stop, the run returned nonzero and recorded the
  failure, and no episode completion marker was published.
- After relocation, native control replay rendered initial, lifted and
  pre-terminal states matching the saved cube/robot poses. Explicit restoration
  of the saved terminal state also matched and rendered valid RGB/depth.
  Interactive window operation was not exercised on the headless host.

## Focused regression coverage

The demo executor, media spool, USD-scene admission, example packaging and
example CLI suites passed **185 tests**, with four native tests excluded from
that CPU lane. Existing production cases were extended for normal/safe-stop
observers, exact float32 depth with infinity, and Gaussian visual-subtree
selection in USD and standard ZIP preflight. Native acceptance above exercises
the real Library through the actual example CLI. No example-import test harness
or new SDK recording API was introduced.

Public API documentation coverage passed at **4913/4913** exports. Existing
simulation/environment/media context guidance remains accurate: this example
composes the established contracts, and its runnable/recording details are
owned by the example README.

## OptiX sensor denoising comparison

The same native build and GPU were used at 960x540, Hybrid, `spp=1`, with
only offscreen denoising changed. Both fresh seed-42 task recordings completed
478 actions and 479 same-frame RGB/depth observations. Native logs confirmed
the OptiX offscreen denoiser was active.

| Measurement | Off | OptiX | Change |
| --- | ---: | ---: | ---: |
| Completed camera call median | 8.224 ms | 10.290 ms | +25.1% |
| Completed camera call P95 | 9.733 ms | 11.137 ms | +14.4% |
| Warm native device-memory delta | 2642 MiB | 2792 MiB | +150 MiB |
| One full recording, excluding setup/audit | 19.59 s | 21.31 s | +8.8% |

Camera timing used four fresh processes in off/optix/optix/off order, restoring
the exact initial/lifted/terminal states from one trajectory. Each mode has
360 timed captures after ten warmups per pose. The timed call includes state
publication, native rendering and CPU RGB/depth readback; physics, writes and
image inspection are excluded. Device memory includes native graphics/CUDA
allocations and is sampled separately. End-to-end recording time is one sample
per mode, so it includes ordinary host/filesystem variability.

Matching-pose images show much less grain on the mesh robot and tabletop.
The Library background ROI (x=650:850, y=140:230) was pixel-identical between
modes at all three inspected poses. Fixed-clock rendering repeats the same
sampling pattern, so temporal variance is not used as a noise-quality metric.
The native denoiser's output and the raw output are both retained; MP4 previews
are encoded from those actual sensor frames at the original 25 fps.

The reproducible harness is
`scripts/benchmark/rendering/library_denoising.py`; it records raw latency
samples, native versions, source hashes, memory scope and matching-pose PNGs.
The renderer/configuration and example-CLI regression suites passed 263 tests
after exposing the `--denoising off|optix` example option.

## DLSS compatibility and default Balanced comparison

DLSS is not excluded by the Gaussian admission contract. The general rendering
configuration defaults to DLSS RR, while the Gaussian preset chooses OptiX for
offscreen cameras. The initial comparison followed that preset; it did not
establish any DLSS incompatibility.

The same Library task subsequently completed with `--denoising dlss`: 478
actions / 479 valid RGB-D frames. Native NGX initialization succeeded, and
`get_rt_pipeline_state()` reported effective `DLSS_RR`, matching the requested
mode. Native logs reported **557x313 internal / 960x540 output** at default
Balanced quality 2. Off and OptiX use native 960x540 input. Therefore this is a
resolution/reconstruction tradeoff, not equal internal shading workloads.

A new six-process pass in off/optix/dlss/dlss/optix/off order measured 360
captures per mode at the same three recorded poses:

| Mode | Internal mesh resolution | Camera median | P95 | Warm device-memory delta |
| --- | --- | ---: | ---: | ---: |
| Off | 960x540 | 8.262 ms | 8.841 ms | 2642 MiB |
| OptiX | 960x540 | 10.355 ms | 10.988 ms | 2792 MiB |
| DLSS RR Balanced | 557x313, reconstructed to 960x540 | 9.118 ms | 10.057 ms | 2875.3 MiB |

Matching-pose screenshots and complete sensor videos are retained for all three
modes. The default DLSS task recording took 19.11 s in one sample, with the same
setup/audit exclusion as the previous recording timings. Native pipeline state
is saved in the benchmark JSON so configured mode can be distinguished from
capability fallback. DLAA quality 5 would retain native internal resolution;
that alternative was not qualified in this comparison.

## Depth geometry audit — failed

The complete off, OptiX and DLSS recordings were inspected (479 depth images
each). All are float32 540x960, positive and finite, with no NaN or negative
values. Capture identity/cadence also passes. Nevertheless, **each recording
contains only one unique depth image**, and the three modes have exactly the
same image hash:
`e5eaedb14a4715fd9581a1c095a289215d658b640a3e15bdd84d10fb53a7a8a3`.
The image does not show the robot, tabletop or cube moving in RGB.

An independent camera-ray/plane and ray/box oracle used the recorded OpenGL
camera pose, intrinsics and object poses. Tabletop samples are strictly inside
the top face with a 2 cm edge margin; pixel centers use the +0.5 convention.
An identical-pose native experiment then hid and removed the GS visual while
retaining the original meshes, camera and physics state:

| Check | GS + mesh | Mesh-only control |
| --- | ---: | ---: |
| Tabletop median absolute view-Z error | 0.2554 m | 0.0000051 m |
| Initial cube interior median absolute error | 0.6918 m | 0.00000078 m |
| Depth pixels changed between initial/lifted poses | 0% | 9.65% |

Hiding and removing the GS produced identical mesh-only controls. This rules out
the camera coordinate convention and Python buffer copying as explanations for
the composite discrepancy. Mesh-only background uses the SDK's large invalid
sentinel and was masked in the diagnostic visualization.

The native contract explicitly describes **first accepted Gaussian contribution
view-Z**, with `selected_surface_depth=false`. A dominant composite value,
1.1799105 m, matches the view-Z of a large source Gaussian (PLY index 1584068,
largest activated scale about 1.37 m). The evidence localizes the limitation to
Gaussian depth composition/selection; it does not establish a CUDA-kernel defect
against that first-contribution contract. Changing denoisers cannot fix it.

The example's current verifier checks storage/frame integrity and measured task
poses. Its `valid` result must not be interpreted as certification of metric
depth. A selected/opacity-aware Gaussian surface-depth policy with correct mesh
composition, or explicitly separate mesh-only depth, requires further native
integration. RGB and physical task acceptance remain independently verified.
