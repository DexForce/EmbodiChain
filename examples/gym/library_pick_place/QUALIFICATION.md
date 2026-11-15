# Native qualification — 2026-11-15

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
