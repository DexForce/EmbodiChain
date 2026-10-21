# Visual-training engineering qualification

These records qualify components and the end-to-end implementation. They are
separate from the formal U128/B32 comparison and must not be used as its
performance results. [manifest.json](manifest.json) records each copied input's
original location and SHA256. Historical failed development attempts remain in
their original directories and will be included in the study's log archive.

- The pinned RLInf encoder matches intermediate features, spatial aggregation,
  projection and gradients with its pretrained/freeze/detach behavior removed.
  Maximum projection error is 6.56e-7; maximum parameter-gradient error is 8.11e-6.
- Image-only fitting holds proprioception constant. Both camera conditions fit
  the fixed targets; removing either required view breaks the dual-view fit.
  CUDA qualification includes a real B32/H2/F32 dual-camera update and CPU/CUDA
  FP32 agreement, with device-context auditing.
- Standard LeRobot/AV1/TorchCodec tests cover actual temporal decoding, exact
  frame selection, lossless numeric integrity, lossy image quality, dynamic
  recording counts, backpressure and normal four-worker shutdown. Complete
  contexts use 33 numeric frames and only two historical camera frames.
- Metric fixtures reject untrained coverage and invalid continuity; report
  fixtures reject mismatched initialization, common initial data, budgets,
  hardware and pending/failed post-validation. Synthetic fixtures are not
  formal benchmark runs.
- A real forkserver regression reproduces the previous exit-status race, where
  two readers changed actual status 0 into 255. Production now serializes
  status consumption; four real-process regression tests cover normal exit,
  concurrent shutdown, hard exit and original worker exceptions.
- Six fresh physical smoke runs completed: single/dual camera, each with
  offline, overlapping ODS and serial ODS. All passed corpus decoding and two
  independent rollout scenes, with normal producer/reader shutdown. For both
  camera modes, overlap and serial have identical initial/final weights,
  normalization and source/anchor update traces.
- A subsequent real dual-camera probe verifies explicit quota waiting and
  post-training generation drain, requested evaluation accounting and timed
  CPU/GPU activity. GPU rates are whole-board samples, including foreign work;
  continuing-PID CPU estimates can miss short-lived/new/disappeared work.

Validation used conda `py311`, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128,
LeRobot 0.4.4 and TorchCodec 0.5. The affected production/model/context selection
passed **400 tests**. API coverage is 2794/2794, Python 3.10 AST checks pass,
and the repository was formatted with Black 26.3.1. The shared native runtime
and exact experiment-source hashes are retained with each supervised suite.

An additional scheduling review found that the first U128 matrix unnecessarily
waited for recording submission before requesting a new generation in its
offline and serial controls. That matrix was cancelled and marked superseded;
its four completed timings are diagnostic evidence, not final efficiency
comparisons. The corrected schedule permits recording/generation overlap in
every arm while still excluding generation/optimizer overlap in serial.
Nineteen CPU state/ownership cases and four old-code negative controls pass.
Three real single-camera C8/P2/B32/U40 runs also pass, with equal overlap/serial
final model hashes and traces. The companion dual-camera attempt failed idle
GPU admission before launch and remains in the record; it is not a passing
dual-camera validation. See the `scheduler-*.json` receipts.

Final metrology qualification adds actual writer-thread SDK service and
prefetched reader-worker service intervals, a fail-closed formal report
contract, and full-frame software video integrity. Fifty-two report negative
cases pass. Single/dual video fixtures reject corrupt headers, truncated tails
and clean frame-count mismatches; receipt collisions and ordinary/symlink
corpus containment preserve old evidence.

All three final dual-camera C8/P2/B32/U40 pipelines pass SDK/full-video audits
and two rollout scenes. Overlap and serial have identical initial/final
models, normalization and training traces. Their 24-episode corpora each decode
23,664 camera frames; Offline-16 decodes 15,572. These are engineering checks,
not the U128 formal matrix. A separately named strengthened audit also passes
on the prior single-camera corpus without replacing its original receipt.
