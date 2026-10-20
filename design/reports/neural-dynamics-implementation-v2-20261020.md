# NeRD implementation audit and remediation — physics signature v2

Status: implementation progressed and bounded evidence refreshed; complete
qualification is **not achieved**. The acceptance contract remains
`design/neural-dynamics-solvers.md`.

## Correctness fixes

The audit reproduced acceptance of a changed physical system by the original
signature: a 180-degree fixed Cartpole root rotation, changed Ant robot contact
properties, and changed collision/topology relationships could escape validation.
The v2 signature now binds fixed-root orientation, parent/child topology, body/
joint flags, relevant material properties and effective collision relationships.
Direct model and public metadata construction agree. Replica count and harmless
contact-free Cartpole translation do not change the signature.

The provider explicitly requires the v2 schema and FP32 model weights/buffers.
Old manifests must not be relabeled. The offline model API still supports FP64;
Newton execution rejects that dtype before stepping. All-false functional reset
masks now preserve the supplied graph and avoid native reset/publication.

Training manifest overrides may add annotations, but cannot change/drop measured
dataset provenance or introduce missing reserved physical fields. Legitimate
fine-tuning on a new measured dataset remains supported.

## Verification coverage

These are separate focused groups, not a deduplicated aggregate test count:

- Runtime audit: **24 passed**, including **6 CUDA cases** across Cartpole/Ant:
  five-step initial q/qd and control gradients, all named network parameters,
  explicit incoming history gradients, partial reset and native-storage isolation.
- Follow-up runtime/collection/schema tests: **13 passed**, including regenerated
  real data and content-hash invariance across replay batch sizes/output paths.
- Input pipeline, trainer and reference group: **70 passed**, with the actual
  official trainer and CPU reference collection enabled.
- Root manager/CLI/comparison regression: **231 passed, 2 skipped**.
- Report qualification tests: **17 passed** after the final provenance changes.
- FP64 fail-fast and actual collector runtime-identity checks: **2 passed**.
- The original official-main two-update training comparison was independently
  rerun after batch changes: **1 passed**.

FP32 numerical receipts still pass for both robots. The two historical matched
training experiments were reread under their `report-audit/` directories:
recorded teacher provenance and numerical parity pass; overall qualification is
pending. Historical experiments retain their original code/data scope.

## Refreshed current-physics pipeline

Both robots ran through the actual CLI with newly generated v2 data:
64 trajectories × 32 steps, zero rejected trajectories; 100 updates of the small
one-layer smoke network; export; offline evaluation; then passive, random-effort
and fixed-policy rollouts. Each rollout uses four trajectories, two environments
and horizons 1/10/32. All six runs completed with zero nonfinite trajectories.
This proves executable flow, not low prediction error or official quality.

The final reports separately identify teacher physics and actual neural
execution device, Python/Torch/CUDA/software versions and adapter source hashes.
They retain selected teacher-content digests, per-trajectory indices, trace files,
model/manifest hashes and policy identity. Reporter gates verify those artifacts
and reject mismatched teacher domains, test populations or fixed policies.
Undefined/nonfinite errors cannot become zero-valued passing evidence. Bare
Boolean numerical pass flags are insufficient.

Compact exact results and artifact paths:
[refreshed smoke receipts](neural-dynamics-v2-smoke-20261020.json).
Full artifacts: `/tmp/embodichain-nerd-v2-qualification/{cartpole,ant}`.

## Performance and delivery

The synchronous grouped batch reader preserves order, duplicates, prefix masks,
owned outputs and exact resume. It groups I/O without prefetch or a full-data
cache. Genuine B512 read/collate/encode timings improved about 11–20× on the
current small Newton datasets. The distinct-trajectory, many-shard stress case
improved only 1.86× and still took 181.72 ms per batch. Full-scale training
throughput remains unmeasured; the former 25.31-day estimate is compute-only.
See [the bounded input benchmark](neural-dynamics-input-20261020.md).

The replacement DexSim artifact is
`0.5.1+g4f093c4a88ff.d2bf6eb9a5b93fd36`; its independent installation passed
**157 tests, 21 deselected**. The refreshed EmbodiChain wheel contains v2 code
and an explicit dependency metadata overlay for this exact local SDK version.
[Artifact hashes and overlay](neural-dynamics-local-artifacts-v2-20261020.json)
identify the pair. Fresh paired installation verified exact dependency matching,
installed package/provider import paths and the FP32 fail-fast case. Proof is
`output/neural-solver-wheel/validation-final-pair/manifest.json` in the DexSim
checkout. The source dependency pin remains rc1, so these are paired
local development artifacts; a published release/pin update is a separate handoff.

Black 26.3.1, diff whitespace, context validation and **2747/2747** public API
coverage passed. A complete Sphinx dummy build succeeded on an isolated committed
source snapshot, avoiding the architecture generator's clean-tree precondition
without committing the user's worktree. It emitted 785 repository warnings;
none mentioned the new NeRD/dynamics pages or module symbols. This is not a
warning-clean documentation claim. The previously isolated NeRD page build also
passed with warnings treated as errors.

## Remaining requirements

- Full-size matched data distributions, five paired seeds and the declared
  4,995,000-update training budget have not been executed.
- The 2048-trajectory, 1/10/100/1000-step complete evaluation and final
  noninferiority/stability/policy-return gates remain pending.
- Official historical Warp data has no compatible v2 current-Newton deployment
  profile. Official-domain closed-loop evaluation needs a separately faithful
  reference execution path; changing metadata is not an implementation.
- Representative full-scale end-to-end throughput is unverified. The bounded
  optimization does not remove the resource-budget question for full training.

No full-budget training job has been launched. The original goal remains active;
unit tests, local wheels and smoke results are not substituted for these gates.
