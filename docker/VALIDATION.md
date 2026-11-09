# Container qualification and published images

> Historical qualification below includes Python 3.10 results from an earlier
> dependency profile. This checkout requires Python 3.11 or newer;
> the main CI explicitly activates Python 3.11 in the dual-profile image;
> the public image and the internal image's interactive default use Python 3.12.
> The current Docker locks select DexSim 0.6.0rc1; the published image digests
> and qualification runs below describe the earlier DexSim 0.5.1rc1 images.

Source commit: `c42e8a0b0857ae00aeeb21d414d30318abb2e171`.
Implementation: [PR #914](https://github.com/DexForce/EmbodiChain/pull/914),
which incorporates [PR #911](https://github.com/DexForce/EmbodiChain/pull/911).

| Target | Published reference | Compressed layers |
| --- | --- | ---: |
| Public py312 | `dexforce/embodichain:ubuntu22.04-cuda12.8-py312` | 9.86 GB |
| Public compatibility alias | `dexforce/embodichain:ubuntu22.04-cuda12.8` | 9.86 GB |
| Internal py311/py312 | `192.168.3.13:5000/dexsdk:ubuntu22.04-cuda12.8.0-h5ffmpeg-v4-py311-py312` | 16.44 GB |

Public digest:
`sha256:7e0627b9ab3f7da3cb81109e8bd7fcc6e9231492d250026130ea518426b366dc`.
Docker Hub and the staged public target have identical digests.

Internal digest:
`sha256:c3514d70890af98e413fdfe5ba2859c039d51f5b23be24639c3fed14cf31f322`.

Both public tags identify the same image. The original candidate tags ending
in `-c42e8a0b0857` remain available with these same digests.

## Actual image qualification

[Container qualification](https://github.com/DexForce/EmbodiChain/actions/runs/41282836376)
passed on RTX 3090. Public py312, internal py311 and internal py312 each passed
**162 behavior tests and 17 native GPU tests**, with no skipped required cases.
Python 3.11.16 uses Torch 2.7.0+cu128 / LeRobot 0.4.4; Python 3.12.14 uses
Torch 2.11.0+cu128 / LeRobot 0.6.1.

Additional real checks passed for H.264/TorchCodec, HDF5, five isolated Blender
operations, CUDA ACT optimizer/checkpoint behavior and traced ONNX CUDA
execution. Native cases cover default/Newton contacts, CPU/CUDA/OptiX cameras,
cuRobo planning and lossless depth video. Startup, login shells and interactive
conda environment switching also passed.

Independent clean and conda installations passed on RTX PRO 5000 / driver
580.126.20; both profiles passed 18 Pink/Pinocchio compatibility cases. The 90
focused bridge, metadata-integrity, workflow, context and docs tests passed.
Black 26.10.1, actionlint and API coverage (3307/3307) passed. The Sphinx dummy
build succeeded with existing cross-reference warnings.

## Installed-wheel qualification follow-up

The historical image run above executed its reused pytest suites from a source
checkout. The corrected verifier stages only tests and supporting scripts,
ignores inherited Python paths in pytest and its subprocesses, and checks the
core, tasks and robot import locations before and after pytest. Configuration
checks now load the installed package resource.

A wheel built from the updated PR passed **162 behavior tests and 17 native GPU
tests per Python profile**, with no required skips, in independent Python
3.11.16/3.12.14 environments on RTX PRO 5000 / driver 580.126.20. Their reports
confirmed all three bundled package roots inside each environment's
`site-packages`. H.264/TorchCodec, HDF5, CUDA ACT optimization/checkpoint loading
and traced ONNX CUDA checks also passed.

Fresh installation exposed incompatible OpenCV Python/native files from the two
variants required by DexSim and LeRobot. Both now select 4.12.0.88 on Python
3.10/3.11 and 4.13.0.92 on Python 3.12. The normal CPU resolver succeeds for all
three SDK matrix versions; qualification also checks the imported `cv2` version.

DexSim SHA256 pins were checked against the published index and all **1,926
payload files in each wheel** matched the previously qualified installation.
`uv` rejected deliberately incorrect hashes for both Python artifacts. The
three source-shadowing, missing-resource and editable-install regressions fail
against the original verifier and pass after correction. These and the existing
workflow/metadata checks pass **35 tests**; Black 26.10.1, actionlint and API
coverage (3307/3307) also pass. No context update is needed: qualification remains
owned by `docker/`, with the existing Blender boundary unchanged.

This follow-up validates fresh installed-wheel environments. The published
`c42e8a0b0857` image digests above have not been rebuilt with these corrections.

## Publication and rollout

[Docker Hub publication](https://github.com/DexForce/EmbodiChain/actions/runs/41289076794)
succeeded. The public digest and source revision were verified after publishing.
The approved temporary GitHub secret was deleted and its absence verified.

[Formal manifest promotion](https://github.com/DexForce/EmbodiChain/actions/runs/41487856201)
succeeded after both CI workflows passed at implementation commit
`df878c6eb913d00f25f4b2c73226304814fb1c57`:

- [Main py311 CI](https://github.com/DexForce/EmbodiChain/actions/runs/41471747933):
  10,830 CPU, 499 GPU and 409 simulation cases passed; 38/8/5 cases were skipped
  respectively. Formatting, API coverage, documentation tests, the Sphinx build
  and the final test gate passed.
- [SDK compatibility](https://github.com/DexForce/EmbodiChain/actions/runs/41471747895):
  all seven jobs passed, including 141 SDK cases each on Python 3.10, 3.11 and
  3.12, all three offline-tool jobs and the standalone exporter.

Promotion copied the existing manifests without rebuilding the images. Their
bundled wheel and source labels remain at `c42e8a0b0857`; the CI runs above
installed and tested the updated PR source in those environments. Use the
source-development instructions in [README.md](README.md) to install a newer
checkout into a running container.

Independent registry reads verified all three formal tags against the candidate
digests. The previous public image is preserved as
`dexforce/embodichain:ubuntu22.04-cuda12.8-legacy-5cfd8f52237d`, with digest
`sha256:5cfd8f52237d8da7bf2461239295f4c136c482e30dd3c588ee283f882036ff6c`.
The internal `v3-py311` tag still identifies its original
`sha256:3ed6575c47aa3015f1b24b1be4355804f2e96ce71b0111cd021fbb47989b9b06` image.
Temporary registry credentials were removed from the publishing runner, the
GitHub secret was deleted and its absence verified, and local Docker login
credentials were deleted.

At that historical promotion, the main, dataset and documentation workflows
used the formal internal tag with explicit `py311` activation, and the SDK
matrix selected Python 3.10/3.11/3.12. The current checkout's SDK matrix selects
Python 3.11 and 3.12; its main test job still explicitly activates `py311`.

## DexSim 0.6.0rc1 build and dependency follow-up — 2026-11-09

Runtime and Dockerfile revision:
`c1af874bae7f0a6bc53691a8b2b9d2e063d80e57`.
Build host: Linux amd64, Docker 29.6.0, BuildKit 0.34.2. The Dockerfile retains
the CUDA 12.8 Ubuntu 22.04 base-image digest recorded in source.

### Python distribution checks

The current package requires Python `>=3.11`. The installation guide now
lists Python 3.11/3.12, and distinguishes the `py312` interactive image default
from the main workflow's explicit `py311` activation.

The published DexSim 0.6.0rc1 wheels resolved successfully with isolated pip
dry-runs targeting CPython 3.11 and 3.12 on `manylinux_2_31_x86_64`.
Downloaded hashes matched both the profiles and locks:

| ABI | SHA256 |
| --- | --- |
| cp311 | `f3cef56bd987b3166b871d7d9f29b2f2d6ceb274cf47ca9a598c51d6a6785718` |
| cp312 | `f84c0f1763689b6d57f63308b417932237325666c20d1e3379bafbb7cdad7076` |

Both published wheel metadata records require `newton[sim]==1.7.0` and
`warp-lang==1.17.0` on Linux. A newly built EmbodiChain 0.3.2 wheel accepted
the Python 3.11/3.12 targets and rejected Python 3.10 through `Requires-Python`.
The wheel's dependency metadata also passed the PyPI direct-URL check.

Both profile inputs were re-resolved with uv 0.12.5 against the checked-in
constraints, including the `gensim` and `policy-deploy` extras. The 309 py311
and 298 py312 requirements matched their lock files without dependency drift.

### Actual container builds

Both Dockerfile targets completed, including wheel installation with
`--no-deps` and the subsequent `uv pip check`. Final checks reported:

| Image/environment | Compatible installed packages |
| --- | ---: |
| public / py312 | 299 |
| public / isolated blender311 | 28 |
| internal / py311, including test tools | 312 |
| internal / py312, including test tools | 301 |

Local OCI manifest digests:

| Target | Digest |
| --- | --- |
| public | `sha256:75c8cb268415817394a012e1de2543da6017cec53cba44fafc0d5e2cdb538010` |
| internal | `sha256:0fd4288482984fa5e04a668b24b3296c5d28a2d31f7b4c715734c054e78afd9d` |

Both exported OCI configurations were checked against their config digests,
the runtime revision label and the `EMBODICHAIN_ENV=py312` default.
BuildKit cache and OCI exports used a dedicated secondary-disk directory;
the existing Docker builder remained selected.

The first build reached the Miniforge download and failed on an upstream
HTTP 504. The Dockerfile now retries the official alias within the same pinned
26.9.4-1 release. Both paths retain the original installer SHA256 check
(`571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc`).
The successful builds used that fallback. An interrupted internal build was
resumed from its completed dependency cache before the final successful export.

The build procedure used a dedicated `docker-container` Buildx builder:

```bash
for target in public internal; do
  docker buildx build --builder pr829-validation --platform linux/amd64 \
    --network host --allow network.host --progress plain \
    --file docker/Dockerfile --target "$target" \
    --build-arg SOURCE_REVISION=c1af874bae7f0a6bc53691a8b2b9d2e063d80e57 \
    --output "type=oci,dest=$target.oci.tar" .
done
```

Release/workflow regressions passed 46 tests. The Sphinx dummy build completed
with existing repository diagnostics; the updated installation page retained
no system messages and included the corrected version matrix and image-scope
note. Black 26.10.1 and API documentation coverage (4440/4440) passed.

This follow-up qualifies distribution availability, resolution, container
builds and installed dependencies. It does not rerun the full GPU, media and
Blender behavior checks in `docker/validate.sh`. The candidates were exported
locally; published image tags and the historical native qualification above
were not promoted or reused as evidence for DexSim 0.6.0rc1.
