---
pretty_name: EmbodiChain Technical Report Assets
language:
  - en
tags:
  - robotics
  - simulation
  - benchmark
  - reproducibility
  - embodichain
---

# EmbodiChain Technical Report Assets

Public asset repository: [DexForceAI/embodichain_reports](https://huggingface.co/datasets/DexForceAI/embodichain_reports).

The dataset stores report-specific benchmark inputs, published measurements,
figures, and videos. The [empty bootstrap index](index.json) in this checkout
is for repository initialization. The dataset index at the pinned revision
lists the available inputs. The [download design](DOWNLOAD_DESIGN.md) also
describes the planned general benchmark CLI.

## Runtime inputs for this report

The release [simulation-report-2026-10-28](https://huggingface.co/datasets/DexForceAI/embodichain_reports/tree/ba51ac54a4c28417691bbed5a23848b01cb21396/reports/simulation-report-2026-10-28)
contains six experiment input bundles and their checksums. Its root index is
read at the same commit as the files. The authored L-02 USDs and the CC BY 4.0
ForgeCAD cabinet, including USD dependencies and attribution, are included.
The NVIDIA 5.1 material license applies to the robot USDs. L-03 and P-03
fetch the declared files from NVIDIA, verify the published hashes, and
preserve their dependency layout. P-03 derives the commands and finger
geometry locally from the verified Franka USD.

`python -m scripts.benchmark.tech_reports.assets.runtime <experiment>`
prepares those inputs. Use `--cache-dir` to choose the cache or `--offline`
to resolve a populated cache. Existing inputs and derived files are verified
on reuse. See the [report runner](../README.md#run-the-published-comparison).

## Ownership

- Benchmark scripts, suite configurations, and runtime implementations remain in
  [DexForce/EmbodiChain](https://github.com/DexForce/EmbodiChain).
- General robot and scene assets continue to use the existing EmbodiChain asset
  resolver and [embodichain_data](https://huggingface.co/datasets/DexForceAI/embodichain_data).
- Model weights continue to use their existing model repositories and download
  helpers. This repository does not duplicate those assets or change their
  access requirements.
- This repository owns report-specific inputs and supporting results/media.
  Downloading benchmark inputs will not automatically download results or media.

The three initialization files are maintained together in
`scripts/benchmark/tech_reports/assets/` in the code repository. Publish those three files
to this dataset's root together;
asset directories are created only when real files are uploaded. The code
repository copy of `index.json` is the empty bootstrap template; after assets
are published, the dataset index at the selected commit is authoritative and
must not be replaced with that empty template during documentation updates.

## Benchmark source assets

The code repository reserves these directories under `scripts/benchmark/tech_reports/assets/`
for small authored inputs, input descriptions, and references:

```text
shared/
loading/
rendering/
physics/
expert_generation/
demonstrations/
```

These match the L/R/P/G/D families in
[issue #679](https://github.com/DexForce/EmbodiChain/issues/679). Put future
experiment-specific material under the appropriate family and experiment ID;
shared definitions belong in `shared/`. These are source directories. The dataset uses the report layout below. Publish the three named bootstrap files explicitly, not
the entire source directory or its `.gitkeep` files.

Large input binaries remain in the selected asset repository and download to
the external cache. Generated run artifacts and reports belong in
`outputs/benchmarks/`. The published input bundles declare the matching IDs and hashes.

## Layout

```text
README.md
DOWNLOAD_DESIGN.md
index.json
reports/<report-id>/
  README.md
  inputs/<benchmark-id>/...
  results/<benchmark-id>/...
  figures/...
  videos/...
```

Each report declares the directories it publishes. Hugging Face may also
maintain repository metadata such as
`.gitattributes`.

## Access

The dataset is public and supports anonymous downloads. Publishing changes
requires an account or token with write access to this repository. For the
project's existing `huggingface_hub>=0.35.3,<1.0` dependency, publishers can use
`hf auth login` or provide `HF_TOKEN` through the execution environment.
Credentials must not be written into report files, commands recorded in reports,
or benchmark provenance. Downloads use `https://huggingface.co` directly.

A client can retrieve the published index without logging in using the existing
Hugging Face SDK; this is not the proposed EmbodiChain download helper:

```python
from huggingface_hub import HfApi, hf_hub_download

repo_id = "DexForceAI/embodichain_reports"
endpoint = "https://huggingface.co"
revision = HfApi(endpoint=endpoint, token=False).dataset_info(repo_id).sha
index_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="index.json",
    revision=revision,
    endpoint=endpoint,
    token=False,
)
```

For reproducible benchmark runs, use a tested full commit SHA instead of
discovering the latest revision. See the [version and cache contract](DOWNLOAD_DESIGN.md#versions-authentication-and-cache).

## Publish report assets

1. Assign a stable report ID and benchmark ID. Prepare the input files and
   supporting outputs under the corresponding report directories.
2. Add `reports/<report-id>/README.md` recording the report title/version,
   EmbodiChain Git commit, benchmark/suite identity, exact run commands, seeds,
   hardware/software environment, input sources, and output interpretation.
3. Record each file's source and license in that report README, with attribution
   or license files where required. No blanket asset license is assigned by
   this initialization; the code repository's license does not automatically
   license report data or third-party media.
4. Add input bundles to `index.json` following the
   [index contract](DOWNLOAD_DESIGN.md#asset-index). Compute byte sizes and
   SHA-256 values from the actual uploaded bytes. Keep results/media outside
   input bundles.
5. Publish inputs, report metadata, and their index entries in one dataset
   commit. Review files and checksums at that commit before declaring it usable.
6. Record the verified dataset commit in the report release metadata and in
   the future benchmark integration's tested default. An optional release tag
   is a human-readable alias; record the resolved full SHA with results.

Later corrections create a new dataset commit and a new validated pin. Report
publication, migration of historical assets, and implementation of the download
interfaces are separate follow-up work.
