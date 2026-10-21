# EmbodiChain report asset download design

Status: design only. The dataset is initialized with documentation and an empty
index. The interfaces in this document do not exist yet, and existing benchmark
scripts retain their current behavior. The first real asset publication will
provide the inputs needed to implement and validate those interfaces.

## Goal and ownership

All EmbodiChain benchmark families use one protocol for report-specific input
assets from the public dataset `DexForceAI/embodichain_reports`. Benchmark
implementations and suites remain in GitHub; reusable robot/scene assets and
model weights retain their existing resolvers and repository ownership.

The common downloader will belong to the benchmark utilities under
`scripts/benchmark/assets/`, with the benchmark command wired through
the existing `scripts.benchmark.__main__` dispatcher.
It will use the project's existing `huggingface_hub>=0.35.3,<1.0` dependency.
No dependency change, runtime implementation, benchmark adapter, or new
benchmark result format is part of this initialization.

## Preparation flow

```text
selected benchmark / suite
  -> declared report input bundle IDs (none: no report-repository request)
  -> selected revision resolved once to a full Hugging Face commit SHA
  -> index.json from that exact commit
  -> only the selected bundles' files from that same commit
  -> file-size and SHA-256 verification
  -> local asset paths and provenance
  -> benchmark initialization, warmup, then measured execution
```

Preparation finishes before simulation/planner initialization and all timed
regions. Downloads, index processing, hashing, and repair are excluded from
benchmark latency and memory measurements. Existing benchmark timing,
quality thresholds, and report/leaderboard contracts remain unchanged.

Only benchmarks or suites explicitly declaring report input bundle IDs use
this repository. A benchmark using only existing common assets or generated
inputs must not acquire a new report-repository dependency. Report-specific
suite declarations will be added with their first real bundles; no speculative
declarations are added now. Direct Python entry points and the unified CLI must
eventually call the same preparation helper.

## Asset index

The root `index.json` initially contains exactly:

```json
{
  "schema_version": 1,
  "bundles": {}
}
```

`bundles` maps stable bundle IDs to input bundle entries. An empty mapping means
there are no downloadable bundles; it is a valid initialized repository, not
a signal to invent or download a default bundle.

Each future entry has these fields:

| Field | Meaning |
| --- | --- |
| `report_id` | Stable report directory ID, for example `technical-report-v1`. |
| `benchmark_id` | Benchmark identity, for example `motion-generation`. |
| `files` | Nonempty list of the exact input files required by this bundle. |
| `files[].path` | Repository-relative POSIX file path under `reports/<report_id>/inputs/<benchmark_id>/`. |
| `files[].size_bytes` | Nonnegative integer byte count of the file. |
| `files[].sha256` | SHA-256 of the file bytes, as 64 lowercase hexadecimal characters. |

For example, a future bundle named
`technical-report-v1-motion-generation-inputs` could contain
`reports/technical-report-v1/inputs/motion-generation/cases.npz`. This is a naming
example only; no such bundle or file exists at initialization.

IDs use lowercase letters, digits, hyphens, or underscores and must start with
a letter or digit. Paths must be normalized relative file paths without `..`,
empty components, backslashes, or URL schemes. Duplicate file paths within an
entry are invalid. Reject unsupported schema versions, malformed entries,
unknown bundle IDs, and mismatched report/benchmark path prefixes before
downloading bundle files. Results, figures, and videos live outside input
bundles and are retrieved separately when requested.

## Versions, authentication, and cache

- Dataset requests use `repo_type="dataset"`, the fixed repository ID, and the
  official endpoint `https://huggingface.co`. Existing common-asset mirror
  settings do not select the endpoint for this report repository.
- The future benchmark integration supplies a tested full dataset commit SHA
  as its default. There is no default asset revision to advertise while the
  index is empty. The Python helper requires `revision` explicitly.
- An explicit branch or tag resolves to a full commit once before downloading
  the index. All file requests use the resolved commit; do not mix a pinned
  index with files from `main`. Changes to the code or asset pin receive review
  together when the first report integration is implemented.
- Public reads must work anonymously; a token is not required. If credentials
  are used, rely on standard SDK discovery (`HF_TOKEN` or local login), without
  custom token storage. Publishing requires write access. Do not send
  credentials to mirrors or include them in diagnostics/provenance. Offline
  mode consumes locally cached content and makes no remote permission or
  metadata request.
- Default `cache_dir` is `~/.cache/embodichain/report-assets`; a caller can
  override it. Use that root as the Hub cache directory so repository and
  commit snapshots are isolated by the SDK's normal cache layout. Returned
  paths are read-only inputs and may use the SDK's normal blob symlinks.
- Offline mode accepts a full commit or a locally cached revision reference.
  Resolve cached references without a network request. If the reference,
  index, or any required file is absent, fail with the missing identifier and
  the corresponding pre-download command. Do not silently select another
  commit or partially run a benchmark.
- Reuse SDK download locks and interrupted-download handling. Expose bundle
  paths only after every required file passes byte-size and SHA-256 checks;
  a partially downloaded bundle is never ready. Concurrent consumers validate
  their requested bundles before proceeding.
- An online checksum failure triggers one forced re-download of the affected
  file through the SDK, followed by revalidation. A second failure terminates
  preparation. Offline corruption terminates immediately. Do not hand-edit
  Hub cache files or return unverified inputs.

These policies build on the official
[download API](https://huggingface.co/docs/huggingface_hub/v0.35.0/en/package_reference/file_download)
and [cache layout](https://huggingface.co/docs/huggingface_hub/v0.35.0/en/guides/manage-cache).

## Proposed interfaces

### Python

Future interface, not available in this change:

```python
from pathlib import Path

def download_report_assets(
    bundle_id: str,
    *,
    revision: str,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> tuple[Path, dict[str, object]]:
    ...
```

The path is the downloaded snapshot's
`reports/<report_id>/inputs/<benchmark_id>/` directory. Only files explicitly
listed in the bundle are guaranteed to be present. The metadata records
`repo_id`, resolved `revision`, `bundle_id`, `report_id`, `benchmark_id`, the
index SHA-256, and each file's path, verified size, and SHA-256. Benchmarks save
that provenance with existing run metadata; local paths are not source IDs.

Failures distinguish invalid index/bundle selection, unavailable credentials
or repository permission, missing cached files, unavailable remote revision or
file, transport errors, and checksum mismatch. Preparation failure exits
nonzero before timing; it is not counted as an algorithm success or failure.

### CLI

Future commands, not runnable in this initialization:

```text
embodichain benchmark assets list --revision <dataset-commit>
embodichain benchmark assets download <bundle-id> --revision <dataset-commit>
embodichain benchmark assets download <bundle-id> --revision <dataset-commit> --offline
```

Both subcommands support `--revision`, `--cache-dir`, and `--offline`. The CLI
uses a tested pin when a future integration defines one; otherwise require
`--revision`. `list` reads only the index and displays bundle/report/benchmark
IDs and total input bytes. An empty index is a successful empty list.
`download` calls the same helper used by benchmark preparation and prints the
local bundle path and resolved commit. Unknown IDs fail nonzero and list
available IDs. `--offline` maps to `local_files_only=True` for every file request
and also disables online revision resolution.

## Publication and adoption

Upload real report inputs, the report README, and index entries together, then
verify downloads at the resulting dataset commit. Reference that commit from
the report release metadata and a future reviewed benchmark integration.
Report READMEs record the code commit, exact run commands, benchmark settings,
seeds, environment, sources, and per-file licensing. Subsequent dataset commits
preserve earlier snapshots for report reproducibility.

Adopt the shared helper only where actual report input dependencies exist.
Motion generation/Atomic Skills, kinematic solvers, grasp generation, workspace,
RL, rendering, scene, data-pipeline, and other benchmark scripts can use the same
protocol without forcing an input bundle onto workloads that do not need one.
Model access remains subject to the model repository's existing requirements.

The initialization ships only `README.md`, `DOWNLOAD_DESIGN.md`, and the empty
`index.json`. It creates no placeholder report directories, release tag, sample
asset, model copy, or automatic publication job. Repository visibility changes
and historical asset migration require separate work.

## Validation

Initialization acceptance checks:

- Repository owner is `DexForceAI`, type is dataset, and visibility is public.
- The authorized account can read all three files at the committed SHA and
  retrieve exactly the uploaded bytes.
- Anonymous metadata and file requests can read the repository contents;
  publishing still requires write authorization.
- `index.json` parses to schema version `1` with an empty `bundles` mapping;
  Markdown links and layout examples are consistent.

Future implementation tests, with fake Hub clients for protocol failures and
a separate real selected-bundle download check:

| Scenario | Required outcome |
| --- | --- |
| Cold cache | Fetch pinned index and only the selected input files; verify all hashes. |
| Anonymous access | Public metadata and input files download without a token. |
| Warm cache | Reuse the same commit snapshot without re-downloading file bodies. |
| Offline complete cache | Read index and files without any network request. |
| Offline missing ref/index/file | Fail before benchmark initialization with a pre-download hint. |
| Version isolation | Two commits cannot mix index or input bytes, including when a tag moves. |
| Concurrent/interrupted downloads | SDK locking/recovery leaves no partial bundle exposed as ready. |
| Corrupt input | Detect size/hash mismatch; repair once online or fail immediately offline. |
| Invalid index | Reject unsupported schema, duplicate/unsafe paths, malformed fields, or unknown IDs. |
| Permission/revision/network failure | Give a specific actionable diagnostic and exit nonzero before timing. |
| No declared report dependencies | Make no request to the report repository. |
| Measurement boundary | Fetching, hashing, and repair never enter latency/memory samples. |
| Provenance | Save actual resolved dataset commit and verified file identities without credentials. |

No runtime tests or performance claims are added by this documentation-only
initialization. Existing benchmark output schemas and rankings are preserved.
