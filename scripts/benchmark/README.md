# EmbodiChain benchmarks

Run benchmarks from a source checkout through `embodichain benchmark` or
`python -m scripts.benchmark`. Use `embodichain benchmark --help` to list the
available benchmark families and each subcommand's `--help` for its options.
Individual benchmark directories contain their own workload and result guidance.

## Technical report assets

The [report asset guide](report_assets/README.md) describes the public
[DexForceAI/embodichain_reports dataset](https://huggingface.co/datasets/DexForceAI/embodichain_reports),
its directory conventions, publishing process, and empty bootstrap index.
The [download design](report_assets/DOWNLOAD_DESIGN.md) defines future shared
benchmark input preparation, revision pinning, cache behavior, and provenance.

These conventions and the proposed download helper are owned by the benchmark
scripts. The helper and asset CLI are design only; existing benchmark commands
keep their current behavior until actual report inputs are integrated.
