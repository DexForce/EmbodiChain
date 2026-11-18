# Task distribution report

Run from an EmbodiChain environment with the existing artifact reader dependencies:

```bash
python -m embodichain.lab.expansion._distribution \
  --before /absolute/job-a \
  --after /absolute/job-a /absolute/job-b \
  --entity cube --cell-size 0.1 --minimum 5 \
  --output /absolute/new-report-directory
```

Inputs must be job directories containing `job.json` and confirmed artifact receipts.
Repeated input paths/commit identities do not multiply samples. Conflicting commit
identities fail. Output must be new; inputs are never overwritten. Both snapshots
must have the same task, robot installation, topology and configured scene domain.

Open `report.html`. Select a spatial region or joint-distribution cell to inspect
recorded object paths. The evidence panel identifies the full artifact and the
existing native preview command (which retains its original preload limit).
The report itself needs no server or external network. `summary.json` carries
version, snapshot hash, domain, distributions, observed-bin deficits and limitations;
`before-episodes.jsonl` and `after-episodes.jsonl` carry source references and metrics.

Version 2 adds full orientation statistics: normalized sign-canonical xyzw,
initial/final SO(3) rotation vectors, angular path length, net rotation and angular
speed quantiles. Three-dimensional rotation-vector bins have 30-degree edges;
they are not equal-volume and no object symmetry is assumed. Original full XYZ
and quaternion poses remain in every episode record.

TCP is explicitly derived, not directly measured: recorded full joint states and
root poses drive the recorded URDF chain, followed by the single declared gripper
root-to-TCP calibration. The record includes the asset digest, link, transform and
method. Missing/ambiguous calibration returns unavailable; malformed transforms
fail. FK runs in 256-frame batches. Scaled robots are not supported. This is the
same recorded-URDF FK approach used by the existing preview, extended with TCP
calibration; no physics World is created.

`summary.json` now holds aggregate statistics and a lightweight episode index,
not all trajectory details. `pages/{snapshot}-{page}.json` and matching local JS
files contain 100 records per page; the latter lets the offline HTML open without
an HTTP service. Agent tools use JSON. The viewer loads details on demand, retains
at most two detail pages, and displays 50 episode options per page. Joint heatmaps
are paged in 20-by-20 blocks; orientation counts are paged in rows of 20. Filters
operate on the full index before pagination. Index memory remains O(N); extracting
features also retains O(N) bounded descriptors, not full raw trajectories. Page
files and JSONL contain identical episode records.

World-XY maps remain the spatial projection; trajectory display switches between
object and TCP XY paths. Raw numeric evidence is never changed. Joint path length
is in joint-coordinate units and must not be interpreted as Cartesian metres.
Whole-episode metrics and existing phase-aware geometry buckets remain separate.

Counts cover observed cells only; no continuous workspace coverage or unobserved
feasibility claim is made. Different physical parents remain separate samples.
Use physical collection roots, not visual-child datasets. Comparison is cumulative
only if every before commit occurs in after; other comparisons are independent.
No automatic collection, arbitrary LeRobot import or learned clustering is added.

Validation dataset: the complete `formal-qualified` pick-place matrix has 40
physical job roots (two scenes, five seeds, four sampling/scheduling variants).
For each scene, the before snapshot is variant A across five seeds; after is
the cumulative union A+B+C+D. Interrupted diagnostics and visual children are
excluded. This compares dataset composition, not a causal learning experiment.
Use all matching job roots in `--before` and `--after`; retain scene separation.
