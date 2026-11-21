# RoboCasa365 physical replay

Run the example with EmbodiChain on a CUDA-capable NVIDIA GPU. The default
demonstration opens a microwave.

```bash
embodichain run-example sim/robocasa365/replay
```

In a source checkout, the script can also be launched with
`python examples/sim/robocasa365/replay.py`.

The first run downloads the selected episode and its robot, kitchen and object
assets from the public RoboCasa365 and RoboSuite sources. It fetches the needed
archive members, checks their SHA-256 digests and converts the scene to USD
locally. The converter is installed in an isolated Python environment in the
cache. Later runs can use prepared episodes offline. The cache is under
`$EMBODICHAIN_DATA_ROOT/robocasa365` (by default
`~/.cache/embodichain_data/robocasa365`). Use `--cache-dir` for another disk.

| Task | Episode | Layout | Style |
| --- | ---: | ---: | ---: |
| `CollectWashingSupplies` | 93 | 13 | 28 |
| `RefillCondimentStation` | 30 | 15 | 49 |
| `TongBuffetSetup` | 39 | 48 | 39 |
| `OpenMicrowave` | 86 | 56 | 15 |
| `SetBowlsForSoup` | 30 | 51 | 41 |
| `LoadPreparedFood` | 34 | 57 | 12 |

```bash
embodichain run-example sim/robocasa365/replay \
  --task CollectWashingSupplies --report CollectWashingSupplies.json
```

The headless example resets the original PandaOmron and kitchen to the
demonstration's initial state. It follows the recorded joint trajectory and
gripper inputs with effort control. Doors, drawers and transported objects
respond to contact in the Newton backend through DexUni's CUDA MJ Warp branch.
The JSON report includes the measured base movement and task success predicates.
A failed predicate exits with status 1.

## Replay an existing bundle

```bash
embodichain run-example sim/robocasa365/replay \
  --episode-dir /path/to/prepared/CollectWashingSupplies \
  --report CollectWashingSupplies.json
```

A prepared directory contains `KitchenEpisode.usdc`, `Textures/`,
`reference.npz` and `episode.json`. Automatic preparation also copies the
[third-party credits and notices](ATTRIBUTION.md). Older bundles with a
CPU-specific collision setting are adapted in a temporary USD stage for
CUDA replay.

For preparation from downloaded archives in a source checkout, see
`python scripts/tools/prepare_robocasa365.py --help`. The offline tool takes
the original LeRobot archive (including `extras/`), the matching RoboCasa and
RoboSuite asset roots and the path to an isolated Python environment with the
converter installed.

## Source index

[`_sources.json`](_sources.json) pins the source archive URLs and sizes and the
SHA-256 digest of each downloaded member after decompression. The offsets
select file data inside the upstream TAR and ZIP archives. RoboSuite files use
paths under the pinned commit. When updating a source revision, regenerate
offsets and hashes against the original archives and verify the resulting
episodes before changing the index.
