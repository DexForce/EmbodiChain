# Frozen motion inputs

The published JSON/NPZ bundles have moved to the versioned Hugging Face demo resource:
[demo/real2sim_carton](https://huggingface.co/datasets/DexForceAI/embodichain_data/tree/main/demo/real2sim_carton).
The original per-file hashes remain in [manifest.json](manifest.json).

```bash
embodichain data download --name Real2SimCarton
```

```python
from pathlib import Path
from embodichain.data import get_data_path
from embodichain.data_pipeline.real2sim import load_prototype

inputs = Path(get_data_path("Real2SimCarton/inputs"))
measured_pick = load_prototype(inputs / "measured/0/prototype.json")
```

Indices 0/1/2 identify PickUp, MoveHeldObject and Place. Measured derivatives
retain DROID CC BY 4.0 attribution; independently synthetic inputs retain their
Apache-2.0 terms. The resource also includes byte-identical raw recording files,
manual annotations and typed source evidence. Loading a frozen prototype alone
does not qualify a live scene or create a confirmed collection episode.

The official task uses only `measured/`. Historical synthetic files in the immutable v1 archive are not formal motion sources or collected candidates.
