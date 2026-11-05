# Sharpa Wave 01 assets

`SharpaWave` registers the official left/right hand descriptions as an `eef`
asset. Each hand has 22 active joints. This is an asset-only integration;
it does not register a RobotCfg, controller, tactile model or RL task.

- **Manufacturer and original publisher:** [sharpa-robotics](https://github.com/sharpa-robotics).
- **Pinned source:** [`0d19cac602f46456b819e4b6a2c09a74982c9a3e`](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml/tree/0d19cac602f46456b819e4b6a2c09a74982c9a3e).
- **Model:** Sharpa Wave 01, left and right hands.
- **Asset license:** Apache-2.0; original LICENSE.txt and NOTICE.txt are retained.
- **Preparation:** The two URDF mesh URI prefixes `package://left_sharpa_wave/` and `package://right_sharpa_wave/` become relative paths. The original URDFs remain under `source/`. Joint/link names, frames, limits, inertials, mesh bytes and MJCF files are unchanged.
- **Bundle:** `eef_assets/SharpaWaveV1.zip`; cache prefix `SharpaWaveV1`.
- **Archive MD5:** `6a03c433d16cfc307e0c8dda2b70006e`.
- **Archive SHA256:** `238bf8d7cd93b65bc80dad8c7eb4f0d57493eff17ba9f7f5bc4c3605434fcf36`.
- **Evidence:** `manifest.json` records source paths, upstream Git blob IDs,
  original/prepared SHA256 hashes and joint order. `ATTRIBUTION.md` records
  the preparation changes. `UPSTREAM_README.md` preserves publisher guidance.

The downloader uses `EMBODICHAIN_DOWNLOAD_PREFIX` first and the official
`DexForceAI/embodichain_data` Hub as fallback. The versioned archive is
[published on Hugging Face](https://huggingface.co/datasets/DexForceAI/embodichain_data/blob/541650b05674ead90b58e9edcac650e87d1a35a6/eef_assets/SharpaWaveV1.zip).

```python
from embodichain.data import get_data_path

right_urdf = get_data_path("SharpaWave/right_sharpa_wave/right_sharpa_wave.urdf")
left_urdf = get_data_path("SharpaWave/left_sharpa_wave/left_sharpa_wave.urdf")
right_mjcf = get_data_path("SharpaWave/right_sharpa_wave/right_sharpa_wave.xml")
```

```bash
embodichain data download --name SharpaWave
```

The URDF mesh paths resolve relative to the selected model. Native joint names,
limits, transforms and inertials are preserved. The packaged MJCF is an
upstream resource; EmbodiChain's standard articulation interface consumes URDF.
Descriptions and declared dynamics do not establish hardware calibration or
contact-rich manipulation success in either simulation backend.
