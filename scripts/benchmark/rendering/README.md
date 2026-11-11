# Rendering benchmark

Compare native group sharing for three 640×480 RGB-only cameras in the fixed
scene:

```bash
python -m scripts.benchmark.rendering.camera_grouping \
  --config scripts/benchmark/rendering/settings/three_rgb_640x480.yaml
```

The paired run alternates three independent groups with one shared group at
each configured environment count. It reports completed RGB camera images per
second (`frames × environments × 3 / seconds`), completed render and fetch
latency, sampled GPU memory, and a corresponding-view image comparison. Scene
state updates and initialization are outside the timed window. Raw worker
results and the report are under `outputs/benchmarks/camera_grouping_<timestamp>/`.
The paired runner refuses to start while another GPU compute process is present
and checks again between workers. Use `--allow-shared-gpu` only for exploratory
runs whose throughput will be reported as shared-machine measurements.

The rendering benchmark compares configured offscreen rendering settings using
one deterministic reflective scene. The scene is deliberately fixed today;
the scene builder and its name provide the extension point for future scenes.

Run the default configuration:

```bash
embodichain benchmark rendering --config scripts/benchmark/rendering/settings/default.yaml
```

CLI values override the configuration file. For example:

```bash
embodichain benchmark rendering \
  --config scripts/benchmark/rendering/settings/default.yaml \
  --camera-count 3 --num-envs 8 --resolution 1280x720 \
  --denoising-modes off optix dlss nrd
```

Run one combined environment-count sweep:

```bash
embodichain benchmark rendering \
  --resolution 640x480 --camera-count 1 \
  --env-sweep 1 2 4 8 16 32 64 128 \
  --denoising-modes optix dlss nrd
```

`camera_count` is the number of camera views per environment. `num_envs` is the
number of cloned environments. All modes use the same camera trajectories and
scene animation. The configuration supports JSON and YAML and defines the
renderer, samples per pixel, warmup and measurement frame counts, quality frame
count, DLSS quality, render GPU, and runtime device.

Each run creates an `outputs/benchmarks/rendering_<timestamp>/` directory with:

- `benchmark_settings.json`: resolved settings for reproducing the run.
- `<mode>_performance.json`: raw timing and memory measurements.
- `<mode>_frames.npz`: quality frames in
  `(frames, environments, cameras, height, width, RGBA)` order.
- `comparison.png`: mode images and amplified differences for every camera
  in the first environment.
- `report.md`: the benchmark settings and the existing three-table report
  contract (`Time & Memory`, `Success & Other Metrics`, and `Leaderboard`).

For an environment sweep, the report keeps one row per environment count and
mode, while the leaderboard averages each mode across the sweep. Individual
run reports and raw artifacts are under `runs/`.

Quality aggregation retains the reference and at most one candidate sequence.
Temporal errors use adjacent images from one environment/camera at a time,
with reusable float32 buffers and float64 reduction, instead of converting
whole sequences to floating point. Comparison images copy only the selected
views and release their source sequences. NPZ archives still decompress a
complete sequence, so raw-image RAM scales with the resolution, environment
count, camera count, and number of quality frames; three-mode aggregation
does not retain all three sequences simultaneously.

`fps` describes complete render batches; `camera_env_fps` counts individual
camera images across all environments. Quality metrics compare each mode with
the configured reference mode. If the reference is omitted from the evaluated
modes, the first mode is used. Comparison images use that same resolved
reference, and their columns follow the configured denoising-mode order.
These are consistency metrics, not a comparison
with a ground-truth image. Timing excludes world/scene construction and scene
state updates and includes rendering, GPU completion, and image capture.

Sweep output-path notes describe possible device-dependent behavior. Actual
renderer warnings appear in worker console output; the report does not infer
that a warning occurred from environment counts alone.
