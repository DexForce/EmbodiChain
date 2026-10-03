# Rendering benchmark

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

`fps` describes complete render batches; `camera_env_fps` counts individual
camera images across all environments. Quality metrics compare each mode with
the configured reference mode. If the reference is omitted from the evaluated
modes, the first mode is used. These are consistency metrics, not a comparison
with a ground-truth image. Timing excludes world/scene construction and scene
state updates and includes rendering, GPU completion, and image capture.
