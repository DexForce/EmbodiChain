# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Run heterogeneous benchmark cases in lockstep in one PhysX scene."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from copy import copy
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

__all__ = []


def _grid_order(jobs):
    """Walk diagonals of the object/hand product to mix both axes in each page."""
    objects = {}
    hands = {}
    for spec, _, _ in jobs:
        key = (spec["category"], spec["object"])
        objects.setdefault(key, len(objects))
        hands.setdefault(spec["gripper"], len(hands))

    def key(job):
        spec, method, _ = job
        h = hands[spec["gripper"]]
        o = objects[spec["category"], spec["object"]]
        return spec["seed"], method, spec["placement"], (o - h) % len(objects), h

    return sorted(jobs, key=key)


def _advance_trials(sim, trials, view_extent):
    """Write every active hand's targets before advancing the common scene clock."""
    from embodichain.lab.sim.grasp_validation import DT

    results = {}
    active = trials.copy()
    while active:
        began = time.monotonic()
        for index, trial in list(active.items()):
            try:
                next(trial)
            except StopIteration as done:
                results[index] = done.value
                del active[index]
        if active:
            if view_extent is not None and sim.viewer is None:
                if not sim.open_viewer():
                    raise RuntimeError("Could not open the grasp grid window")
                sim.viewer.set_look_at(
                    [0, -view_extent, view_extent], [0, 0, 0.05], [0, 0, 1]
                )
            if view_extent is not None and not sim.is_viewer_open:
                raise KeyboardInterrupt("Grasp grid window closed")
            sim.update(step=1)
            if view_extent is not None:
                time.sleep(max(0, DT - (time.monotonic() - began)))
    return results


def _offset_target(cfg, x, y):
    """Compose a grid translation before the effective configured or USD root pose."""
    from embodichain.lab.sim.spawn.descriptors import _pose_from_cfg

    cfg = cfg.copy()
    source_mode = cfg.source.pose_mode if cfg.source else "auto"
    mode = cfg.source_pose_mode
    if source_mode != "auto":
        if mode not in ("auto", source_mode):
            raise ValueError("Conflicting USD source pose modes")
        mode = source_mode
    inherit = mode == "preserve" or (
        mode == "auto"
        and cfg.init_local_pose is None
        and not np.any(cfg.init_pos)
        and not np.any(cfg.init_rot)
    )
    pose = np.eye(4) if inherit else _pose_from_cfg(cfg).copy()
    pose[:2, 3] += (x, y)
    cfg.init_local_pose = pose
    cfg.init_pos = tuple(pose[:3, 3])
    cfg.source_pose_mode = "offset" if inherit or mode == "offset" else "config"
    if cfg.source:
        cfg.source.pose_mode = "auto"
    return cfg


def _prepare_stations(sim, temp, pending, args):
    import torch
    from embodichain.lab.sim.grasp_validation import (
        _GraspStation,
        _settled_poses,
    )

    if not args.headless:
        from embodichain.lab.sim.cfg import VisualObjectCfg
        from embodichain.lab.sim.shapes import CubeCfg

        sim.add_visual_object(
            VisualObjectCfg(
                uid="grid_table",
                shape=CubeCfg(
                    size=(
                        args.grid_side * args.grid_spacing,
                        args.grid_side * args.grid_spacing,
                        0.1,
                    )
                ),
                init_pos=(0.0, 0.0, -0.05),
            )
        )
    stations = {}
    for i, (_, options, x, y) in pending.items():
        directory = Path(temp) / str(i)
        directory.mkdir()
        # Only the scene owner opens the viewer and advances physics.
        stations[i] = _GraspStation(
            sim,
            directory=str(directory),
            uid_suffix=f"_{i}",
            mount_xy=(x, y),
            **(
                options
                | {
                    "headless": True,
                    "object_cfg": _offset_target(options["object_cfg"], x, y),
                }
            ),
        )
    sim.prepare()
    placements = {}
    rotated = []
    for i, station in stations.items():
        # PhysX suppresses non-global pairs with distinct filter group IDs.
        group = torch.tensor([[i + 1, 1, 0, 0]], dtype=torch.int32)
        station.support.set_collision_filter(group)
        station.hand.set_collision_filter(group)
        station.obj.set_collision_filter(group)
        # Native scene preparation can advance overlapping targets before the
        # filters exist. Start the settling window from the declared USD/config pose.
        placements[i] = station._tensor(station.obj.cfg.init_local_pose)[None].clone()
        station._reset_target(placements[i])
        station._read_geometry()
        if station._rotation is not None:
            placements[i] = station._separate_rotated_target(station._rotation)
            rotated.append(i)
    if rotated:
        sim.update(step=2)
        for i in rotated:
            stations[i]._reset_target(placements[i])
    initial = _settled_poses(sim, stations.values())
    for station, pose in zip(stations.values(), initial):
        station.initial = pose.clone()
        station._bind_trial(args.physics)
    return stations, placements


def _resettle_targets(sim, active, stations, placements):
    """Rebuild the resting contacts after the preceding grasp moved the objects."""
    import torch
    from embodichain.lab.sim.grasp_validation import _settled_poses

    for i in active:
        station = stations[i]
        placement = placements[i]
        parked = torch.eye(4, device=placement.device, dtype=placement.dtype)
        parked[:2, 3] = placement[0, :2, 3]
        parked[2, 3] = 1.5
        station.hand.set_local_pose(parked[None])
        separated = placement.clone()
        separated[:, 2, 3] += 0.5
        station._reset_target(separated)
    sim.update(step=2, render_final_step=False)
    for i in active:
        stations[i]._reset_target(placements[i])
    _settled_poses(sim, [stations[i] for i in active])


def _run_candidates(sim, stations, placements, pending, cases, args):
    view_extent = None if args.headless else args.grid_side * args.grid_spacing
    replies = stations.copy()
    first_batch = True
    while replies:
        trials = {}
        for i, reply in replies.items():
            try:
                batch = pending[i][0].send(reply)
            except StopIteration as done:
                cases[i] = done.value
                continue
            trials[i] = stations[i]._trial_steps(batch)
        if trials:
            if not first_batch:
                _resettle_targets(sim, trials, stations, placements)
            first_batch = False
            print(f"Grid: {len(trials)} simultaneous grasp trials", flush=True)
        replies = _advance_trials(sim, trials, view_extent)


def _run_page(jobs, root, args):
    from .cli import _case_trials

    side = args.grid_side
    pitch = args.grid_spacing
    cases = {}
    pending = {}
    for i, (spec, method, source) in jobs:
        row, col = divmod(i, side)
        x = (col - (side - 1) / 2) * pitch
        y = ((side - 1) / 2 - row) * pitch
        cell = dict(
            capacity=args.grid,
            rows=side,
            columns=side,
            spacing_m=pitch,
            index=i,
            xy_m=[x, y],
            support_size_m=4.0,
        )
        work = _case_trials(spec, method, source, root, args.output, args, cell=cell)
        try:
            options = next(work)
        except StopIteration as done:
            cases[i] = done.value
            continue
        pending[i] = (work, options, x, y)
    if pending:
        from embodichain.lab.sim.grasp_validation import (
            SimulationManager,
            _validation_scene,
        )

        with TemporaryDirectory() as temp:
            sim = _validation_scene(1, args.physics, args.headless)
            try:
                stations, placements = _prepare_stations(sim, temp, pending, args)
                _run_candidates(sim, stations, placements, pending, cases, args)
            finally:
                sim.destroy(exit_process=False)
                SimulationManager.flush_cleanup_queue()
    return [cases[i] for i, _ in jobs]


def run_grid(jobs: list, root: Path, args: argparse.Namespace) -> list[dict]:
    """Run bounded pages and return ordinary case records for reporting."""
    ordered = _grid_order(jobs)
    capacity = args.grid
    cases = []
    pages = deque(
        (list(enumerate(ordered[start : start + capacity])), args.resume)
        for start in range(0, len(ordered), capacity)
    )
    pool = None
    try:
        while pages:
            page, resume = pages.popleft()
            run_args = copy(args)
            run_args.resume = resume
            if pool is None:
                pool = ProcessPoolExecutor(
                    max_workers=1, mp_context=multiprocessing.get_context("spawn")
                )
            print(
                f"Preparing {len(page)} grid cells ({len(cases)}/{len(ordered)} cases complete)",
                flush=True,
            )
            try:
                results = pool.submit(_run_page, page, root, run_args).result()
            except Exception as exc:
                if isinstance(exc, BrokenProcessPool):
                    pool.shutdown()
                    pool = None
                if len(page) == 1:
                    _, (spec, method, _) = page[0]
                    print(
                        f"Grid case {spec['gripper']} / {spec['object']}: {exc}",
                        flush=True,
                    )
                    cases.append(
                        dict(spec, algorithm=method, error=str(exc), count=0, trials=[])
                    )
                    continue
                # Keep original cell coordinates and checkpoints when isolating a bad asset.
                middle = len(page) // 2
                print(f"Retrying smaller groups after grid error: {exc}", flush=True)
                pages.appendleft((page[middle:], True))
                pages.appendleft((page[:middle], True))
            else:
                cases.extend(results)
    finally:
        if pool is not None:
            pool.shutdown()
    return cases
