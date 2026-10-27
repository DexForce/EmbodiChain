# Copyright (c) 2026 Zhiyang Dou. SPDX-License-Identifier: MIT
# Author reference evaluator, pinned at 160fe0852c51cffae7b7440452b58d0b49db6df6.
# Import mapping only; full attribution is in the model THIRD_PARTY_LICENSES.

"""
Rollout evaluator computing RMSE metrics on test/validation sets.

Metrics (HopNet-aligned):
- Position RMSE: area-weighted center of mass of the predicted mesh vs. the
  ground-truth trajectory, in meters.
- Rotation error: quaternion angular distance in degrees, obtained by Kabsch
  alignment of predicted vertices against the canonical (frame-0) mesh.
- Checkpoints: physical frames 25, 50, 75, 100, 200, 300, 400.

Frame convention:
- Checkpoints use direct array indexing and include the ground-truth warmup
  frames ("@25 steps" = array index 25), matching HopNet. The warmup length
  is configured via eval_warmup_frames.
- Checkpoints are evaluated only on the step-size grid: physical frame f is
  network step f / skip_step_size. A checkpoint between two predicted frames,
  or at one of the given warm-up (context) frames, is not evaluated and is
  reported as "/" (step size 10: @25 and @75).

Degenerate meshes (zero total surface area) raise a detailed error rather
than being skipped or regularized with an epsilon.
"""

from __future__ import annotations

import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
import trimesh
import roma  # For rotation matrix to quaternion conversion
from embodichain_models.dynamics.rigidformer._native.data.scene_frames import load_frame

logger = logging.getLogger(__name__)


def _volume_center_of_mass(V: np.ndarray, F: np.ndarray):
    """Centre of mass of closed triangle meshes V (..., Nv, 3) with faces F (Nf, 3).

    Signed-tetrahedron decomposition about the origin: identical to trimesh's
    ``Trimesh.center_mass`` (the same divergence-theorem integrals), vectorised over
    leading dimensions. Returns (com (..., 3), volume (...)).
    """
    v0 = V[..., F[:, 0], :]
    v1 = V[..., F[:, 1], :]
    v2 = V[..., F[:, 2], :]
    det = np.einsum("...i,...i->...", v0, np.cross(v1, v2))  # 6 x signed tetra volume
    vol = det.sum(axis=-1) / 6.0
    first_moment = (det[..., None] * (v0 + v1 + v2)).sum(axis=-2) / 24.0
    return first_moment / vol[..., None], vol


def _hopnet_object_poses(
    pred_world: np.ndarray, ref_positions_list, faces_list, Nobj: int
):
    """HopNet-aligned object poses from predicted vertices (utils/rollout.py::shape_matching).

    For each object: qi = canonical vertices - canonical centre of mass, pi = predicted vertices -
    predicted centre of mass, Apq = mean_i pi qi^T, R = polar(Apq) = U V^T, and the object origin
    t = com_pred - R @ com_canonical. Returns (positions (T, Nobj, 3), rotations list of (T, 3, 3)).
    """
    T = pred_world.shape[0]
    positions = np.zeros((T, Nobj, 3), dtype=np.float64)
    rotations = []
    for obj_idx in range(Nobj):
        C = np.asarray(
            ref_positions_list[obj_idx], dtype=np.float64
        )  # (Nv, 3) canonical
        F = np.asarray(faces_list[obj_idx])
        Nv = C.shape[0]
        P = np.asarray(pred_world[:, obj_idx, :Nv, :], dtype=np.float64)  # (T, Nv, 3)
        com_c, vol_c = _volume_center_of_mass(C, F)
        if not np.isfinite(vol_c) or abs(vol_c) < 1e-12:
            raise ValueError(
                f"HopNet position estimator: object {obj_idx} has a degenerate volume ({vol_c}); "
                f"the collision mesh must be closed."
            )
        com_w, _ = _volume_center_of_mass(P, F)  # (T, 3)
        q = C - com_c  # (Nv, 3)
        pmat = P - com_w[:, None, :]  # (T, Nv, 3)
        Apq = np.einsum("tni,nj->tij", pmat, q) / Nv  # (T, 3, 3)
        U, _, Vt = np.linalg.svd(Apq)
        R = U @ Vt  # polar decomposition (as HopNet)
        positions[:, obj_idx, :] = com_w - np.einsum("tij,j->ti", R, com_c)
        rotations.append(R)
    return positions.astype(np.float32), rotations


class Evaluator:
    """
    Evaluator for computing RMSE metrics on test/validation sets.

    Key features:
    - Batch-wise evaluation (no drop_last, processes all samples)
    - HopNet-aligned checkpoints (@25, @50, etc.)
    - TensorBoard logging
    - Cached dataset indexing (reuse across epochs)
    """

    def __init__(self, trainer):
        """
        Initialize evaluator with reference to trainer.

        Args:
            trainer: Main trainer instance (provides model, config, device, etc.)
        """
        self.trainer = trainer
        self.config = trainer.config
        # data.object_scale_file: the canonical frame-0 meshes read below must carry the same scale as the model inputs
        from embodichain_models.dynamics.rigidformer._native.data.object_scale import (
            from_config as _object_scale_from_config,
        )

        self._object_scale = _object_scale_from_config(self.config)
        # data.partial_view: the inputs are partial point clouds of a simulated depth camera (src/data/partial_view.py);
        # each observed object's rigid motion moves its full collision mesh before scoring, unobserved objects are
        # left out of the metrics
        _pv = self.config["data"].get("partial_view") or {}
        self._partial_view = dict(_pv) if _pv.get("enable", False) else None
        self._partial_counts = (
            0,
            0,
        )  # observed objects, all objects (last evaluated split and step size)
        # Unwrap DDP wrapper to access custom methods (e.g., rollout)
        self.model = (
            trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
        )
        self.device = trainer.device
        self.writer = trainer.writer
        self.log_dir = trainer.log_dir
        self.epoch = 0  # Updated by trainer before calling evaluate

        # Metrics configuration
        metrics_cfg = self.config["training"].get("metrics_test", {})
        self.metrics_enabled = metrics_cfg.get("enable", False)
        self.print_every_n_epochs = metrics_cfg.get("print_every_n_epochs", 5)

        # Evaluation settings
        self.eval_num_frames = metrics_cfg.get("eval_num_frames", 480)
        self.eval_warmup_frames = metrics_cfg.get(
            "eval_warmup_frames", 2
        )  # Ground-truth frames before autoregressive rollout
        self.hopnet_checkpoints_physical = metrics_cfg.get(
            "hopnet_checkpoints", [25, 50, 75, 100, 200, 300, 400]
        )
        # Step sizes > 1 start from the model's own step-size-1 rollout instead of ground-truth warm-up frames
        # (_warmup_from_step1): every step size then sees only the ground truth of the step-size-1 protocol
        self.warmup_from_step1 = bool(metrics_cfg.get("warmup_from_step1", False))
        if self.warmup_from_step1 and self._partial_view is not None:
            raise ValueError(
                "metrics_test.warmup_from_step1 is not defined for partial-view inputs (the observation "
                "frame differs between step sizes)"
            )
        # Object-pose extraction from predicted vertices.
        #   'area_centroid': area-weighted surface centroid vs. the ground-truth origin.
        #   'hopnet':        HopNet's shape matching: volume centre of mass + polar-decomposition
        #                    rotation, object origin = com_pred - R @ com_canonical (exact for a perfect
        #                    prediction, no constant offset).
        self.position_estimator = str(
            metrics_cfg.get("position_estimator", "area_centroid")
        )
        if self.position_estimator not in ("area_centroid", "hopnet"):
            raise ValueError(
                f"metrics_test.position_estimator must be 'area_centroid' or 'hopnet'; got {self.position_estimator}"
            )

        # Cached datasets (per skip_step_size)
        self._test_eval_datasets = {}
        self._val_eval_datasets = {}

        # DDP info
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _protocol_text(self, split: str, n_scenes: int) -> str:
        """What every reported number of this evaluation depends on besides the checkpoint. A split file derived for
        evaluating another partition through the test list (scripts/eval_cross_dataset.py --split val) names that
        partition in 'evaluated_partition'."""
        d = self.config["data"]
        with open(d["split_file"]) as f:
            partition = json.load(f).get("evaluated_partition", split)
        text = (
            f"position estimator {self.position_estimator}; scene set {d.get('scene_set', 'benchmark')} "
            f"({os.path.basename(str(d.get('split_file')))}), {n_scenes} {partition} scenes; "
            f"rotation error = RMS over objects, rotation MAE = mean absolute angle over objects"
        )
        if self.warmup_from_step1:
            text += (
                f"; step sizes > 1: warm-up frames s .. {self.eval_warmup_frames - 1}s predicted by the model at "
                f"step size 1 from ground-truth frames 0-{self.eval_warmup_frames - 1} (only frame 0 is ground "
                f"truth at step size s)"
            )
        if self._partial_view is not None:
            from src.data.partial_view import PartialView

            pv = PartialView(
                {**self._partial_view, "observe_index": self.eval_warmup_frames - 1},
                Path(__file__).resolve().parents[2],
            )
            seen, total = self._partial_counts
            text += (
                f"; inputs: {pv.describe(train=False)}; metrics over the {seen} observed of {total} objects "
                f"(full collision meshes moved by the predicted rigid motions)"
            )
        return text

    def should_evaluate(self, epoch: int) -> bool:
        """Check if RMSE evaluation should run this epoch."""
        if not self.metrics_enabled:
            return False
        # Evaluate at epoch 0 (initial baseline) and every N epochs thereafter
        return (epoch % self.print_every_n_epochs) == 0

    def evaluate_rollout_rmse(self, epoch: int):
        """
        Main entry point for RMSE evaluation.

        Evaluates on validation and/or test sets based on config.
        Multi-timescale aware: evaluates at each step_size in eval_step_sizes.

        Called by trainer every N epochs.

        Args:
            epoch: Current training epoch
        """
        if not self.should_evaluate(epoch):
            return

        self.epoch = epoch
        if self.rank == 0:
            logger.info(f"{'='*80}")
            logger.info(f"[RMSE Evaluation] Epoch {epoch}")
            logger.info(f"{'='*80}")

        # Get split file
        split_file = self.config["data"].get("split_file", "dataset_split.json")
        if not os.path.exists(split_file):
            if self.rank == 0:
                logger.warning(
                    f"Split file not found: {split_file}, skipping RMSE evaluation"
                )
            return

        with open(split_file, "r") as f:
            split_info = json.load(f)

        val_datasets = split_info.get("val_datasets", [])
        test_datasets = split_info.get("test_datasets", [])

        # Multi-timescale evaluation: loop over the configured eval_step_sizes,
        # falling back to the single skip_step_size when unset
        eval_step_sizes = self.config["data"].get(
            "eval_step_sizes", [self.config["data"].get("skip_step_size", 1)]
        )

        if self.rank == 0:
            logger.info(
                f"[RMSE Evaluation] Evaluating at step_sizes: {eval_step_sizes}"
            )

        all_test_metrics = {}  # Aggregate test metrics across all step_sizes

        for step_size in eval_step_sizes:
            if self.rank == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"[RMSE Evaluation] Evaluating at step_size = {step_size}")
                logger.info(f"{'='*80}")

            # Evaluate validation set at this step_size
            enable_val = self.config["data"].get("enable_val_split", False)
            if enable_val and len(val_datasets) > 0:
                if self.rank == 0:
                    logger.info(
                        f"[RMSE Evaluation] Validation set: {len(val_datasets)} datasets (step_size={step_size})"
                    )
                self._evaluate_rmse_on_split(
                    val_datasets, split="val", step_size=step_size
                )

            # Evaluate test set at this step_size (always evaluate if exists)
            if len(test_datasets) > 0:
                if self.rank == 0:
                    logger.info(
                        f"[RMSE Evaluation] Test set: {len(test_datasets)} datasets (step_size={step_size})"
                    )
                test_metrics_ss = self._evaluate_rmse_on_split(
                    test_datasets, split="test", step_size=step_size
                )

                # Aggregate test metrics (already include step_size suffix from _evaluate_rmse_on_split)
                for metric_name, value in test_metrics_ss.items():
                    all_test_metrics[metric_name] = value

        if self.rank == 0:
            logger.info(
                f"\n[RMSE Evaluation] Epoch {epoch} completed for all step_sizes"
            )

        # Return aggregated test metrics for best checkpoint selection
        return all_test_metrics

    def _evaluate_rmse_on_split(
        self, dataset_names: List[str], split: str, step_size: int
    ):
        """
        Evaluate RMSE on a split (val or test) using batched processing.

        DDP-aware: Each rank processes a shard of data, then Rank 0 gathers results.

        Args:
            dataset_names: List of dataset names to evaluate
            split: 'val' or 'test'
            step_size: Step size for frame sampling (1, 2, 5, 10, etc.)
        """
        from embodichain_models.dynamics.rigidformer._native.data.dataset import (
            MultiObjectDynamics,
            make_collate_fn,
        )
        import torch.distributed as dist

        skip_step_size = step_size

        # Calculate sequence length for evaluation (network steps)
        eval_sequence_length = self.eval_num_frames // skip_step_size

        # Only Rank 0 logs dataset creation info
        if self.rank == 0:
            logger.info(
                f"[RMSE Dataset] Creating {split} dataset: "
                f"physical_frames={self.eval_num_frames}, skip_step={skip_step_size}, "
                f"sequence_length={eval_sequence_length}"
            )

            # Display the checkpoints: evaluated on the step-size grid after the warm-up frames, "/" otherwise
            logger.info(
                f"\n[Checkpoints] skip_step_size={skip_step_size}, warm-up frames 0 to {self.eval_warmup_frames - 1}"
            )
            logger.info(
                f"{'Checkpoint':^12} | {'Network Step':^13} | {'Rollout predictions':^20}"
            )
            logger.info("-" * 52)
            for physical_cp in self.hopnet_checkpoints_physical:
                if physical_cp % skip_step_size:
                    logger.info(
                        f"{physical_cp:^12} | {'-':^13} | {'/ (not on the grid)':^20}"
                    )
                elif physical_cp // skip_step_size < self.eval_warmup_frames:
                    logger.info(
                        f"{physical_cp:^12} | {physical_cp // skip_step_size:^13} | {'/ (warm-up frame)':^20}"
                    )
                else:
                    network_step = physical_cp // skip_step_size
                    logger.info(
                        f"{physical_cp:^12} | {network_step:^13} | {network_step - self.eval_warmup_frames + 1:^20}"
                    )
            logger.info("")

        eval_dataset = self._get_eval_dataset(
            split, skip_step_size, eval_sequence_length
        )
        batch_size = self.config["training"].get("eval_batch_size", 8)
        eval_loader = self._eval_loader(eval_dataset, batch_size)
        # metrics_test.warmup_from_step1: for a step size s > 1 the warm-up frames s, 2s, ... are the model's own
        # step-size-1 rollout from the step-size-1 warm-up (ground-truth frames 0 .. warm-up - 1), instead of
        # ground-truth frames; every step size then uses exactly the ground truth of the step-size-1 protocol
        step1_loader = None
        if self.warmup_from_step1 and skip_step_size > 1:
            step1_loader = self._eval_loader(
                self._get_eval_dataset(split, 1, self.eval_num_frames), batch_size
            )
            if len(step1_loader) != len(eval_loader):
                raise ValueError(
                    f"warmup_from_step1: {len(step1_loader)} step-size-1 batches vs {len(eval_loader)}"
                )

        if self.rank == 0:
            logger.info(
                f"[RMSE Dataset] {split} set: {len(eval_dataset)} samples, "
                f"batch_size={batch_size}, {len(eval_loader)} batches"
                + (
                    ", warm-up from the step-size-1 rollout"
                    if step1_loader is not None
                    else ""
                )
            )

        return self._evaluate_rmse_loop(
            eval_dataset,
            eval_loader,
            step1_loader,
            split,
            skip_step_size,
            eval_sequence_length,
        )

    def _get_eval_dataset(
        self, split: str, skip_step_size: int, eval_sequence_length: int
    ):
        """The cached evaluation dataset of a split at a step size (one sample per scene, warm-up frames only)."""
        from embodichain_models.dynamics.rigidformer._native.data.dataset import (
            MultiObjectDynamics,
        )

        # Create or reuse cached dataset
        cache_key = f"_eval_dataset_{split}_step{skip_step_size}"
        if cache_key not in (
            self._test_eval_datasets if split == "test" else self._val_eval_datasets
        ):
            if self.rank == 0:
                logger.info(f"[RMSE Dataset] Creating new {split} dataset (first time)")

            # Load only the warmup frames; rollout() generates the remaining
            # predictions autoregressively.
            eval_load_frames = max(
                self.eval_warmup_frames, 8
            )  # At least 8 frames for safety margin

            if self.rank == 0:
                logger.info(
                    f"[RMSE Eval] Loading only {eval_load_frames} frames (not {eval_sequence_length})"
                )

            eval_dataset = MultiObjectDynamics(
                data_path=self.config["data"]["dataset_path"],
                sequence_length=eval_load_frames,  # Load only the warmup frames
                stride=999999,  # Large stride yields one sample per dataset (no sliding windows)
                skip_step_size=skip_step_size,
                split=split,
                split_file=self.config["data"].get("split_file", "dataset_split.json"),
                max_vertices_per_object=self.config["data"].get(
                    "max_vertices_per_object", 1150
                ),
                max_objects_per_frame=self.config["data"].get(
                    "max_objects_per_frame", 20
                ),
                anchor_k=self.config["data"].get("anchor_k", 32),
                if_ground_mesh=self.config["data"].get("if_ground_mesh", False),
                ground_mesh_vertex_num=self.config["data"].get(
                    "ground_mesh_vertex_num", 968
                ),
                if_debug_normalization_log=False,  # Disable verbose logging during evaluation
                random_step_size={
                    "enable": False
                },  # Disable random sampling for evaluation
                augmentation={"enable": False},  # Disable augmentation for evaluation
                anchor_fps_start=self.config["data"].get("anchor_fps_start", "index0"),
                fps_order=self.config["data"].get("fps_order", False),
                object_scale_file=self.config["data"].get("object_scale_file"),
                # the same physics-parameter validation as the training data (missing entries are errors)
                if_phys_param=self.config.get("model", {}).get("if_phys_param", False),
                # partial inputs observed at the last warm-up frame
                partial_view=(
                    {**self._partial_view, "observe_index": self.eval_warmup_frames - 1}
                    if self._partial_view is not None
                    else None
                ),
            )

            # Cache the dataset
            if split == "test":
                self._test_eval_datasets[cache_key] = eval_dataset
            else:
                self._val_eval_datasets[cache_key] = eval_dataset
        else:
            if self.rank == 0:
                logger.info(f"[RMSE Dataset] Reusing cached {split} dataset")
            eval_dataset = (
                self._test_eval_datasets if split == "test" else self._val_eval_datasets
            )[cache_key]
        return eval_dataset

    def _eval_loader(self, eval_dataset, batch_size: int):
        """Evaluation DataLoader: scenes in dataset order, every scene on exactly one rank."""
        from embodichain_models.dynamics.rigidformer._native.data.dataset import (
            make_collate_fn,
        )

        # Create DataLoader with DDP-aware sampler
        collate_fn = make_collate_fn(
            max_objects_per_frame=self.config["data"].get("max_objects_per_frame", 20),
            max_vertices_per_object=self.config["data"].get(
                "max_vertices_per_object", 1150
            ),
        )

        # DDP-aware: every scene on exactly one rank (strided shards without the padding a DistributedSampler adds,
        # which would evaluate some scenes twice whenever the scene count is not a multiple of the rank count). The
        # rollout contains no collective, so ranks may process different numbers of scenes.
        if self.world_size > 1:
            eval_loader = DataLoader(
                torch.utils.data.Subset(
                    eval_dataset,
                    list(range(self.rank, len(eval_dataset), self.world_size)),
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Avoid DDP deadlock: use main process only for evaluation
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        else:
            # Single GPU: no sampler needed
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Single process loading for evaluation
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_fn,
            )
        return eval_loader

    def _warmup_from_step1(
        self, batch: dict, batch1: dict, skip_step_size: int
    ) -> dict:
        """Replace the warm-up frames s, 2s, ... of a step-size-s batch by the model's step-size-1 rollout of the same
        scenes, started from the step-size-1 warm-up (ground-truth frames 0 .. warm-up - 1).
        """
        W = self.eval_warmup_frames
        if list(batch.get("dataset_name", [])) != list(batch1.get("dataset_name", [])):
            raise ValueError(
                f"warmup_from_step1: scene order differs between the step-size-{skip_step_size} and "
                f"step-size-1 evaluation batches"
            )
        frames = [k * skip_step_size for k in range(1, W)]
        out = self.model.rollout(
            initial_batch=batch1,
            num_steps=frames[-1] + 1,
            dt=1.0,
            return_extras=False,
            warmup_frames=W,
        )
        pos = batch["obj_positions"].clone()
        pos[:, 1:W] = out["obj_positions"][:, frames].to(pos.dtype)
        return {**batch, "obj_positions": pos}

    def _evaluate_rmse_loop(
        self,
        eval_dataset,
        eval_loader,
        step1_loader,
        split: str,
        skip_step_size: int,
        eval_sequence_length: int,
    ):
        import torch.distributed as dist

        step1_batches = iter(step1_loader) if step1_loader is not None else None

        # Synchronize all ranks before starting evaluation loop (avoid DDP deadlock)
        if self.world_size > 1:
            if self.rank == 0:
                import sys

                print(f"[RMSE Eval] Rank 0: Entering barrier sync...", flush=True)
                sys.stdout.flush()
            dist.barrier()
            if self.rank == 0:
                print(
                    f"[RMSE Eval] Rank 0: Barrier sync complete, starting eval loop",
                    flush=True,
                )

        # Evaluation loop
        self.model.eval()

        all_position_rmse = {}  # checkpoint -> list of RMSE values
        all_rotation_error = (
            {}
        )  # checkpoint -> list of rotation errors (per-scene RMS over objects)
        all_rotation_mae = (
            {}
        )  # checkpoint -> list of per-scene mean absolute rotation errors over objects
        checkpoint_mapping = (
            {}
        )  # checkpoint -> (evaluated physical frame, warm-up frames) when not (cp, main warm-up)
        not_evaluated = (
            {}
        )  # checkpoint -> reason ("/"): off the step-size grid or at a warm-up (context) frame
        total_samples = 0
        partial_seen, partial_total = (
            0,
            0,
        )  # data.partial_view: observed objects, all objects

        # Flash Attention handles dtype conversion inside the decoder; the rest
        # of the model runs in FP32, so no global autocast is needed.
        with torch.no_grad():
            import sys

            if self.rank == 0:
                print(f"[RMSE Eval] Creating tqdm progress bar...", flush=True)

            pbar = tqdm(
                eval_loader, desc=f"RMSE Eval ({split})", disable=(self.rank != 0)
            )

            if self.rank == 0:
                print(
                    f"[RMSE Eval] tqdm created, entering for loop (num_batches={len(eval_loader)})...",
                    flush=True,
                )
                sys.stdout.flush()

            for batch_idx, batch in enumerate(pbar):
                import sys

                if self.rank == 0:
                    print(
                        f"[RMSE Eval] Batch {batch_idx+1}/{len(eval_loader)} starting...",
                        flush=True,
                    )
                    sys.stdout.flush()

                # Move batch to device
                batch = {
                    k: (
                        v.to(self.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }
                if step1_batches is not None:
                    batch1 = {
                        k: (
                            v.to(self.device, non_blocking=True)
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in next(step1_batches).items()
                    }
                    batch = self._warmup_from_step1(batch, batch1, skip_step_size)

                # Get dataset names for this batch
                batch_dataset_names = batch.get("dataset_name", None)
                if batch_dataset_names is None:
                    logger.warning(
                        f"[RMSE] Batch {batch_idx} missing 'dataset_name', skipping"
                    )
                    continue

                # Autoregressive rollout; return_extras=False skips storing
                # intermediate tensors to reduce memory.
                num_frames = eval_sequence_length
                if self.rank == 0:
                    logger.info(
                        f"[RMSE Eval] Batch {batch_idx+1}: Starting rollout ({num_frames} frames, warmup={self.eval_warmup_frames})..."
                    )

                import time

                rollout_start = time.time()
                rollout_out = self.model.rollout(
                    initial_batch=batch,
                    num_steps=num_frames,
                    dt=1.0,
                    return_extras=False,
                    warmup_frames=self.eval_warmup_frames,
                )
                rollout_time = time.time() - rollout_start

                if self.rank == 0:
                    logger.info(
                        f"[RMSE Eval] Batch {batch_idx+1}: Rollout done in {rollout_time:.1f}s"
                    )

                # Extract predictions
                pred_full = (
                    rollout_out["obj_positions"].cpu().numpy()
                )  # (B, T, Nobj, Nv, 3)
                B_actual = pred_full.shape[0]

                # Process each sample in batch
                for b_idx in range(B_actual):
                    dataset_name = (
                        batch_dataset_names[b_idx]
                        if isinstance(batch_dataset_names, list)
                        else batch_dataset_names[b_idx].item()
                    )
                    pred_single = pred_full[b_idx]  # (T, Nobj, Nv, 3)
                    object_mask = None
                    if self._partial_view is not None:
                        pred_single, object_mask = self._full_from_partial(
                            pred_single, batch, b_idx
                        )
                        partial_seen += int(object_mask.sum())
                        partial_total += int(object_mask.size)

                    # Compute RMSE for this sample
                    rmse_dict = self._compute_rmse_for_sample(
                        pred_single,
                        dataset_name,
                        eval_dataset,
                        skip_step_size,
                        self.hopnet_checkpoints_physical,
                        object_mask=object_mask,
                    )

                    # Aggregate results (position RMSE and rotation error)
                    for checkpoint, value in rmse_dict.items():
                        if checkpoint == "step_size":
                            continue

                        # A checkpoint off the step-size grid or at a warm-up (context) frame: not evaluated
                        if isinstance(checkpoint, str) and checkpoint.startswith(
                            "_na_"
                        ):
                            not_evaluated[int(checkpoint[len("_na_") :])] = value
                            continue

                        # Mapping metadata: the frame / warm-up actually evaluated for a checkpoint
                        if isinstance(checkpoint, str) and checkpoint.startswith(
                            "_mapping_"
                        ):
                            checkpoint_mapping[int(checkpoint[len("_mapping_") :])] = (
                                tuple(value)
                            )
                            continue

                        if isinstance(checkpoint, str) and checkpoint.startswith(
                            "rotation_mae_deg_"
                        ):
                            all_rotation_mae.setdefault(
                                int(checkpoint[len("rotation_mae_deg_") :]), []
                            ).append(value)
                            continue

                        # Check if this is a rotation error key (starts with 'rotation_deg_')
                        if isinstance(checkpoint, str) and checkpoint.startswith(
                            "rotation_deg_"
                        ):
                            # Extract physical frame number from key
                            physical_frame = int(
                                checkpoint.replace("rotation_deg_", "")
                            )
                            if physical_frame not in all_rotation_error:
                                all_rotation_error[physical_frame] = []
                            all_rotation_error[physical_frame].append(value)
                        else:
                            # Position RMSE, keyed by target checkpoint
                            if checkpoint not in all_position_rmse:
                                all_position_rmse[checkpoint] = []
                            all_position_rmse[checkpoint].append(value)

                    total_samples += 1

                # Update progress bar
                pbar.set_postfix({"samples": total_samples})

        # Synchronize all ranks before gathering results (ensure all finished evaluation)
        if self.world_size > 1:
            dist.barrier()

        # DDP: Gather results from all ranks to Rank 0
        if self.world_size > 1:
            # Convert dict of lists to list of dicts for gathering
            position_rmse_list = [
                {"checkpoint": k, "values": v} for k, v in all_position_rmse.items()
            ]
            rotation_error_list = [
                {"checkpoint": k, "values": v} for k, v in all_rotation_error.items()
            ]

            # Gather all results to Rank 0
            gathered_position = [None] * self.world_size
            gathered_rotation = [None] * self.world_size
            gathered_extra = [None] * self.world_size

            dist.all_gather_object(gathered_position, position_rmse_list)
            dist.all_gather_object(gathered_rotation, rotation_error_list)
            dist.all_gather_object(
                gathered_extra,
                {
                    "rotation_mae": all_rotation_mae,
                    "mapping": checkpoint_mapping,
                    "not_evaluated": not_evaluated,
                    "samples": total_samples,
                    "partial": (partial_seen, partial_total),
                },
            )

            if self.rank == 0:
                # Merge results from all ranks
                all_position_rmse = {}
                all_rotation_error = {}
                total_samples = 0

                for rank_position in gathered_position:
                    for item in rank_position:
                        checkpoint = item["checkpoint"]
                        if checkpoint not in all_position_rmse:
                            all_position_rmse[checkpoint] = []
                        all_position_rmse[checkpoint].extend(item["values"])

                for rank_rotation in gathered_rotation:
                    for item in rank_rotation:
                        checkpoint = item["checkpoint"]
                        if checkpoint not in all_rotation_error:
                            all_rotation_error[checkpoint] = []
                        all_rotation_error[checkpoint].extend(item["values"])

                all_rotation_mae = {}
                for _ex in gathered_extra:
                    for checkpoint, vals in _ex["rotation_mae"].items():
                        all_rotation_mae.setdefault(checkpoint, []).extend(vals)
                    checkpoint_mapping.update(_ex["mapping"])
                    not_evaluated.update(_ex["not_evaluated"])

                # Count total samples across all ranks
                total_samples = sum(_ex["samples"] for _ex in gathered_extra)
                partial_seen = sum(_ex["partial"][0] for _ex in gathered_extra)
                partial_total = sum(_ex["partial"][1] for _ex in gathered_extra)
        self._partial_counts = (partial_seen, partial_total)

        # Only Rank 0 logs and writes to TensorBoard
        if self.rank == 0:
            if total_samples != len(eval_dataset):
                raise RuntimeError(
                    f"[RMSE Eval] evaluated {total_samples} scenes, the {split} set has {len(eval_dataset)}"
                )
            # Aggregate and log results
            logger.info(f"[RMSE Results] {split} set ({total_samples} samples):")

            # Log position RMSE
            logger.info(f"\n[Position RMSE] (actual physical frames evaluated)")
            for physical_frame in sorted(all_position_rmse.keys()):
                rmse_values = all_position_rmse[physical_frame]
                mean_rmse = np.mean(rmse_values)

                logger.info(
                    f"  Pos RMSE @ frame {physical_frame:3d}: {mean_rmse:.6f} m"
                )

                # Keyed by target checkpoint so all step sizes align in TensorBoard
                self.writer.add_scalar(
                    f"metrics/p_rmse_@{physical_frame:03d}/s={skip_step_size}",
                    mean_rmse,
                    self.epoch,
                )

            # Log rotation error
            if len(all_rotation_error) > 0:
                logger.info(f"\n[Rotation Error] (actual physical frames evaluated)")
                for physical_frame in sorted(all_rotation_error.keys()):
                    rot_values = all_rotation_error[physical_frame]
                    mean_rot = np.mean(rot_values)

                    logger.info(
                        f"  Rot Error @ frame {physical_frame:3d}: {mean_rot:.4f}°"
                    )

                    # Keyed by target checkpoint so all step sizes align in TensorBoard
                    self.writer.add_scalar(
                        f"metrics/r_error_@{physical_frame:03d}/s={skip_step_size}",
                        mean_rot,
                        self.epoch,
                    )
            # Rotation error as the mean absolute angle over objects (the statistic HopNet's reporting code prints;
            # "Rot Error" above is the RMS over objects)
            if len(all_rotation_mae) > 0:
                logger.info(
                    f"\n[Rotation MAE] (mean absolute angle over objects, per scene; averaged over scenes)"
                )
                for physical_frame in sorted(all_rotation_mae.keys()):
                    mean_mae = np.mean(all_rotation_mae[physical_frame])
                    logger.info(
                        f"  Rot MAE @ frame {physical_frame:3d}: {mean_mae:.4f}°"
                    )
                    self.writer.add_scalar(
                        f"metrics/r_mae_@{physical_frame:03d}/s={skip_step_size}",
                        mean_mae,
                        self.epoch,
                    )
            _mapping_text = ", ".join(
                f"@{cp} -> frame {fr} (warm-up {wu})"
                for cp, (fr, wu) in sorted(checkpoint_mapping.items())
            )
            _na_text = ", ".join(
                f"@{cp} ({why})" for cp, why in sorted(not_evaluated.items())
            )
            logger.info(
                f"[Checkpoint mapping] step_size={skip_step_size}, rollout warm-up {self.eval_warmup_frames} frames; "
                f"checkpoints evaluated elsewhere: {_mapping_text or 'none'}; "
                f"not evaluated (/): {_na_text or 'none'}"
            )
            logger.info(f"[Protocol] {self._protocol_text(split, total_samples)}")

            logger.info(f"\n[RMSE Results] {split} set completed")

            # Append results to eval.txt for post-hoc analysis
            eval_txt_path = os.path.join(self.log_dir, "eval.txt")
            with open(eval_txt_path, "a") as f:
                import datetime

                f.write(f"\n{'='*80}\n")
                f.write(
                    f"Epoch {self.epoch} | {split} | step_size={skip_step_size} | {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"{'='*80}\n")

                def _rows(values, fmt, unit):
                    # one row per checkpoint, "/" for those not evaluated
                    for physical_frame in sorted(set(values) | set(not_evaluated)):
                        if physical_frame in values:
                            f.write(
                                f"  frame {physical_frame:3d}: {np.mean(values[physical_frame]):{fmt}}{unit}\n"
                            )
                        else:
                            f.write(f"  frame {physical_frame:3d}: /\n")

                f.write(f"[Position RMSE]\n")
                _rows(all_position_rmse, ".6f", " m")
                if len(all_rotation_error) > 0:
                    f.write(f"[Rotation Error]\n")
                    _rows(all_rotation_error, ".4f", "°")
                if len(all_rotation_mae) > 0:
                    f.write(f"[Rotation MAE]\n")
                    _rows(all_rotation_mae, ".4f", "°")
                f.write(
                    f"  mapping: warm-up {self.eval_warmup_frames}; "
                    + (
                        ", ".join(
                            [
                                f"@{cp} -> frame {fr} (warm-up {wu})"
                                for cp, (fr, wu) in sorted(checkpoint_mapping.items())
                            ]
                            + [
                                f"@{cp} / ({why})"
                                for cp, why in sorted(not_evaluated.items())
                            ]
                        )
                        or "all checkpoints at their own frame"
                    )
                    + "\n"
                )
                f.write(f"  protocol: {self._protocol_text(split, total_samples)}\n")
                f.write(f"\n")
            logger.info(f"[Eval] Results appended to {eval_txt_path}")

            # Aggregated metrics for best-checkpoint selection, named to match
            # the TensorBoard tags
            metrics = {}
            for physical_frame in sorted(all_position_rmse.keys()):
                rmse_values = all_position_rmse[physical_frame]
                mean_rmse = np.mean(rmse_values)
                metrics[f"p_rmse_@{physical_frame:03d}/s={skip_step_size}"] = mean_rmse

            for physical_frame in sorted(all_rotation_error.keys()):
                rot_values = all_rotation_error[physical_frame]
                mean_rot = np.mean(rot_values)
                metrics[f"r_error_@{physical_frame:03d}/s={skip_step_size}"] = mean_rot
            for physical_frame in sorted(all_rotation_mae.keys()):
                metrics[f"r_mae_@{physical_frame:03d}/s={skip_step_size}"] = np.mean(
                    all_rotation_mae[physical_frame]
                )

            return metrics
        else:
            # Non-rank-0: return empty dict
            return {}

    def _full_from_partial(self, pred_single: np.ndarray, batch: dict, b_idx: int):
        """Full-mesh prediction of a rollout from partial point clouds (data.partial_view).

        The rollout moves each observed object's points rigidly; Kabsch from the points at the observation frame (the
        last warm-up frame, ground truth) to the points of every frame recovers that motion, which moves the object's
        full collision mesh of the observation frame. Unobserved objects keep their observation-frame mesh as a
        placeholder and are masked out of the metrics. Returns ((T, Nobj, Nv, 3), observed mask (Nobj,)).
        """
        from src.data.partial_view import kabsch

        observed = np.asarray(batch["partial_obj_index"][b_idx], dtype=np.int64)
        k = int(batch["partial_obs_index"][b_idx])
        full = batch["partial_full_obs"][b_idx]  # (Nv_i, 3) per object of the scene
        vmask = batch["valid_vtx_mask"][b_idx].detach().cpu().numpy()  # (T0, Nobs, Np)
        T = pred_single.shape[0]
        nv = max(V.shape[0] for V in full)
        out = np.zeros((T, len(full), nv, 3), dtype=np.float32)
        for i, V in enumerate(full):
            out[:, i, : V.shape[0]] = V
        for j, i in enumerate(observed):
            m = vmask[k, j]
            R, t = kabsch(pred_single[k, j, m], pred_single[:, j, m])
            out[:, i, : full[i].shape[0]] = (
                np.einsum("tab,nb->tna", R, full[i].astype(np.float64)) + t[:, None, :]
            ).astype(np.float32)
        mask = np.zeros((len(full),), dtype=bool)
        mask[observed] = True
        return out, mask

    def _compute_rmse_for_sample(
        self,
        pred_single: np.ndarray,
        dataset_name: str,
        eval_dataset,
        skip_step_size: int,
        physical_checkpoints: List[int],
        object_mask: Optional[np.ndarray] = None,
    ) -> Dict[int, float]:
        """
        Compute position RMSE and rotation error for one sample at the checkpoints.

        Args:
            pred_single: Predicted positions (T, Nobj, Nv, 3) with the main warmup
            dataset_name: Name of the dataset
            eval_dataset: Dataset instance for normalization
            skip_step_size: Frame subsampling factor
            physical_checkpoints: Physical frame checkpoints (e.g., [25, 50, 75, ...])

        Returns:
            Dict keyed by target checkpoint (position RMSE), with additional
            'rotation_deg_<cp>' entries and '_na_<cp>' (the reason) for a
            checkpoint that is not evaluated: off the step-size grid, or at a
            warm-up (context) frame.
        """
        # Load GT trajectory
        dataset_path = Path(self.config["data"]["dataset_path"]) / dataset_name
        traj_path = dataset_path / "vis_traj.npy"

        if not traj_path.exists():
            raise FileNotFoundError(f"Missing trajectory file: {traj_path}")

        traj_full = np.load(traj_path)[0]  # (T_total, Nobj, 7) - remove batch dim

        # Subsample trajectory to match skip_step_size
        traj_sampled = traj_full[::skip_step_size]  # (T_sampled, Nobj, 7)

        T_eval = min(pred_single.shape[0], traj_sampled.shape[0])
        gt_com = traj_sampled[:T_eval, :, :3].astype(np.float32)  # (T_eval, Nobj, 3)

        # Get object count
        Nobj_gt = gt_com.shape[1]

        # Trim predictions to match GT object count
        if pred_single.shape[1] > Nobj_gt:
            pred_single = pred_single[:T_eval, :Nobj_gt]
        else:
            pred_single = pred_single[:T_eval]

        # Predictions are already in world coordinates (DISABLE_NORMALIZERS=1)
        pred_world = pred_single  # (T_eval, Nobj, Nv, 3)

        # Load faces for area-weighted COM computation
        faces_list = []
        meshes_dir = dataset_path / "meshes"
        for obj_idx in range(Nobj_gt):
            faces_file = meshes_dir / f"obj_{obj_idx}_faces.npy"
            if not faces_file.exists():
                raise FileNotFoundError(f"Missing faces file: {faces_file}")
            faces = np.load(faces_file)
            faces_list.append(faces)

        R_hopnet = None
        if self.position_estimator == "hopnet":
            # Canonical meshes (object frame, from the ground-truth frame-0 pose) for shape matching
            _gt_quat_eval = traj_sampled[:T_eval, :Nobj_gt, 3:7].astype(np.float32)
            _ref_list = []
            _frame0 = load_frame(
                dataset_path, 0
            )  # either layout (src/data/scene_frames.py)
            for obj_idx in range(Nobj_gt):
                _p0 = dataset_path / "frame_00000" / f"obj_{obj_idx}_pos.npy"
                _pos0 = _frame0[obj_idx]
                if self._object_scale is not None:
                    _pos0 = self._object_scale.apply(_p0, _pos0)
                _q0 = _gt_quat_eval[0, obj_idx]
                _q0_xyzw = (
                    torch.tensor([_q0[1], _q0[2], _q0[3], _q0[0]]).float().unsqueeze(0)
                )
                _R0 = roma.unitquat_to_rotmat(_q0_xyzw).squeeze(0).numpy()
                _ref_list.append((_pos0 - gt_com[0, obj_idx]) @ _R0)
            pred_com, R_hopnet = _hopnet_object_poses(
                pred_world, _ref_list, faces_list, Nobj_gt
            )
        else:
            # Vectorized area-weighted COM across all timesteps (avoids building a
            # trimesh object per frame and per object)
            pred_com = np.zeros((T_eval, Nobj_gt, 3), dtype=np.float32)
            for obj_idx in range(Nobj_gt):
                F = faces_list[obj_idx]  # (Nf, 3)
                # Process all timesteps for this object at once
                V_all_t = pred_world[:, obj_idx, :, :]  # (T_eval, Nv, 3)

                # Vectorized COM calculation across all timesteps
                # Get triangle vertices: (T, Nf, 3, 3)
                v0 = V_all_t[:, F[:, 0], :]  # (T, Nf, 3)
                v1 = V_all_t[:, F[:, 1], :]  # (T, Nf, 3)
                v2 = V_all_t[:, F[:, 2], :]  # (T, Nf, 3)

                # Triangle areas: 0.5 * ||cross(v1-v0, v2-v0)||
                cross = np.cross(v1 - v0, v2 - v0)  # (T, Nf, 3)
                areas = 0.5 * np.linalg.norm(cross, axis=-1)  # (T, Nf)

                # Triangle centroids: (v0 + v1 + v2) / 3
                centroids = (v0 + v1 + v2) / 3.0  # (T, Nf, 3)

                # Area-weighted COM: sum(area * centroid) / sum(area)
                total_area = np.sum(areas, axis=1, keepdims=True)  # (T, 1)

                # Degenerate mesh is a hard error
                if np.any(total_area <= 0):
                    zero_frames = np.where(total_area.squeeze() <= 0)[0]
                    raise ValueError(
                        f"Degenerate mesh detected for object {obj_idx} in dataset {dataset_name}!\n"
                        f"Total area is zero or negative at frames: {zero_frames.tolist()}\n"
                        f"This indicates collapsed triangles or invalid mesh geometry.\n"
                        f"Check mesh file: meshes/obj_{obj_idx}_faces.npy"
                    )

                weighted_sum = np.sum(
                    areas[:, :, np.newaxis] * centroids, axis=1
                )  # (T, 3)
                pred_com[:, obj_idx, :] = weighted_sum / total_area  # (T, 3)

        # Compute position error
        pos_err = np.sum((pred_com - gt_com) ** 2, axis=-1)  # (T_eval, Nobj)

        # Compute per-step RMSE (data.partial_view: over the observed objects)
        if object_mask is not None:
            pos_err = pos_err[:, np.asarray(object_mask, dtype=bool)]
        position_rmse_all_steps = []
        for step in range(T_eval):
            mse_step = np.mean(pos_err[step])  # Average over objects
            rmse_step = np.sqrt(mse_step)
            position_rmse_all_steps.append(rmse_step)

        # ============================================================
        # Rotation error (HopNet-aligned): GT vs. predicted quaternions
        # ============================================================
        # traj_full shape: (T_total, Nobj, 7), where [:, :, 3:7] = quaternions WXYZ
        gt_quaternions_full = traj_full[:, :, 3:7].astype(
            np.float32
        )  # (T_total, Nobj, 4) WXYZ

        # Subsample GT quaternions to match evaluation frames
        gt_quaternions_sampled = gt_quaternions_full[
            ::skip_step_size
        ]  # (T_sampled, Nobj, 4)
        gt_quaternions_eval = gt_quaternions_sampled[
            :T_eval, :Nobj_gt
        ]  # (T_eval, Nobj_gt, 4)

        # Load canonical mesh for each object (inverse transform from frame 0)
        # Canonical mesh = rotation-free reference at origin
        ref_positions_list = []
        _frame0 = load_frame(
            dataset_path, 0
        )  # either layout (src/data/scene_frames.py)
        for obj_idx in range(Nobj_gt):
            # Load frame 0 positions
            _p0 = dataset_path / "frame_00000" / f"obj_{obj_idx}_pos.npy"
            obj_pos_0 = _frame0[obj_idx]
            if self._object_scale is not None:
                obj_pos_0 = self._object_scale.apply(_p0, obj_pos_0)

            # Get frame 0 quaternion and COM from GT trajectory
            quat_0 = gt_quaternions_eval[0, obj_idx]  # (4,) WXYZ
            com_0 = gt_com[0, obj_idx]  # (3,)

            # Convert quaternion to rotation matrix (WXYZ format)
            quat_0_torch = torch.from_numpy(quat_0).float().unsqueeze(0)  # (1, 4) WXYZ
            # roma expects XYZW, convert from WXYZ
            quat_0_xyzw = (
                torch.tensor([quat_0[1], quat_0[2], quat_0[3], quat_0[0]])
                .float()
                .unsqueeze(0)
            )  # (1, 4) XYZW
            R_0 = roma.unitquat_to_rotmat(quat_0_xyzw).squeeze(0).numpy()  # (3, 3)

            # Inverse transform: canonical = R_0^T @ (pos_0 - com_0)
            canonical_pos = (obj_pos_0 - com_0) @ R_0  # (Nv, 3)
            ref_positions_list.append(canonical_pos)

        # Convert GT quaternions to torch
        gt_quat_torch = (
            torch.from_numpy(gt_quaternions_eval).float().to(self.device)
        )  # (T, Nobj, 4) WXYZ

        import math

        # Network steps needed for the requested checkpoints (on the step-size grid, after the warm-up frames)
        checkpoint_network_steps = set()
        for physical_cp in physical_checkpoints:
            network_step = physical_cp // skip_step_size
            if (
                physical_cp % skip_step_size == 0
                and self.eval_warmup_frames <= network_step < T_eval
            ):
                checkpoint_network_steps.add(network_step)

        # Pre-transfer reference positions to GPU (once, not per-frame)
        ref_positions_gpu = []
        for obj_idx in range(Nobj_gt):
            ref_canonical = ref_positions_list[obj_idx]
            ref_positions_gpu.append(
                torch.from_numpy(ref_canonical).float().to(self.device)
            )

        # Compute rotation error only at checkpoint timesteps
        rotation_error_at_checkpoint = {}  # network_step -> RMS over objects
        rotation_mae_at_checkpoint = (
            {}
        )  # network_step -> mean absolute angle over objects

        for network_step in sorted(checkpoint_network_steps):
            rotation_errors_t = []

            for obj_idx in range(Nobj_gt):
                if object_mask is not None and not object_mask[obj_idx]:
                    continue
                ref_pts = ref_positions_gpu[obj_idx]  # Already on GPU
                Nv_actual = ref_pts.shape[0]
                if Nv_actual < 3:
                    continue

                # Get predicted vertices for this timestep
                V_pred = pred_world[network_step, obj_idx]  # (Nv_padded, 3)
                pred_pts = torch.from_numpy(V_pred[:Nv_actual]).float().to(self.device)

                if R_hopnet is not None:
                    R_pred = (
                        torch.from_numpy(R_hopnet[obj_idx][network_step])
                        .float()
                        .to(self.device)
                        .unsqueeze(0)
                    )
                else:
                    # Center points
                    ref_centered = ref_pts - ref_pts.mean(dim=0, keepdim=True)
                    pred_centered = pred_pts - pred_pts.mean(dim=0, keepdim=True)
                    # Kabsch alignment
                    R_pred, _ = roma.rigid_points_registration(
                        ref_centered.unsqueeze(0), pred_centered.unsqueeze(0)
                    )

                # Convert to quaternion and compute geodesic distance
                quat_pred_xyzw = roma.rotmat_to_unitquat(R_pred.squeeze(0))
                quat_pred_wxyz = torch.stack(
                    [
                        quat_pred_xyzw[3],
                        quat_pred_xyzw[0],
                        quat_pred_xyzw[1],
                        quat_pred_xyzw[2],
                    ]
                )

                quat_gt = gt_quat_torch[network_step, obj_idx]
                quat_pred_conj = torch.stack(
                    [
                        quat_pred_wxyz[0],
                        -quat_pred_wxyz[1],
                        -quat_pred_wxyz[2],
                        -quat_pred_wxyz[3],
                    ]
                )

                w0, x0, y0, z0 = quat_pred_conj
                w1, x1, y1, z1 = quat_gt
                q_err_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
                q_err_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
                q_err_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

                imag_norm = torch.sqrt(q_err_x**2 + q_err_y**2 + q_err_z**2)
                angle_rad = 2.0 * torch.arcsin(torch.clamp(imag_norm, 0.0, 1.0))
                angle_deg = angle_rad * 180.0 / np.pi
                rotation_errors_t.append(angle_deg.item())

            if rotation_errors_t:
                rotation_error_at_checkpoint[network_step] = np.sqrt(
                    np.mean(np.square(rotation_errors_t))
                )
                rotation_mae_at_checkpoint[network_step] = float(
                    np.mean(rotation_errors_t)
                )
            else:
                rotation_error_at_checkpoint[network_step] = 0.0
                rotation_mae_at_checkpoint[network_step] = 0.0

        # Sparse per-step rotation errors; only checkpoint steps carry values
        rotation_error_all_steps = [0.0] * T_eval
        for network_step, rmse in rotation_error_at_checkpoint.items():
            rotation_error_all_steps[network_step] = rmse

        # ============================================================
        # Extract checkpoints for both position and rotation
        # ============================================================
        import math

        rmse_dict = {"step_size": skip_step_size}

        # HopNet convention: direct indexing, so checkpoints include warmup frames. Only frames on the step-size grid
        # are evaluated: a checkpoint between two predicted frames, or at a given warm-up (context) frame, is "/".
        for physical_cp in physical_checkpoints:
            if physical_cp % skip_step_size:
                rmse_dict[f"_na_{physical_cp}"] = (
                    f"not on the step-size-{skip_step_size} grid"
                )
                continue
            network_step = physical_cp // skip_step_size
            if network_step < self.eval_warmup_frames:
                rmse_dict[f"_na_{physical_cp}"] = "a warm-up frame"
                continue

            if network_step < T_eval:
                # Position RMSE, keyed by target checkpoint so all step sizes
                # align to the same key in TensorBoard
                rmse_dict[physical_cp] = position_rmse_all_steps[network_step]

                # Rotation error (degrees), keyed by target checkpoint
                if network_step < len(rotation_error_all_steps):
                    rot_err_value = rotation_error_all_steps[network_step]
                    rmse_dict[f"rotation_deg_{physical_cp}"] = rot_err_value
                    rmse_dict[f"rotation_mae_deg_{physical_cp}"] = (
                        rotation_mae_at_checkpoint.get(network_step, 0.0)
                    )

        return rmse_dict
