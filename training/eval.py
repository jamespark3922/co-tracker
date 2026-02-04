# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation utilities for CoTracker training.
Uses WandbLogger directly instead of TensorBoard.
"""

import os
import logging
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from cotracker.datasets.utils import collate_fn, collate_fn_train
from cotracker.models.evaluation_predictor import EvaluationPredictor
from cotracker.evaluation.core.evaluator import Evaluator


def get_eval_dataloader(dataset_root: str, ds_name: str) -> DataLoader:
    """
    Create evaluation dataloader for a specific dataset.

    Args:
        dataset_root: Root directory containing evaluation datasets
        ds_name: Dataset name (e.g., 'tapvid_davis_first', 'tapvid_robotap')

    Returns:
        DataLoader for the evaluation dataset
    """
    from cotracker.datasets.tap_vid_datasets import TapVidDataset

    collate_fn_local = collate_fn

    if ds_name == "dynamic_replica":
        from cotracker.datasets.dr_dataset import DynamicReplicaDataset

        eval_dataset = DynamicReplicaDataset(
            root=os.path.join(dataset_root, "dynamic_replica"),
            sample_len=300,
            only_first_n_samples=1,
            rgbd_input=False,
        )
    elif ds_name == "tapvid_davis_first":
        data_root = os.path.join(dataset_root, "tapvid/tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(
            dataset_type="davis", data_root=data_root, queried_first=True
        )
    elif ds_name == "tapvid_davis_strided":
        data_root = os.path.join(dataset_root, "tapvid/tapvid_davis/tapvid_davis.pkl")
        eval_dataset = TapVidDataset(
            dataset_type="davis", data_root=data_root, queried_first=False
        )
    elif ds_name == "tapvid_kinetics_first":
        eval_dataset = TapVidDataset(
            dataset_type="kinetics",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_kinetics"),
        )
    elif ds_name == "tapvid_stacking":
        eval_dataset = TapVidDataset(
            dataset_type="stacking",
            data_root=os.path.join(
                dataset_root, "tapvid", "tapvid_rgb_stacking", "tapvid_rgb_stacking.pkl"
            ),
        )
    elif ds_name == "tapvid_robotap":
        eval_dataset = TapVidDataset(
            dataset_type="robotap",
            data_root=os.path.join(dataset_root, "tapvid", "tapvid_robotap"),
        )
    elif ds_name == "kubric":
        from cotracker.datasets.kubric_movif_dataset import KubricMovifDataset

        eval_dataset = KubricMovifDataset(
            data_root=os.path.join(dataset_root, "kubric/movi_f"),
            traj_per_sample=1024,
            use_augs=False,
            split="valid",
            sample_vis_last_frame=True,
        )
        collate_fn_local = collate_fn_train
    else:
        raise ValueError(f"Unknown eval dataset: {ds_name}")

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn_local,
    )
    return eval_dataloader


def unwrap_model(model):
    """Unwrap model from DDP/Lightning wrappers."""
    while hasattr(model, "module"):
        model = model.module
    return model


def run_eval(
    evaluator: Evaluator,
    model: torch.nn.Module,
    dataloaders: List[Tuple[str, DataLoader]],
    logger,
    step: int,
    query_random: bool = False,
) -> dict:
    """
    Run evaluation on multiple datasets and log to wandb.

    Args:
        evaluator: Evaluator instance
        model: Model to evaluate (can be DDP wrapped)
        dataloaders: List of (dataset_name, dataloader) tuples
        logger: WandbLogger instance
        step: Current training step
        query_random: Whether to use random query sampling

    Returns:
        Dictionary of all metrics
    """
    model.eval()
    all_metrics = {}

    for ds_name, dataloader in dataloaders:
        # Dataset-specific settings
        visualize_every = 1
        grid_size = 5
        num_uniformly_sampled_pts = 0

        if ds_name == "dynamic_replica":
            visualize_every = 8
            grid_size = 0
        elif ds_name == "kubric":
            visualize_every = 5
            grid_size = 0
        elif "davis" in ds_name or "tapvid_stacking" in ds_name:
            visualize_every = 5
        elif "robotap" in ds_name:
            visualize_every = 20
        elif "kinetics" in ds_name:
            visualize_every = 50

        if query_random:
            grid_size = 0
            num_uniformly_sampled_pts = 100

        # Create predictor with unwrapped model
        predictor = EvaluationPredictor(
            unwrap_model(model),
            grid_size=grid_size,
            local_grid_size=0,
            single_point=False,
            num_uniformly_sampled_pts=num_uniformly_sampled_pts,
            n_iters=6,
        )

        if torch.cuda.is_available():
            predictor.model = predictor.model.cuda()

        # Run evaluation
        metrics = evaluator.evaluate_sequence(
            model=predictor,
            test_dataloader=dataloader,
            dataset_name=ds_name,
            train_mode=True,
            writer=None,  # We handle logging ourselves
            step=step,
            visualize_every=visualize_every,
        )

        # Process metrics based on dataset type
        if ds_name == "dynamic_replica" or ds_name == "kubric":
            processed_metrics = {
                f"eval/{ds_name}_avg_{k}": v
                for k, v in metrics["avg"].items()
                if not ("1" in k or "2" in k or "4" in k or "8" in k)
            }
        elif "tapvid" in ds_name:
            processed_metrics = {
                f"eval/{ds_name}_avg_OA": metrics["avg"]["occlusion_accuracy"],
                f"eval/{ds_name}_avg_delta": metrics["avg"]["average_pts_within_thresh"],
                f"eval/{ds_name}_avg_Jaccard": metrics["avg"]["average_jaccard"],
            }
        else:
            processed_metrics = {
                f"eval/{ds_name}_{k}": v for k, v in metrics.get("avg", metrics).items()
            }

        # Log to wandb
        logger.log(processed_metrics, step=step)

        # Store for return
        all_metrics.update(processed_metrics)

        logging.info(f"Eval {ds_name} @ step {step}: {processed_metrics}")

    return all_metrics
