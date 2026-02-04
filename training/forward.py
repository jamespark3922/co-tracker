# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Forward batch computation for CoTracker training.
Extracted from train_on_kubric.py for use with new training infrastructure.
"""

from typing import Dict, Any
from dataclasses import dataclass

import torch
from omegaconf import DictConfig

from cotracker.models.core.cotracker.losses import (
    sequence_loss,
    sequence_BCE_loss,
    sequence_prob_loss,
)


@dataclass
class ForwardConfig:
    """Configuration for forward_batch function."""
    train_iters: int = 4
    offline_model: bool = True
    sliding_window_len: int = 16
    query_sampling_method: str = None
    train_only_on_visible: bool = False
    add_huber_loss: bool = True


def forward_batch(
    batch,
    model: torch.nn.Module,
    cfg: ForwardConfig,
) -> Dict[str, Any]:
    """
    Compute forward pass and losses for a batch.

    Args:
        batch: CoTrackerData batch containing video, trajectory, visibility, valid
        model: CoTracker model
        cfg: Forward configuration

    Returns:
        Dictionary with loss components and predictions
    """
    video = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid

    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device

    __, first_positive_inds = torch.max(vis_g, dim=1)

    # Query sampling
    if cfg.query_sampling_method == "random":
        queries = _sample_queries_random(trajs_g, vis_g, B, N, D, device)
    else:
        queries = _sample_queries_default(trajs_g, vis_g, first_positive_inds, B, T, N, D, device)

    assert B == 1

    # Handle invalid samples
    if (
        torch.isnan(queries).any()
        or torch.isnan(trajs_g).any()
        or queries.abs().max() > 1500
    ):
        print("failed_sample")
        print("queries time", queries[..., 0])
        print("queries ", queries[..., 1:])
        queries = torch.ones_like(queries).to(queries.device).float()
        print("new queries", queries)
        valids = torch.zeros_like(valids).to(valids.device).float()
        print("new valids", valids)

    # Forward pass
    model_output = model(
        video=video,
        queries=queries[..., :3],
        iters=cfg.train_iters,
        is_train=True,
    )

    tracks, visibility, confidence, train_data = model_output
    coord_predictions, vis_predictions, confidence_predictions, valid_mask = train_data

    # Prepare ground truth for loss computation
    vis_gts = []
    invis_gts = []
    traj_gts = []
    valids_gts = []

    if cfg.offline_model:
        S = T
        seq_len = (S // 2) + 1
    else:
        S = cfg.sliding_window_len
        seq_len = T

    for ind in range(0, seq_len - S // 2, S // 2):
        vis_gts.append(vis_g[:, ind : ind + S])
        invis_gts.append(1 - vis_g[:, ind : ind + S])
        traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
        val = valids[:, ind : ind + S]
        if not cfg.offline_model:
            val = val * valid_mask[:, ind : ind + S]
        valids_gts.append(val)

    # Compute losses
    seq_loss_visible = sequence_loss(
        coord_predictions,
        traj_gts,
        valids_gts,
        vis=vis_gts,
        gamma=0.8,
        add_huber_loss=cfg.add_huber_loss,
        loss_only_for_visible=True,
    )

    # BCE loss only supports fp32?
    with torch.amp.autocast('cuda', enabled=False):
        confidence_loss = sequence_prob_loss(
            coord_predictions, confidence_predictions, traj_gts, vis_gts
        )
        vis_loss = sequence_BCE_loss(vis_predictions, vis_gts)

    # Build output dictionary
    output = {
        "flow": {
            "predictions": tracks[0].detach(),
            "loss": seq_loss_visible.mean() * 0.05,
            "queries": queries.clone(),
        },
        "visibility": {
            "loss": vis_loss.mean(),
            "predictions": visibility[0].detach(),
        },
        "confidence": {
            "loss": confidence_loss.mean(),
        },
    }

    if not cfg.train_only_on_visible:
        seq_loss_invisible = sequence_loss(
            coord_predictions,
            traj_gts,
            valids_gts,
            vis=invis_gts,
            gamma=0.8,
            add_huber_loss=False,
            loss_only_for_visible=True,
        )
        output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}

    return output


def _sample_queries_random(trajs_g, vis_g, B, N, D, device):
    """Sample queries from random visible frames."""
    assert B == 1
    true_indices = torch.nonzero(vis_g[0])
    grouped_indices = true_indices[:, 1].unique()

    sampled_points = torch.empty((B, N, D), device=device)
    indices = torch.empty((B, N, 1), device=device)

    for n in grouped_indices:
        t_indices = true_indices[true_indices[:, 1] == n, 0]
        random_index = t_indices[torch.randint(0, len(t_indices), (1,))]
        sampled_points[0, n] = trajs_g[0, random_index, n]
        indices[0, n] = random_index.float()

    queries = torch.cat([indices, sampled_points], dim=2)
    return queries


def _sample_queries_default(trajs_g, vis_g, first_positive_inds, B, T, N, D, device):
    """Sample queries mixing random visibility and first frame."""
    N_rand = N // 4

    nonzero_inds = [
        [torch.nonzero(vis_g[b, :, i]) for i in range(N)] for b in range(B)
    ]

    for b in range(B):
        rand_vis_inds = torch.cat(
            [
                nonzero_row[torch.randint(len(nonzero_row), size=(1,))]
                for nonzero_row in nonzero_inds[b]
            ],
            dim=1,
        )
        first_positive_inds[b] = torch.cat(
            [rand_vis_inds[:, :N_rand], first_positive_inds[b : b + 1, N_rand:]],
            dim=1,
        )

    ind_array_ = torch.arange(T, device=device)
    ind_array_ = ind_array_[None, :, None].repeat(B, 1, N)
    assert torch.allclose(
        vis_g[ind_array_ == first_positive_inds[:, None, :]],
        torch.ones(1, device=device),
    )

    gather = torch.gather(
        trajs_g, 1, first_positive_inds[:, :, None, None].repeat(1, 1, N, D)
    )
    xys = torch.diagonal(gather, dim1=1, dim2=2).permute(0, 2, 1)

    queries = torch.cat([first_positive_inds[:, :, None], xys[:, :, :2]], dim=2)
    return queries


def compute_total_loss(output: Dict[str, Any]) -> torch.Tensor:
    """Compute total loss from output dictionary."""
    loss = torch.tensor(0.0, device=next(iter(output.values()))["loss"].device)
    for k, v in output.items():
        if "loss" in v:
            loss = loss + v["loss"]
    return loss


def create_forward_config(cfg: DictConfig) -> ForwardConfig:
    """Create ForwardConfig from Hydra config."""
    return ForwardConfig(
        train_iters=cfg.training.train_iters,
        offline_model=cfg.training.offline_model,
        sliding_window_len=cfg.training.sliding_window_len,
        query_sampling_method=cfg.training.query_sampling_method,
        train_only_on_visible=cfg.training.train_only_on_visible,
        add_huber_loss=cfg.training.add_huber_loss,
    )
