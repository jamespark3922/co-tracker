# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Wandb-based logger for CoTracker training.
Replaces TensorBoard Logger from original implementation.
"""

import os
import logging
from typing import Dict, Optional, Any, Union
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")


class WandbLogger:
    """
    Wandb-based logger with interface compatible with original CoTracker Logger.

    Supports:
    - Scalar logging
    - Image/video logging
    - Running metric aggregation
    - Model watching for gradient logging
    """

    SUM_FREQ = 100  # Aggregate metrics every N steps

    def __init__(
        self,
        project: str,
        name: str,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        dir: Optional[str] = None,
        model: Optional[nn.Module] = None,
        rank: int = 0,
        enabled: bool = True,
        tags: Optional[list] = None,
        resume: Optional[str] = None,
    ):
        """
        Initialize wandb logger.

        Args:
            project: Wandb project name
            name: Run name
            config: Config dict to log
            entity: Wandb entity (team/user)
            dir: Directory for wandb files
            model: Optional model for gradient watching
            rank: Distributed rank (only rank 0 logs)
            enabled: Whether logging is enabled
            tags: Optional tags for the run
            resume: Resume mode ('allow', 'must', 'never', or run_id)
        """
        self.rank = rank
        self.enabled = enabled and WANDB_AVAILABLE and rank == 0
        self.running_metrics = {}
        self.step_count = 0

        if self.enabled:
            if dir is not None:
                Path(dir).mkdir(parents=True, exist_ok=True)

            # Use name as run ID for deterministic resume
            run_id = name.replace("/", "_").replace(" ", "_")[:64] if resume else None

            wandb.init(
                project=project,
                name=name,
                id=run_id,
                config=config,
                entity=entity,
                dir=dir,
                tags=tags,
                resume=resume,
                reinit=True,
            )

            if model is not None:
                # Watch model for gradient logging
                wandb.watch(model, log="gradients", log_freq=1000)

            logging.info(f"Wandb initialized: {wandb.run.url}")

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number (uses internal counter if not provided)
        """
        if not self.enabled:
            return

        # Flatten nested metrics
        flat_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_metrics[f"{k}/{k2}"] = self._to_scalar(v2)
            else:
                flat_metrics[k] = self._to_scalar(v)

        wandb.log(flat_metrics, step=step)

    def log_scalar(self, name: str, value: Union[float, torch.Tensor], step: int):
        """Log a single scalar value."""
        if not self.enabled:
            return
        wandb.log({name: self._to_scalar(value)}, step=step)

    def log_image(self, name: str, image: Union[torch.Tensor, np.ndarray], step: int, caption: Optional[str] = None):
        """
        Log an image to wandb.

        Args:
            name: Image name/tag
            image: Image tensor (C, H, W) or numpy array (H, W, C)
            step: Step number
            caption: Optional caption
        """
        if not self.enabled:
            return

        if isinstance(image, torch.Tensor):
            image = image.cpu()
            if image.dim() == 3 and image.shape[0] in [1, 3, 4]:
                # C, H, W -> H, W, C
                image = image.permute(1, 2, 0)
            image = image.numpy()

        # Normalize to 0-255 if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        wandb.log({name: wandb.Image(image, caption=caption)}, step=step)

    def log_video(
        self,
        name: str,
        video: Union[torch.Tensor, np.ndarray],
        step: int,
        fps: int = 10,
        caption: Optional[str] = None,
    ):
        """
        Log a video to wandb.

        Args:
            name: Video name/tag
            video: Video tensor (T, C, H, W) or numpy array (T, H, W, C)
            step: Step number
            fps: Frames per second
            caption: Optional caption
        """
        if not self.enabled:
            return

        if isinstance(video, torch.Tensor):
            video = video.cpu()
            if video.dim() == 4 and video.shape[1] in [1, 3, 4]:
                # T, C, H, W -> T, H, W, C
                video = video.permute(0, 2, 3, 1)
            video = video.numpy()

        # Normalize to 0-255 if needed
        if video.max() <= 1.0:
            video = (video * 255).astype(np.uint8)

        wandb.log({name: wandb.Video(video, fps=fps, caption=caption)}, step=step)

    def push(self, metrics: Dict[str, float], task: str):
        """
        Accumulate metrics for aggregation (compatibility with original Logger).

        Args:
            metrics: Dict of metric name -> value
            task: Task name prefix
        """
        self.step_count += 1

        for key, value in metrics.items():
            task_key = f"{task}/{key}"
            if task_key not in self.running_metrics:
                self.running_metrics[task_key] = 0.0
            self.running_metrics[task_key] += value

        # Log aggregated metrics periodically
        if self.step_count % self.SUM_FREQ == self.SUM_FREQ - 1:
            avg_metrics = {
                k: v / self.SUM_FREQ
                for k, v in self.running_metrics.items()
            }
            self.log(avg_metrics, step=self.step_count)
            self.running_metrics = {}

    def log_config(self, config: Dict):
        """Update wandb config."""
        if not self.enabled:
            return
        wandb.config.update(config, allow_val_change=True)

    def log_artifact(self, path: str, name: str, type: str = "model"):
        """Log a file as a wandb artifact."""
        if not self.enabled:
            return
        artifact = wandb.Artifact(name, type=type)
        artifact.add_file(path)
        wandb.log_artifact(artifact)

    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()
            logging.info("Wandb run finished")

    def _to_scalar(self, value: Any) -> float:
        """Convert value to scalar."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return float(value)

    @property
    def run_url(self) -> Optional[str]:
        """Get wandb run URL."""
        if self.enabled and wandb.run is not None:
            return wandb.run.url
        return None


class DummyLogger:
    """Dummy logger that does nothing (for non-rank-0 processes)."""

    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def log_scalar(self, *args, **kwargs):
        pass

    def log_image(self, *args, **kwargs):
        pass

    def log_video(self, *args, **kwargs):
        pass

    def push(self, *args, **kwargs):
        pass

    def log_config(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def create_logger(
    project: str,
    name: str,
    config: Optional[Dict] = None,
    rank: int = 0,
    enabled: bool = True,
    resume: Optional[str] = "allow",
    **kwargs,
) -> Union[WandbLogger, DummyLogger]:
    """
    Create appropriate logger based on rank.

    Args:
        project: Wandb project name
        name: Run name
        config: Config dict
        rank: Distributed rank
        enabled: Whether logging is enabled
        resume: Wandb resume mode ('allow', 'must', 'never', None)
                'allow' = resume if run exists, else create new

    Returns:
        WandbLogger for rank 0, DummyLogger for other ranks
    """
    if rank == 0 and enabled:
        return WandbLogger(
            project=project,
            name=name,
            config=config,
            rank=rank,
            enabled=enabled,
            resume=resume,
            **kwargs,
        )
    return DummyLogger()
