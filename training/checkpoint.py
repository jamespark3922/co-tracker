# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint utilities with backward compatibility for legacy CoTracker checkpoints.

Supports loading:
1. Plain state_dict (old format)
2. {'model': ..., 'optimizer': ...} dict format
3. Keys with 'module.' prefix (DDP wrapped)
4. Filtering out 'time_emb', 'pos_emb' keys (matching original behavior)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    strict: bool = False,
    filter_keys: Optional[List[str]] = None,
) -> int:
    """
    Load checkpoint with legacy format support.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into (should NOT be DDP wrapped yet)
        optimizer: Optional optimizer to restore state
        scheduler: Optional scheduler to restore state
        scaler: Optional GradScaler to restore state
        strict: Whether to strictly enforce state_dict key matching
        filter_keys: List of substrings to filter out from state_dict keys

    Returns:
        total_steps: Training step count (0 if not found in checkpoint)
    """
    if filter_keys is None:
        filter_keys = ["time_emb", "pos_emb"]

    logging.info(f"Loading checkpoint from {path}")
    state_dict = torch.load(path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in state_dict:
        model_state = state_dict["model"]
    else:
        # Plain state_dict (old format)
        model_state = state_dict

    # Handle DDP 'module.' prefix
    if len(model_state) > 0:
        first_key = list(model_state.keys())[0]
        if first_key.startswith("module."):
            logging.info("Stripping 'module.' prefix from checkpoint keys")
            model_state = {
                k.replace("module.", ""): v
                for k, v in model_state.items()
            }

    # Filter out specified keys (e.g., time_emb, pos_emb)
    if filter_keys:
        original_len = len(model_state)
        model_state = {
            k: v for k, v in model_state.items()
            if not any(fk in k for fk in filter_keys)
        }
        if len(model_state) < original_len:
            logging.info(f"Filtered out {original_len - len(model_state)} keys containing {filter_keys}")

    # Load model state
    missing, unexpected = model.load_state_dict(model_state, strict=strict)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    # Load optimizer state if available and requested
    total_steps = 0
    if optimizer is not None and "optimizer" in state_dict:
        logging.info("Loading optimizer state")
        try:
            optimizer.load_state_dict(state_dict["optimizer"])
        except Exception as e:
            logging.warning(f"Failed to load optimizer state: {e}")

    if scheduler is not None and "scheduler" in state_dict:
        logging.info("Loading scheduler state")
        try:
            scheduler.load_state_dict(state_dict["scheduler"])
        except Exception as e:
            logging.warning(f"Failed to load scheduler state: {e}")

    if scaler is not None and "scaler" in state_dict:
        logging.info("Loading scaler state")
        try:
            scaler.load_state_dict(state_dict["scaler"])
        except Exception as e:
            logging.warning(f"Failed to load scaler state: {e}")

    if "total_steps" in state_dict:
        total_steps = state_dict["total_steps"]
        logging.info(f"Resuming from step {total_steps}")

    logging.info("Checkpoint loaded successfully")
    return total_steps


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    total_steps: int,
    epoch: int,
    config: Optional[DictConfig] = None,
    rank: int = 0,
):
    """
    Save checkpoint (only on rank 0).

    Args:
        path: Path to save checkpoint
        model: Model (can be DDP wrapped)
        optimizer: Optimizer
        scheduler: Scheduler
        scaler: GradScaler
        total_steps: Current training step
        epoch: Current epoch
        config: Optional config to save
        rank: Current rank (only saves on rank 0)
    """
    if rank != 0:
        return

    # Unwrap DDP if needed
    model_to_save = model.module if hasattr(model, "module") else model

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "total_steps": total_steps,
        "epoch": epoch,
    }

    if config is not None:
        state["config"] = OmegaConf.to_container(config, resolve=True)

    # Create directory if needed
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Save with backup
    tmp_path = str(path) + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)

    logging.info(f"Saved checkpoint to {path}")


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.

    Handles both epoch-based (model_*.pth) and step-based (step_*.pth) checkpoints.
    Returns the checkpoint with the highest step number.

    Args:
        ckpt_dir: Directory to search

    Returns:
        Path to latest checkpoint or None if no checkpoints found
    """
    import re

    if not os.path.exists(ckpt_dir):
        return None

    ckpt_files = [
        f for f in os.listdir(ckpt_dir)
        if f.endswith(".pth") and "final" not in f and not os.path.isdir(os.path.join(ckpt_dir, f))
    ]

    if not ckpt_files:
        return None

    def extract_step(filename):
        """Extract step number from checkpoint filename."""
        # Try step_XXXXXX.pth pattern first
        match = re.search(r'step_(\d+)\.pth', filename)
        if match:
            return int(match.group(1))

        # Try model_*_XXXXXX.pth pattern (epoch-based)
        match = re.search(r'_(\d{6})\.pth$', filename)
        if match:
            return int(match.group(1))

        # Fallback: try to find any number sequence
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[-1])

        return 0

    # Find checkpoint with highest step number
    latest = max(ckpt_files, key=extract_step)
    latest_step = extract_step(latest)
    logging.info(f"Found latest checkpoint: {latest} (step {latest_step})")

    return os.path.join(ckpt_dir, latest)


def save_final_model(
    path: str,
    model: nn.Module,
    rank: int = 0,
):
    """
    Save final model weights only (no optimizer/scheduler state).

    Args:
        path: Path to save model
        model: Model (can be DDP wrapped)
        rank: Current rank (only saves on rank 0)
    """
    if rank != 0:
        return

    model_to_save = model.module if hasattr(model, "module") else model

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_to_save.state_dict(), path)

    logging.info(f"Saved final model to {path}")
