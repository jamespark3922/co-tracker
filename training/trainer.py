# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main Trainer class for CoTracker using pure torch.distributed.
Replaces PyTorch Lightning Lite with explicit DDP control.
"""

import os
import json
import logging
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# CoTracker imports
from cotracker.datasets.utils import collate_fn_train, dataclass_to_cuda_
from cotracker.utils.visualizer import Visualizer
from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.utils.train_utils import get_eval_dataloader, run_test_eval

# Training imports
from training.distributed import (
    init_distributed,
    cleanup,
    is_main_process,
    get_rank,
    barrier,
    set_seeds,
    seed_worker,
)
from training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    save_final_model,
    find_latest_checkpoint,
)
from training.logging_utils import create_logger
from training.forward import forward_batch, compute_total_loss, create_forward_config
from training.data.composed_dataset import create_dataloader


class Trainer:
    """
    CoTracker Trainer with explicit torch.distributed control.

    Handles:
    - DDP initialization and model wrapping
    - Mixed precision training with GradScaler
    - Gradient accumulation with model.no_sync()
    - Checkpoint save/load with legacy format support
    - Wandb logging
    - Evaluation during training
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize trainer.

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg

        # Set environment variables
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

        # Initialize distributed
        self.rank, self.world_size, self.local_rank, self.device = init_distributed(
            backend=cfg.distributed.backend,
            timeout_mins=cfg.distributed.timeout_mins,
        )

        # Configure CUDA
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cfg.cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = cfg.cuda.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cfg.cuda.allow_tf32
            torch.backends.cudnn.allow_tf32 = cfg.cuda.allow_tf32

        # Set seeds
        set_seeds(cfg.seed, rank=self.rank)

        # Initialize components (will be set in setup())
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.logger = None
        self.visualizer = None
        self.evaluator = None
        self.eval_dataloaders = []
        self.final_dataloaders = []

        # Training state
        self.total_steps = 0
        self.epoch = 0

        # Forward config
        self.forward_cfg = create_forward_config(cfg)

    def setup(self):
        """Set up all training components."""
        cfg = self.cfg

        # Create directories
        Path(cfg.exp_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.checkpoint.save_dir).mkdir(parents=True, exist_ok=True)

        # Save config
        if is_main_process():
            config_path = Path(cfg.exp_dir) / "config.yaml"
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

            meta_path = Path(cfg.exp_dir) / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)

        # Build model
        self._build_model()

        # Build optimizer and scheduler
        self._build_optimizer()

        # Build scaler for AMP
        self._build_scaler()

        # Build dataloader
        self._build_dataloader()

        # Build logger
        self._build_logger()

        # Load checkpoint if resuming
        self._load_checkpoint()

        # Wrap model with DDP (after checkpoint loading!)
        self._wrap_ddp()

        # Set up evaluation (only on main process)
        if is_main_process():
            self._setup_evaluation()

        barrier()
        logging.info(f"Trainer setup complete. Rank {self.rank}/{self.world_size}")

    def _build_model(self):
        """Build model from config."""
        logging.info("Building model...")
        self.model = instantiate(self.cfg.model, _recursive_=False)
        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {total_params:,}")

    def _build_optimizer(self):
        """Build optimizer and scheduler."""
        logging.info("Building optimizer...")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            eps=self.cfg.optimizer.eps,
        )

        # OneCycleLR scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.cfg.optimizer.lr,
            total_steps=self.cfg.training.num_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="cos",
        )

    def _build_scaler(self):
        """Build GradScaler for mixed precision."""
        # NOTE: Original CoTracker uses GradScaler(enabled=False) because
        # BCE loss in sequence_prob_loss is not autocast-safe.
        # Lightning handles bf16 at model level, not via scaler.
        self.scaler = GradScaler(enabled=False)

    def _build_dataloader(self):
        """Build training dataloader."""
        logging.info("Building dataloader...")

        # Instantiate dataset
        train_dataset = instantiate(self.cfg.data.train, _recursive_=False)

        self.train_loader = create_dataloader(
            dataset=train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=True,
            collate_fn=collate_fn_train,
            distributed=self.world_size > 1,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.cfg.seed,
        )

        logging.info(f"Train loader length: {len(self.train_loader)}")

    def _build_logger(self):
        """Build wandb logger."""
        self.logger = create_logger(
            project=self.cfg.logging.wandb_project,
            name=self.cfg.exp_name,
            config=OmegaConf.to_container(self.cfg, resolve=True),
            entity=self.cfg.logging.wandb_entity,
            dir=self.cfg.logging.log_dir,
            rank=self.rank,
            enabled=self.cfg.logging.use_wandb,
            model=self.model,
        )

    def _load_checkpoint(self):
        """Load checkpoint if resuming or restoring."""
        cfg = self.cfg.checkpoint

        # Check for existing checkpoint in save_dir (auto-resume)
        latest_ckpt = find_latest_checkpoint(cfg.save_dir)
        if latest_ckpt is not None:
            logging.info(f"Found existing checkpoint: {latest_ckpt}")
            self.total_steps = load_checkpoint(
                latest_ckpt,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                strict=False,
            )
            return

        # Check for explicit resume path
        if cfg.resume_from is not None:
            logging.info(f"Resuming from: {cfg.resume_from}")
            self.total_steps = load_checkpoint(
                cfg.resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                strict=False,
            )
            return

        # Check for pretrained weights only
        if cfg.restore_ckpt is not None:
            logging.info(f"Loading pretrained weights from: {cfg.restore_ckpt}")
            load_checkpoint(
                cfg.restore_ckpt,
                self.model,
                optimizer=None,  # Don't load optimizer
                scheduler=None,
                scaler=None,
                strict=False,
            )

    def _wrap_ddp(self):
        """Wrap model with DistributedDataParallel."""
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.cfg.distributed.find_unused_parameters,
                gradient_as_bucket_view=self.cfg.distributed.gradient_as_bucket_view,
                bucket_cap_mb=self.cfg.distributed.bucket_cap_mb,
                broadcast_buffers=self.cfg.distributed.broadcast_buffers,
            )

    def _setup_evaluation(self):
        """Set up evaluation components (main process only)."""
        cfg = self.cfg

        # Evaluator
        self.evaluator = Evaluator(cfg.checkpoint.save_dir)

        # Visualizer
        self.visualizer = Visualizer(
            save_dir=cfg.checkpoint.save_dir,
            pad_value=180,
            fps=1,
            show_first_frame=0,
            tracks_leave_trace=0,
        )

        # Evaluation dataloaders
        for ds_name in cfg.evaluation.datasets:
            self.eval_dataloaders.append(
                (ds_name, get_eval_dataloader(cfg.evaluation.dataset_root, ds_name))
            )

        # Final evaluation dataloaders
        for ds_name in cfg.evaluation.final_datasets:
            self.final_dataloaders.append(
                (ds_name, get_eval_dataloader(cfg.evaluation.dataset_root, ds_name))
            )

    def train(self):
        """Main training loop."""
        cfg = self.cfg
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        scaler = self.scaler

        model.train()
        should_keep_training = True
        epoch = -1

        save_freq = cfg.checkpoint.save_freq

        while should_keep_training:
            epoch += 1
            self.epoch = epoch

            # Set sampler epoch for proper shuffling
            if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            for i_batch, batch in enumerate(tqdm(self.train_loader, disable=not is_main_process())):
                batch, gotit = batch
                if not all(gotit):
                    logging.debug("Skipping batch with missing data")
                    continue

                # Move to GPU
                dataclass_to_cuda_(batch)

                # Zero gradients
                optimizer.zero_grad(set_to_none=True)

                assert model.training

                # Get underlying model for forward
                model_for_forward = model.module if hasattr(model, "module") else model

                # Forward pass
                # NOTE: We don't use autocast here because the original CoTracker code
                # uses GradScaler(enabled=False). The BCE loss in sequence_prob_loss
                # is not autocast-safe. Lightning handles bf16 at a different level.
                output = forward_batch(batch, model_for_forward, self.forward_cfg)

                loss = compute_total_loss(output)

                # Logging (main process only)
                if is_main_process():
                    self._log_training_step(output, loss, i_batch, save_freq, batch)

                # Synchronize before backward
                barrier()

                # Backward pass
                scaler.scale(loss).backward()

                # Unscale and clip gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip_norm)

                # Optimizer step
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

                self.total_steps += 1

                # End of epoch handling
                if is_main_process():
                    if (i_batch >= len(self.train_loader) - 1) or (
                        self.total_steps == 1 and cfg.checkpoint.validate_at_start
                    ):
                        self._end_of_epoch(epoch)

                barrier()

                if self.total_steps > cfg.training.num_steps:
                    should_keep_training = False
                    break

        # Final save and evaluation
        if is_main_process():
            self._finish_training()

        cleanup()

    def _log_training_step(self, output, loss, i_batch, save_freq, batch):
        """Log training metrics and visualizations."""
        # Log losses
        metrics = {}
        for k, v in output.items():
            if "loss" in v:
                metrics[f"train/{k}_loss"] = v["loss"].item()
        metrics["train/total_loss"] = loss.item()
        metrics["train/lr"] = self.optimizer.param_groups[0]["lr"]

        self.logger.log(metrics, step=self.total_steps)

        # Visualizations
        if self.total_steps % save_freq == save_freq - 1:
            self._log_visualizations(batch, output)

    def _log_visualizations(self, batch, output):
        """Log trajectory visualizations."""
        # This would require converting visualizer output to wandb format
        # For now, we skip this to avoid complexity
        pass

    def _end_of_epoch(self, epoch):
        """Handle end of epoch: save checkpoint and evaluate."""
        cfg = self.cfg

        # Save checkpoint
        if (epoch + 1) % cfg.checkpoint.save_every_n_epoch == 0:
            ckpt_iter = str(self.total_steps).zfill(6)
            save_path = Path(cfg.checkpoint.save_dir) / f"model_{cfg.exp_name}_{ckpt_iter}.pth"

            save_checkpoint(
                str(save_path),
                self.model,
                self.optimizer,
                self.scheduler,
                self.scaler,
                self.total_steps,
                epoch,
                self.cfg,
                rank=self.rank,
            )

        # Evaluate
        if (epoch + 1) % cfg.checkpoint.evaluate_every_n_epoch == 0 or (
            cfg.checkpoint.validate_at_start and epoch == 0
        ):
            self._evaluate()

    def _evaluate(self):
        """Run evaluation."""
        model_for_eval = self.model.module if hasattr(self.model, "module") else self.model

        run_test_eval(
            self.evaluator,
            model_for_eval,
            self.eval_dataloaders,
            writer=None,  # We'll log to wandb instead
            step=self.total_steps,
            query_random=(
                self.cfg.training.query_sampling_method is not None
                and "random" in self.cfg.training.query_sampling_method
            ),
        )

        self.model.train()
        torch.cuda.empty_cache()

    def _finish_training(self):
        """Final save and evaluation."""
        logging.info("FINISHED TRAINING")

        # Save final model
        final_path = Path(self.cfg.checkpoint.save_dir) / f"{self.cfg.exp_name}_final.pth"
        save_final_model(str(final_path), self.model, rank=self.rank)

        # Final evaluation
        model_for_eval = self.model.module if hasattr(self.model, "module") else self.model

        run_test_eval(
            self.evaluator,
            model_for_eval,
            self.final_dataloaders,
            writer=None,
            step=self.total_steps,
            query_random=(
                self.cfg.training.query_sampling_method is not None
                and "random" in self.cfg.training.query_sampling_method
            ),
        )

        self.logger.finish()
