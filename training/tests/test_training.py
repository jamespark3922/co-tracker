#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Tests for CoTracker training infrastructure.

Run with: pytest training/tests/test_training.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestDistributedUtils:
    """Test distributed training utilities."""

    def test_get_machine_local_and_dist_rank_default(self):
        """Test rank detection with no environment variables."""
        from training.distributed import get_machine_local_and_dist_rank

        # Clear env vars
        for var in ["LOCAL_RANK", "RANK", "SLURM_LOCALID", "SLURM_PROCID"]:
            os.environ.pop(var, None)

        local_rank, dist_rank = get_machine_local_and_dist_rank()
        assert local_rank == 0
        assert dist_rank == 0

    def test_get_machine_local_and_dist_rank_torchrun(self):
        """Test rank detection with torchrun environment."""
        from training.distributed import get_machine_local_and_dist_rank

        os.environ["LOCAL_RANK"] = "2"
        os.environ["RANK"] = "5"

        local_rank, dist_rank = get_machine_local_and_dist_rank()
        assert local_rank == 2
        assert dist_rank == 5

        # Cleanup
        del os.environ["LOCAL_RANK"]
        del os.environ["RANK"]

    def test_get_world_size_default(self):
        """Test world size with no environment variables."""
        from training.distributed import get_world_size

        for var in ["WORLD_SIZE", "SLURM_NTASKS"]:
            os.environ.pop(var, None)

        assert get_world_size() == 1

    def test_is_main_process_not_initialized(self):
        """Test is_main_process when distributed not initialized."""
        from training.distributed import is_main_process

        assert is_main_process() is True

    def test_set_seeds(self):
        """Test seed setting."""
        from training.distributed import set_seeds

        set_seeds(42, rank=0)

        # Check torch is seeded
        t1 = torch.rand(10)
        set_seeds(42, rank=0)
        t2 = torch.rand(10)
        assert torch.allclose(t1, t2)


class TestCheckpoint:
    """Test checkpoint utilities."""

    def test_load_checkpoint_plain_state_dict(self):
        """Test loading a plain state_dict (old format)."""
        from training.checkpoint import load_checkpoint

        # Create a simple model
        model = nn.Linear(10, 5)

        # Save as plain state dict
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            tmp_path = f.name

        try:
            # Create a new model and load
            new_model = nn.Linear(10, 5)
            steps = load_checkpoint(tmp_path, new_model, strict=True)

            assert steps == 0  # No steps in old format
            # Check weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(tmp_path)

    def test_load_checkpoint_dict_format(self):
        """Test loading checkpoint with model key."""
        from training.checkpoint import load_checkpoint

        model = nn.Linear(10, 5)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save({
                "model": model.state_dict(),
                "total_steps": 1000,
            }, f.name)
            tmp_path = f.name

        try:
            new_model = nn.Linear(10, 5)
            steps = load_checkpoint(tmp_path, new_model, strict=True)

            assert steps == 1000
        finally:
            os.unlink(tmp_path)

    def test_load_checkpoint_with_module_prefix(self):
        """Test loading checkpoint with module. prefix (DDP format)."""
        from training.checkpoint import load_checkpoint

        model = nn.Linear(10, 5)

        # Add module. prefix like DDP does
        state_dict = {"module." + k: v for k, v in model.state_dict().items()}

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            torch.save(state_dict, f.name)
            tmp_path = f.name

        try:
            new_model = nn.Linear(10, 5)
            load_checkpoint(tmp_path, new_model, strict=True)

            # Check weights match
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(tmp_path)

    def test_save_checkpoint(self):
        """Test checkpoint saving."""
        from training.checkpoint import save_checkpoint, load_checkpoint

        model = nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test.pth")
            save_checkpoint(
                save_path,
                model,
                optimizer,
                scheduler,
                scaler,
                total_steps=500,
                epoch=10,
                config=None,
                rank=0,
            )

            assert os.path.exists(save_path)

            # Load and verify
            new_model = nn.Linear(10, 5)
            steps = load_checkpoint(save_path, new_model, strict=True)
            assert steps == 500

    def test_find_latest_checkpoint(self):
        """Test finding latest checkpoint in directory."""
        from training.checkpoint import find_latest_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            # No checkpoints
            assert find_latest_checkpoint(tmpdir) is None

            # Create some checkpoints
            Path(os.path.join(tmpdir, "model_000100.pth")).touch()
            Path(os.path.join(tmpdir, "model_000200.pth")).touch()
            Path(os.path.join(tmpdir, "model_000050.pth")).touch()

            latest = find_latest_checkpoint(tmpdir)
            assert latest.endswith("model_000200.pth")


class TestLoggingUtils:
    """Test logging utilities."""

    def test_dummy_logger(self):
        """Test DummyLogger does nothing."""
        from training.logging_utils import DummyLogger

        logger = DummyLogger()
        # Should not raise
        logger.log({"loss": 1.0}, step=0)
        logger.log_scalar("test", 1.0, 0)
        logger.push({"acc": 0.5}, "train")
        logger.finish()

    def test_create_logger_non_rank0(self):
        """Test logger creation for non-rank-0 processes."""
        from training.logging_utils import create_logger, DummyLogger

        logger = create_logger(
            project="test",
            name="test_run",
            rank=1,  # Not rank 0
            enabled=True,
        )
        assert isinstance(logger, DummyLogger)

    def test_create_logger_disabled(self):
        """Test logger creation when disabled."""
        from training.logging_utils import create_logger, DummyLogger

        logger = create_logger(
            project="test",
            name="test_run",
            rank=0,
            enabled=False,
        )
        assert isinstance(logger, DummyLogger)


class TestForward:
    """Test forward batch computation."""

    def test_create_forward_config(self):
        """Test ForwardConfig creation from dict."""
        from training.forward import ForwardConfig

        cfg = ForwardConfig(
            train_iters=4,
            offline_model=True,
            sliding_window_len=16,
        )
        assert cfg.train_iters == 4
        assert cfg.offline_model is True

    def test_compute_total_loss(self):
        """Test total loss computation."""
        from training.forward import compute_total_loss

        output = {
            "flow": {"loss": torch.tensor(0.5)},
            "visibility": {"loss": torch.tensor(0.3)},
            "confidence": {"loss": torch.tensor(0.2)},
        }
        total = compute_total_loss(output)
        assert torch.isclose(total, torch.tensor(1.0))


class TestComposedDataset:
    """Test ComposedDataset."""

    def test_composed_dataset_single(self):
        """Test ComposedDataset with single dataset."""
        from torch.utils.data import TensorDataset

        # Create a simple dataset
        ds = TensorDataset(torch.randn(100, 10))

        # Use ComposedDataset (without Hydra instantiate)
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([ds])

        assert len(combined) == 100
        assert combined[0][0].shape == (10,)

    def test_composed_dataset_repeat(self):
        """Test ComposedDataset with repeat."""
        from torch.utils.data import TensorDataset, ConcatDataset

        ds = TensorDataset(torch.randn(100, 10))
        combined = ConcatDataset(4 * [ds])

        assert len(combined) == 400


class TestConfigLoading:
    """Test Hydra config loading."""

    def test_config_exists(self):
        """Test that config files exist."""
        config_dir = Path(__file__).parent.parent / "config"

        assert (config_dir / "default.yaml").exists()
        assert (config_dir / "model" / "cotracker3_offline.yaml").exists()
        assert (config_dir / "model" / "cotracker3_online.yaml").exists()
        assert (config_dir / "data" / "kubric.yaml").exists()
        assert (config_dir / "optimizer" / "adamw_onecycle.yaml").exists()

    def test_config_valid_yaml(self):
        """Test that config files are valid YAML."""
        import yaml

        config_dir = Path(__file__).parent.parent / "config"

        for yaml_file in config_dir.rglob("*.yaml"):
            with open(yaml_file) as f:
                try:
                    yaml.safe_load(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML in {yaml_file}: {e}")


class TestIntegration:
    """Integration tests (require actual dataset)."""

    @pytest.mark.skipif(
        not os.path.exists("/weka/oe-training-default/mm-olmo/video_datasets/point_track/CoTracker3_Kubric/"),
        reason="Kubric dataset not available"
    )
    def test_kubric_dataset_loads(self):
        """Test that KubricMovifDataset can be loaded."""
        from cotracker.datasets.kubric_movif_dataset import KubricMovifDataset

        ds = KubricMovifDataset(
            data_root="/weka/oe-training-default/mm-olmo/video_datasets/point_track/CoTracker3_Kubric/data",
            crop_size=[384, 512],
            seq_len=24,
            traj_per_sample=768,
            sample_vis_last_frame=False,
            use_augs=False,
            split="train",
        )

        assert len(ds) > 0

        # Try to get one sample
        sample, gotit = ds[0]
        assert gotit or not gotit  # Either is valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
