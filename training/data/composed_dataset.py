# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
ComposedDataset for CoTracker training.
Wraps existing datasets with support for:
- Multiple data sources
- Dataset repetition (matching original ConcatDataset pattern)
- Weighted sampling (optional)
"""

from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, DistributedSampler
from hydra.utils import instantiate


class ComposedDataset(Dataset):
    """
    Composes multiple datasets with optional repetition and weighting.

    Supports Hydra instantiation of nested datasets.
    """

    def __init__(
        self,
        datasets: List[Dict[str, Any]],
        repeat: int = 1,
        weights: Optional[List[float]] = None,
        **kwargs,
    ):
        """
        Initialize ComposedDataset.

        Args:
            datasets: List of dataset configs (Hydra format with _target_)
            repeat: Number of times to repeat the combined dataset
            weights: Optional weights for weighted sampling (not used with repeat)
            **kwargs: Additional arguments passed to child datasets
        """
        super().__init__()

        self.repeat = repeat
        self.weights = weights

        # Instantiate each dataset from config
        instantiated_datasets = []
        for ds_config in datasets:
            ds = instantiate(ds_config, _recursive_=False)
            instantiated_datasets.append(ds)

        # Combine with repetition (matching original 4x ConcatDataset pattern)
        if repeat > 1:
            self.combined = ConcatDataset(repeat * instantiated_datasets)
        else:
            self.combined = ConcatDataset(instantiated_datasets)

        self._datasets = instantiated_datasets

    def __len__(self) -> int:
        return len(self.combined)

    def __getitem__(self, idx: int):
        return self.combined[idx]

    @property
    def datasets(self) -> List[Dataset]:
        """Get list of underlying datasets."""
        return self._datasets


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn=None,
    distributed: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """
    Create a DataLoader with optional distributed sampling.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        collate_fn: Custom collate function
        distributed: Whether to use distributed sampler
        rank: Distributed rank
        world_size: Number of distributed processes
        seed: Random seed for reproducibility

    Returns:
        DataLoader instance
    """
    from training.distributed import seed_worker

    sampler = None
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
        )
        # Don't shuffle in DataLoader when using sampler
        shuffle = False

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return loader
