# CoTracker Training Infrastructure (torch.distributed)

This is a refactored training infrastructure that replaces PyTorch Lightning Lite with pure `torch.distributed`. It uses Hydra for configuration and wandb for logging.

## Why This Refactor?

| Original (Lightning Lite) | New (torch.distributed) |
|---------------------------|-------------------------|
| Magic wrapping with `self.setup()` | Explicit `DistributedDataParallel()` |
| Hidden DDP sync behavior | Visible `model.no_sync()` for gradient accumulation |
| Argparse with 40+ flags | Hydra YAML configs with composition |
| TensorBoard logging | wandb logging |

## Quick Start

```bash
# Single GPU
python training/launch.py exp_name=my_exp

# 8 GPUs
torchrun --nproc_per_node=8 training/launch.py exp_name=my_exp

# With online model instead of offline
torchrun --nproc_per_node=8 training/launch.py model=cotracker3_online

# Override dataset path
torchrun --nproc_per_node=8 training/launch.py evaluation.dataset_root=/path/to/data
```

## Directory Structure

```
training/
├── launch.py              # Entry point (Hydra main)
├── trainer.py             # Trainer class with DDP/AMP
├── forward.py             # forward_batch() - loss computation
├── distributed.py         # DDP utilities (init, rank, barrier)
├── checkpoint.py          # Save/load with legacy support
├── logging_utils.py       # WandbLogger
├── config/
│   ├── default.yaml       # Main config
│   ├── model/
│   │   ├── cotracker3_offline.yaml
│   │   └── cotracker3_online.yaml
│   ├── data/
│   │   └── kubric.yaml
│   └── optimizer/
│       └── adamw_onecycle.yaml
├── data/
│   └── composed_dataset.py
└── tests/
    └── test_training.py
```

## Key Files Explained

### `trainer.py` - Main Training Logic

The `Trainer` class handles:
- **DDP initialization**: `dist.init_process_group()` + `DistributedDataParallel()`
- **Mixed precision**: `torch.cuda.amp.GradScaler` + `autocast`
- **Training loop**: Explicit epoch/batch iteration with gradient clipping
- **Checkpointing**: Save/resume with legacy format support
- **Evaluation**: Calls existing `run_test_eval()` from cotracker

```python
# Key pattern: explicit DDP wrapping
self.model = DDP(
    self.model,
    device_ids=[self.local_rank],
    find_unused_parameters=False,
)

# Key pattern: AMP with scaler
with autocast(enabled=True, dtype=torch.bfloat16):
    output = forward_batch(batch, model, cfg)
loss = compute_total_loss(output)
scaler.scale(loss).backward()
```

### `forward.py` - Loss Computation

Extracted from original `train_on_kubric.py:79-227`. Computes:
- **Coordinate loss** (visible points): `sequence_loss()` weighted by 0.05
- **Coordinate loss** (invisible points): `sequence_loss()` weighted by 0.01
- **Visibility loss**: `sequence_BCE_loss()`
- **Confidence loss**: `sequence_prob_loss()`

### `distributed.py` - DDP Utilities

```python
# Initialize distributed (handles torchrun and SLURM)
rank, world_size, local_rank, device = init_distributed()

# Sync barrier
barrier()

# Check if main process (for logging)
if is_main_process():
    logger.log(...)
```

### `checkpoint.py` - Legacy Compatibility

Handles all checkpoint formats:
1. Plain `state_dict` (old format)
2. `{'model': ..., 'optimizer': ...}` dict
3. Keys with `module.` prefix (DDP wrapped)
4. Filters out `time_emb`, `pos_emb` keys

```python
# Load old checkpoint into new training
load_checkpoint("old_model.pth", model, strict=False)
```

### `config/default.yaml` - Configuration

```yaml
defaults:
  - model: cotracker3_offline  # or cotracker3_online
  - data: kubric
  - optimizer: adamw_onecycle

exp_name: cotracker3_kubric
seed: 42

training:
  num_steps: 200000
  batch_size: 1
  mixed_precision: true

logging:
  use_wandb: true
  wandb_project: cotracker3
```

Override anything from command line:
```bash
torchrun ... training/launch.py training.num_steps=100000 training.batch_size=2
```

## Comparison with Original

### Original `train_on_kubric.py`
```python
class Lite(LightningLite):
    def run(self, args):
        model, optimizer = self.setup(model, optimizer)  # Magic DDP
        train_loader = self.setup_dataloaders(train_loader)
        self.backward(loss)  # Hidden scaler
        self.barrier()
```

### New `trainer.py`
```python
class Trainer:
    def setup(self):
        self.model = DDP(self.model, device_ids=[self.local_rank])  # Explicit

    def train(self):
        with autocast(enabled=True):
            output = forward_batch(batch, model, cfg)
        scaler.scale(loss).backward()  # Visible scaler
        barrier()
```

## Unchanged Code

The following are **imported but not modified**:
- `cotracker/datasets/kubric_movif_dataset.py` - KubricMovifDataset
- `cotracker/models/core/cotracker/` - Model architectures
- `cotracker/models/core/cotracker/losses.py` - Loss functions
- `cotracker/utils/train_utils.py` - `run_test_eval()`
- `cotracker/evaluation/` - Evaluation code

Original `train_on_kubric.py` still works if you prefer Lightning.

## Running Tests

```bash
pytest training/tests/test_training.py -v
```

Tests cover:
- Distributed utilities (rank detection, seeding)
- Checkpoint loading (all legacy formats)
- Logger (disabled mode)
- Config file validation

## Common Issues

### "KubricMovifDataset got unexpected keyword argument"
Check `training/config/data/kubric.yaml` uses correct parameter names:
- `sample_vis_last_frame` (not `sample_vis_1st_frame`)
- `split: train`

### wandb not logging
- Only rank 0 logs to wandb
- Check `logging.use_wandb: true` in config
- Install wandb: `pip install wandb`

### Checkpoint not loading
The loader handles all formats automatically. If issues persist:
```python
# Check what's in the checkpoint
ckpt = torch.load("model.pth")
print(ckpt.keys())
print(list(ckpt.get('model', ckpt).keys())[:5])
```
