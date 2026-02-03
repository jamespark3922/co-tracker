#!/bin/bash
# CoTracker Training Launch Script
#
# Usage:
#   ./scripts/train.sh                           # Default: 8 GPUs, offline model
#   ./scripts/train.sh my_exp 4                  # 4 GPUs with custom name
#   ./scripts/train.sh my_exp 8 online           # 8 GPUs, online model
#   bash scripts/train.sh my_exp 1 cotracker3_offline
#
# For SLURM:
#   sbatch scripts/train.sh my_exp 8 offline

set -e

# Configuration
EXP_NAME=${1:-"cotracker3_kubric_offline"}
NUM_GPUS=${2:-8}
MODEL_TYPE=${3:-"cotracker3_offline"}  # cotracker3_offline or cotracker3_online
DATASET_ROOT=${DATASET_ROOT:-"/weka/oe-training-default/mm-olmo/video_datasets/point_track/CoTracker3_Kubric/"}

# Derived settings
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================"
echo "CoTracker Training"
echo "============================================"
echo "Experiment: $EXP_NAME"
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_TYPE"
echo "Dataset root: $DATASET_ROOT"
echo "Project root: $PROJECT_ROOT"
echo "============================================"

cd "$PROJECT_ROOT"

# Check if running under SLURM
if [ -n "$SLURM_JOB_ID" ]; then
    echo "Running under SLURM (Job ID: $SLURM_JOB_ID)"

    # SLURM multi-node setup
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=${MASTER_PORT:-29500}

    echo "Master: $MASTER_ADDR:$MASTER_PORT"
    echo "Nodes: $SLURM_NNODES"
    echo "Tasks per node: $SLURM_NTASKS_PER_NODE"

    srun torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$NUM_GPUS \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        training/launch.py \
        exp_name=$EXP_NAME \
        model=$MODEL_TYPE \
        evaluation.dataset_root=$DATASET_ROOT \
        "$@"
else
    # Single node launch
    echo "Running single-node training"

    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=${MASTER_PORT:-29500} \
        training/launch.py \
        exp_name=$EXP_NAME \
        model=$MODEL_TYPE \
        evaluation.dataset_root=$DATASET_ROOT \
        "${@:4}"  # Pass additional arguments
fi

echo "Training complete!"
