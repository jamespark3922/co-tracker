#!/bin/bash

# bash launch_training_kubric_offline_local.sh <EXP_DIR> <EXP_NAME> <DATASET_ROOT> <NUM_STEPS>

# bash launch_training_kubric_offline_local.sh exp kubric-offline  \
# /weka/oe-training-default/mm-olmo/video_datasets/point_track/CoTracker3_Kubric/ 50000


set -x

EXP_DIR=$1
EXP_NAME=$2
DATASET_ROOT=$3
NUM_STEPS=$4

DATE=$(date +%Y%m%d_%H%M%S)

NUM_NODES=1


echo `which python`

mkdir -p ${EXP_DIR}/${DATE}_${EXP_NAME}/logs/;
# mkdir ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3;
# find . \( -name "*.sh" -o -name "*.py" \) -type f -exec cp --parents {} ${EXP_DIR}/${DATE}_${EXP_NAME}/cotracker3 \;

python train_on_kubric.py --batch_size 1 \
--num_steps ${NUM_STEPS} --ckpt_path ${EXP_DIR}/${DATE}_${EXP_NAME} --model_name cotracker_three \
--save_freq 200 --sequence_len 60 --eval_datasets tapvid_davis_first tapvid_stacking \
--traj_per_sample 512 --sliding_window_len 60 --train_datasets kubric \
--save_every_n_epoch 5 --evaluate_every_n_epoch 5 --model_stride 4 --dataset_root ${DATASET_ROOT} --num_nodes ${NUM_NODES} \
--num_virtual_tracks 64 --mixed_precision --offline_model --random_frame_rate --query_sampling_method random \
--corr_radius 3 --wdecay 0.0005 --random_seq_len --linear_layer_for_vis_conf --validate_at_start --add_huber_loss
