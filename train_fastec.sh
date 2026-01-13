#
#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
# Get absolute path of the project root (where this script is located)
PROJECT_ROOT=$(pwd)

# Default paths assuming dataset is at ../trainfastec relative to project
# You can override these variables or edit them here
fastec_dataset_type=Fastec
fastec_root_path_training_data="${PROJECT_ROOT}/../trainfastec"

# Log directories relative to project
log_dir_pretrained_GS="${PROJECT_ROOT}/Pretrain_models_SelfSoftsplat"
log_dir="${PROJECT_ROOT}/logs"
#
cd deep_unroll_net


python train_SelfRSSR.py \
          --dataset_type=$fastec_dataset_type \
          --dataset_root_dir=$fastec_root_path_training_data \
          --log_dir_pretrained_GS=$log_dir_pretrained_GS \
          --log_dir=$log_dir \
          --gamma=1.0 \
          --lamda_L1=10 \
          --lamda_perceptual=1 \
          --lamda_flow_smoothness=0.1 \
          --img_H=480 \
          #--continue_train=True \
          #--start_epoch=81 \
          #--model_label=80


