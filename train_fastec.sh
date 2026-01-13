#
#!/bin/bash

# !! Updata the path to the dataset and directory to 
# !! save your trained models with your own local path !!
# Get absolute path of the project root (where this script is located)
PROJECT_ROOT=$(pwd)

# Default paths assuming dataset is at ../trainfastec relative to project
# You can override these variables or edit them here
fastec_dataset_type=Fastec
fastec_root_path_training_data="${PROJECT_ROOT}/../fastec_rs_train"

# Log directories relative to project
log_dir_pretrained_GS="${PROJECT_ROOT}/Pretrain_models_SelfSoftsplat"
log_dir="${PROJECT_ROOT}/logs"
#
cd deep_unroll_net

# Add project root to PYTHONPATH so package_core can be found
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Attempt to add CUDA libraries to LD_LIBRARY_PATH
# 1. Try standard locations
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda-12.2/lib64:/usr/local/cuda-12/lib64:/usr/local/cuda-11/lib64

# 2. Try to find pip-installed nvidia-cuda-runtime-cu11
# 2. Try to find any pip-installed nvidia libraries (cuda_runtime, cublas, etc.)
echo "Attempting to locate pip-installed CUDA libraries..."
PIP_CUDA_PATHS=$(python -c "import os, glob, site; 
try:
    site_packages = site.getsitepackages()[0]
    # Look for nvidia/*/lib directories
    libs = glob.glob(os.path.join(site_packages, 'nvidia', '*', 'lib'))
    print(':'.join(libs))
except:
    pass" 2>/dev/null)

if [ ! -z "$PIP_CUDA_PATHS" ]; then
    echo "Found pip-installed CUDA libraries at: $PIP_CUDA_PATHS"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PIP_CUDA_PATHS
else
    echo "WARNING: Could not locate pip-installed CUDA libraries."
fi

echo "LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH"

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


