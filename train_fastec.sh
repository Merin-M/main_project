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
echo "Attempting to locate pip-installed CUDA runtime..."
PIP_CUDA_PATH=$(python -c "import os, sys; 
try: 
    import nvidia.cuda_runtime; 
    print(os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib')) 
except: 
    pass" 2>/dev/null)

# Fallback: Find it manually in site-packages if python import fails
if [ -z "$PIP_CUDA_PATH" ]; then
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
    if [ ! -z "$SITE_PACKAGES" ]; then
        PIP_CUDA_PATH=$(find $SITE_PACKAGES -name "libcudart.so.11.0" -printf "%h\n" | head -n 1)
    fi
fi

if [ ! -z "$PIP_CUDA_PATH" ]; then
    echo "Found pip-installed CUDA runtime at: $PIP_CUDA_PATH"
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PIP_CUDA_PATH
else
    echo "WARNING: Could not locate pip-installed CUDA runtime. 'cupy' might fail."
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


