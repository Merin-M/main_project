# Start Training from Pretrained VFI Models
# This script will train for 50 epochs with automatic checkpoint saving

# Dataset paths
$fastec_train = "C:\Users\hp\Downloads\fastec_rs_train\train"
$fastec_val = "C:\Users\hp\Downloads\fastec_rs_train\val"

# Log directory (checkpoints will save here)
$log_dir = "..\logs\MyTraining_Fastec\"

# Navigate to training directory
cd deep_unroll_net

# Activate virtual environment
..\venv_selfrssplat\Scripts\Activate.ps1

# Start training
python -u train_SelfRSSR.py `
    --dataset_type=Fastec `
    --dataset_root_dir=$fastec_train `
    --dataset_val_root_dir=$fastec_val `
    --log_dir=$log_dir `
    --log_dir_pretrained_GS="../logs/Pretrained/pretrain_vfi/" `
    --batch_sz=2 `
    --gamma=1.0 `
    --lamda_L1=10 `
    --lamda_perceptual=1 `
    --lamda_flow_smoothness=0.1 `
    --img_H=480 `
    --max_epochs=50 `
    --save_freq=1

# Checkpoints will save EVERY EPOCH in logs/MyTraining_Fastec/
# To resume training, use r_training.ps1
