# Train SelfRSSplat on Windows with Fastec RS dataset

# Dataset paths
$fastec_dataset_type = "Fastec"
$fastec_root_path_training_data = "C:\Users\hp\Downloads\fastec_rs_train\train"
$fastec_root_path_validation_data = "C:\Users\hp\Downloads\fastec_rs_test\test"

# Log directories
# NOTE: If pretrained models are not available, the training will start with random initialization
# You can download pretrained GMFlow models from: https://github.com/haofeixu/gmflow
$log_dir_pretrained_GS = "C:\Users\hp\Desktop\MainProject\SelfRSSplat\logs\Pretrained\pretrain_vfi\"
$log_dir = "C:\Users\hp\Desktop\MainProject\SelfRSSplat\logs\"

# Change to deep_unroll_net directory
Set-Location -Path "C:\Users\hp\Desktop\MainProject\SelfRSSplat\deep_unroll_net"

# Run training with reduced batch size for 6GB VRAM and log to file
python train_SelfRSSR.py `
    --dataset_type=$fastec_dataset_type `
    --dataset_root_dir=$fastec_root_path_training_data `
    --dataset_val_root_dir=$fastec_root_path_validation_data `
    --log_dir_pretrained_GS=$log_dir_pretrained_GS `
    --log_dir=$log_dir `
    --batch_sz=2 `
    --gamma=1.0 `
    --lamda_L1=10 `
    --lamda_perceptual=1 `
    --lamda_flow_smoothness=0.1 `
    --img_H=480 *>&1 | Tee-Object -FilePath "..\training_log.txt"

