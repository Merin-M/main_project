# Resume Training from a Checkpoint
# Edit the variables below to match your checkpoint

# Which epoch to resume from (e.g., 25 if you stopped at epoch 25)
$resume_epoch = 25

# Dataset paths
$fastec_train = "C:\Users\hp\Downloads\fastec_rs_train\train"
$fastec_val = "C:\Users\hp\Downloads\fastec_rs_train\val"

# Log directory (same as where you started training)
$log_dir = "..\logs\MyTraining_Fastec\"

# Navigate to training directory
cd deep_unroll_net

# Activate virtual environment
..\venv_selfrssplat\Scripts\Activate.ps1

# Resume training
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
    --save_freq=1 `
    --continue_train=True `
    --start_epoch=$resume_epoch `
    --model_label=$resume_epoch

# This will load checkpoint from epoch $resume_epoch and continue to epoch 50
