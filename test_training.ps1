# Quick test training with minimal settings
cd C:\Users\hp\Desktop\MainProject\SelfRSSplat\deep_unroll_net

python -u train_SelfRSSR.py `
    --dataset_type=Fastec `
    --dataset_root_dir="C:\Users\hp\Downloads\fastec_rs_train\train" `
    --dataset_val_root_dir="C:\Users\hp\Downloads\fastec_rs_test\test" `
    --log_dir="../logs_test/" `
    --batch_sz=1 `
    --img_H=320 `
    --crop_sz_H=256 `
    --crop_sz_W=192 `
    --gamma=1.0 `
    --lamda_L1=10 `
    --lamda_perceptual=1 `
    --lamda_flow_smoothness=0.1 `
    --max_epochs=2 `
    --save_freq=1
