# Demo Video Generation Script for Windows
# Generates demo videos using pretrained Fastec model

# Create output directory
New-Item -ItemType Directory -Path "experiments\results_demo_fastec_video" -Force | Out-Null

# Navigate to deep_unroll_net
cd deep_unroll_net

# Activate virtual environment
..\venv_selfrssplat\Scripts\Activate.ps1

# Run inference to generate demo video
python inference_demo_video.py `
    --model_label='pre' `
    --results_dir="../experiments/results_demo_fastec_video" `
    --data_dir='../demo_video/Fastec' `
    --img_H=480 `
    --gamma=1.0 `
    --log_dir="C:\Users\hp\Downloads\Pretrain_models_SelfSoftsplat\Pretrain_models_SelfSoftsplat\deep_unroll_weights\pre_fastec_ft"

Write-Host ""
Write-Host "Demo video generation complete!" -ForegroundColor Green
Write-Host "Results saved to: experiments\results_demo_fastec_video" -ForegroundColor Cyan
