# Quick Training Status Check
# Run this to see current training progress

Write-Host ""
Write-Host "=== Training Status Check ===" -ForegroundColor Cyan

# Check if training is running
$trainingProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.MainWindowTitle -like "*train*"}
if ($trainingProcess) {
    Write-Host "Training is RUNNING (PID: $($trainingProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "No training process found" -ForegroundColor Yellow
}

# Check saved checkpoints
Write-Host ""
Write-Host "=== Saved Checkpoints ===" -ForegroundColor Cyan
$checkpointDir = ".\logs\MyTraining_Fastec\"
if (Test-Path $checkpointDir) {
    $checkpoints = Get-ChildItem $checkpointDir -Filter "*_net_*.pth" -ErrorAction SilentlyContinue | 
                   Select-Object Name,@{Name='SizeMB';Expression={[math]::Round($_.Length/1MB,1)}},LastWriteTime |
                   Sort-Object LastWriteTime -Descending
    if ($checkpoints) {
        $checkpoints | Format-Table -AutoSize
        $latestEpoch = ($checkpoints[0].Name -split '_')[0]
        Write-Host "Latest checkpoint: Epoch $latestEpoch" -ForegroundColor Green
    } else {
        Write-Host "No checkpoints saved yet (training just started)" -ForegroundColor Yellow
    }
} else {
    Write-Host "Training directory not created yet" -ForegroundColor Yellow
}

# Check GPU usage
Write-Host ""
Write-Host "=== GPU Status ===" -ForegroundColor Cyan
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader

Write-Host ""
