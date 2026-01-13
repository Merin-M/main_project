# GPU Temperature Monitor
# Run this to check current GPU temperature

Write-Host "`n=== GPU Temperature Check ===" -ForegroundColor Cyan

# Get GPU temperature
$temp = nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits

Write-Host "Current GPU Temperature: $temp°C" -ForegroundColor White

# Temperature status
if ($temp -lt 70) {
    Write-Host "Status: SAFE (Cool)" -ForegroundColor Green
} elseif ($temp -lt 80) {
    Write-Host "Status: NORMAL (Warm)" -ForegroundColor Yellow
} elseif ($temp -lt 85) {
    Write-Host "Status: CAUTION (Hot)" -ForegroundColor Yellow
    Write-Host "  → Watch closely, ensure good ventilation" -ForegroundColor Yellow
} else {
    Write-Host "Status: WARNING (Very Hot!)" -ForegroundColor Red
    Write-Host "  → Consider stopping training and cooling down" -ForegroundColor Red
}

# Get GPU utilization
$util = nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
$mem = nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits

Write-Host "`nGPU Usage: $util%" -ForegroundColor Cyan
Write-Host "Memory: $mem%" -ForegroundColor Cyan

Write-Host "`n"
