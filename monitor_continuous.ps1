# Continuous GPU Temperature Monitor
# This will check temperature every 60 seconds
# Press Ctrl+C to stop monitoring

Write-Host "`n=== Continuous GPU Monitor ===" -ForegroundColor Cyan
Write-Host "Checking temperature every 60 seconds..." -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop`n" -ForegroundColor Yellow

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    $tempRaw = nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
    $temp = $tempRaw.Trim()
    
    # Check if we got a valid temperature
    if ([string]::IsNullOrWhiteSpace($temp)) {
        Write-Host "[$timestamp] GPU: Unable to read temperature" -ForegroundColor Red
    } else {
        $tempNum = [int]$temp
        
        # Color based on temperature
        if ($tempNum -lt 70) {
            Write-Host "[$timestamp] GPU: ${tempNum}°C - SAFE" -ForegroundColor Green
        } elseif ($tempNum -lt 80) {
            Write-Host "[$timestamp] GPU: ${tempNum}°C - Normal" -ForegroundColor Yellow
        } elseif ($tempNum -lt 85) {
            Write-Host "[$timestamp] GPU: ${tempNum}°C - CAUTION!" -ForegroundColor Yellow
        } else {
            Write-Host "[$timestamp] GPU: ${tempNum}°C - WARNING! HOT!" -ForegroundColor Red
            # Optional: beep
            [console]::beep(1000,500)
        }
    }
    
    Start-Sleep -Seconds 60
}
