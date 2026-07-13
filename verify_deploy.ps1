# verify_deploy.ps1 — paste your Render URL on the first line, then run.
$URL = "https://YOUR-SERVICE-NAME.onrender.com"   # <-- edit me

Write-Host "`n--- GET $URL/api ---" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest "$URL/api" -UseBasicParsing
    Write-Host "Status: $($r.StatusCode)"
    $r.Content | ConvertFrom-Json | Format-List
} catch { Write-Host "ERR: $_" -ForegroundColor Red }

Write-Host "`n--- GET $URL/api/diag ---" -ForegroundColor Cyan
try {
    $r = Invoke-WebRequest "$URL/api/diag" -UseBasicParsing
    Write-Host "Status: $($r.StatusCode)"
    $j = $r.Content | ConvertFrom-Json
    Write-Host "demo_mode: $($j.demo_mode)"
    Write-Host "last_error: $($j.last_error)"
    Write-Host "bundle_path: $($j.bundle_path)"
    Write-Host "sklearn_version: $($j.sklearn_version)"
    Write-Host "numpy_version: $($j.numpy_version)"
    Write-Host "model_name_length: $($j.model_name_length)"
    Write-Host "model_name_crystallinity: $($j.model_name_crystallinity)"
} catch { Write-Host "ERR: $_" -ForegroundColor Red }

Write-Host "`n--- POST $URL/api/predict ---" -ForegroundColor Cyan
try {
    $body = '{"cellulose_group":"Wood / Pulp-based","acid_conc_wt_percent":60,"temp_c":50,"time_min":60}'
    $r = Invoke-WebRequest -Method POST "$URL/api/predict" -ContentType 'application/json' -Body $body -UseBasicParsing
    Write-Host "Status: $($r.StatusCode)"
    $r.Content | ConvertFrom-Json | Format-List
} catch { Write-Host "ERR: $_" -ForegroundColor Red }

Write-Host "`nOpen $URL in a browser to use the UI." -ForegroundColor Green