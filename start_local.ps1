$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$envFile = Join-Path $projectRoot ".env.local"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host "Virtual environment not found: $pythonPath" -ForegroundColor Red
    Write-Host "Create .venv and install requirements.txt first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

function Import-LocalEnvironment {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        if ($trimmed -match '^([^=]+)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

Import-LocalEnvironment -Path $envFile

if (-not $env:MATHLLM_API_BASE_URL) {
    $env:MATHLLM_API_BASE_URL = Read-Host "Enter API base URL (for example https://api.openai.com/v1)"
}
if (-not $env:MATHLLM_MODEL_NAME) {
    $env:MATHLLM_MODEL_NAME = Read-Host "Enter model name (for example deepseek-chat)"
}
if (-not $env:MATHLLM_API_KEY) {
    $secureKey = Read-Host "Enter API Key (input hidden)" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:MATHLLM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

if (-not $env:MATHLLM_API_HOST) {
    $env:MATHLLM_API_HOST = "127.0.0.1"
}
if (-not $env:MATHLLM_API_PORT) {
    $env:MATHLLM_API_PORT = "8080"
}
if (-not $env:MATHLLM_ALLOW_ORIGINS) {
    $env:MATHLLM_ALLOW_ORIGINS = "http://127.0.0.1:7860,http://localhost:7860"
}

Write-Host "Starting FastAPI (http://localhost:$env:MATHLLM_API_PORT)..." -ForegroundColor Cyan
Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot -ArgumentList @("-m", "app.api.main")

Start-Sleep -Seconds 2

Write-Host "Starting web frontend (http://localhost:7860)..." -ForegroundColor Cyan
Start-Process -FilePath $pythonPath -WorkingDirectory $projectRoot -ArgumentList @("-m", "http.server", "7860", "--directory", (Join-Path $projectRoot "web"))

Start-Sleep -Seconds 3
$cacheBust = [DateTimeOffset]::Now.ToUnixTimeSeconds()
Start-Process "http://localhost:7860/?launcher=$cacheBust#solve"

Write-Host ""
Write-Host "MathLLM started." -ForegroundColor Green
Write-Host "Frontend: http://localhost:7860"
Write-Host "Backend: http://localhost:$env:MATHLLM_API_PORT"
Write-Host "The API key is kept in the current process environment and is not written to project files."
