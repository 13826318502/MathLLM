param([switch]$Configure)

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

# Open the configuration dialog on first run, or when explicitly asked to.
if ($Configure -or -not (Test-Path -LiteralPath $envFile)) {
    Write-Host "Opening the model configuration dialog..." -ForegroundColor Cyan
    & $pythonPath -m app.tools.config_ui --env-file $envFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Configuration saved without starting, or cancelled." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 0
    }
    Import-LocalEnvironment -Path $envFile
}

# Solver endpoint: new names first, legacy names as fallback.
$solverBaseUrl = $env:MATHLLM_SOLVER_BASE_URL
if (-not $solverBaseUrl) { $solverBaseUrl = $env:MATHLLM_API_BASE_URL }
$solverModelName = $env:MATHLLM_SOLVER_MODEL
if (-not $solverModelName) { $solverModelName = $env:MATHLLM_MODEL_NAME }
$solverApiKey = $env:MATHLLM_SOLVER_API_KEY
if (-not $solverApiKey) { $solverApiKey = $env:MATHLLM_API_KEY }

if ($solverBaseUrl -eq "http://127.0.0.1:11434/v1" -or
    $solverBaseUrl -eq "http://localhost:11434/v1") {
    try {
        $ollamaTags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5
        $installedModels = @($ollamaTags.models | ForEach-Object { $_.name })
        $modelReady = $installedModels -contains $solverModelName -or
            $installedModels -contains "$($solverModelName):latest" -or
            ($installedModels | Where-Object { $_ -like "$($solverModelName):*" }).Count -gt 0
        if (-not $modelReady) {
            Write-Host "Ollama is running, but model '$solverModelName' is not installed." -ForegroundColor Red
            Write-Host "Available models: $($installedModels -join ', ')" -ForegroundColor Yellow
            Write-Host "Install it with: ollama create $solverModelName -f models/cpu/Modelfile" -ForegroundColor Yellow
            Read-Host "Press Enter to exit"
            exit 1
        }
        Write-Host "Ollama model ready: $solverModelName" -ForegroundColor Green
    }
    catch {
        Write-Host "Cannot reach Ollama at http://127.0.0.1:11434." -ForegroundColor Red
        Write-Host "Start Ollama first, then run this launcher again." -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
}

if (-not $solverBaseUrl) {
    $solverBaseUrl = Read-Host "Enter the solver API base URL (for example https://api.openai.com/v1)"
    $env:MATHLLM_SOLVER_BASE_URL = $solverBaseUrl
}
if (-not $solverModelName) {
    $solverModelName = Read-Host "Enter the solver model name (for example deepseek-chat)"
    $env:MATHLLM_SOLVER_MODEL = $solverModelName
}
if (-not $solverApiKey) {
    $secureKey = Read-Host "Enter the solver API Key (input hidden)" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $env:MATHLLM_SOLVER_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
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

if ($env:MATHLLM_ORCHESTRATOR_BASE_URL -and $env:MATHLLM_ORCHESTRATOR_MODEL) {
    Write-Host "Orchestrator: $env:MATHLLM_ORCHESTRATOR_MODEL @ $env:MATHLLM_ORCHESTRATOR_BASE_URL" -ForegroundColor Cyan
}
else {
    Write-Host "Orchestrator: not configured, everything runs on the local solver." -ForegroundColor Yellow
}
Write-Host "Solver: $solverModelName @ $solverBaseUrl" -ForegroundColor Cyan

# Knowledge base: build the Chroma vector index on first run so that
# knowledge-base questions work without a manual indexing step.
Write-Host "Checking knowledge base index..." -ForegroundColor Cyan
& $pythonPath -m app.services.rag_service --ensure
if ($LASTEXITCODE -ne 0) {
    Write-Host "Knowledge base index is unavailable; knowledge-base questions will fail until it is built." -ForegroundColor Yellow
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
