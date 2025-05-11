# Set project directories
$PROJECT_DIR = Get-Location
$VENV_DIR = Join-Path -Path $PROJECT_DIR -ChildPath "venv"
$REQUIREMENTS_FILE = Join-Path -Path $PROJECT_DIR -ChildPath "requirements.txt"

# Check if virtual environment exists
if (-not (Test-Path $VENV_DIR)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VENV_DIR
    if (-not $?) {
        Write-Host "Failed to create virtual environment. Please make sure venv is installed."
        exit 1
    }
    Write-Host "Virtual environment created at $VENV_DIR"
} else {
    Write-Host "Virtual environment already exists at $VENV_DIR"
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
$ACTIVATE_SCRIPT = Join-Path -Path $VENV_DIR -ChildPath "Scripts\Activate.ps1"
& $ACTIVATE_SCRIPT

# Install requirements
Write-Host "Installing requirements..."
pip install -r $REQUIREMENTS_FILE
if (-not $?) {
    Write-Host "Failed to install some requirements."
    exit 1
}

Write-Host "Setup complete! Virtual environment is activated and all requirements are installed."
Write-Host "To deactivate the virtual environment, run 'deactivate'"