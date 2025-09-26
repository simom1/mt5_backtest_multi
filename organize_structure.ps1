# Create directories
$directories = @(
    "core",
    "strategies",
    "gui",
    "utils",
    "config",
    "data",
    "outputs",
    "tests"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "Created directory: $dir"
    }
}

# Move files to appropriate directories
$fileMappings = @{
    # Core files
    "main.py" = "core"
    "tuner.py" = "core"
    
    # GUI files
    "main_window.py" = "gui"
    "gui_app.py" = "gui"
    "test_gui.py" = "gui"
    "ui_components.py" = "gui"
    
    # Strategy files
    "strategy.py" = "strategies"
    
    # Config files
    "config.py" = "config"
    
    # Utility files (example, add more as needed)
    "重构说明.md" = "docs"
}

# Process each file mapping
foreach ($file in $fileMappings.Keys) {
    $source = $file
    $destination = $fileMappings[$file]
    
    if (Test-Path $source) {
        $destPath = Join-Path $destination (Split-Path $source -Leaf)
        
        # Ensure destination directory exists
        if (-not (Test-Path $destination)) {
            New-Item -ItemType Directory -Path $destination | Out-Null
        }
        
        Move-Item -Path $source -Destination $destPath -Force
        Write-Host "Moved: $source -> $destPath"
    }
}

# Create __init__.py files in package directories
$packageDirs = @("core", "strategies", "gui", "utils", "config")
foreach ($dir in $packageDirs) {
    $initFile = Join-Path $dir "__init__.py"
    if (-not (Test-Path $initFile)) {
        New-Item -ItemType File -Path $initFile -Force | Out-Null
        Write-Host "Created: $initFile"
    }
}

Write-Host "Project structure has been organized successfully!" -ForegroundColor Green
