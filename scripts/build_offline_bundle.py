#!/usr/bin/env python3
"""
Build an offline installation bundle for Windows.

This script creates a complete package that can be transferred via USB
to computers without internet access.

Usage:
    python scripts/build_offline_bundle.py --platform win_amd64 --python 3.11
    
The output will be in dist/offline-bundle-windows/
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def run_command(cmd: list, cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"  STDOUT: {result.stdout}")
        print(f"  STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    return result


def build_wheel(project_root: Path, dist_dir: Path) -> Path:
    """Build the wheel for the package."""
    print("Building wheel...")
    
    # Clean old builds
    build_dir = project_root / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # Ensure dist_dir exists
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Try using uv first (preferred for this project)
    try:
        result = run_command(
            ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
            cwd=project_root,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to pip build
        try:
            result = run_command(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "-w", str(dist_dir), str(project_root)],
                cwd=project_root,
            )
        except subprocess.CalledProcessError:
            # Last resort: try build module
            result = run_command(
                [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
                cwd=project_root,
            )
    
    # Find the built wheel
    wheels = list(dist_dir.glob("ephyalign-*.whl"))
    if not wheels:
        raise RuntimeError("No wheel built!")
    
    return wheels[0]


def download_dependencies(
    project_root: Path,
    output_dir: Path,
    target_platform: str,
    python_version: str,
) -> None:
    """Download all dependencies as wheels for the target platform."""
    print(f"Downloading dependencies for {target_platform} (Python {python_version})...")
    
    wheels_dir = output_dir / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    
    # Map platform names for pip
    platform_map = {
        "win_amd64": "win_amd64",
        "win32": "win32",
        "macosx_x86_64": "macosx_10_9_x86_64",
        "macosx_arm64": "macosx_11_0_arm64",
        "manylinux_x86_64": "manylinux2014_x86_64",
    }
    
    pip_platform = platform_map.get(target_platform, target_platform)
    
    # Read dependencies from pyproject.toml
    import tomllib
    pyproject_path = project_root / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)
    
    dependencies = pyproject.get("project", {}).get("dependencies", [])
    
    # Create a temporary requirements file
    req_file = output_dir / "requirements.txt"
    req_file.write_text("\n".join(dependencies))
    
    print(f"  Dependencies to download: {', '.join(d.split('>=')[0].split('==')[0] for d in dependencies)}")
    
    # Ensure pip is available using uv pip install pip, then use uv export + pip
    # First, try using uv pip compile to get resolved dependencies
    lock_file = output_dir / "requirements.lock"
    
    # Method 1: Use uv pip compile + download in separate venv
    success = False
    
    # Create a temporary venv with pip for downloads
    temp_venv = output_dir / ".temp_venv"
    print(f"  Creating temporary venv with pip...")
    
    try:
        # Create venv
        run_command(["uv", "venv", str(temp_venv)], cwd=project_root, check=True)
        
        # Install pip into the temp venv
        run_command(["uv", "pip", "install", "--python", str(temp_venv / "bin" / "python"), "pip"], 
                    cwd=project_root, check=True)
        
        # Use the temp venv's pip to download
        temp_pip = temp_venv / "bin" / "pip"
        
        cmd = [
            str(temp_pip), "download",
            "-d", str(wheels_dir),
            "--platform", pip_platform,
            "--python-version", python_version,
            "--only-binary=:all:",
            "-r", str(req_file),
        ]
        
        run_command(cmd, cwd=project_root, check=True)
        success = True
    except subprocess.CalledProcessError as e:
        print(f"  Platform-specific download failed, trying without platform constraint...")
        try:
            cmd = [
                str(temp_pip), "download",
                "-d", str(wheels_dir),
                "-r", str(req_file),
            ]
            run_command(cmd, cwd=project_root, check=True)
            success = True
        except subprocess.CalledProcessError as e2:
            print(f"  Warning: pip download failed: {e2}")
    finally:
        # Cleanup temp venv
        if temp_venv.exists():
            shutil.rmtree(temp_venv)
    
    # Clean up
    if req_file.exists():
        req_file.unlink()
    
    wheel_count = len(list(wheels_dir.glob("*.whl")))
    tar_count = len(list(wheels_dir.glob("*.tar.gz")))
    print(f"  Downloaded {wheel_count} wheel files, {tar_count} source distributions")
    
    if wheel_count == 0 and tar_count == 0:
        print("  WARNING: No packages downloaded! The offline bundle may not work.")
        print("  You may need to manually download dependencies from PyPI.")


def create_install_scripts(output_dir: Path, project_name: str = "ephyalign") -> None:
    """Create installation scripts for the target system."""
    
    # Windows batch script
    bat_script = output_dir / "install.bat"
    bat_script.write_text(f'''@echo off
REM Offline installation script for {project_name}
REM Run this script from Command Prompt or PowerShell

echo ========================================
echo {project_name} Offline Installer
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10 or later from python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Install from local wheels (no internet required)
echo Installing {project_name} and dependencies...
pip install --no-index --find-links=wheels {project_name}
if errorlevel 1 (
    echo ERROR: Installation failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo ========================================
echo.
echo To use {project_name}:
echo   1. Open Command Prompt
echo   2. Navigate to this folder
echo   3. Run: venv\\Scripts\\activate.bat
echo   4. Run: {project_name} --help
echo.
echo Or run: venv\\Scripts\\{project_name}.exe --help
echo.
pause
''')
    
    # PowerShell script
    ps_script = output_dir / "install.ps1"
    ps_script.write_text(f'''# Offline installation script for {project_name}
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1

Write-Host "========================================"
Write-Host "{project_name} Offline Installer"
Write-Host "========================================"
Write-Host ""

# Check Python
try {{
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion"
}} catch {{
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from python.org"
    Read-Host "Press Enter to exit"
    exit 1
}}

# Create virtual environment
Write-Host "Creating virtual environment..."
python -m venv venv

# Activate and install
Write-Host "Installing {project_name}..."
.\\venv\\Scripts\\Activate.ps1
pip install --no-index --find-links=wheels {project_name}

if ($LASTEXITCODE -eq 0) {{
    Write-Host ""
    Write-Host "========================================"
    Write-Host "Installation complete!" -ForegroundColor Green
    Write-Host "========================================"
    Write-Host ""
    Write-Host "To use {project_name}:"
    Write-Host "  1. Run: .\\venv\\Scripts\\Activate.ps1"
    Write-Host "  2. Run: {project_name} --help"
}} else {{
    Write-Host "Installation failed!" -ForegroundColor Red
}}

Read-Host "Press Enter to exit"
''')
    
    # Shell script for macOS/Linux
    sh_script = output_dir / "install.sh"
    sh_script.write_text(f'''#!/bin/bash
# Offline installation script for {project_name}

echo "========================================"
echo "{project_name} Offline Installer"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

echo "Found: $(python3 --version)"
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install
source venv/bin/activate
pip install --no-index --find-links=wheels {project_name}

if [ $? -eq 0 ]; then
    echo
    echo "========================================"
    echo "Installation complete!"
    echo "========================================"
    echo
    echo "To use {project_name}:"
    echo "  1. Run: source venv/bin/activate"
    echo "  2. Run: {project_name} --help"
else
    echo "Installation failed!"
fi
''')
    sh_script.chmod(0o755)


def create_readme(output_dir: Path) -> None:
    """Create README for the offline bundle."""
    readme = output_dir / "README.txt"
    readme.write_text('''
================================================================================
                    EPHYALIGN OFFLINE INSTALLATION BUNDLE
================================================================================

This bundle contains everything needed to install ephyalign on a computer
WITHOUT internet access.

REQUIREMENTS:
  - Python 3.10 or later must be installed
  - Download Python from: https://www.python.org/downloads/
  - IMPORTANT: Check "Add Python to PATH" during installation!

INSTALLATION:

  Windows:
    Option 1: Double-click install.bat
    Option 2: Right-click install.ps1 > Run with PowerShell

  macOS/Linux:
    Run: ./install.sh

WHAT'S INCLUDED:
  - wheels/           All required Python packages
  - install.bat       Windows batch installer
  - install.ps1       Windows PowerShell installer  
  - install.sh        macOS/Linux installer
  - README.txt        This file

AFTER INSTALLATION:

  Windows:
    1. Open Command Prompt
    2. Navigate to the installation folder
    3. Run: venv\\Scripts\\activate.bat
    4. Run: ephyalign --help

  macOS/Linux:
    1. Run: source venv/bin/activate
    2. Run: ephyalign --help

EXAMPLE USAGE:

    # Get info about an ABF file
    ephyalign info my_recording.abf
    
    # Process a recording
    ephyalign process my_recording.abf
    
    # Process with custom settings
    ephyalign process my_recording.abf --pre-time 0.5 --post-time 3.0

TROUBLESHOOTING:

  "Python is not recognized..."
    - Install Python from python.org
    - Make sure "Add Python to PATH" is checked during installation
    - Restart Command Prompt after installation

  "pip is not recognized..."
    - Python may not be installed correctly
    - Try reinstalling Python with "Add Python to PATH" checked

  Installation fails:
    - Make sure you have write permissions in the folder
    - Try running Command Prompt as Administrator

================================================================================
''')


def create_bundle(
    target_platform: str = "win_amd64",
    python_version: str = "3.11",
) -> Path:
    """Create complete offline installation bundle."""
    
    project_root = get_project_root()
    
    # Create output directory
    bundle_name = f"ephyalign-offline-{target_platform}-py{python_version}"
    output_dir = project_root / "dist" / bundle_name
    
    if output_dir.exists():
        print(f"Removing existing bundle: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True)
    
    print(f"\nBuilding offline bundle: {bundle_name}")
    print("=" * 50)
    
    # 1. Build the wheel
    wheel_path = build_wheel(project_root, output_dir / "wheels")
    print(f"  Built: {wheel_path.name}")
    
    # 2. Download dependencies
    download_dependencies(project_root, output_dir, target_platform, python_version)
    
    # 3. Create installation scripts
    create_install_scripts(output_dir)
    print("  Created installation scripts")
    
    # 4. Create README
    create_readme(output_dir)
    print("  Created README")
    
    # 5. Create ZIP archive
    zip_path = project_root / "dist" / f"{bundle_name}.zip"
    print(f"\nCreating ZIP archive: {zip_path.name}")
    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        output_dir.parent,
        bundle_name,
    )
    
    print("\n" + "=" * 50)
    print("Bundle created successfully!")
    print(f"  Folder: {output_dir}")
    print(f"  ZIP:    {zip_path}")
    print("\nTo deploy:")
    print("  1. Copy the ZIP file to a USB drive")
    print("  2. Extract on the target Windows computer")
    print("  3. Run install.bat")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build offline installation bundle for ephyalign"
    )
    parser.add_argument(
        "--platform",
        choices=["win_amd64", "win32", "macosx_x86_64", "macosx_arm64", "current"],
        default="win_amd64",
        help="Target platform (default: win_amd64)",
    )
    parser.add_argument(
        "--python",
        default="3.11",
        help="Python version (default: 3.11)",
    )
    
    args = parser.parse_args()
    
    target_platform = args.platform
    if target_platform == "current":
        # Detect current platform
        system = platform.system().lower()
        machine = platform.machine().lower()
        if system == "windows":
            target_platform = "win_amd64" if "64" in machine else "win32"
        elif system == "darwin":
            target_platform = "macosx_arm64" if machine == "arm64" else "macosx_x86_64"
        else:
            target_platform = "manylinux_x86_64"
    
    create_bundle(target_platform, args.python)


if __name__ == "__main__":
    main()
