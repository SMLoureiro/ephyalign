# Offline Installation Guide

This guide explains how to install ephyalign on Windows computers that do not have internet access.

## Overview

The process involves two steps:
1. **On a computer with internet**: Create an offline installation bundle
2. **On the offline computer**: Run the installer from USB

---

## Step 1: Create the Offline Bundle (Internet Required)

On a macOS or Linux computer with internet access:

### Prerequisites

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Option A: Using the Build Script (Recommended)

```bash
cd /path/to/ephyalign

# Install build dependencies
uv pip install build

# Build for Windows 64-bit, Python 3.11
uv run python scripts/build_offline_bundle.py --platform win_amd64 --python 3.11

# Or for Windows 32-bit
uv run python scripts/build_offline_bundle.py --platform win32 --python 3.11
```

This creates:
- `dist/ephyalign-offline-win_amd64-py3.11/` - The bundle folder
- `dist/ephyalign-offline-win_amd64-py3.11.zip` - ZIP archive for USB transfer

### Option B: Manual Bundle Creation

If the script doesn't work, you can create the bundle manually:

```bash
# 1. Create output directory
mkdir -p dist/offline-windows/wheels

# 2. Build the package wheel
uv build --wheel --out-dir dist/offline-windows/wheels

# 3. Download Windows wheels for all dependencies
pip download \
    --dest dist/offline-windows/wheels \
    --platform win_amd64 \
    --python-version 3.11 \
    --only-binary=:all: \
    .

# 4. Copy the install scripts
cp scripts/install_windows/* dist/offline-windows/
```

---

## Step 2: Prepare the USB Drive

1. Copy the `ephyalign-offline-win_amd64-py3.11.zip` to a USB drive
2. Optionally, include the Python 3.11 Windows installer:
   - Download from: https://www.python.org/downloads/windows/
   - Get the "Windows installer (64-bit)" for Python 3.11.x

### Recommended USB Contents

```
USB Drive/
├── ephyalign-offline-win_amd64-py3.11.zip
├── python-3.11.x-amd64.exe  (optional, from python.org)
└── README.txt
```

---

## Step 3: Install on Windows (Offline)

### 3.1 Install Python (if not already installed)

If Python is not installed on the Windows computer:

1. Run `python-3.11.x-amd64.exe` from the USB drive
2. **IMPORTANT**: Check ✅ "Add Python to PATH"
3. Click "Install Now"
4. Restart Command Prompt after installation

To verify Python is installed:
```cmd
python --version
```

### 3.2 Extract and Install ephyalign

1. Copy `ephyalign-offline-win_amd64-py3.11.zip` to a folder (e.g., `C:\Tools\`)
2. Extract the ZIP file
3. Open Command Prompt and navigate to the folder:

```cmd
cd C:\Tools\ephyalign-offline-win_amd64-py3.11
```

4. Run the installer:

**Option A: Batch file**
```cmd
install.bat
```

**Option B: PowerShell**
```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

**Option C: Manual installation**
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install --no-index --find-links=wheels ephyalign
```

---

## Step 4: Using ephyalign

After installation, always activate the virtual environment first:

```cmd
cd C:\Tools\ephyalign-offline-win_amd64-py3.11
venv\Scripts\activate.bat
```

Now you can use the CLI:

```cmd
# Get help
ephyalign --help

# View file info
ephyalign info D:\data\recording.abf

# Process a recording
ephyalign process D:\data\recording.abf

# Process with custom settings
ephyalign process D:\data\recording.abf --pre-time 0.5 --post-time 3.0

# Batch process multiple files
ephyalign batch D:\data\*.abf --output-dir D:\results
```

---

## Creating a Desktop Shortcut

To make it easier to launch ephyalign:

1. Right-click on Desktop → New → Shortcut
2. Enter this command:
   ```
   cmd.exe /k "cd /d C:\Tools\ephyalign-offline-win_amd64-py3.11 && venv\Scripts\activate.bat"
   ```
3. Name it "ephyalign Terminal"
4. Click Finish

Now double-clicking opens a terminal ready for ephyalign.

---

## Troubleshooting

### "python is not recognized as an internal or external command"

Python is not in your PATH. Either:
1. Reinstall Python and check "Add Python to PATH"
2. Or use the full path: `C:\Users\<username>\AppData\Local\Programs\Python\Python311\python.exe`

### "pip install" fails with dependency errors

The bundle might be missing some wheels. On the internet-connected computer:
1. Check which package is missing
2. Download it manually: `pip download <package> --platform win_amd64`
3. Add to the `wheels/` folder

### Scripts are blocked by execution policy

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy RemoteSigned
```

### The installer creates a virtual environment but ephyalign doesn't work

Make sure to activate the virtual environment before each use:
```cmd
venv\Scripts\activate.bat
```

You should see `(venv)` at the start of your command prompt.

---

## Alternative: Standalone Executable (Advanced)

For completely portable installation without requiring Python:

```bash
# Install PyInstaller
uv pip install pyinstaller

# Create standalone executable
uv run pyinstaller --onefile --name ephyalign src/ephyalign/cli.py
```

This creates `dist/ephyalign.exe` which can be run directly without Python.
Note: The executable is larger (~50-100MB) and startup is slower.

---

## Updating on Offline Computers

To update ephyalign on an offline computer:

1. On internet-connected computer: Rebuild the bundle
2. Copy new ZIP to USB
3. On offline computer:
   ```cmd
   cd C:\Tools\ephyalign-offline-win_amd64-py3.11
   venv\Scripts\activate.bat
   pip install --no-index --find-links=wheels --upgrade ephyalign
   ```

---

## Technical Details

### What's in the Bundle

The offline bundle contains:

| File/Folder | Description |
|-------------|-------------|
| `wheels/` | All Python package wheels (.whl files) |
| `install.bat` | Windows batch installer |
| `install.ps1` | PowerShell installer |
| `README.txt` | Quick-start instructions |

### Supported Platforms

| Platform | Build flag |
|----------|------------|
| Windows 64-bit | `--platform win_amd64` |
| Windows 32-bit | `--platform win32` |
| macOS Intel | `--platform macosx_x86_64` |
| macOS Apple Silicon | `--platform macosx_arm64` |

### Python Version Compatibility

- Python 3.10, 3.11, 3.12, or 3.13
- Recommended: Python 3.11 (best compatibility with scientific packages)
