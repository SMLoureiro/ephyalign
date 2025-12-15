# ephyalign

[![Tests](https://github.com/SMLoureiro/ephyalign/actions/workflows/test.yml/badge.svg)](https://github.com/SMLoureiro/ephyalign/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/ephyalign.svg)](https://badge.fury.io/py/ephyalign)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Electrophysiology Response Alignment Toolkit**

A Python package for automatically detecting and aligning stimulus-evoked responses in ABF (Axon Binary Format) electrophysiological recordings with sub-millisecond precision.

## Features

- 🎯 **Automatic stimulus detection** - Detects stimulus artifacts from the signal derivative without requiring a separate trigger channel
- ⚡ **Sub-millisecond alignment** - Refines epoch alignment by locking to the exact artifact onset
- 📊 **Comprehensive metrics** - Calculates peak amplitude, rise time, time-to-peak, AUC, and more
- 🔧 **Highly configurable** - YAML/TOML configuration files, environment variables, or programmatic API
- 📁 **Multiple output formats** - NPZ (Python), ATF (Stimfit text), HDF5 (Stimfit binary)
- 📈 **Diagnostic plots** - Automatic generation of overlay, average, zoom, and concatenated plots
- 🖥️ **Professional CLI** - Full-featured command-line interface for batch processing
- 🧪 **Well tested** - Comprehensive test suite with pytest

## Installation

### From PyPI (when published)

```bash
pip install ephyalign
```

### From source

```bash
git clone https://github.com/SMLoureiro/ephyalign.git
cd ephyalign
pip install -e .
```

### With optional dependencies

```bash
# With YAML config support
pip install ephyalign[yaml]

# With development tools
pip install ephyalign[dev]

# Everything
pip install ephyalign[all]
```

### Offline Installation (Windows without Internet)

For Windows computers without internet access, we provide an offline installation bundle:

1. **On a computer with internet**, create the bundle:
   ```bash
   cd ephyalign
   uv pip install build
   uv run python scripts/build_offline_bundle.py --platform win_amd64 --python 3.11
   ```

2. **Copy to USB**: Transfer `dist/ephyalign-offline-win_amd64-py3.11.zip` to USB

3. **On the offline Windows computer**:
   - Extract the ZIP file
   - Run `install.bat`

📖 See [docs/OFFLINE_INSTALLATION.md](docs/OFFLINE_INSTALLATION.md) for detailed instructions.

## Quick Start

### Command Line

```bash
# Process a single file
ephyalign process data/recording.abf

# Process with custom parameters
ephyalign process data/recording.abf --pre-time 0.5 --post-time 3.0 --channel 0

# Process multiple files
ephyalign batch data/*.abf --output-dir results/

# Get file information
ephyalign info data/recording.abf

# Generate example config
ephyalign init-config my_config.yaml
```

### Python API

```python
import ephyalign

# Simple one-liner
result = ephyalign.align_recording("data/recording.abf")
print(f"Found {result.n_epochs} epochs with {result.jitter_ms:.2f}ms jitter")

# Access aligned data
epochs = result.epochs_aligned  # Shape: (n_epochs, epoch_length)
time = result.alignment.time_axis  # Time axis in seconds

# Access metrics
print(f"Peak amplitude: {result.metrics.peak_amp_mean:.2f} ± {result.metrics.peak_amp_std:.2f}")
```

### With Custom Configuration

```python
from ephyalign import AlignmentConfig, align_recording

# Create configuration
config = AlignmentConfig(
    input_file="data/recording.abf",
    reference_channel=0,
)

# Customize detection parameters
config.detection.threshold_multiplier = 5.0
config.detection.min_interval_s = 3.0

# Customize epoch extraction
config.epoch.pre_time_s = 0.5
config.epoch.post_time_s = 3.0

# Customize output
config.output.save_plots = True
config.output.save_hdf5 = True

# Run alignment
result = align_recording(config=config)
```

### Using Configuration Files

Create a `config.yaml`:

```yaml
input_file: data/recording.abf
reference_channel: 0

detection:
  threshold_multiplier: 5.0
  min_interval_s: 3.0

epoch:
  pre_time_s: 0.5
  post_time_s: 3.0

output:
  save_npz: true
  save_atf: true
  save_hdf5: true
  save_plots: true
```

Then use it:

```python
from ephyalign import AlignmentConfig, align_recording

config = AlignmentConfig.from_yaml("config.yaml")
result = align_recording(config=config)
```

Or from CLI:

```bash
ephyalign process data/recording.abf -c config.yaml
```

## Output Structure

```
aligned/<recording_name>/
├── <name>_aligned.npz     # NumPy archive for Python analysis
├── <name>_aligned.atf     # Stimfit-compatible text format
├── <name>_aligned.h5      # Stimfit-compatible HDF5 format
└── plots/
    ├── <name>_zoom_alignment.png    # Alignment quality check
    ├── <name>_overlay_epochs.png    # All epochs overlaid
    ├── <name>_average_response.png  # Mean response ± SEM
    ├── <name>_concatenated_epochs.png
    └── <name>_stats.txt             # Statistics report
```

## Key Concepts

### How Detection Works

1. Computes the derivative of the signal to detect sharp transients
2. Sets threshold at `threshold_multiplier × std(derivative)`
3. Finds samples exceeding threshold with minimum inter-stimulus interval
4. Reports timing statistics (mean ISI, detection rate)

### How Alignment Works

1. Extracts raw epochs around each detected stimulus
2. Searches early portion of each epoch for exact artifact onset
3. Re-cuts all epochs to start at detected onset
4. Reports alignment jitter (sub-millisecond precision)

### Multi-Channel Support

For multi-channel recordings, detection is performed on a reference channel and the same alignment is applied to all channels:

```python
result = align_recording(
    "data/multichannel.abf",
    reference_channel=0,  # Use channel 0 for detection
)

# Access all channels
all_channels = result.epochs_all_aligned  # Shape: (n_channels, n_epochs, length)
```

## API Reference

### Main Functions

- `align_recording(input_file, config=None, **kwargs)` - Process a single recording
- `batch_align(input_files, config=None)` - Process multiple files

### Core Classes

- `AlignmentConfig` - Main configuration class
- `PipelineResult` - Container for all processing results
- `EpochMetrics` - Response metrics container

### Low-Level Functions

```python
from ephyalign.core import (
    load_recording,      # Load ABF file
    detect_stim_onsets,  # Detect stimuli
    build_epochs,        # Extract epochs
    refine_alignment,    # Refine alignment
    compute_epoch_metrics,  # Calculate metrics
)
```

## Development

### Setup

```bash
git clone https://github.com/SMLoureiro/ephyalign.git
cd ephyalign
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest                    # Run all tests
pytest -v                 # Verbose
pytest --cov=ephyalign    # With coverage
pytest -m "not slow"      # Skip slow tests
```

### Code Quality

```bash
black src tests           # Format code
ruff check src tests      # Lint
mypy src                  # Type check
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{ephyalign,
  author = {Loureiro, Samuel Matthews Mckay},
  title = {ephyalign: Electrophysiology Response Alignment Toolkit},
  year = {2025},
  url = {https://github.com/SMLoureiro/ephyalign}
}
```

## Acknowledgments

- Built with [pyabf](https://github.com/swharden/pyABF) for ABF file reading
- HDF5 output compatible with [Stimfit](https://github.com/neurodroid/stimfit)
- Inspired by the need for reliable stimulus-locked averaging in neuroscience research
