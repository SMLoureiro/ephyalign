"""
Input/Output modules for ephyalign.

This package provides exporters for saving aligned epochs in various formats:
- ATF: Axon Text Format (Stimfit-compatible)
- HDF5: Hierarchical Data Format (Stimfit-compatible binary)
- NPZ: NumPy compressed archive
"""

from ephyalign.io.exporters import (
    save_atf,
    save_hdf5,
    save_npz,
    save_all_formats,
    write_stats_report,
)
from ephyalign.io.paths import build_output_paths, OutputPaths

__all__ = [
    "save_atf",
    "save_hdf5",
    "save_npz",
    "save_all_formats",
    "write_stats_report",
    "build_output_paths",
    "OutputPaths",
]
