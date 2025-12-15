"""
Output path management.

This module handles the creation and management of output directory structures
for saving alignment results.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class OutputPaths:
    """
    Container for output file paths.
    
    Provides a structured way to access all output locations for a single
    alignment run.
    
    Attributes:
        root: Root output directory
        plots: Subdirectory for plots
        npz: Path for NumPy archive
        atf: Path for ATF file
        hdf5: Path for HDF5 file
        stats: Path for statistics report
        base_name: Base name derived from input file
    """
    
    root: Path
    plots: Path
    npz: Path
    atf: Path
    hdf5: Path
    stats: Path
    base_name: str
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.plots.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output directories: {self.root}")
    
    def exists(self, format: str = "all") -> bool:
        """
        Check if output files exist.
        
        Args:
            format: 'all', 'npz', 'atf', 'hdf5', or 'any'
        
        Returns:
            True if specified files exist
        """
        if format == "all":
            return self.npz.exists() and self.atf.exists() and self.hdf5.exists()
        elif format == "any":
            return self.npz.exists() or self.atf.exists() or self.hdf5.exists()
        elif format == "npz":
            return self.npz.exists()
        elif format == "atf":
            return self.atf.exists()
        elif format == "hdf5":
            return self.hdf5.exists()
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def get_plot_path(self, plot_name: str, extension: str = "png") -> Path:
        """Get path for a specific plot."""
        return self.plots / f"{self.base_name}_{plot_name}.{extension}"
    
    def __repr__(self) -> str:
        return f"OutputPaths(root={self.root}, base_name='{self.base_name}')"


def build_output_paths(
    input_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    create: bool = True,
) -> OutputPaths:
    """
    Build output paths based on input file and optional output directory.
    
    Default structure:
        aligned/<basename>/
            <basename>_aligned.npz
            <basename>_aligned.atf
            <basename>_aligned.h5
            plots/
                <basename>_*.png
                <basename>_stats.txt
    
    Args:
        input_file: Path to input ABF file
        output_dir: Optional custom output directory
        create: Whether to create directories immediately
    
    Returns:
        OutputPaths with all file paths configured
    
    Example:
        >>> paths = build_output_paths("data/recording.abf")
        >>> print(paths.npz)
        aligned/recording/recording_aligned.npz
    """
    input_path = Path(input_file)
    base_name = input_path.stem
    
    if output_dir is not None:
        root = Path(output_dir)
    else:
        # Default: aligned/<basename>/
        root = Path("aligned") / base_name
    
    plots = root / "plots"
    
    paths = OutputPaths(
        root=root,
        plots=plots,
        npz=root / f"{base_name}_aligned.npz",
        atf=root / f"{base_name}_aligned.atf",
        hdf5=root / f"{base_name}_aligned.h5",
        stats=plots / f"{base_name}_stats.txt",
        base_name=base_name,
    )
    
    if create:
        paths.create_directories()
    
    logger.debug(f"Built output paths for '{base_name}'")
    
    return paths


def ensure_output_dir(
    path: Union[str, Path],
    create: bool = True,
) -> Path:
    """
    Ensure output directory exists.
    
    Args:
        path: Directory path
        create: Whether to create if missing
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    
    if create:
        path.mkdir(parents=True, exist_ok=True)
    
    return path
