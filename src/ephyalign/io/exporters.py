"""
File format exporters for aligned epochs.

This module provides functions for saving aligned epochs in various formats:
- ATF: Axon Text Format (tab-delimited, Stimfit-compatible)
- HDF5: Hierarchical Data Format (Stimfit-compatible binary structure)
- NPZ: NumPy compressed archive (for Python analysis)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tables

from ephyalign.io.paths import OutputPaths
from ephyalign.core.metrics import EpochMetrics

logger = logging.getLogger(__name__)


# HDF5 table descriptions for Stimfit compatibility
class RecordingDescription(tables.IsDescription):
    """HDF5 table structure for recording-level metadata."""
    channels = tables.Int32Col()
    date = tables.StringCol(128)
    time = tables.StringCol(128)


class ChannelDescription(tables.IsDescription):
    """HDF5 table structure for channel-level metadata."""
    n_sections = tables.Int32Col()


class SectionDescription(tables.IsDescription):
    """HDF5 table structure for section (sweep) metadata."""
    dt = tables.Float64Col()
    xunits = tables.StringCol(16)
    yunits = tables.StringCol(16)


def save_npz(
    path: Union[str, Path],
    epochs: np.ndarray,
    time_axis: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save aligned epochs to NumPy compressed archive.
    
    Args:
        path: Output file path
        epochs: Epoch data (2D or 3D array)
        time_axis: Time axis in seconds
        metadata: Optional additional data to include
    
    Returns:
        Path to saved file
    
    The NPZ file contains:
        - 'epochs': The aligned epoch data
        - 'time': Time axis in seconds
        - Any additional metadata passed
    """
    path = Path(path)
    
    data = {
        "epochs": epochs,
        "time": time_axis,
    }
    
    if metadata is not None:
        data.update(metadata)
    
    np.savez(path, **data)
    
    logger.info(f"Saved NPZ to {path}")
    return path


def save_atf(
    path: Union[str, Path],
    epochs: np.ndarray,
    dt: float,
    y_units: str = "mV",
    channel_name: str = "Response",
) -> Path:
    """
    Save aligned epochs in ATF format (Stimfit-compatible text).
    
    ATF (Axon Text Format) is a tab-delimited format that can be imported
    directly into Stimfit for analysis.
    
    Args:
        path: Output file path
        epochs: 2D array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        y_units: Units for Y axis (e.g., 'mV', 'pA')
        channel_name: Name for the channel
    
    Returns:
        Path to saved file
    
    File format:
        ATF    1.0
        n_header_lines    n_columns
        "Time (ms)"    "Sweep1 (units)"    "Sweep2 (units)"    ...
        0.0    value1    value2    ...
        ...
    """
    path = Path(path)
    
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    
    n_epochs, epoch_length = epochs.shape
    time_ms = np.arange(epoch_length) * dt * 1000.0
    
    with path.open("w") as f:
        # ATF header
        f.write("ATF\t1.0\n")
        f.write(f"1\t{n_epochs + 1}\n")
        
        # Column headers
        headers = ['"Time (ms)"'] + [
            f'"Sweep{i+1} ({y_units})"' for i in range(n_epochs)
        ]
        f.write("\t".join(headers) + "\n")
        
        # Data rows
        for i in range(epoch_length):
            row = [f"{time_ms[i]:.6f}"] + [
                f"{epochs[j, i]:.6f}" for j in range(n_epochs)
            ]
            f.write("\t".join(row) + "\n")
    
    logger.info(f"Saved ATF to {path} ({n_epochs} sweeps)")
    return path


def save_hdf5(
    path: Union[str, Path],
    epochs: np.ndarray,
    dt: float,
    channel_names: Optional[List[str]] = None,
    channel_units: Optional[List[str]] = None,
    comment: str = "Aligned epochs from ephyalign",
    date: str = "",
    time_str: str = "",
) -> Path:
    """
    Save aligned epochs in Stimfit-compatible HDF5 format.
    
    The file structure follows Stimfit's HDF5 layout:
    https://github.com/neurodroid/stimfit/blob/master/src/stimfit/py/hdf5tools.py
    
    Args:
        path: Output file path
        epochs: Epoch data. Can be:
            - 2D (n_epochs, epoch_length) for single channel
            - 3D (n_channels, n_epochs, epoch_length) for multi-channel
        dt: Sampling interval in seconds
        channel_names: Names for each channel
        channel_units: Units for each channel (e.g., 'mV')
        comment: File comment
        date: Recording date string
        time_str: Recording time string
    
    Returns:
        Path to saved file
    """
    path = Path(path)
    
    # Handle input dimensions
    if epochs.ndim == 2:
        epochs = epochs[np.newaxis, :, :]  # Add channel dimension
    
    n_channels, n_sections, epoch_length = epochs.shape
    
    # Default channel names/units
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
    if channel_units is None:
        channel_units = ["mV"] * n_channels
    
    # Extend lists if needed
    while len(channel_names) < n_channels:
        channel_names.append(f"ch{len(channel_names)}")
    while len(channel_units) < n_channels:
        channel_units.append("mV")
    
    # Zero-padding for section names
    max_log10 = int(np.log10(n_sections - 1)) if n_sections > 1 else 0
    
    with tables.open_file(str(path), mode="w", title=comment) as h5file:
        # Root description table
        root_table = h5file.create_table(
            h5file.root,
            "description",
            RecordingDescription,
            "Recording description",
        )
        root_row = root_table.row
        root_row["channels"] = n_channels
        root_row["date"] = date
        root_row["time"] = time_str
        root_row.append()
        root_table.flush()
        
        # Comment group
        comment_group = h5file.create_group("/", "comment", "File comment")
        h5file.create_array(comment_group, "comment", [comment], "Comment")
        
        # Channels group (channel names)
        channels_group = h5file.create_group("/", "channels", "Channel names")
        
        for ch in range(n_channels):
            ch_name = channel_names[ch]
            ch_unit = channel_units[ch]
            
            # Register channel name
            h5file.create_array(channels_group, f"ch{ch}", [ch_name], "Channel name")
            
            # Create channel group
            ch_group = h5file.create_group("/", ch_name, f"Channel {ch}")
            
            # Channel description
            ch_desc = h5file.create_table(
                ch_group,
                "description",
                ChannelDescription,
                f"Description of {ch_name}",
            )
            ch_row = ch_desc.row
            ch_row["n_sections"] = n_sections
            ch_row.append()
            ch_desc.flush()
            
            # Create sections (sweeps)
            for sec in range(n_sections):
                # Zero-padded section name
                if sec == 0:
                    n10 = 0
                else:
                    n10 = int(np.log10(sec))
                padding = "0" * (max_log10 - n10)
                section_name = f"section_{padding}{sec}"
                
                sec_group = h5file.create_group(ch_group, section_name, f"sec{sec}")
                
                # Data array
                h5file.create_array(
                    sec_group,
                    "data",
                    epochs[ch, sec, :].astype(np.float32),
                    f"Data for section {sec}",
                )
                
                # Section description
                sec_desc = h5file.create_table(
                    sec_group,
                    "description",
                    SectionDescription,
                    f"Description of sec{sec}",
                )
                sec_row = sec_desc.row
                sec_row["dt"] = dt * 1000.0  # Convert to ms
                sec_row["xunits"] = "ms"
                sec_row["yunits"] = ch_unit
                sec_row.append()
                sec_desc.flush()
    
    logger.info(f"Saved HDF5 to {path} ({n_channels} channels, {n_sections} sweeps)")
    return path


def save_all_formats(
    paths: OutputPaths,
    epochs: np.ndarray,
    dt: float,
    time_axis: np.ndarray,
    channel_names: Optional[List[str]] = None,
    channel_units: Optional[List[str]] = None,
    save_npz_flag: bool = True,
    save_atf_flag: bool = True,
    save_hdf5_flag: bool = True,
) -> Dict[str, Path]:
    """
    Save epochs in all configured formats.
    
    Args:
        paths: OutputPaths with file locations
        epochs: Epoch data
        dt: Sampling interval
        time_axis: Time axis in seconds
        channel_names: Names for channels
        channel_units: Units for channels
        save_npz_flag: Whether to save NPZ
        save_atf_flag: Whether to save ATF
        save_hdf5_flag: Whether to save HDF5
    
    Returns:
        Dictionary mapping format names to saved file paths
    """
    saved = {}
    
    # Determine reference channel epochs for ATF (always 2D)
    if epochs.ndim == 3:
        epochs_2d = epochs[0]  # Use first channel for ATF
    else:
        epochs_2d = epochs
    
    if save_npz_flag:
        saved["npz"] = save_npz(
            paths.npz,
            epochs,
            time_axis,
            {"dt": dt},
        )
    
    if save_atf_flag:
        y_units = channel_units[0] if channel_units else "mV"
        saved["atf"] = save_atf(
            paths.atf,
            epochs_2d,
            dt,
            y_units=y_units,
        )
    
    if save_hdf5_flag:
        saved["hdf5"] = save_hdf5(
            paths.hdf5,
            epochs,
            dt,
            channel_names=channel_names,
            channel_units=channel_units,
        )
    
    return saved


def write_stats_report(
    path: Union[str, Path],
    config_dict: Dict[str, Any],
    metrics: EpochMetrics,
    detection_info: Dict[str, Any],
    recording_info: Dict[str, Any],
    jitter_ms: float = 0.0,
) -> Path:
    """
    Write comprehensive statistics report.
    
    Args:
        path: Output file path
        config_dict: Configuration used for processing
        metrics: EpochMetrics from analysis
        detection_info: Detection results info
        recording_info: Recording metadata
        jitter_ms: Alignment jitter in milliseconds
    
    Returns:
        Path to saved file
    """
    path = Path(path)
    
    with path.open("w") as f:
        f.write("=" * 60 + "\n")
        f.write("EPHYALIGN STATISTICS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Recording info
        f.write("RECORDING INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"File: {recording_info.get('file_path', 'N/A')}\n")
        f.write(f"Sampling rate (Hz): {recording_info.get('sampling_rate', 'N/A'):.2f}\n")
        f.write(f"Duration (s): {recording_info.get('duration_s', 'N/A'):.3f}\n")
        f.write(f"Channels: {recording_info.get('n_channels', 'N/A')}\n")
        f.write(f"Reference channel: {config_dict.get('reference_channel', 0)}\n")
        f.write("\n")
        
        # Detection info
        f.write("DETECTION RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Stimuli detected: {detection_info.get('n_detected', 0)}\n")
        f.write(f"Mean ISI (s): {detection_info.get('mean_isi_s', float('nan')):.3f}\n")
        f.write(f"ISI std (s): {detection_info.get('std_isi_s', float('nan')):.3f}\n")
        if detection_info.get('mean_isi_s', 0) > 0:
            rate = 1.0 / detection_info['mean_isi_s']
            f.write(f"Stimulus rate (Hz): {rate:.3f}\n")
        f.write("\n")
        
        # Epoch info
        f.write("EPOCH EXTRACTION\n")
        f.write("-" * 40 + "\n")
        epoch_cfg = config_dict.get("epoch", {})
        f.write(f"Pre-stimulus window (s): {epoch_cfg.get('pre_time_s', 0.5):.3f}\n")
        f.write(f"Post-stimulus window (s): {epoch_cfg.get('post_time_s', 3.0):.3f}\n")
        f.write(f"Usable epochs: {metrics.n_epochs}\n")
        f.write("\n")
        
        # Alignment quality
        f.write("ALIGNMENT QUALITY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Alignment jitter (ms): {jitter_ms:.3f}\n")
        f.write(f"Raw jitter (ms): {metrics.jitter_ms:.3f}\n")
        f.write("\n")
        
        # Response metrics
        f.write("RESPONSE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Peak amplitude: {metrics.peak_amp_mean:.3f} ± {metrics.peak_amp_std:.3f}\n")
        f.write(f"Peak amplitude CV: {metrics.peak_amp_cv:.3f}\n")
        f.write(f"Time to peak (ms): {metrics.time_to_peak_mean_ms:.3f} ± {metrics.time_to_peak_std_ms:.3f}\n")
        f.write(f"Rise time 10-90% (ms): {metrics.rise_time_mean_ms:.3f} ± {metrics.rise_time_std_ms:.3f}\n")
        f.write(f"Baseline noise (SD): {metrics.baseline_noise:.3f}\n")
        f.write(f"AUC: {metrics.auc_mean:.3e} ± {metrics.auc_std:.3e}\n")
        f.write("\n")
        
        # Per-sweep table
        f.write("PER-SWEEP METRICS\n")
        f.write("-" * 40 + "\n")
        f.write("#\tPeakAmp\tTime2Peak(ms)\tRiseTime(ms)\tAUC\n")
        
        for i in range(metrics.n_epochs):
            pa = metrics.peak_amplitude[i] if i < len(metrics.peak_amplitude) else float("nan")
            tp = metrics.time_to_peak_s[i] * 1000 if i < len(metrics.time_to_peak_s) else float("nan")
            rt = metrics.rise_time_s[i] * 1000 if i < len(metrics.rise_time_s) else float("nan")
            au = metrics.auc[i] if i < len(metrics.auc) else float("nan")
            f.write(f"{i+1}\t{pa:.3f}\t{tp:.3f}\t{rt:.3f}\t{au:.3e}\n")
    
    logger.info(f"Saved statistics report to {path}")
    return path
