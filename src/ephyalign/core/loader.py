"""ABF file loading and data extraction.

This module handles reading Axon Binary Format (ABF) files and extracting
continuous electrophysiological recordings for further processing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyabf

logger = logging.getLogger(__name__)


@dataclass
class RecordingData:
    """Container for loaded recording data and metadata.

    Attributes:
        data: Recording data array of shape (n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        dt: Sampling interval in seconds
        channel_names: Names of each channel
        channel_units: Units for each channel (e.g., 'mV', 'pA')
        n_channels: Number of channels
        n_samples: Total number of samples per channel
        duration_s: Total recording duration in seconds
        file_path: Path to the source file
        abf_version: ABF file version
        recording_datetime: Date and time of recording

    """

    data: np.ndarray
    sampling_rate: float
    dt: float
    channel_names: list[str]
    channel_units: list[str]
    n_channels: int
    n_samples: int
    duration_s: float
    file_path: Path
    abf_version: dict
    recording_datetime: datetime | None

    # Keep reference to ABF object for additional metadata access
    _abf: pyabf.ABF | None = None

    def get_channel_data(self, channel: int) -> np.ndarray:
        """Get data for a specific channel."""
        if channel < 0 or channel >= self.n_channels:
            raise ValueError(
                f"Channel {channel} out of range. Available channels: 0-{self.n_channels - 1}"
            )
        return self.data[channel]

    def get_channel_by_name(self, name: str) -> np.ndarray:
        """Get data for a channel by its name."""
        try:
            idx = self.channel_names.index(name)
            return self.data[idx]
        except ValueError as err:
            raise ValueError(
                f"Channel '{name}' not found. Available channels: {self.channel_names}"
            ) from err

    def get_time_axis(self) -> np.ndarray:
        """Get time axis in seconds."""
        return np.arange(self.n_samples) * self.dt

    def __repr__(self) -> str:
        """Return string representation of RecordingData."""
        return (
            f"RecordingData("
            f"channels={self.n_channels}, "
            f"samples={self.n_samples}, "
            f"duration={self.duration_s:.2f}s, "
            f"rate={self.sampling_rate}Hz)"
        )


def load_recording(
    file_path: str | Path,
    channels: list[int] | None = None,
) -> RecordingData:
    """Load an ABF recording file.

    Args:
        file_path: Path to the ABF file
        channels: Specific channels to load (None = all channels)

    Returns:
        RecordingData object containing the loaded data and metadata

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If specified channels are invalid

    Example:
        >>> recording = load_recording("data/experiment.abf")
        >>> print(f"Loaded {recording.n_channels} channels, {recording.duration_s:.1f}s")

    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ABF file not found: {file_path}")

    logger.info(f"Loading ABF file: {file_path}")

    abf = pyabf.ABF(str(file_path))

    # Extract basic metadata
    sampling_rate = float(abf.dataRate)
    dt = 1.0 / sampling_rate
    n_channels = abf.channelCount

    logger.debug(f"Sampling rate: {sampling_rate} Hz")
    logger.debug(f"Sweep count: {abf.sweepCount}")
    logger.debug(f"Channel count: {n_channels}")

    # Validate channel selection
    if channels is not None:
        for ch in channels:
            if ch < 0 or ch >= n_channels:
                raise ValueError(
                    f"Channel {ch} out of range. "
                    f"File has {n_channels} channels (0-{n_channels - 1})"
                )
        channels_to_load = channels
    else:
        channels_to_load = list(range(n_channels))

    # Load data for all channels
    # Handle both single-sweep and multi-sweep recordings
    if abf.sweepCount == 1:
        # Single continuous sweep
        channel_data = []
        for ch in channels_to_load:
            abf.setSweep(0, channel=ch)
            channel_data.append(abf.sweepY.copy())
        data = np.vstack(channel_data)
    else:
        # Multiple sweeps - concatenate
        channel_data = []
        for ch in channels_to_load:
            sweep_arrays = []
            for sweep_idx in range(abf.sweepCount):
                abf.setSweep(sweep_idx, channel=ch)
                sweep_arrays.append(abf.sweepY.copy())
            channel_data.append(np.concatenate(sweep_arrays))
        data = np.vstack(channel_data)

    n_samples = data.shape[1]
    duration_s = n_samples * dt

    # Extract channel names and units
    channel_names = []
    channel_units = []
    for ch in channels_to_load:
        name = abf.adcNames[ch] if ch < len(abf.adcNames) else f"Ch{ch}"
        # Clean up non-printable characters
        name = "".join(c for c in name if c.isprintable()) or f"Ch{ch}"
        channel_names.append(name)

        unit = abf.adcUnits[ch] if ch < len(abf.adcUnits) else "?"
        unit = "".join(c for c in unit if c.isprintable()) or "?"
        channel_units.append(unit)

    logger.info(f"Loaded {len(channels_to_load)} channels, {n_samples} samples ({duration_s:.2f}s)")

    return RecordingData(
        data=data,
        sampling_rate=sampling_rate,
        dt=dt,
        channel_names=channel_names,
        channel_units=channel_units,
        n_channels=len(channels_to_load),
        n_samples=n_samples,
        duration_s=duration_s,
        file_path=file_path,
        abf_version=abf.abfVersion,
        recording_datetime=abf.abfDateTime,
        _abf=abf,
    )


def get_file_info(file_path: str | Path) -> dict:
    """Get metadata about an ABF file without loading all data.

    Args:
        file_path: Path to the ABF file

    Returns:
        Dictionary with file metadata

    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"ABF file not found: {file_path}")

    abf = pyabf.ABF(str(file_path))

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "sampling_rate_hz": abf.dataRate,
        "sweep_count": abf.sweepCount,
        "channel_count": abf.channelCount,
        "channel_names": abf.adcNames,
        "channel_units": abf.adcUnits,
        "abf_version": abf.abfVersion,
        "recording_datetime": abf.abfDateTime,
        "sweep_length_points": len(abf.sweepY) if abf.sweepCount > 0 else 0,
        "total_duration_s": (
            abf.sweepCount * len(abf.sweepY) / abf.dataRate if abf.sweepCount > 0 else 0
        ),
    }
