"""Epoch extraction and alignment.

This module provides functions for extracting time-locked epochs around
stimulus events and refining their alignment with sub-millisecond precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ephyalign.config import EpochConfig

logger = logging.getLogger(__name__)


@dataclass
class EpochData:
    """Container for extracted epoch data.

    Attributes:
        epochs: Epoch data array of shape (n_epochs, epoch_length) or
                (n_channels, n_epochs, epoch_length) for multi-channel
        time_axis: Time axis in seconds relative to stimulus onset
        n_epochs: Number of epochs extracted
        epoch_length: Length of each epoch in samples
        pre_samples: Number of samples before stimulus
        post_samples: Number of samples after stimulus
        dt: Sampling interval in seconds
        valid_indices: Stimulus indices that were successfully extracted

    """

    epochs: np.ndarray
    time_axis: np.ndarray
    n_epochs: int
    epoch_length: int
    pre_samples: int
    post_samples: int
    dt: float
    valid_indices: np.ndarray

    @property
    def is_multichannel(self) -> bool:
        """Check if this is multi-channel epoch data."""
        return self.epochs.ndim == 3

    @property
    def n_channels(self) -> int:
        """Number of channels (1 for single-channel data)."""
        return self.epochs.shape[0] if self.is_multichannel else 1

    def get_channel(self, channel: int) -> np.ndarray:
        """Get epochs for a specific channel."""
        if not self.is_multichannel:
            if channel != 0:
                raise ValueError("Single-channel data, use channel=0")
            return self.epochs
        return self.epochs[channel]

    def get_mean(self, channel: int | None = None) -> np.ndarray:
        """Get mean epoch (optionally for specific channel)."""
        if channel is not None:
            return self.get_channel(channel).mean(axis=0)
        if self.is_multichannel:
            return self.epochs.mean(axis=1)  # Shape: (n_channels, epoch_length)
        return self.epochs.mean(axis=0)

    def get_std(self, channel: int | None = None) -> np.ndarray:
        """Get standard deviation across epochs."""
        if channel is not None:
            return self.get_channel(channel).std(axis=0)
        if self.is_multichannel:
            return self.epochs.std(axis=1)
        return self.epochs.std(axis=0)

    def __repr__(self) -> str:
        """Return string representation of EpochData."""
        shape_str = f"({self.n_channels}ch, " if self.is_multichannel else "("
        return f"EpochData{shape_str}{self.n_epochs} epochs, {self.epoch_length} samples)"


@dataclass
class AlignmentResult:
    """Results from epoch alignment refinement.

    Attributes:
        epochs: Aligned epoch data
        time_axis: Time axis for aligned epochs
        onset_positions: Detected onset position within each raw epoch
        jitter_ms: Standard deviation of onset positions (alignment jitter)
        original_epochs: Original (unaligned) epochs for reference

    """

    epochs: np.ndarray
    time_axis: np.ndarray
    onset_positions: np.ndarray
    jitter_ms: float
    original_epochs: np.ndarray | None = None

    @property
    def n_epochs(self) -> int:
        """Number of aligned epochs."""
        return self.epochs.shape[0] if self.epochs.ndim == 2 else self.epochs.shape[1]

    @property
    def epoch_length(self) -> int:
        """Length of each aligned epoch in samples."""
        return self.epochs.shape[-1]


def build_epochs(
    data: np.ndarray,
    stim_indices: np.ndarray,
    dt: float,
    config: EpochConfig | None = None,
    pre_time_s: float | None = None,
    post_time_s: float | None = None,
) -> EpochData:
    """Extract epochs around each detected stimulus.

    Args:
        data: Continuous recording data. Can be:
              - 1D array (n_samples,) for single channel
              - 2D array (n_channels, n_samples) for multi-channel
        stim_indices: Sample indices of stimulus onsets
        dt: Sampling interval in seconds
        config: EpochConfig with extraction parameters
        pre_time_s: Time before stimulus to include (overrides config)
        post_time_s: Time after stimulus to include (overrides config)

    Returns:
        EpochData containing extracted epochs and metadata

    Example:
        >>> epochs = build_epochs(data, stim_indices, dt=0.0001, pre_time_s=0.5)
        >>> print(f"Extracted {epochs.n_epochs} epochs")

    """
    if config is None:
        config = EpochConfig()

    # Allow parameter overrides
    pre_s = pre_time_s if pre_time_s is not None else config.pre_time_s
    post_s = post_time_s if post_time_s is not None else config.post_time_s

    # Handle 1D vs 2D input
    data = np.asarray(data)
    is_multichannel = data.ndim == 2

    if is_multichannel:
        n_channels, n_samples = data.shape
    else:
        n_channels = 1
        n_samples = len(data)
        data = data.reshape(1, -1)  # Convert to 2D for uniform processing

    # Calculate sample counts
    pre_samples = int(round(pre_s / dt))
    post_samples = int(round(post_s / dt))
    epoch_length = pre_samples + post_samples

    logger.debug(
        f"Epoch window: {pre_samples} pre + {post_samples} post = "
        f"{epoch_length} samples ({pre_s + post_s:.2f}s)"
    )

    # Find valid stimulus indices (not too close to edges)
    valid_indices_list: list[int] = []
    for idx in stim_indices:
        start = idx - pre_samples
        end = idx + post_samples
        if start >= 0 and end <= n_samples:
            valid_indices_list.append(int(idx))

    valid_indices = np.asarray(valid_indices_list, dtype=np.int64)
    n_epochs = len(valid_indices)

    if n_epochs == 0:
        logger.warning("No valid epochs found (stimuli too close to recording edges)")
        return EpochData(
            epochs=np.empty(
                (n_channels, 0, epoch_length) if is_multichannel else (0, epoch_length)
            ),
            time_axis=(np.arange(epoch_length) - pre_samples) * dt,
            n_epochs=0,
            epoch_length=epoch_length,
            pre_samples=pre_samples,
            post_samples=post_samples,
            dt=dt,
            valid_indices=valid_indices,
        )

    logger.info(f"Extracting {n_epochs} epochs ({len(stim_indices) - n_epochs} rejected at edges)")

    # Extract epochs for all channels
    epochs = np.zeros((n_channels, n_epochs, epoch_length), dtype=data.dtype)

    for ch in range(n_channels):
        for i, idx in enumerate(valid_indices):
            start = idx - pre_samples
            end = idx + post_samples
            epochs[ch, i, :] = data[ch, start:end]

    # Apply baseline subtraction if configured
    if config.baseline_subtract:
        baseline_samples = int(config.baseline_window_ms / 1000.0 / dt)
        baseline_samples = min(baseline_samples, pre_samples)
        for ch in range(n_channels):
            for i in range(n_epochs):
                baseline = epochs[ch, i, :baseline_samples].mean()
                epochs[ch, i, :] -= baseline
        logger.debug(f"Baseline-subtracted using first {baseline_samples} samples")

    # Time axis relative to stimulus onset
    time_axis = (np.arange(epoch_length) - pre_samples) * dt

    # Return single-channel format if input was single-channel
    if not is_multichannel:
        epochs = epochs[0]  # Shape: (n_epochs, epoch_length)

    return EpochData(
        epochs=epochs,
        time_axis=time_axis,
        n_epochs=n_epochs,
        epoch_length=epoch_length,
        pre_samples=pre_samples,
        post_samples=post_samples,
        dt=dt,
        valid_indices=valid_indices,
    )


def refine_alignment(
    epochs: np.ndarray,
    dt: float,
    search_ms: float = 1.0,
    keep_original: bool = False,
) -> AlignmentResult:
    """Refine epoch alignment by locking to the sharp artifact onset.

    This function provides sub-millisecond alignment by detecting the
    exact artifact onset within each epoch and re-cutting all epochs
    to start at the same relative position.

    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        search_ms: Time window to search for artifact onset (milliseconds)
        keep_original: Whether to keep original epochs in result

    Returns:
        AlignmentResult with refined epochs and alignment statistics

    Algorithm:
        1. For each epoch, find max derivative in search window
        2. Record onset position for each epoch
        3. Re-cut epochs to start at detected onset
        4. Truncate all epochs to same length

    """
    if epochs.size == 0:
        return AlignmentResult(
            epochs=epochs,
            time_axis=np.array([]),
            onset_positions=np.array([], dtype=np.int64),
            jitter_ms=0.0,
            original_epochs=epochs if keep_original else None,
        )

    # Handle single epoch edge case
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)

    n_epochs, epoch_length = epochs.shape
    search_samples = int(search_ms / 1000.0 / dt)
    search_samples = max(1, min(search_samples, epoch_length - 1))

    logger.debug(f"Refining alignment with {search_samples} sample search window")

    refined_list: list[np.ndarray] = []
    onset_positions_list: list[int] = []

    for ep in epochs:
        # Search for artifact in early part of epoch
        segment = ep[:search_samples]
        derivative = np.diff(segment)

        # Find peak of absolute derivative (handles both polarities)
        peak_idx = int(np.argmax(np.abs(derivative)))
        onset_positions_list.append(peak_idx)

        # Re-cut epoch starting from detected onset
        refined_list.append(ep[peak_idx:])

    onset_positions = np.asarray(onset_positions_list, dtype=np.int64)

    # Calculate jitter (alignment quality metric)
    jitter_samples = float(np.std(onset_positions))
    jitter_ms = jitter_samples * dt * 1000.0

    logger.info(f"Alignment jitter: {jitter_ms:.3f} ms")

    # Truncate all epochs to same length
    min_length = min(len(ep) for ep in refined_list)
    refined_epochs = np.vstack([ep[:min_length] for ep in refined_list])

    # New time axis (starts at 0 after alignment)
    time_axis = np.arange(min_length) * dt

    return AlignmentResult(
        epochs=refined_epochs,
        time_axis=time_axis,
        onset_positions=onset_positions,
        jitter_ms=jitter_ms,
        original_epochs=epochs if keep_original else None,
    )


def apply_alignment(
    epochs: np.ndarray,
    onset_positions: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """Apply a pre-computed alignment to epochs.

    This is useful for aligning additional channels using onset positions
    detected from a reference channel.

    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        onset_positions: Onset position for each epoch (from refine_alignment)
        target_length: Desired output epoch length

    Returns:
        Aligned epochs of shape (n_epochs, target_length)

    Example:
        >>> # Align reference channel
        >>> result = refine_alignment(epochs_ch0, dt)
        >>> # Apply same alignment to other channels
        >>> epochs_ch1_aligned = apply_alignment(
        ...     epochs_ch1, result.onset_positions, result.epoch_length
        ... )

    """
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)

    n_epochs = epochs.shape[0]

    if len(onset_positions) != n_epochs:
        raise ValueError(
            f"Onset positions length ({len(onset_positions)}) doesn't match "
            f"number of epochs ({n_epochs})"
        )

    aligned = np.zeros((n_epochs, target_length), dtype=epochs.dtype)

    for i, (ep, onset) in enumerate(zip(epochs, onset_positions, strict=False)):
        end_idx = onset + target_length
        if end_idx <= len(ep):
            aligned[i, :] = ep[onset:end_idx]
        else:
            # Handle edge case where epoch is too short
            available = len(ep) - onset
            aligned[i, :available] = ep[onset:]
            # Pad with last value (could also use NaN or 0)
            if available < target_length:
                aligned[i, available:] = ep[-1]

    return aligned


def align_multichannel(
    epochs_all: np.ndarray,
    dt: float,
    reference_channel: int = 0,
    search_ms: float = 1.0,
) -> tuple[np.ndarray, AlignmentResult]:
    """Align multi-channel epochs using a reference channel.

    Detects artifact onset on the reference channel and applies the same
    alignment to all channels.

    Args:
        epochs_all: 3D array of shape (n_channels, n_epochs, epoch_length)
        dt: Sampling interval in seconds
        reference_channel: Channel to use for onset detection
        search_ms: Search window for artifact detection

    Returns:
        Tuple of:
        - Aligned epochs (n_channels, n_epochs, aligned_length)
        - AlignmentResult from reference channel

    """
    if epochs_all.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {epochs_all.shape}")

    n_channels, n_epochs, epoch_length = epochs_all.shape

    if reference_channel < 0 or reference_channel >= n_channels:
        raise ValueError(f"Reference channel {reference_channel} out of range (0-{n_channels - 1})")

    logger.info(f"Aligning {n_channels} channels using channel {reference_channel}")

    # Align reference channel
    ref_result = refine_alignment(
        epochs_all[reference_channel],
        dt,
        search_ms=search_ms,
    )

    target_length = ref_result.epoch_length

    # Apply same alignment to all channels
    aligned_all = np.zeros((n_channels, n_epochs, target_length), dtype=epochs_all.dtype)

    for ch in range(n_channels):
        if ch == reference_channel:
            aligned_all[ch] = ref_result.epochs
        else:
            aligned_all[ch] = apply_alignment(
                epochs_all[ch],
                ref_result.onset_positions,
                target_length,
            )

    return aligned_all, ref_result
