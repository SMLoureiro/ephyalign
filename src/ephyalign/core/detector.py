"""Stimulus artifact detection.

This module provides algorithms for automatically detecting stimulus onset
times from electrophysiological recordings, based on the characteristic
sharp transients (artifacts) that stimulation produces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ephyalign.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Results from stimulus detection.

    Attributes:
        stim_indices: Sample indices of detected stimulus onsets
        stim_times_s: Times of detected stimuli in seconds
        n_detected: Number of stimuli detected
        threshold_used: Actual threshold value used for detection
        derivative_std: Standard deviation of the signal derivative
        inter_stim_intervals_s: Inter-stimulus intervals in seconds
        mean_isi_s: Mean inter-stimulus interval
        std_isi_s: Standard deviation of ISI

    """

    stim_indices: np.ndarray
    stim_times_s: np.ndarray
    n_detected: int
    threshold_used: float
    derivative_std: float
    inter_stim_intervals_s: np.ndarray | None
    mean_isi_s: float
    std_isi_s: float

    def __repr__(self) -> str:
        """Return string representation of DetectionResult."""
        return (
            f"DetectionResult("
            f"n_detected={self.n_detected}, "
            f"mean_isi={self.mean_isi_s:.3f}s ± {self.std_isi_s:.3f}s)"
        )


def detect_stim_onsets(
    data: np.ndarray,
    dt: float,
    config: DetectionConfig | None = None,
    min_interval_s: float | None = None,
    threshold_multiplier: float | None = None,
) -> DetectionResult:
    """Detect stimulus onset artifacts in a continuous recording.

    Uses derivative thresholding to detect sharp transients characteristic
    of electrical stimulation artifacts. Works well for capacitive transients
    and other sharp onset responses.

    Args:
        data: 1D array of continuous recording data (single channel)
        dt: Sampling interval in seconds
        config: DetectionConfig with detection parameters
        min_interval_s: Minimum interval between stimuli (overrides config)
        threshold_multiplier: Threshold multiplier (overrides config)

    Returns:
        DetectionResult with detected stimulus indices and timing statistics

    Algorithm:
        1. Compute derivative of signal
        2. Set threshold at (threshold_multiplier * std(derivative))
        3. Find samples where derivative exceeds threshold
        4. Apply minimum interval constraint to avoid double detections

    Example:
        >>> result = detect_stim_onsets(data, dt=0.0001, min_interval_s=3.0)
        >>> print(f"Found {result.n_detected} stimuli")

    """
    if config is None:
        config = DetectionConfig()

    # Allow parameter overrides
    min_interval = min_interval_s if min_interval_s is not None else config.min_interval_s
    threshold_mult = (
        threshold_multiplier if threshold_multiplier is not None else config.threshold_multiplier
    )

    # Ensure data is 1D
    data = np.asarray(data).flatten()

    logger.debug(f"Detecting stimuli in {len(data)} samples")

    # Compute derivative
    derivative = np.diff(data)
    derivative_std = float(np.std(derivative))
    threshold = derivative_std * threshold_mult

    logger.debug(f"Derivative std: {derivative_std:.4f}")
    logger.debug(f"Threshold: {threshold:.4f}")

    # Find candidate indices
    if config.use_absolute_derivative:
        candidates = np.where(np.abs(derivative) > threshold)[0]
    else:
        candidates = np.where(derivative > threshold)[0]

    logger.debug(f"Found {len(candidates)} candidate samples")

    # Apply minimum interval constraint
    min_interval_samples = int(min_interval / dt)

    stim_indices_list: list[int] = []
    last_idx = -np.inf

    for idx in candidates:
        if idx - last_idx >= min_interval_samples:
            stim_indices_list.append(int(idx))
            last_idx = idx

    stim_indices = np.asarray(stim_indices_list, dtype=np.int64)
    n_detected = len(stim_indices)

    logger.info(f"Detected {n_detected} stimulus artifacts")

    # Calculate timing statistics
    stim_times_s = stim_indices * dt

    if n_detected > 1:
        isi = np.diff(stim_indices) * dt
        mean_isi = float(np.mean(isi))
        std_isi = float(np.std(isi, ddof=1))
    else:
        isi = None
        mean_isi = float("nan")
        std_isi = float("nan")

    if n_detected > 1:
        stim_rate = 1.0 / mean_isi if mean_isi > 0 else float("nan")
        logger.info(f"Mean ISI: {mean_isi:.3f}s (rate: {stim_rate:.3f} Hz)")

    return DetectionResult(
        stim_indices=stim_indices,
        stim_times_s=stim_times_s,
        n_detected=n_detected,
        threshold_used=threshold,
        derivative_std=derivative_std,
        inter_stim_intervals_s=isi,
        mean_isi_s=mean_isi,
        std_isi_s=std_isi,
    )


def detect_stim_onsets_multichannel(
    data: np.ndarray,
    dt: float,
    reference_channel: int = 0,
    config: DetectionConfig | None = None,
) -> DetectionResult:
    """Detect stimulus onsets using a reference channel from multi-channel data.

    Args:
        data: 2D array of shape (n_channels, n_samples)
        dt: Sampling interval in seconds
        reference_channel: Channel to use for detection
        config: Detection configuration

    Returns:
        DetectionResult with detected stimulus indices

    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape}")

    n_channels, n_samples = data.shape

    if reference_channel < 0 or reference_channel >= n_channels:
        raise ValueError(f"Reference channel {reference_channel} out of range (0-{n_channels - 1})")

    logger.info(f"Using channel {reference_channel} for detection")

    return detect_stim_onsets(data[reference_channel], dt, config)


def refine_onset_positions(
    data: np.ndarray,
    stim_indices: np.ndarray,
    dt: float,
    search_window_samples: int = 10,
) -> tuple[np.ndarray, float]:
    """Refine stimulus onset positions with sub-sample precision.

    After initial detection, this function can be used to fine-tune
    the exact onset position within a small search window around
    each detected stimulus.

    Args:
        data: 1D continuous recording data
        stim_indices: Initial stimulus indices from detection
        dt: Sampling interval in seconds
        search_window_samples: Window size for refinement

    Returns:
        Tuple of (refined_indices, jitter_std_samples)
        where jitter_std_samples is the standard deviation of the
        offset corrections applied.

    """
    data = np.asarray(data).flatten()
    refined = []
    offsets = []

    for idx in stim_indices:
        # Define search window
        start = max(0, idx - search_window_samples // 2)
        end = min(len(data) - 1, idx + search_window_samples // 2)

        if end <= start:
            refined.append(idx)
            offsets.append(0)
            continue

        # Find peak of derivative in window
        segment = data[start:end]
        derivative = np.diff(segment)
        peak_offset = int(np.argmax(np.abs(derivative)))

        refined_idx = start + peak_offset
        refined.append(refined_idx)
        offsets.append(refined_idx - idx)

    refined_arr = np.asarray(refined, dtype=np.int64)
    jitter_std = float(np.std(offsets)) if len(offsets) > 1 else 0.0

    logger.debug(f"Refinement jitter std: {jitter_std * dt * 1000:.3f} ms")

    return refined_arr, jitter_std
