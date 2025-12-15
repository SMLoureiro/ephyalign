"""Signal filtering and epoch quality control.

Provides filtering utilities for electrophysiology signals and
quality control functions for epoch rejection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """Configuration for signal filtering."""

    # Lowpass filter
    lowpass_hz: float | None = None
    lowpass_order: int = 4

    # Highpass filter
    highpass_hz: float | None = None
    highpass_order: int = 4

    # Bandpass filter (alternative to separate low/high)
    bandpass_hz: tuple[float, float] | None = None
    bandpass_order: int = 4

    # Notch filter (e.g., for 50/60Hz line noise)
    notch_hz: float | None = None
    notch_q: float = 30.0

    # Median filter (for artifact removal)
    median_kernel: int | None = None

    # Savitzky-Golay filter (for smoothing)
    savgol_window: int | None = None
    savgol_order: int = 3


@dataclass
class QualityConfig:
    """Configuration for epoch quality control."""

    # Enable/disable quality control
    enabled: bool = True

    # Maximum baseline noise (std of pre-stimulus period)
    max_baseline_noise: float | None = None

    # Maximum absolute amplitude
    max_amplitude: float | None = None

    # Minimum amplitude (reject flat epochs)
    min_amplitude: float | None = None

    # Maximum epoch-to-epoch amplitude change
    max_amplitude_change: float | None = None

    # Z-score threshold for outlier detection
    outlier_zscore: float = 3.0

    # Correlation threshold with template (0-1)
    min_template_correlation: float | None = None

    # Minimum number of epochs required
    min_epochs: int = 2


@dataclass
class QualityResult:
    """Results of quality control analysis."""

    n_input: int
    n_passed: int
    n_rejected: int
    passed_indices: np.ndarray
    rejected_indices: np.ndarray
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def rejection_rate(self) -> float:
        """Fraction of epochs rejected."""
        return self.n_rejected / self.n_input if self.n_input > 0 else 0.0

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "Quality Control Summary:",
            f"  Input epochs: {self.n_input}",
            f"  Passed: {self.n_passed}",
            f"  Rejected: {self.n_rejected} ({self.rejection_rate:.1%})",
        ]
        if self.rejection_reasons:
            lines.append("  Rejection reasons:")
            for reason in set(self.rejection_reasons):
                count = self.rejection_reasons.count(reason)
                lines.append(f"    - {reason}: {count}")
        return "\n".join(lines)


def apply_filter(
    data: np.ndarray,
    dt: float,
    config: FilterConfig | None = None,
    lowpass_hz: float | None = None,
    highpass_hz: float | None = None,
    notch_hz: float | None = None,
) -> np.ndarray:
    """Apply digital filters to signal data.

    Args:
        data: 1D or 2D signal array
        dt: Sampling interval in seconds
        config: Filter configuration object
        lowpass_hz: Lowpass cutoff frequency (overrides config)
        highpass_hz: Highpass cutoff frequency (overrides config)
        notch_hz: Notch filter frequency (overrides config)

    Returns:
        Filtered signal array (same shape as input)

    """
    if config is None:
        config = FilterConfig()

    # Apply overrides
    if lowpass_hz is not None:
        config.lowpass_hz = lowpass_hz
    if highpass_hz is not None:
        config.highpass_hz = highpass_hz
    if notch_hz is not None:
        config.notch_hz = notch_hz

    fs = 1.0 / dt
    nyq = fs / 2.0

    result = data.copy()

    # Apply bandpass if specified
    if config.bandpass_hz is not None:
        low, high = config.bandpass_hz
        if high > nyq:
            logger.warning(f"Bandpass high frequency {high} Hz exceeds Nyquist {nyq} Hz")
            high = nyq * 0.99
        sos = signal.butter(
            config.bandpass_order,
            [low / nyq, high / nyq],
            btype="band",
            output="sos",
        )
        result = signal.sosfiltfilt(sos, result, axis=-1)
        logger.info(f"Applied bandpass filter: {low}-{high} Hz")

    else:
        # Apply highpass
        if config.highpass_hz is not None:
            if config.highpass_hz > nyq:
                logger.warning(f"Highpass frequency {config.highpass_hz} Hz exceeds Nyquist")
            else:
                sos = signal.butter(
                    config.highpass_order,
                    config.highpass_hz / nyq,
                    btype="high",
                    output="sos",
                )
                result = signal.sosfiltfilt(sos, result, axis=-1)
                logger.info(f"Applied highpass filter: {config.highpass_hz} Hz")

        # Apply lowpass
        if config.lowpass_hz is not None:
            if config.lowpass_hz > nyq:
                logger.warning(f"Lowpass frequency {config.lowpass_hz} Hz exceeds Nyquist")
            else:
                sos = signal.butter(
                    config.lowpass_order,
                    config.lowpass_hz / nyq,
                    btype="low",
                    output="sos",
                )
                result = signal.sosfiltfilt(sos, result, axis=-1)
                logger.info(f"Applied lowpass filter: {config.lowpass_hz} Hz")

    # Apply notch filter
    if config.notch_hz is not None:
        if config.notch_hz > nyq:
            logger.warning(f"Notch frequency {config.notch_hz} Hz exceeds Nyquist")
        else:
            b, a = signal.iirnotch(config.notch_hz, config.notch_q, fs)
            result = signal.filtfilt(b, a, result, axis=-1)
            logger.info(f"Applied notch filter: {config.notch_hz} Hz (Q={config.notch_q})")

    # Apply median filter
    if config.median_kernel is not None:
        from scipy.ndimage import median_filter

        result = median_filter(result, size=config.median_kernel)
        logger.info(f"Applied median filter: kernel={config.median_kernel}")

    # Apply Savitzky-Golay filter
    if config.savgol_window is not None:
        if config.savgol_window % 2 == 0:
            config.savgol_window += 1  # Must be odd
        result = signal.savgol_filter(
            result,
            config.savgol_window,
            config.savgol_order,
            axis=-1,
        )
        logger.info(f"Applied Savitzky-Golay filter: window={config.savgol_window}")

    return result


def baseline_correct(
    epochs: np.ndarray,
    dt: float,
    pre_time_s: float,
    method: Literal["mean", "median", "polynomial", "linear"] = "mean",
    poly_order: int = 1,
) -> np.ndarray:
    """Apply baseline correction to epochs.

    Args:
        epochs: Array of epochs (n_epochs, epoch_length) or (n_channels, n_epochs, epoch_length)
        dt: Sampling interval in seconds
        pre_time_s: Duration of pre-stimulus period for baseline estimation
        method: Baseline estimation method
        poly_order: Order for polynomial fitting

    Returns:
        Baseline-corrected epochs

    """
    pre_samples = int(pre_time_s / dt)

    if epochs.ndim == 2:
        # Single channel: (n_epochs, epoch_length)
        baseline_region = epochs[:, :pre_samples]

        if method == "mean":
            baseline = np.mean(baseline_region, axis=1, keepdims=True)
        elif method == "median":
            baseline = np.median(baseline_region, axis=1, keepdims=True)
        elif method == "linear":
            # Fit line to baseline, extrapolate
            x = np.arange(pre_samples)
            corrected = epochs.copy()
            for i in range(epochs.shape[0]):
                coeffs = np.polyfit(x, baseline_region[i], 1)
                x_full = np.arange(epochs.shape[1])
                baseline_fit = np.polyval(coeffs, x_full)
                corrected[i] -= baseline_fit
            return corrected
        elif method == "polynomial":
            x = np.arange(pre_samples)
            corrected = epochs.copy()
            for i in range(epochs.shape[0]):
                coeffs = np.polyfit(x, baseline_region[i], poly_order)
                x_full = np.arange(epochs.shape[1])
                baseline_fit = np.polyval(coeffs, x_full)
                corrected[i] -= baseline_fit
            return corrected
        else:
            raise ValueError(f"Unknown baseline method: {method}")

        return epochs - baseline

    elif epochs.ndim == 3:
        # Multi-channel: (n_channels, n_epochs, epoch_length)
        corrected = np.empty_like(epochs)
        for ch in range(epochs.shape[0]):
            corrected[ch] = baseline_correct(epochs[ch], dt, pre_time_s, method, poly_order)
        return corrected

    else:
        raise ValueError(f"Expected 2D or 3D array, got {epochs.ndim}D")


def quality_control(
    epochs: np.ndarray,
    dt: float,
    pre_time_s: float,
    config: QualityConfig | None = None,
) -> tuple[np.ndarray, QualityResult]:
    """Apply quality control to epochs, rejecting those that fail criteria.

    Args:
        epochs: Array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        pre_time_s: Duration of pre-stimulus period
        config: Quality control configuration

    Returns:
        Tuple of (filtered_epochs, quality_result)

    """
    if config is None:
        config = QualityConfig()

    if not config.enabled:
        passed = np.arange(epochs.shape[0])
        return epochs, QualityResult(
            n_input=epochs.shape[0],
            n_passed=epochs.shape[0],
            n_rejected=0,
            passed_indices=passed,
            rejected_indices=np.array([], dtype=int),
        )

    n_epochs = epochs.shape[0]
    pre_samples = int(pre_time_s / dt)

    passed_mask = np.ones(n_epochs, dtype=bool)
    rejection_reasons = []

    # Calculate baseline statistics
    baseline_region = epochs[:, :pre_samples]
    baseline_noise = np.std(baseline_region, axis=1)

    # Calculate amplitude metrics
    amplitudes = np.ptp(epochs, axis=1)  # Peak-to-peak amplitude
    max_abs = np.max(np.abs(epochs), axis=1)

    # Apply baseline noise threshold
    if config.max_baseline_noise is not None:
        noisy = baseline_noise > config.max_baseline_noise
        for idx in np.where(noisy)[0]:
            if passed_mask[idx]:
                passed_mask[idx] = False
                rejection_reasons.append(f"baseline_noise (epoch {idx})")

    # Apply maximum amplitude threshold
    if config.max_amplitude is not None:
        too_large = max_abs > config.max_amplitude
        for idx in np.where(too_large)[0]:
            if passed_mask[idx]:
                passed_mask[idx] = False
                rejection_reasons.append(f"max_amplitude (epoch {idx})")

    # Apply minimum amplitude threshold
    if config.min_amplitude is not None:
        too_small = amplitudes < config.min_amplitude
        for idx in np.where(too_small)[0]:
            if passed_mask[idx]:
                passed_mask[idx] = False
                rejection_reasons.append(f"min_amplitude (epoch {idx})")

    # Apply outlier detection using z-scores
    if config.outlier_zscore is not None and n_epochs > 2:
        # Z-score based on amplitude
        amp_mean = np.mean(amplitudes)
        amp_std = np.std(amplitudes)
        if amp_std > 0:
            z_scores = np.abs((amplitudes - amp_mean) / amp_std)
            outliers = z_scores > config.outlier_zscore
            for idx in np.where(outliers)[0]:
                if passed_mask[idx]:
                    passed_mask[idx] = False
                    rejection_reasons.append(
                        f"amplitude_outlier (epoch {idx}, z={z_scores[idx]:.1f})"
                    )

    # Apply template correlation threshold
    if config.min_template_correlation is not None and passed_mask.sum() >= 2:
        # Calculate template as mean of passed epochs so far
        template = np.mean(epochs[passed_mask], axis=0)
        template_std = np.std(template)
        if template_std > 0:
            for idx in range(n_epochs):
                if passed_mask[idx]:
                    epoch_std = np.std(epochs[idx])
                    if epoch_std > 0:
                        corr = np.corrcoef(epochs[idx], template)[0, 1]
                        if corr < config.min_template_correlation:
                            passed_mask[idx] = False
                            rejection_reasons.append(f"template_corr (epoch {idx}, r={corr:.2f})")

    passed_indices = np.where(passed_mask)[0]
    rejected_indices = np.where(~passed_mask)[0]

    result = QualityResult(
        n_input=n_epochs,
        n_passed=len(passed_indices),
        n_rejected=len(rejected_indices),
        passed_indices=passed_indices,
        rejected_indices=rejected_indices,
        rejection_reasons=rejection_reasons,
    )

    if result.n_passed < config.min_epochs:
        logger.warning(f"Only {result.n_passed} epochs passed QC (minimum: {config.min_epochs})")
    else:
        logger.info(
            f"Quality control: {result.n_passed}/{result.n_input} epochs passed "
            f"({result.rejection_rate:.1%} rejected)"
        )

    return epochs[passed_mask], result


def remove_stimulus_artifact(
    epochs: np.ndarray,
    dt: float,
    artifact_start_ms: float = 0.0,
    artifact_duration_ms: float = 2.0,
    method: Literal["linear", "zero", "median"] = "linear",
    pre_samples: int = 5,
    post_samples: int = 5,
) -> np.ndarray:
    """Remove or interpolate over stimulus artifacts in epochs.

    Args:
        epochs: Array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        artifact_start_ms: Start of artifact relative to epoch start (ms)
        artifact_duration_ms: Duration of artifact to remove (ms)
        method: Interpolation method
        pre_samples: Number of samples before artifact for interpolation
        post_samples: Number of samples after artifact for interpolation

    Returns:
        Epochs with artifacts removed/interpolated

    """
    result = epochs.copy()

    artifact_start_idx = int(artifact_start_ms / 1000.0 / dt)
    artifact_end_idx = artifact_start_idx + int(artifact_duration_ms / 1000.0 / dt)

    if artifact_end_idx >= epochs.shape[1]:
        logger.warning("Artifact extends beyond epoch, cannot remove")
        return result

    for i in range(epochs.shape[0]):
        if method == "zero":
            result[i, artifact_start_idx:artifact_end_idx] = 0

        elif method == "median":
            # Replace with median of surrounding samples
            pre_region = epochs[i, max(0, artifact_start_idx - pre_samples) : artifact_start_idx]
            post_region = epochs[i, artifact_end_idx : artifact_end_idx + post_samples]
            replacement = np.median(np.concatenate([pre_region, post_region]))
            result[i, artifact_start_idx:artifact_end_idx] = replacement

        elif method == "linear":
            # Linear interpolation between pre and post artifact
            pre_idx = max(0, artifact_start_idx - pre_samples)
            post_idx = min(epochs.shape[1], artifact_end_idx + post_samples)

            pre_val = np.mean(epochs[i, pre_idx:artifact_start_idx])
            post_val = np.mean(epochs[i, artifact_end_idx:post_idx])

            artifact_len = artifact_end_idx - artifact_start_idx
            interp = np.linspace(pre_val, post_val, artifact_len)
            result[i, artifact_start_idx:artifact_end_idx] = interp

        else:
            raise ValueError(f"Unknown artifact removal method: {method}")

    artifact_end = artifact_start_ms + artifact_duration_ms
    logger.info(f"Removed artifacts: {artifact_start_ms:.1f}-{artifact_end:.1f} ms ({method})")

    return result


def detect_artifact_bounds(
    epochs: np.ndarray,
    dt: float,
    threshold_std: float = 10.0,
    search_window_ms: float = 10.0,
) -> tuple[float, float]:
    """Automatically detect artifact boundaries in epochs.

    Args:
        epochs: Array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        threshold_std: Threshold as multiple of baseline std
        search_window_ms: Window to search for artifact

    Returns:
        Tuple of (artifact_start_ms, artifact_end_ms)

    """
    # Use median epoch to reduce noise
    median_epoch = np.median(epochs, axis=0)

    # Estimate baseline noise from first part of epoch
    search_samples = int(search_window_ms / 1000.0 / dt)
    baseline_std = np.std(median_epoch[: search_samples // 2])

    # Find artifact region using derivative
    deriv = np.abs(np.diff(median_epoch))
    threshold = threshold_std * baseline_std / dt

    above_threshold = deriv > threshold

    if not any(above_threshold):
        return 0.0, 0.0

    # Find first and last crossing
    crossings = np.where(above_threshold)[0]
    start_idx = crossings[0]
    end_idx = crossings[-1] + 1

    start_ms = start_idx * dt * 1000.0
    end_ms = end_idx * dt * 1000.0

    logger.info(f"Detected artifact bounds: {start_ms:.2f} - {end_ms:.2f} ms")

    return start_ms, end_ms
