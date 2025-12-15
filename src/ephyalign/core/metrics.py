"""
Response metrics calculation.

This module provides functions for computing electrophysiological response
metrics from aligned epochs, including peak amplitude, timing, rise time,
and area under curve measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ephyalign.config import MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """
    Comprehensive metrics computed from aligned epochs.
    
    Contains both per-sweep (trial) metrics and summary statistics.
    
    Attributes:
        n_epochs: Number of epochs analyzed
        
        # Per-sweep metrics (lists, one value per epoch)
        baseline_mean: Mean baseline value for each epoch
        baseline_std: Standard deviation of baseline for each epoch
        peak_amplitude: Peak amplitude (relative to baseline) for each epoch
        time_to_peak_s: Time from stimulus to peak for each epoch
        rise_time_s: 10-90% rise time for each epoch
        auc: Area under curve for each epoch
        artifact_index: Detected artifact sample index for each epoch
        
        # Summary statistics
        baseline_noise: Overall mean baseline noise (SD)
        peak_amp_mean: Mean peak amplitude across epochs
        peak_amp_std: Std of peak amplitude
        peak_amp_cv: Coefficient of variation of peak amplitude
        time_to_peak_mean_ms: Mean time to peak in milliseconds
        time_to_peak_std_ms: Std of time to peak
        rise_time_mean_ms: Mean 10-90% rise time in milliseconds
        rise_time_std_ms: Std of rise time
        auc_mean: Mean AUC
        auc_std: Std of AUC
        jitter_ms: Artifact timing jitter (std of artifact positions)
    """
    
    n_epochs: int
    
    # Per-sweep metrics
    baseline_mean: List[float] = field(default_factory=list)
    baseline_std: List[float] = field(default_factory=list)
    peak_amplitude: List[float] = field(default_factory=list)
    time_to_peak_s: List[float] = field(default_factory=list)
    rise_time_s: List[float] = field(default_factory=list)
    auc: List[float] = field(default_factory=list)
    artifact_index: List[int] = field(default_factory=list)
    
    # Summary statistics
    baseline_noise: float = float("nan")
    peak_amp_mean: float = float("nan")
    peak_amp_std: float = float("nan")
    peak_amp_cv: float = float("nan")
    time_to_peak_mean_ms: float = float("nan")
    time_to_peak_std_ms: float = float("nan")
    rise_time_mean_ms: float = float("nan")
    rise_time_std_ms: float = float("nan")
    auc_mean: float = float("nan")
    auc_std: float = float("nan")
    jitter_ms: float = float("nan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "n_epochs": self.n_epochs,
            "baseline_mean_per_sweep": self.baseline_mean,
            "baseline_std_per_sweep": self.baseline_std,
            "peak_amp_per_sweep": self.peak_amplitude,
            "time_to_peak_s_per_sweep": self.time_to_peak_s,
            "rise_time_s_per_sweep": self.rise_time_s,
            "auc_per_sweep": self.auc,
            "artifact_indices": self.artifact_index,
            "baseline_noise_overall": self.baseline_noise,
            "peak_amp_mean": self.peak_amp_mean,
            "peak_amp_std": self.peak_amp_std,
            "peak_amp_cv": self.peak_amp_cv,
            "time_to_peak_mean_ms": self.time_to_peak_mean_ms,
            "time_to_peak_std_ms": self.time_to_peak_std_ms,
            "rise_time_mean_ms": self.rise_time_mean_ms,
            "rise_time_std_ms": self.rise_time_std_ms,
            "auc_mean": self.auc_mean,
            "auc_std": self.auc_std,
            "jitter_early_ms": self.jitter_ms,
        }
    
    def __repr__(self) -> str:
        return (
            f"EpochMetrics(n={self.n_epochs}, "
            f"peak_amp={self.peak_amp_mean:.3f}±{self.peak_amp_std:.3f}, "
            f"t_peak={self.time_to_peak_mean_ms:.2f}ms)"
        )


def _nanmean(arr: np.ndarray) -> float:
    """Compute mean ignoring NaN values."""
    return float(np.nanmean(arr)) if arr.size > 0 else float("nan")


def _nanstd(arr: np.ndarray, ddof: int = 1) -> float:
    """Compute std ignoring NaN values."""
    if arr.size <= ddof:
        return float("nan")
    return float(np.nanstd(arr, ddof=ddof))


def compute_epoch_metrics(
    epochs: np.ndarray,
    dt: float,
    config: Optional[MetricsConfig] = None,
    baseline_ms: Optional[float] = None,
    peak_window_ms: Optional[float] = None,
) -> EpochMetrics:
    """
    Compute comprehensive metrics from aligned epochs.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        config: MetricsConfig with analysis parameters
        baseline_ms: Baseline window in ms (overrides config)
        peak_window_ms: Peak search window in ms (overrides config)
    
    Returns:
        EpochMetrics containing per-sweep and summary statistics
    
    Metrics computed:
        - Baseline mean and noise (SD) for each epoch
        - Peak amplitude (relative to baseline)
        - Time to peak from artifact
        - 10-90% rise time
        - Area under curve (baseline-subtracted)
        - Artifact timing jitter
    
    Example:
        >>> metrics = compute_epoch_metrics(epochs, dt=0.0001)
        >>> print(f"Peak amplitude: {metrics.peak_amp_mean:.2f} ± {metrics.peak_amp_std:.2f}")
    """
    if config is None:
        config = MetricsConfig()
    
    # Allow parameter overrides
    baseline_window_ms = baseline_ms if baseline_ms is not None else config.baseline_ms
    peak_search_ms = peak_window_ms if peak_window_ms is not None else config.peak_window_ms
    
    # Handle empty input
    if epochs.size == 0:
        logger.warning("Empty epochs array, returning empty metrics")
        return EpochMetrics(n_epochs=0)
    
    # Handle 1D input (single epoch)
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    
    n_epochs, n_samples = epochs.shape
    
    # Convert time windows to samples
    baseline_samples = int(baseline_window_ms / 1000.0 / dt)
    search_samples = int(config.search_ms / 1000.0 / dt)
    peak_window_samples = int(peak_search_ms / 1000.0 / dt)
    
    # Ensure reasonable bounds
    baseline_samples = max(1, min(baseline_samples, n_samples // 4))
    search_samples = max(1, min(search_samples, n_samples // 4))
    
    logger.debug(
        f"Metrics params: baseline={baseline_samples}samp, "
        f"search={search_samples}samp, peak_window={peak_window_samples}samp"
    )
    
    # Initialize per-sweep lists
    baseline_means = []
    baseline_stds = []
    peak_amps = []
    time_to_peak_s = []
    rise_times_s = []
    aucs = []
    artifact_indices = []
    
    for ep in epochs:
        # --- Baseline analysis ---
        baseline = ep[:baseline_samples]
        b_mean = float(np.mean(baseline))
        b_std = float(np.std(baseline))
        baseline_means.append(b_mean)
        baseline_stds.append(b_std)
        
        # --- Artifact detection ---
        seg = ep[:search_samples]
        derivative = np.diff(seg)
        art_idx = int(np.argmax(np.abs(derivative)))
        artifact_indices.append(art_idx)
        
        # --- Peak search ---
        start_peak = art_idx
        end_peak = min(n_samples, art_idx + peak_window_samples)
        
        if end_peak <= start_peak:
            peak_amps.append(float("nan"))
            time_to_peak_s.append(float("nan"))
            rise_times_s.append(float("nan"))
            aucs.append(float("nan"))
            continue
        
        local = ep[start_peak:end_peak]
        
        # Find peak (direction depends on config)
        if config.upward_responses:
            local_peak_rel = int(np.argmax(local))
        else:
            local_peak_rel = int(np.argmin(local))
        
        peak_idx = start_peak + local_peak_rel
        peak_val = float(ep[peak_idx])
        peak_amp = peak_val - b_mean
        peak_amps.append(peak_amp)
        
        # Time to peak (from artifact to peak)
        t_peak = (peak_idx - art_idx) * dt
        time_to_peak_s.append(t_peak)
        
        # --- Rise time 10-90% ---
        level10 = b_mean + 0.1 * peak_amp
        level90 = b_mean + 0.9 * peak_amp
        post_seg = ep[art_idx:peak_idx + 1]
        
        try:
            if config.upward_responses:
                idx10 = int(np.where(post_seg >= level10)[0][0])
                idx90 = int(np.where(post_seg >= level90)[0][0])
            else:
                idx10 = int(np.where(post_seg <= level10)[0][0])
                idx90 = int(np.where(post_seg <= level90)[0][0])
            rise_time = abs(idx90 - idx10) * dt
        except IndexError:
            rise_time = float("nan")
        rise_times_s.append(rise_time)
        
        # --- Area under curve ---
        auc = float(np.trapezoid(ep[start_peak:end_peak] - b_mean, dx=dt))
        aucs.append(auc)
    
    # Convert to arrays for statistics
    artifact_arr = np.asarray(artifact_indices, dtype=np.int64)
    peak_arr = np.asarray(peak_amps, dtype=np.float64)
    ttp_arr = np.asarray(time_to_peak_s, dtype=np.float64)
    rise_arr = np.asarray(rise_times_s, dtype=np.float64)
    auc_arr = np.asarray(aucs, dtype=np.float64)
    baseline_std_arr = np.asarray(baseline_stds, dtype=np.float64)
    
    # Compute summary statistics
    jitter_ms = float(np.std(artifact_arr) * dt * 1000.0) if len(artifact_arr) > 1 else 0.0
    
    peak_mean = _nanmean(peak_arr)
    peak_std = _nanstd(peak_arr)
    peak_cv = peak_std / abs(peak_mean) if peak_mean != 0 and not np.isnan(peak_mean) else float("nan")
    
    metrics = EpochMetrics(
        n_epochs=n_epochs,
        baseline_mean=baseline_means,
        baseline_std=baseline_stds,
        peak_amplitude=peak_amps,
        time_to_peak_s=time_to_peak_s,
        rise_time_s=rise_times_s,
        auc=aucs,
        artifact_index=artifact_indices,
        baseline_noise=_nanmean(baseline_std_arr),
        peak_amp_mean=peak_mean,
        peak_amp_std=peak_std,
        peak_amp_cv=peak_cv,
        time_to_peak_mean_ms=_nanmean(ttp_arr) * 1000.0,
        time_to_peak_std_ms=_nanstd(ttp_arr) * 1000.0,
        rise_time_mean_ms=_nanmean(rise_arr) * 1000.0,
        rise_time_std_ms=_nanstd(rise_arr) * 1000.0,
        auc_mean=_nanmean(auc_arr),
        auc_std=_nanstd(auc_arr),
        jitter_ms=jitter_ms,
    )
    
    logger.info(
        f"Computed metrics for {n_epochs} epochs: "
        f"peak={peak_mean:.3f}±{peak_std:.3f}"
    )
    
    return metrics


def compute_signal_to_noise(
    epochs: np.ndarray,
    dt: float,
    baseline_ms: float = 10.0,
    signal_window_ms: tuple = (0, 50),
) -> float:
    """
    Compute signal-to-noise ratio for the response.
    
    SNR is defined as (peak - baseline) / baseline_std
    
    Args:
        epochs: 2D array of epochs
        dt: Sampling interval
        baseline_ms: Baseline window duration
        signal_window_ms: Tuple of (start, end) for signal window in ms
    
    Returns:
        Signal-to-noise ratio
    """
    if epochs.size == 0:
        return float("nan")
    
    if epochs.ndim == 1:
        epochs = epochs.reshape(1, -1)
    
    baseline_samples = int(baseline_ms / 1000.0 / dt)
    signal_start = int(signal_window_ms[0] / 1000.0 / dt)
    signal_end = int(signal_window_ms[1] / 1000.0 / dt)
    
    # Mean across epochs
    mean_epoch = epochs.mean(axis=0)
    
    # Baseline stats
    baseline = mean_epoch[:baseline_samples]
    baseline_mean = np.mean(baseline)
    baseline_std = np.std(baseline)
    
    if baseline_std == 0:
        return float("nan")
    
    # Signal peak
    signal = mean_epoch[signal_start:signal_end]
    peak = np.max(np.abs(signal - baseline_mean))
    
    snr = peak / baseline_std
    
    logger.debug(f"SNR: {snr:.2f}")
    
    return float(snr)


def compute_response_consistency(
    epochs: np.ndarray,
    dt: float,
    response_window_ms: tuple = (0, 100),
) -> Dict[str, float]:
    """
    Compute measures of response consistency across trials.
    
    Args:
        epochs: 2D array of epochs
        dt: Sampling interval
        response_window_ms: Time window to analyze
    
    Returns:
        Dictionary with consistency metrics
    """
    if epochs.size == 0:
        return {"correlation_mean": float("nan"), "correlation_std": float("nan")}
    
    if epochs.ndim == 1:
        return {"correlation_mean": 1.0, "correlation_std": 0.0}
    
    n_epochs = epochs.shape[0]
    
    start_samp = int(response_window_ms[0] / 1000.0 / dt)
    end_samp = int(response_window_ms[1] / 1000.0 / dt)
    end_samp = min(end_samp, epochs.shape[1])
    
    # Extract response windows
    responses = epochs[:, start_samp:end_samp]
    
    # Compute mean response
    mean_response = responses.mean(axis=0)
    
    # Correlate each trial with mean
    correlations = []
    for i in range(n_epochs):
        if np.std(responses[i]) > 0 and np.std(mean_response) > 0:
            corr = np.corrcoef(responses[i], mean_response)[0, 1]
            correlations.append(corr)
    
    if len(correlations) == 0:
        return {"correlation_mean": float("nan"), "correlation_std": float("nan")}
    
    corr_arr = np.asarray(correlations)
    
    return {
        "correlation_mean": float(np.mean(corr_arr)),
        "correlation_std": float(np.std(corr_arr)),
        "n_valid": len(correlations),
    }
