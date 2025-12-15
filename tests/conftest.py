"""
Pytest configuration and fixtures for ephyalign tests.
"""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def sample_rate() -> float:
    """Standard sampling rate for test data."""
    return 10000.0  # 10 kHz


@pytest.fixture
def dt(sample_rate: float) -> float:
    """Sampling interval in seconds."""
    return 1.0 / sample_rate


@pytest.fixture
def synthetic_recording(sample_rate: float) -> np.ndarray:
    """
    Generate synthetic electrophysiology data with stimulus artifacts.

    Returns:
        1D array of synthetic data with clear stimulus artifacts
    """
    duration_s = 30.0  # 30 seconds of recording
    n_samples = int(duration_s * sample_rate)
    1.0 / sample_rate

    # Generate baseline noise
    np.random.seed(42)
    data = np.random.randn(n_samples) * 0.1  # Low baseline noise

    # Add stimulus artifacts every 5 seconds
    stim_interval_s = 5.0
    stim_times = np.arange(1.0, duration_s - 1.0, stim_interval_s)

    for stim_time in stim_times:
        stim_idx = int(stim_time * sample_rate)

        # Sharp artifact (capacitive transient)
        artifact_duration = int(0.002 * sample_rate)  # 2ms artifact
        for i in range(artifact_duration):
            if stim_idx + i < n_samples:
                data[stim_idx + i] += 5.0 * np.exp(-i / (0.0005 * sample_rate))

        # Response (slower, after artifact)
        response_start = stim_idx + artifact_duration
        response_duration = int(0.05 * sample_rate)  # 50ms response
        for i in range(response_duration):
            if response_start + i < n_samples:
                # Rising then decaying response
                t = i / sample_rate
                response = 2.0 * np.exp(-t / 0.02) * np.sin(t * 100)
                data[response_start + i] += response

    return data


@pytest.fixture
def synthetic_multichannel(synthetic_recording: np.ndarray) -> np.ndarray:
    """
    Generate multi-channel synthetic data.

    Returns:
        2D array of shape (3, n_samples)
    """
    # Create 3 channels with slightly different characteristics
    ch0 = synthetic_recording.copy()
    ch1 = synthetic_recording * 0.8 + np.random.randn(len(synthetic_recording)) * 0.05
    ch2 = synthetic_recording * 1.2 + np.random.randn(len(synthetic_recording)) * 0.15

    return np.vstack([ch0, ch1, ch2])


@pytest.fixture
def expected_stim_count() -> int:
    """Expected number of stimuli in synthetic data."""
    return 5  # Based on 30s recording with 5s intervals


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Temporary directory for output files."""
    output_dir = tmp_path / "ephyalign_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def sample_epochs(synthetic_recording: np.ndarray, sample_rate: float) -> np.ndarray:
    """Pre-extracted sample epochs for testing."""
    1.0 / sample_rate
    pre_samples = int(0.5 * sample_rate)  # 500ms pre
    post_samples = int(3.0 * sample_rate)  # 3s post
    pre_samples + post_samples

    # Extract epochs at known positions
    stim_indices = [int(t * sample_rate) for t in [5.0, 10.0, 15.0, 20.0, 25.0]]

    epochs = []
    for idx in stim_indices:
        start = idx - pre_samples
        end = idx + post_samples
        if start >= 0 and end <= len(synthetic_recording):
            epochs.append(synthetic_recording[start:end])

    return np.array(epochs)
