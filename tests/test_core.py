"""Tests for core processing modules."""

import numpy as np
import pytest

from ephyalign.core.detector import detect_stim_onsets, DetectionResult
from ephyalign.core.aligner import (
    build_epochs,
    refine_alignment,
    apply_alignment,
    EpochData,
    AlignmentResult,
)
from ephyalign.core.metrics import compute_epoch_metrics, EpochMetrics
from ephyalign.config import DetectionConfig, EpochConfig


class TestDetectStimOnsets:
    """Tests for stimulus detection."""
    
    def test_detects_stimuli(self, synthetic_recording, dt):
        result = detect_stim_onsets(synthetic_recording, dt, min_interval_s=3.0)
        
        assert isinstance(result, DetectionResult)
        assert result.n_detected > 0
        assert len(result.stim_indices) == result.n_detected
        assert len(result.stim_times_s) == result.n_detected
    
    def test_respects_min_interval(self, synthetic_recording, dt):
        # With large minimum interval, should detect fewer
        result_large = detect_stim_onsets(
            synthetic_recording, dt, min_interval_s=10.0
        )
        result_small = detect_stim_onsets(
            synthetic_recording, dt, min_interval_s=2.0
        )
        
        assert result_large.n_detected <= result_small.n_detected
    
    def test_threshold_affects_detection(self, synthetic_recording, dt):
        # Higher threshold should detect fewer (or same)
        result_high = detect_stim_onsets(
            synthetic_recording, dt, threshold_multiplier=20.0
        )
        result_low = detect_stim_onsets(
            synthetic_recording, dt, threshold_multiplier=2.0
        )
        
        assert result_high.n_detected <= result_low.n_detected
    
    def test_calculates_isi(self, synthetic_recording, dt):
        result = detect_stim_onsets(synthetic_recording, dt, min_interval_s=3.0)
        
        if result.n_detected > 1:
            assert not np.isnan(result.mean_isi_s)
            assert not np.isnan(result.std_isi_s)
            assert result.inter_stim_intervals_s is not None
    
    def test_empty_data(self, dt):
        empty = np.zeros(1000)
        result = detect_stim_onsets(empty, dt)
        # Should handle gracefully (may detect 0 or few false positives)
        assert isinstance(result, DetectionResult)
    
    def test_uses_config(self, synthetic_recording, dt):
        config = DetectionConfig(
            threshold_multiplier=3.0,
            min_interval_s=4.0,
        )
        result = detect_stim_onsets(synthetic_recording, dt, config=config)
        
        assert isinstance(result, DetectionResult)


class TestBuildEpochs:
    """Tests for epoch extraction."""
    
    def test_extracts_epochs(self, synthetic_recording, dt):
        stim_indices = np.array([50000, 100000, 150000])  # Known positions
        
        epochs = build_epochs(
            synthetic_recording,
            stim_indices,
            dt,
            pre_time_s=0.1,
            post_time_s=0.5,
        )
        
        assert isinstance(epochs, EpochData)
        assert epochs.n_epochs > 0
        assert epochs.epochs.shape[0] == epochs.n_epochs
    
    def test_epoch_length(self, synthetic_recording, dt):
        stim_indices = np.array([50000, 100000])
        pre_s, post_s = 0.2, 0.8
        
        epochs = build_epochs(
            synthetic_recording,
            stim_indices,
            dt,
            pre_time_s=pre_s,
            post_time_s=post_s,
        )
        
        expected_samples = int((pre_s + post_s) / dt)
        assert epochs.epoch_length == expected_samples
    
    def test_rejects_edge_epochs(self, synthetic_recording, dt):
        # Stimulus at very start - should be rejected
        stim_indices = np.array([100, 50000, 100000])
        
        epochs = build_epochs(
            synthetic_recording,
            stim_indices,
            dt,
            pre_time_s=0.5,  # 0.5s = 5000 samples at 10kHz
            post_time_s=0.5,
        )
        
        # First stimulus at idx=100 should be rejected (needs 5000 pre samples)
        assert epochs.n_epochs < len(stim_indices)
    
    def test_multichannel_input(self, synthetic_multichannel, dt):
        stim_indices = np.array([50000, 100000])
        
        epochs = build_epochs(
            synthetic_multichannel,
            stim_indices,
            dt,
            pre_time_s=0.1,
            post_time_s=0.5,
        )
        
        assert epochs.is_multichannel
        assert epochs.n_channels == 3
    
    def test_time_axis(self, synthetic_recording, dt):
        stim_indices = np.array([50000])
        pre_s = 0.1
        
        epochs = build_epochs(
            synthetic_recording,
            stim_indices,
            dt,
            pre_time_s=pre_s,
            post_time_s=0.5,
        )
        
        # Time axis should start negative (pre-stimulus)
        assert epochs.time_axis[0] < 0
        # Should cross zero
        assert epochs.time_axis[-1] > 0


class TestRefineAlignment:
    """Tests for alignment refinement."""
    
    def test_refines_epochs(self, sample_epochs, dt):
        result = refine_alignment(sample_epochs, dt, search_ms=5.0)
        
        assert isinstance(result, AlignmentResult)
        assert result.epochs.shape[0] == sample_epochs.shape[0]
        assert len(result.onset_positions) == sample_epochs.shape[0]
    
    def test_calculates_jitter(self, sample_epochs, dt):
        result = refine_alignment(sample_epochs, dt)
        
        # Jitter should be a non-negative number
        assert result.jitter_ms >= 0
    
    def test_all_epochs_same_length(self, sample_epochs, dt):
        result = refine_alignment(sample_epochs, dt)
        
        # All refined epochs should have same length
        assert result.epochs.shape[1] == result.epoch_length
    
    def test_empty_input(self, dt):
        empty = np.array([]).reshape(0, 100)
        result = refine_alignment(empty, dt)
        
        assert result.epochs.size == 0
    
    def test_keeps_original(self, sample_epochs, dt):
        result = refine_alignment(sample_epochs, dt, keep_original=True)
        
        assert result.original_epochs is not None
        assert result.original_epochs.shape == sample_epochs.shape


class TestApplyAlignment:
    """Tests for applying alignment to additional channels."""
    
    def test_applies_offsets(self, sample_epochs, dt):
        # First refine reference channel
        ref_result = refine_alignment(sample_epochs, dt)
        
        # Create another "channel" with same epochs
        other_epochs = sample_epochs + np.random.randn(*sample_epochs.shape) * 0.1
        
        # Apply same alignment
        aligned = apply_alignment(
            other_epochs,
            ref_result.onset_positions,
            ref_result.epoch_length,
        )
        
        assert aligned.shape == ref_result.epochs.shape
    
    def test_mismatched_length_raises(self, sample_epochs, dt):
        ref_result = refine_alignment(sample_epochs, dt)
        
        # Wrong number of epochs
        wrong_epochs = sample_epochs[:2]
        
        with pytest.raises(ValueError):
            apply_alignment(
                wrong_epochs,
                ref_result.onset_positions,
                ref_result.epoch_length,
            )


class TestComputeEpochMetrics:
    """Tests for metrics calculation."""
    
    def test_computes_metrics(self, sample_epochs, dt):
        metrics = compute_epoch_metrics(sample_epochs, dt)
        
        assert isinstance(metrics, EpochMetrics)
        assert metrics.n_epochs == sample_epochs.shape[0]
    
    def test_per_sweep_metrics(self, sample_epochs, dt):
        metrics = compute_epoch_metrics(sample_epochs, dt)
        
        assert len(metrics.baseline_mean) == metrics.n_epochs
        assert len(metrics.peak_amplitude) == metrics.n_epochs
        assert len(metrics.time_to_peak_s) == metrics.n_epochs
    
    def test_summary_statistics(self, sample_epochs, dt):
        metrics = compute_epoch_metrics(sample_epochs, dt)
        
        # Should have summary stats
        assert not np.isnan(metrics.peak_amp_mean) or metrics.n_epochs == 0
        assert not np.isnan(metrics.baseline_noise) or metrics.n_epochs == 0
    
    def test_to_dict(self, sample_epochs, dt):
        metrics = compute_epoch_metrics(sample_epochs, dt)
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert "n_epochs" in d
        assert "peak_amp_mean" in d
    
    def test_empty_input(self, dt):
        empty = np.array([]).reshape(0, 100)
        metrics = compute_epoch_metrics(empty, dt)
        
        assert metrics.n_epochs == 0


class TestIntegration:
    """Integration tests combining multiple core functions."""
    
    def test_detect_and_extract(self, synthetic_recording, dt):
        # Detect
        detection = detect_stim_onsets(synthetic_recording, dt, min_interval_s=3.0)
        
        if detection.n_detected == 0:
            pytest.skip("No stimuli detected in synthetic data")
        
        # Extract
        epochs = build_epochs(
            synthetic_recording,
            detection.stim_indices,
            dt,
            pre_time_s=0.5,
            post_time_s=3.0,
        )
        
        assert epochs.n_epochs > 0
    
    def test_full_pipeline(self, synthetic_recording, dt):
        # Detect
        detection = detect_stim_onsets(synthetic_recording, dt, min_interval_s=3.0)
        
        if detection.n_detected == 0:
            pytest.skip("No stimuli detected")
        
        # Extract
        epochs = build_epochs(
            synthetic_recording,
            detection.stim_indices,
            dt,
        )
        
        if epochs.n_epochs == 0:
            pytest.skip("No epochs extracted")
        
        # Align
        alignment = refine_alignment(epochs.epochs, dt)
        
        # Metrics
        metrics = compute_epoch_metrics(alignment.epochs, dt)
        
        assert metrics.n_epochs == alignment.n_epochs
