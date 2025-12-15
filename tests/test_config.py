"""Tests for configuration management."""

import pytest
from pathlib import Path

from ephyalign.config import (
    AlignmentConfig,
    DetectionConfig,
    EpochConfig,
    MetricsConfig,
    OutputConfig,
    PlotConfig,
    get_default_config,
)


class TestDetectionConfig:
    """Tests for DetectionConfig."""
    
    def test_default_values(self):
        config = DetectionConfig()
        assert config.threshold_multiplier == 5.0
        assert config.min_interval_s == 3.0
        assert config.search_window_ms == 1.0
        assert config.use_absolute_derivative is True
    
    def test_custom_values(self):
        config = DetectionConfig(
            threshold_multiplier=10.0,
            min_interval_s=5.0,
        )
        assert config.threshold_multiplier == 10.0
        assert config.min_interval_s == 5.0


class TestEpochConfig:
    """Tests for EpochConfig."""
    
    def test_default_values(self):
        config = EpochConfig()
        assert config.pre_time_s == 0.5
        assert config.post_time_s == 3.0
        assert config.baseline_subtract is False
    
    def test_epoch_duration(self):
        config = EpochConfig(pre_time_s=1.0, post_time_s=5.0)
        total = config.pre_time_s + config.post_time_s
        assert total == 6.0


class TestAlignmentConfig:
    """Tests for main AlignmentConfig."""
    
    def test_default_config(self):
        config = AlignmentConfig()
        assert config.input_file is None
        assert config.reference_channel == 0
        assert isinstance(config.detection, DetectionConfig)
        assert isinstance(config.epoch, EpochConfig)
    
    def test_convenience_properties(self):
        config = AlignmentConfig()
        assert config.min_interval_s == config.detection.min_interval_s
        assert config.pre_time_s == config.epoch.pre_time_s
        assert config.post_time_s == config.epoch.post_time_s
    
    def test_to_dict(self):
        config = AlignmentConfig()
        d = config.to_dict()
        
        assert isinstance(d, dict)
        assert "detection" in d
        assert "epoch" in d
        assert "reference_channel" in d
    
    def test_from_dict(self):
        data = {
            "reference_channel": 1,
            "detection": {"threshold_multiplier": 10.0},
            "epoch": {"pre_time_s": 1.0},
        }
        
        config = AlignmentConfig.from_dict(data)
        assert config.reference_channel == 1
        assert config.detection.threshold_multiplier == 10.0
        assert config.epoch.pre_time_s == 1.0
    
    def test_roundtrip(self):
        original = AlignmentConfig(reference_channel=2)
        original.detection.threshold_multiplier = 7.5
        original.epoch.post_time_s = 4.0
        
        d = original.to_dict()
        restored = AlignmentConfig.from_dict(d)
        
        assert restored.reference_channel == 2
        assert restored.detection.threshold_multiplier == 7.5
        assert restored.epoch.post_time_s == 4.0


class TestGetDefaultConfig:
    """Tests for get_default_config helper."""
    
    def test_returns_alignment_config(self):
        config = get_default_config()
        assert isinstance(config, AlignmentConfig)
    
    def test_has_all_subconfigs(self):
        config = get_default_config()
        assert config.detection is not None
        assert config.epoch is not None
        assert config.metrics is not None
        assert config.output is not None
        assert config.plot is not None


class TestOutputConfig:
    """Tests for OutputConfig."""
    
    def test_default_save_flags(self):
        config = OutputConfig()
        assert config.save_npz is True
        assert config.save_atf is True
        assert config.save_hdf5 is True
        assert config.save_plots is True
    
    def test_output_dir_conversion(self):
        config = OutputConfig(output_dir="/some/path")
        assert isinstance(config.output_dir, Path)


class TestPlotConfig:
    """Tests for PlotConfig."""
    
    def test_default_values(self):
        config = PlotConfig()
        assert config.dpi == 200
        assert config.format == "png"
        assert config.overlay_alpha == 0.3
    
    def test_figsize_is_tuple(self):
        config = PlotConfig()
        assert isinstance(config.figsize, tuple)
        assert len(config.figsize) == 2
