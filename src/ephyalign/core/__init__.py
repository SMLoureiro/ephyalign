"""
Core processing modules for ephyalign.

This package contains the fundamental signal processing components:
- loader: ABF file loading and data extraction
- detector: Stimulus artifact detection
- aligner: Epoch extraction and alignment
- metrics: Response metrics calculation
"""

from ephyalign.core.loader import load_recording, RecordingData
from ephyalign.core.detector import detect_stim_onsets
from ephyalign.core.aligner import build_epochs, refine_alignment, apply_alignment
from ephyalign.core.metrics import compute_epoch_metrics, EpochMetrics

__all__ = [
    "load_recording",
    "RecordingData",
    "detect_stim_onsets",
    "build_epochs",
    "refine_alignment",
    "apply_alignment",
    "compute_epoch_metrics",
    "EpochMetrics",
]
