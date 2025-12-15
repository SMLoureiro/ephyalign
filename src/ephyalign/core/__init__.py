"""Core processing modules for ephyalign.

This package contains the fundamental signal processing components:
- loader: ABF file loading and data extraction
- detector: Stimulus artifact detection
- aligner: Epoch extraction and alignment
- metrics: Response metrics calculation
"""

from ephyalign.core.aligner import apply_alignment, build_epochs, refine_alignment
from ephyalign.core.detector import detect_stim_onsets
from ephyalign.core.loader import RecordingData, load_recording
from ephyalign.core.metrics import EpochMetrics, compute_epoch_metrics

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
