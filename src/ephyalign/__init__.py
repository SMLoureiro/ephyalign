"""
ephyalign - Electrophysiology Response Alignment Toolkit

A Python package for aligning electrophysiological responses to stimulation events
in ABF (Axon Binary Format) recordings. Automatically detects stimulus artifacts
and aligns epochs with sub-millisecond precision.

Copyright (c) 2025 Samuel Matthews Mckay Loureiro
License: MIT
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ephyalign")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

from ephyalign.config import AlignmentConfig
from ephyalign.core.loader import load_recording
from ephyalign.core.detector import detect_stim_onsets
from ephyalign.core.aligner import build_epochs, refine_alignment
from ephyalign.core.metrics import compute_epoch_metrics
from ephyalign.pipeline import align_recording, AlignmentResult

__all__ = [
    "__version__",
    "AlignmentConfig",
    "AlignmentResult",
    "align_recording",
    "load_recording",
    "detect_stim_onsets",
    "build_epochs",
    "refine_alignment",
    "compute_epoch_metrics",
]
