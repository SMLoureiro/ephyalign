"""Visualization module for ephyalign.

Provides functions for creating diagnostic and publication-quality plots
of aligned electrophysiological responses.
"""

from ephyalign.visualization.plots import (
    plot_all_diagnostics,
    plot_average,
    plot_concatenated,
    plot_overlay,
    plot_zoom_alignment,
)

__all__ = [
    "plot_overlay",
    "plot_average",
    "plot_zoom_alignment",
    "plot_concatenated",
    "plot_all_diagnostics",
]
