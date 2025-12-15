"""
Plotting functions for aligned epochs.

This module provides functions for creating diagnostic and analysis plots
from aligned electrophysiological recordings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ephyalign.config import PlotConfig
from ephyalign.io.paths import OutputPaths

logger = logging.getLogger(__name__)


def plot_overlay(
    epochs: np.ndarray,
    time_axis: np.ndarray,
    y_label: str = "Response",
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    show_average: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """
    Plot all epochs overlaid on same axes.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        time_axis: Time axis in seconds
        y_label: Label for Y axis
        title: Plot title
        config: PlotConfig for styling
        ax: Optional existing axes to plot on
        show_average: Whether to show average trace
    
    Returns:
        Tuple of (Figure, Axes)
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()
    
    # Plot individual epochs
    for ep in epochs:
        ax.plot(time_axis, ep, alpha=config.overlay_alpha, linewidth=0.8)
    
    # Plot average
    if show_average and epochs.shape[0] > 1:
        mean_trace = epochs.mean(axis=0)
        ax.plot(
            time_axis, 
            mean_trace, 
            color="black", 
            linewidth=config.average_linewidth,
            label="Average",
        )
        ax.legend()
    
    # Stimulus onset marker
    ax.axvline(0, linestyle="--", color="gray", alpha=0.7, label="Stimulus")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Aligned Responses ({epochs.shape[0]} epochs)")
    
    return fig, ax


def plot_average(
    epochs: np.ndarray,
    time_axis: np.ndarray,
    y_label: str = "Response",
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    show_sem: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """
    Plot average response with optional SEM shading.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        time_axis: Time axis in seconds
        y_label: Label for Y axis
        title: Plot title
        config: PlotConfig for styling
        ax: Optional existing axes to plot on
        show_sem: Whether to show SEM shading
    
    Returns:
        Tuple of (Figure, Axes)
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()
    
    mean_trace = epochs.mean(axis=0)
    
    # Plot mean
    ax.plot(
        time_axis, 
        mean_trace, 
        color="blue", 
        linewidth=config.average_linewidth,
    )
    
    # SEM shading
    if show_sem and epochs.shape[0] > 1:
        sem = epochs.std(axis=0) / np.sqrt(epochs.shape[0])
        ax.fill_between(
            time_axis,
            mean_trace - sem,
            mean_trace + sem,
            alpha=0.3,
            color="blue",
            label="SEM",
        )
        ax.legend()
    
    # Stimulus onset marker
    ax.axvline(0, linestyle="--", color="gray", alpha=0.7)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Average Response (n={epochs.shape[0]})")
    
    return fig, ax


def plot_zoom_alignment(
    epochs: np.ndarray,
    time_axis: np.ndarray,
    y_label: str = "Response",
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    baseline_subtract: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """
    Plot zoomed view around stimulus onset for alignment verification.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        time_axis: Time axis in seconds
        y_label: Label for Y axis
        title: Plot title
        config: PlotConfig for styling
        ax: Optional existing axes to plot on
        baseline_subtract: Whether to subtract baseline from each epoch
    
    Returns:
        Tuple of (Figure, Axes)
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()
    
    # Convert time to ms for zoom
    time_ms = time_axis * 1000.0
    
    # Plot epochs
    for ep in epochs:
        if baseline_subtract:
            ep_plot = ep - ep[0]
        else:
            ep_plot = ep
        ax.plot(time_ms, ep_plot, alpha=0.7, linewidth=0.8)
    
    # Zoom range
    zoom_start, zoom_end = config.zoom_range_ms
    ax.set_xlim(zoom_start, zoom_end)
    
    # Stimulus onset marker
    ax.axvline(0, linestyle="--", color="gray", alpha=0.7)
    
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title("Zoomed Alignment Around Stimulus")
    
    return fig, ax


def plot_concatenated(
    epochs: np.ndarray,
    dt: float,
    y_label: str = "Response",
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    show_boundaries: bool = True,
) -> Tuple[Figure, plt.Axes]:
    """
    Plot epochs concatenated in sequence.
    
    Useful for visualizing overall response stability across the recording.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        dt: Sampling interval in seconds
        y_label: Label for Y axis
        title: Plot title
        config: PlotConfig for styling
        ax: Optional existing axes to plot on
        show_boundaries: Whether to show epoch boundary markers
    
    Returns:
        Tuple of (Figure, Axes)
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
    else:
        fig = ax.get_figure()
    
    # Concatenate epochs
    concat = epochs.ravel()
    time_axis = np.arange(len(concat)) * dt
    
    ax.plot(time_axis, concat, linewidth=0.8)
    
    # Epoch boundaries
    if show_boundaries:
        epoch_length = epochs.shape[1]
        for k in range(1, epochs.shape[0]):
            boundary_time = k * epoch_length * dt
            ax.axvline(boundary_time, linestyle="--", alpha=0.3, color="gray")
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Concatenated Epochs ({epochs.shape[0]} epochs)")
    
    return fig, ax


def plot_all_diagnostics(
    epochs: np.ndarray,
    time_axis: np.ndarray,
    dt: float,
    output_dir: Union[str, Path, OutputPaths],
    base_name: str,
    y_label: str = "Response",
    config: Optional[PlotConfig] = None,
) -> Dict[str, Path]:
    """
    Generate all diagnostic plots and save to files.
    
    Args:
        epochs: 2D array of epochs (n_epochs, epoch_length)
        time_axis: Time axis in seconds
        dt: Sampling interval
        output_dir: Output directory or OutputPaths
        base_name: Base name for output files
        y_label: Label for Y axis
        config: PlotConfig for styling
    
    Returns:
        Dictionary mapping plot names to saved file paths
    """
    if config is None:
        config = PlotConfig()
    
    # Determine output directory
    if isinstance(output_dir, OutputPaths):
        plot_dir = output_dir.plots
        plot_dir.mkdir(parents=True, exist_ok=True)
    else:
        plot_dir = Path(output_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
    
    saved_plots = {}
    
    # 1. Zoom alignment plot
    fig, ax = plot_zoom_alignment(
        epochs, time_axis, y_label=y_label,
        title=f"Zoomed Alignment ({base_name})",
        config=config,
    )
    path = plot_dir / f"{base_name}_zoom_alignment.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["zoom_alignment"] = path
    logger.debug(f"Saved {path}")
    
    # 2. Overlay plot
    fig, ax = plot_overlay(
        epochs, time_axis, y_label=y_label,
        title=f"Aligned Responses ({base_name})",
        config=config,
    )
    path = plot_dir / f"{base_name}_overlay_epochs.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["overlay_epochs"] = path
    logger.debug(f"Saved {path}")
    
    # 3. Average response plot
    fig, ax = plot_average(
        epochs, time_axis, y_label=y_label,
        title=f"Average Response ({base_name})",
        config=config,
    )
    path = plot_dir / f"{base_name}_average_response.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["average_response"] = path
    logger.debug(f"Saved {path}")
    
    # 4. Concatenated plot
    fig, ax = plot_concatenated(
        epochs, dt, y_label=y_label,
        title=f"Concatenated Epochs ({base_name})",
        config=config,
    )
    path = plot_dir / f"{base_name}_concatenated_epochs.{config.format}"
    fig.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)
    saved_plots["concatenated_epochs"] = path
    logger.debug(f"Saved {path}")
    
    logger.info(f"Saved {len(saved_plots)} plots to {plot_dir}")
    
    return saved_plots


def create_summary_figure(
    epochs: np.ndarray,
    time_axis: np.ndarray,
    dt: float,
    y_label: str = "Response",
    title: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Figure:
    """
    Create a summary figure with all diagnostic plots in a single figure.
    
    Args:
        epochs: 2D array of epochs
        time_axis: Time axis in seconds
        dt: Sampling interval
        y_label: Y-axis label
        title: Overall figure title
        config: PlotConfig for styling
    
    Returns:
        Matplotlib Figure with 2x2 grid of plots
    """
    if config is None:
        config = PlotConfig()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Zoom alignment
    plot_zoom_alignment(
        epochs, time_axis, y_label=y_label,
        title="Alignment Quality",
        config=config, ax=axes[0, 0],
    )
    
    # Top-right: Overlay
    plot_overlay(
        epochs, time_axis, y_label=y_label,
        title="All Epochs Overlaid",
        config=config, ax=axes[0, 1],
    )
    
    # Bottom-left: Average
    plot_average(
        epochs, time_axis, y_label=y_label,
        title="Average Response",
        config=config, ax=axes[1, 0],
    )
    
    # Bottom-right: Concatenated
    plot_concatenated(
        epochs, dt, y_label=y_label,
        title="Response Stability",
        config=config, ax=axes[1, 1],
    )
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    return fig
