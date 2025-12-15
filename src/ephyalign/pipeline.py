"""Main processing pipeline for ephyalign.

This module provides the high-level pipeline that orchestrates all processing
steps: loading, detection, alignment, metrics, and output generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ephyalign.config import AlignmentConfig
from ephyalign.core.aligner import (
    AlignmentResult,
    EpochData,
    align_multichannel,
    build_epochs,
    refine_alignment,
)
from ephyalign.core.detector import DetectionResult, detect_stim_onsets
from ephyalign.core.loader import RecordingData, load_recording
from ephyalign.core.metrics import EpochMetrics, compute_epoch_metrics
from ephyalign.io.exporters import save_all_formats, write_stats_report
from ephyalign.io.paths import OutputPaths, build_output_paths
from ephyalign.visualization.plots import plot_all_diagnostics

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete results from the alignment pipeline.

    Contains all intermediate and final results from processing.

    Attributes:
        config: Configuration used for processing
        recording: Loaded recording data
        detection: Stimulus detection results
        epochs_raw: Raw extracted epochs
        epochs_aligned: Aligned epochs (single channel or reference)
        epochs_all_aligned: All channels aligned (if multi-channel)
        alignment: Alignment result with jitter info
        metrics: Response metrics
        paths: Output file paths
        saved_files: Dictionary of saved file paths

    """

    config: AlignmentConfig
    recording: RecordingData
    detection: DetectionResult
    epochs_raw: EpochData
    epochs_aligned: np.ndarray
    epochs_all_aligned: np.ndarray | None
    alignment: AlignmentResult
    metrics: EpochMetrics
    paths: OutputPaths
    saved_files: dict[str, Path]

    @property
    def n_epochs(self) -> int:
        """Number of aligned epochs."""
        return self.epochs_aligned.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of channels processed."""
        if self.epochs_all_aligned is not None:
            return self.epochs_all_aligned.shape[0]
        return 1

    @property
    def jitter_ms(self) -> float:
        """Alignment jitter in milliseconds."""
        return self.alignment.jitter_ms

    def summary(self) -> str:
        """Generate a text summary of the results."""
        lines = [
            "=" * 50,
            "EPHYALIGN PROCESSING SUMMARY",
            "=" * 50,
            f"Input: {self.config.input_file}",
            f"Channels: {self.n_channels}",
            f"Stimuli detected: {self.detection.n_detected}",
            f"Epochs extracted: {self.n_epochs}",
            f"Alignment jitter: {self.jitter_ms:.3f} ms",
            f"Peak amplitude: {self.metrics.peak_amp_mean:.3f} ± {self.metrics.peak_amp_std:.3f}",
            f"Output directory: {self.paths.root}",
            "=" * 50,
        ]
        return "\n".join(lines)


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
) -> None:
    """Configure logging for the package.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file to write logs to

    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # Set level for our package
    logging.getLogger("ephyalign").setLevel(log_level)


def align_recording(
    input_file: str | Path,
    config: AlignmentConfig | None = None,
    output_dir: str | Path | None = None,
    reference_channel: int | None = None,
    **kwargs: Any,
) -> PipelineResult:
    """Run the complete alignment pipeline on an ABF recording.

    This is the main entry point for processing a recording. It handles
    all steps from loading through output generation.

    Args:
        input_file: Path to ABF file
        config: AlignmentConfig (created with defaults if None)
        output_dir: Optional custom output directory
        reference_channel: Channel for stimulus detection (overrides config)
        **kwargs: Additional config overrides

    Returns:
        PipelineResult with all processing results

    Example:
        >>> result = align_recording("data/experiment.abf")
        >>> print(f"Found {result.n_epochs} epochs with {result.jitter_ms:.2f}ms jitter")

        >>> # With custom config
        >>> config = AlignmentConfig(
        ...     input_file="data/experiment.abf",
        ...     reference_channel=0,
        ... )
        >>> config.epoch.pre_time_s = 1.0
        >>> config.epoch.post_time_s = 5.0
        >>> result = align_recording(config=config)

    """
    # Build or update config
    if config is None:
        config = AlignmentConfig(input_file=input_file, **kwargs)
    elif config.input_file is None:
        config.input_file = Path(input_file)

    if reference_channel is not None:
        config.reference_channel = reference_channel

    # Setup logging
    setup_logging(config.log_level)

    logger.info(f"Starting alignment pipeline for {config.input_file}")

    # Validate input file is set
    if config.input_file is None:
        raise ValueError("Input file must be specified in config or as argument")

    input_path = config.input_file

    # Build output paths
    paths = build_output_paths(
        input_path,
        output_dir=output_dir or config.output.output_dir,
    )

    # Step 1: Load recording
    logger.info("Loading recording...")
    recording = load_recording(input_path, config.channels)

    # Step 2: Detect stimuli
    logger.info("Detecting stimulus artifacts...")
    ref_ch = config.reference_channel
    detection = detect_stim_onsets(
        recording.get_channel_data(ref_ch),
        recording.dt,
        config.detection,
    )

    if detection.n_detected == 0:
        raise ValueError("No stimuli detected. Check detection parameters.")

    # Step 3: Extract epochs
    logger.info("Extracting epochs...")
    epochs_raw = build_epochs(
        recording.data,
        detection.stim_indices,
        recording.dt,
        config.epoch,
    )

    if epochs_raw.n_epochs == 0:
        raise ValueError("No valid epochs extracted. Check epoch windows.")

    # Step 4: Align epochs
    logger.info("Refining alignment...")
    if epochs_raw.is_multichannel:
        # Multi-channel alignment
        epochs_all_aligned, alignment = align_multichannel(
            epochs_raw.epochs,
            recording.dt,
            reference_channel=ref_ch,
            search_ms=config.detection.search_window_ms,
        )
        epochs_aligned = epochs_all_aligned[ref_ch]
    else:
        # Single channel alignment
        alignment = refine_alignment(
            epochs_raw.epochs,
            recording.dt,
            search_ms=config.detection.search_window_ms,
        )
        epochs_aligned = alignment.epochs
        epochs_all_aligned = None

    # Step 5: Compute metrics
    logger.info("Computing response metrics...")
    metrics = compute_epoch_metrics(
        epochs_aligned,
        recording.dt,
        config.metrics,
    )

    # Step 6: Save outputs
    saved_files = {}

    # Save data files
    if any([config.output.save_npz, config.output.save_atf, config.output.save_hdf5]):
        logger.info("Saving output files...")

        # Determine which epochs to save
        epochs_to_save = epochs_all_aligned if epochs_all_aligned is not None else epochs_aligned

        saved_files.update(
            save_all_formats(
                paths,
                epochs_to_save,
                recording.dt,
                alignment.time_axis,
                channel_names=recording.channel_names,
                channel_units=recording.channel_units,
                save_npz_flag=config.output.save_npz,
                save_atf_flag=config.output.save_atf,
                save_hdf5_flag=config.output.save_hdf5,
            )
        )

    # Save plots
    if config.output.save_plots:
        logger.info("Generating plots...")
        y_label = recording.channel_units[ref_ch] if recording.channel_units else "Response"

        plot_paths = plot_all_diagnostics(
            epochs_aligned,
            alignment.time_axis,
            recording.dt,
            paths,
            paths.base_name,
            y_label=y_label,
            config=config.plot,
        )
        saved_files.update({f"plot_{k}": v for k, v in plot_paths.items()})

    # Save stats report
    if config.output.save_stats:
        logger.info("Writing statistics report...")
        detection_info = {
            "n_detected": detection.n_detected,
            "mean_isi_s": detection.mean_isi_s,
            "std_isi_s": detection.std_isi_s,
        }
        recording_info = {
            "file_path": str(config.input_file),
            "sampling_rate": recording.sampling_rate,
            "duration_s": recording.duration_s,
            "n_channels": recording.n_channels,
        }
        saved_files["stats"] = write_stats_report(
            paths.stats,
            config.to_dict(),
            metrics,
            detection_info,
            recording_info,
            jitter_ms=alignment.jitter_ms,
        )

    # Build result
    result = PipelineResult(
        config=config,
        recording=recording,
        detection=detection,
        epochs_raw=epochs_raw,
        epochs_aligned=epochs_aligned,
        epochs_all_aligned=epochs_all_aligned,
        alignment=alignment,
        metrics=metrics,
        paths=paths,
        saved_files=saved_files,
    )

    logger.info("Pipeline complete!")
    if config.verbose:
        print(result.summary())

    return result


def batch_align(
    input_files: list[str | Path],
    config: AlignmentConfig | None = None,
    output_base: str | Path | None = None,
    **kwargs: Any,
) -> list[PipelineResult]:
    """Process multiple ABF files in batch.

    Args:
        input_files: List of ABF file paths
        config: Base configuration (input_file will be updated for each)
        output_base: Base output directory (subdirs created per file)
        **kwargs: Additional config overrides

    Returns:
        List of PipelineResult for each file

    Example:
        >>> from pathlib import Path
        >>> files = list(Path("data").glob("*.abf"))
        >>> results = batch_align(files)
        >>> for r in results:
        ...     print(f"{r.config.input_file.name}: {r.n_epochs} epochs")

    """
    results = []

    for i, file_path in enumerate(input_files):
        logger.info(f"Processing file {i + 1}/{len(input_files)}: {file_path}")

        try:
            # Determine output directory
            if output_base is not None:
                output_dir = Path(output_base) / Path(file_path).stem
            else:
                output_dir = None

            result = align_recording(
                file_path,
                config=config,
                output_dir=output_dir,
                **kwargs,
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            # Continue with other files

    logger.info(f"Batch processing complete: {len(results)}/{len(input_files)} succeeded")

    return results
