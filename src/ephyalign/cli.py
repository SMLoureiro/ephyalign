"""
Command-line interface for ephyalign.

Provides a professional CLI for processing ABF recordings from the terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import click

from ephyalign import __version__


@click.group()
@click.version_option(version=__version__, prog_name="ephyalign")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """
    ephyalign - Electrophysiology Response Alignment Toolkit
    
    Automatically detect and align stimulus-evoked responses in ABF recordings
    with sub-millisecond precision.
    
    Examples:
    
    \b
        # Process a single file
        ephyalign process data/recording.abf
    
    \b
        # Process with custom parameters
        ephyalign process data/recording.abf --pre-time 1.0 --post-time 5.0
    
    \b
        # Process multiple files
        ephyalign batch data/*.abf --output-dir results/
    
    \b
        # Generate example config file
        ephyalign init-config my_config.yaml
    
    \b
        # Get info about an ABF file
        ephyalign info data/recording.abf
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug
    
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)


@cli.command("process")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), help="Output directory")
@click.option("-c", "--config", type=click.Path(exists=True), help="Config file (YAML)")
@click.option("--channel", "-ch", type=int, default=0, help="Reference channel (0-indexed)")
@click.option("--pre-time", type=float, help="Pre-stimulus time (seconds)")
@click.option("--post-time", type=float, help="Post-stimulus time (seconds)")
@click.option("--min-interval", type=float, help="Minimum inter-stimulus interval (seconds)")
@click.option("--threshold", type=float, help="Detection threshold multiplier")
@click.option("--no-plots", is_flag=True, help="Skip plot generation")
@click.option("--no-hdf5", is_flag=True, help="Skip HDF5 output")
@click.option("--no-atf", is_flag=True, help="Skip ATF output")
@click.pass_context
def process_command(
    ctx: click.Context,
    input_file: str,
    output_dir: Optional[str],
    config: Optional[str],
    channel: int,
    pre_time: Optional[float],
    post_time: Optional[float],
    min_interval: Optional[float],
    threshold: Optional[float],
    no_plots: bool,
    no_hdf5: bool,
    no_atf: bool,
) -> None:
    """
    Process a single ABF file.
    
    Detects stimulus artifacts, extracts and aligns epochs, computes metrics,
    and saves results in multiple formats.
    
    Example:
    
    \b
        ephyalign process data/recording.abf
        ephyalign process data/recording.abf -o results/ --pre-time 0.5
    """
    from ephyalign.config import AlignmentConfig
    from ephyalign.pipeline import align_recording
    
    click.echo(f"Processing: {input_file}")
    
    # Load config from file or create new
    if config:
        cfg = AlignmentConfig.from_file(config)
        cfg.input_file = Path(input_file)
    else:
        cfg = AlignmentConfig(input_file=input_file)
    
    # Apply CLI overrides
    cfg.reference_channel = channel
    cfg.verbose = ctx.obj.get("verbose", False)
    
    if ctx.obj.get("debug"):
        cfg.log_level = "DEBUG"
    
    if pre_time is not None:
        cfg.epoch.pre_time_s = pre_time
    if post_time is not None:
        cfg.epoch.post_time_s = post_time
    if min_interval is not None:
        cfg.detection.min_interval_s = min_interval
    if threshold is not None:
        cfg.detection.threshold_multiplier = threshold
    
    cfg.output.save_plots = not no_plots
    cfg.output.save_hdf5 = not no_hdf5
    cfg.output.save_atf = not no_atf
    
    try:
        result = align_recording(input_file, config=cfg, output_dir=output_dir)
        
        click.echo()
        click.echo(click.style("✓ Processing complete!", fg="green", bold=True))
        click.echo(f"  Detected stimuli: {result.detection.n_detected}")
        click.echo(f"  Aligned epochs: {result.n_epochs}")
        click.echo(f"  Alignment jitter: {result.jitter_ms:.3f} ms")
        click.echo(f"  Peak amplitude: {result.metrics.peak_amp_mean:.3f} ± {result.metrics.peak_amp_std:.3f}")
        click.echo(f"  Output: {result.paths.root}")
        
    except Exception as e:
        click.echo(click.style(f"✗ Error: {e}", fg="red", bold=True), err=True)
        if ctx.obj.get("debug"):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command("batch")
@click.argument("input_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output-dir", type=click.Path(), help="Base output directory")
@click.option("-c", "--config", type=click.Path(exists=True), help="Config file (YAML)")
@click.option("--channel", "-ch", type=int, default=0, help="Reference channel (0-indexed)")
@click.option("--continue-on-error", is_flag=True, help="Continue if a file fails")
@click.pass_context
def batch_command(
    ctx: click.Context,
    input_files: tuple,
    output_dir: Optional[str],
    config: Optional[str],
    channel: int,
    continue_on_error: bool,
) -> None:
    """
    Process multiple ABF files in batch.
    
    Example:
    
    \b
        ephyalign batch data/*.abf
        ephyalign batch data/*.abf -o results/
    """
    from ephyalign.config import AlignmentConfig
    from ephyalign.pipeline import align_recording
    
    click.echo(f"Processing {len(input_files)} files...")
    
    # Load base config
    if config:
        base_cfg = AlignmentConfig.from_file(config)
    else:
        base_cfg = AlignmentConfig()
    
    base_cfg.reference_channel = channel
    base_cfg.verbose = False  # Less noise in batch mode
    
    if ctx.obj.get("debug"):
        base_cfg.log_level = "DEBUG"
    
    succeeded = 0
    failed = 0
    
    with click.progressbar(input_files, label="Processing") as files:
        for file_path in files:
            try:
                cfg = AlignmentConfig.from_dict(base_cfg.to_dict())
                cfg.input_file = Path(file_path)
                
                if output_dir:
                    out = Path(output_dir) / Path(file_path).stem
                else:
                    out = None
                
                align_recording(file_path, config=cfg, output_dir=out)
                succeeded += 1
                
            except Exception as e:
                failed += 1
                if not continue_on_error:
                    click.echo(click.style(f"\n✗ Error on {file_path}: {e}", fg="red"), err=True)
                    sys.exit(1)
    
    click.echo()
    click.echo(click.style(f"✓ Complete: {succeeded} succeeded, {failed} failed", 
                           fg="green" if failed == 0 else "yellow", bold=True))


@cli.command("info")
@click.argument("input_file", type=click.Path(exists=True))
def info_command(input_file: str) -> None:
    """
    Display information about an ABF file.
    
    Example:
    
    \b
        ephyalign info data/recording.abf
    """
    from ephyalign.core.loader import get_file_info
    
    try:
        info = get_file_info(input_file)
        
        click.echo(click.style("ABF File Information", bold=True))
        click.echo("-" * 40)
        click.echo(f"File: {info['file_name']}")
        click.echo(f"Path: {info['file_path']}")
        click.echo(f"Sampling rate: {info['sampling_rate_hz']} Hz")
        click.echo(f"Duration: {info['total_duration_s']:.2f} s")
        click.echo(f"Sweeps: {info['sweep_count']}")
        click.echo(f"Channels: {info['channel_count']}")
        
        if info['channel_names']:
            click.echo("\nChannel details:")
            for i, (name, unit) in enumerate(zip(info['channel_names'], info['channel_units'])):
                click.echo(f"  {i}: {name} ({unit})")
        
        if info['recording_datetime']:
            click.echo(f"\nRecorded: {info['recording_datetime']}")
        
        click.echo(f"ABF version: {info['abf_version']}")
        
    except Exception as e:
        click.echo(click.style(f"Error reading file: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command("init-config")
@click.argument("output_file", type=click.Path())
@click.option("--format", "-f", type=click.Choice(["yaml"]), default="yaml", 
              help="Config file format")
def init_config_command(output_file: str, format: str) -> None:
    """
    Generate an example configuration file.
    
    Creates a well-documented config file with all available options.
    
    Example:
    
    \b
        ephyalign init-config my_config.yaml
    """
    from ephyalign.config import generate_example_config
    
    try:
        generate_example_config(output_file, format=format)
        click.echo(click.style(f"✓ Created config file: {output_file}", fg="green"))
        click.echo("  Edit this file and use with: ephyalign process -c {output_file} <input.abf>")
    except Exception as e:
        click.echo(click.style(f"Error: {e}", fg="red"), err=True)
        sys.exit(1)


@cli.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
def validate_command(config_file: str) -> None:
    """
    Validate a configuration file.
    
    Example:
    
    \b
        ephyalign validate my_config.yaml
    """
    from ephyalign.config import AlignmentConfig
    
    try:
        cfg = AlignmentConfig.from_file(config_file)
        click.echo(click.style("✓ Configuration is valid", fg="green"))
        click.echo(f"  Reference channel: {cfg.reference_channel}")
        click.echo(f"  Pre-stim window: {cfg.epoch.pre_time_s}s")
        click.echo(f"  Post-stim window: {cfg.epoch.post_time_s}s")
        click.echo(f"  Min interval: {cfg.detection.min_interval_s}s")
    except Exception as e:
        click.echo(click.style(f"✗ Invalid configuration: {e}", fg="red"), err=True)
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
