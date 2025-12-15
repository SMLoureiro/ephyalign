"""Configuration management for ephyalign.

Provides dataclass-based configuration with support for:
- YAML/TOML configuration files
- Environment variables
- Programmatic configuration
- Sensible defaults for electrophysiology experiments
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Configuration for stimulus artifact detection."""

    # Derivative threshold multiplier (relative to noise SD)
    threshold_multiplier: float = 5.0

    # Minimum interval between detected stimuli (seconds)
    min_interval_s: float = 3.0

    # Search window for artifact refinement (milliseconds)
    search_window_ms: float = 1.0

    # Whether to use absolute value of derivative for detection
    use_absolute_derivative: bool = True


@dataclass
class EpochConfig:
    """Configuration for epoch extraction."""

    # Time before stimulus to include (seconds)
    pre_time_s: float = 0.5

    # Time after stimulus to include (seconds)
    post_time_s: float = 3.0

    # Whether to baseline-subtract each epoch
    baseline_subtract: bool = False

    # Baseline window for subtraction (milliseconds from epoch start)
    baseline_window_ms: float = 10.0


@dataclass
class MetricsConfig:
    """Configuration for response metrics calculation."""

    # Baseline window for metrics (milliseconds)
    baseline_ms: float = 10.0

    # Peak search window after artifact (milliseconds)
    peak_window_ms: float = 50.0

    # Early search window for artifact detection (milliseconds)
    search_ms: float = 5.0

    # Whether responses are expected to be upward (depolarizing)
    upward_responses: bool = True


@dataclass
class OutputConfig:
    """Configuration for output files and formats."""

    # Output directory (None = auto-generate based on input file)
    output_dir: Path | None = None

    # Output formats to generate
    save_npz: bool = True
    save_atf: bool = True
    save_hdf5: bool = True

    # Plotting options
    save_plots: bool = True
    plot_dpi: int = 200
    plot_format: str = "png"

    # Stats report
    save_stats: bool = True

    # Overwrite existing files
    overwrite: bool = False

    def __post_init__(self) -> None:
        """Convert output_dir to Path if provided."""
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


@dataclass
class PlotConfig:
    """Configuration for visualization."""

    # Figure size (width, height in inches)
    figsize: tuple[int, int] = (10, 6)

    # DPI for saved figures
    dpi: int = 200

    # Output format
    format: str = "png"

    # Epoch overlay alpha
    overlay_alpha: float = 0.3

    # Zoom plot range (milliseconds)
    zoom_range_ms: tuple[int, int] = (0, 50)

    # Color scheme
    colormap: str = "viridis"

    # Show average on overlay plots
    show_average_overlay: bool = True

    # Line width for average trace
    average_linewidth: float = 3.0


@dataclass
class AlignmentConfig:
    """Main configuration for the alignment pipeline.

    Combines all sub-configurations and provides file-level settings.

    Example:
        >>> config = AlignmentConfig(
        ...     input_file="data/recording.abf",
        ...     reference_channel=0,
        ... )
        >>> # Or load from file
        >>> config = AlignmentConfig.from_yaml("config.yaml")

    """

    # Input file path (required)
    input_file: str | Path | None = None

    # Reference channel for detection and alignment (0-indexed)
    reference_channel: int = 0

    # Channels to process (None = all channels)
    channels: list[int] | None = None

    # Sub-configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    epoch: EpochConfig = field(default_factory=EpochConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)

    # Logging level
    log_level: str = "INFO"

    # Verbose console output
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        if self.input_file is not None:
            self.input_file = Path(self.input_file)
            if not self.input_file.exists():
                raise FileNotFoundError(f"Input file not found: {self.input_file}")

    @property
    def min_interval_s(self) -> float:
        """Convenience accessor for detection.min_interval_s."""
        return self.detection.min_interval_s

    @property
    def pre_time_s(self) -> float:
        """Convenience accessor for epoch.pre_time_s."""
        return self.epoch.pre_time_s

    @property
    def post_time_s(self) -> float:
        """Convenience accessor for epoch.post_time_s."""
        return self.epoch.post_time_s

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        result = asdict(self)
        # Convert Path objects to strings
        if result["input_file"] is not None:
            result["input_file"] = str(result["input_file"])
        if result["output"]["output_dir"] is not None:
            result["output"]["output_dir"] = str(result["output"]["output_dir"])
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignmentConfig:
        """Create configuration from dictionary."""
        # Extract sub-configurations
        detection = DetectionConfig(**data.pop("detection", {}))
        epoch = EpochConfig(**data.pop("epoch", {}))
        metrics = MetricsConfig(**data.pop("metrics", {}))
        output = OutputConfig(**data.pop("output", {}))
        plot_data = data.pop("plot", {})

        # Handle tuple conversion for figsize and zoom_range_ms
        if "figsize" in plot_data and isinstance(plot_data["figsize"], list):
            plot_data["figsize"] = tuple(plot_data["figsize"])
        if "zoom_range_ms" in plot_data and isinstance(plot_data["zoom_range_ms"], list):
            plot_data["zoom_range_ms"] = tuple(plot_data["zoom_range_ms"])
        plot = PlotConfig(**plot_data)

        return cls(
            detection=detection,
            epoch=epoch,
            metrics=metrics,
            output=output,
            plot=plot,
            **data,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AlignmentConfig:
        """Load configuration from YAML file."""
        try:
            import yaml
        except ImportError as err:
            raise ImportError(
                "PyYAML is required for YAML config files: pip install pyyaml"
            ) from err

        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {path}")
        return cls.from_dict(data or {})

    @classmethod
    def from_toml(cls, path: str | Path) -> AlignmentConfig:
        """Load configuration from TOML file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError as err:
                raise ImportError(
                    "tomli is required for TOML config files on Python < 3.11"
                ) from err

        path = Path(path)
        with path.open("rb") as f:
            data = tomllib.load(f)

        logger.info(f"Loaded configuration from {path}")
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> AlignmentConfig:
        """Load configuration from file, auto-detecting format."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in (".yaml", ".yml"):
            return cls.from_yaml(path)
        elif suffix == ".toml":
            return cls.from_toml(path)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError as err:
            raise ImportError(
                "PyYAML is required for YAML config files: pip install pyyaml"
            ) from err

        path = Path(path)
        with path.open("w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {path}")

    @classmethod
    def from_env(cls, prefix: str = "EPHYALIGN_") -> AlignmentConfig:
        """Create configuration from environment variables.

        Environment variables are expected in the format:
        EPHYALIGN_INPUT_FILE, EPHYALIGN_REFERENCE_CHANNEL, etc.
        """
        data: dict[str, Any] = {}

        env_mappings = {
            "INPUT_FILE": "input_file",
            "REFERENCE_CHANNEL": ("reference_channel", int),
            "LOG_LEVEL": "log_level",
            "VERBOSE": ("verbose", lambda x: x.lower() in ("true", "1", "yes")),
            "DETECTION_THRESHOLD_MULTIPLIER": ("detection.threshold_multiplier", float),
            "DETECTION_MIN_INTERVAL_S": ("detection.min_interval_s", float),
            "EPOCH_PRE_TIME_S": ("epoch.pre_time_s", float),
            "EPOCH_POST_TIME_S": ("epoch.post_time_s", float),
            "OUTPUT_DIR": "output.output_dir",
            "OUTPUT_OVERWRITE": ("output.overwrite", lambda x: x.lower() in ("true", "1", "yes")),
        }

        for env_key, mapping in env_mappings.items():
            env_value = os.environ.get(f"{prefix}{env_key}")
            if env_value is not None:
                if isinstance(mapping, tuple):
                    key, converter = mapping
                    value = converter(env_value)
                else:
                    key = mapping
                    value = env_value

                # Handle nested keys
                if "." in key:
                    parts = key.split(".")
                    d = data
                    for part in parts[:-1]:
                        d = d.setdefault(part, {})
                    d[parts[-1]] = value
                else:
                    data[key] = value

        return cls.from_dict(data) if data else cls()


def get_default_config() -> AlignmentConfig:
    """Get a default configuration instance."""
    return AlignmentConfig()


def generate_example_config(path: str | Path, format: str = "yaml") -> None:
    """Generate an example configuration file with all options documented.

    Args:
        path: Output file path
        format: 'yaml' or 'toml'

    """
    get_default_config()

    if format == "yaml":
        # Check yaml availability for consistency, though we write raw text
        import importlib.util

        if importlib.util.find_spec("yaml") is None:
            raise ImportError("PyYAML required: pip install pyyaml")

        content = """# ephyalign Configuration File
# ============================
# This file contains all available configuration options with their default values.

# Input/Output Settings
# ---------------------
input_file: null  # Path to ABF file (required for processing)
reference_channel: 0  # Channel to use for stimulus detection (0-indexed)
channels: null  # Channels to process (null = all channels)

# Logging
log_level: INFO  # DEBUG, INFO, WARNING, ERROR
verbose: true  # Show progress in console

# Stimulus Detection Settings
detection:
  threshold_multiplier: 5.0  # Derivative threshold (x noise SD)
  min_interval_s: 3.0  # Minimum interval between stimuli
  search_window_ms: 1.0  # Window for artifact refinement
  use_absolute_derivative: true

# Epoch Extraction Settings
epoch:
  pre_time_s: 0.5  # Time before stimulus (seconds)
  post_time_s: 3.0  # Time after stimulus (seconds)
  baseline_subtract: false  # Subtract baseline from each epoch
  baseline_window_ms: 10.0  # Baseline window for subtraction

# Response Metrics Settings
metrics:
  baseline_ms: 10.0  # Baseline window for metrics
  peak_window_ms: 50.0  # Peak search window after artifact
  search_ms: 5.0  # Early search window for artifact
  upward_responses: true  # Expect upward (depolarizing) responses

# Output Settings
output:
  output_dir: null  # Output directory (null = auto-generate)
  save_npz: true  # Save NumPy archive
  save_atf: true  # Save ATF format (Stimfit text)
  save_hdf5: true  # Save HDF5 format (Stimfit binary)
  save_plots: true  # Generate diagnostic plots
  save_stats: true  # Generate statistics report
  overwrite: false  # Overwrite existing files

# Plotting Settings
plot:
  figsize: [10, 6]  # Figure size (inches)
  dpi: 200  # Resolution
  format: png  # Output format
  overlay_alpha: 0.3  # Transparency for overlaid traces
  zoom_range_ms: [0, 50]  # Zoom plot range
  colormap: viridis
  show_average_overlay: true
  average_linewidth: 3.0
"""
        Path(path).write_text(content)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Generated example config at {path}")
