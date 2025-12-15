"""Tests for I/O modules."""

import numpy as np

from ephyalign.io.exporters import save_atf, save_hdf5, save_npz
from ephyalign.io.paths import OutputPaths, build_output_paths


class TestBuildOutputPaths:
    """Tests for output path generation."""

    def test_creates_paths(self, temp_output_dir):
        paths = build_output_paths(
            "test_file.abf",
            output_dir=temp_output_dir,
            create=False,
        )

        assert isinstance(paths, OutputPaths)
        assert paths.base_name == "test_file"

    def test_default_structure(self, tmp_path):
        paths = build_output_paths(
            "recording.abf",
            create=False,
        )

        # Should create aligned/<name>/ structure
        assert "aligned" in str(paths.root) or paths.root.name == "recording"
        assert paths.plots.name == "plots"

    def test_custom_output_dir(self, temp_output_dir):
        paths = build_output_paths(
            "test.abf",
            output_dir=temp_output_dir,
            create=False,
        )

        assert paths.root == temp_output_dir

    def test_creates_directories(self, tmp_path):
        output_dir = tmp_path / "new_dir"

        paths = build_output_paths(
            "test.abf",
            output_dir=output_dir,
            create=True,
        )

        assert paths.root.exists()
        assert paths.plots.exists()

    def test_file_extensions(self, temp_output_dir):
        paths = build_output_paths(
            "test.abf",
            output_dir=temp_output_dir,
            create=False,
        )

        assert paths.npz.suffix == ".npz"
        assert paths.atf.suffix == ".atf"
        assert paths.hdf5.suffix == ".h5"

    def test_get_plot_path(self, temp_output_dir):
        paths = build_output_paths(
            "test.abf",
            output_dir=temp_output_dir,
        )

        plot_path = paths.get_plot_path("overlay", "png")
        assert "test_overlay.png" in str(plot_path)


class TestOutputPathsExists:
    """Tests for OutputPaths.exists() method."""

    def test_exists_none(self, temp_output_dir):
        paths = build_output_paths("test.abf", output_dir=temp_output_dir)

        assert not paths.exists("any")
        assert not paths.exists("npz")

    def test_exists_after_save(self, temp_output_dir, sample_epochs, dt):
        paths = build_output_paths("test.abf", output_dir=temp_output_dir)
        time_axis = np.arange(sample_epochs.shape[1]) * dt

        save_npz(paths.npz, sample_epochs, time_axis)

        assert paths.exists("npz")
        assert paths.exists("any")
        assert not paths.exists("all")


class TestSaveNpz:
    """Tests for NPZ export."""

    def test_saves_file(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.npz"
        time_axis = np.arange(sample_epochs.shape[1]) * dt

        result = save_npz(path, sample_epochs, time_axis)

        assert result.exists()

    def test_contains_data(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.npz"
        time_axis = np.arange(sample_epochs.shape[1]) * dt

        save_npz(path, sample_epochs, time_axis)

        data = np.load(path)
        assert "epochs" in data
        assert "time" in data
        assert data["epochs"].shape == sample_epochs.shape

    def test_includes_metadata(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.npz"
        time_axis = np.arange(sample_epochs.shape[1]) * dt

        save_npz(path, sample_epochs, time_axis, metadata={"dt": dt})

        data = np.load(path)
        assert "dt" in data


class TestSaveAtf:
    """Tests for ATF export."""

    def test_saves_file(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.atf"

        result = save_atf(path, sample_epochs, dt)

        assert result.exists()

    def test_atf_header(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.atf"

        save_atf(path, sample_epochs, dt)

        with open(path) as f:
            first_line = f.readline()
            assert "ATF" in first_line

    def test_correct_columns(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.atf"
        n_epochs = sample_epochs.shape[0]

        save_atf(path, sample_epochs, dt)

        with open(path) as f:
            lines = f.readlines()
            # Header should indicate n_epochs + 1 columns (time + sweeps)
            assert str(n_epochs + 1) in lines[1]

    def test_y_units_in_header(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.atf"

        save_atf(path, sample_epochs, dt, y_units="pA")

        content = path.read_text()
        assert "pA" in content


class TestSaveHdf5:
    """Tests for HDF5 export."""

    def test_saves_file(self, temp_output_dir, sample_epochs, dt):
        path = temp_output_dir / "test.h5"

        result = save_hdf5(path, sample_epochs, dt)

        assert result.exists()

    def test_hdf5_structure(self, temp_output_dir, sample_epochs, dt):
        import tables

        path = temp_output_dir / "test.h5"
        save_hdf5(path, sample_epochs, dt, channel_names=["Ch0"])

        with tables.open_file(str(path), "r") as h5:
            assert hasattr(h5.root, "description")
            assert hasattr(h5.root, "comment")

    def test_multichannel(self, temp_output_dir, dt):
        import tables

        # 3 channels, 5 epochs, 1000 samples
        epochs = np.random.randn(3, 5, 1000)
        path = temp_output_dir / "test_multi.h5"

        save_hdf5(
            path,
            epochs,
            dt,
            channel_names=["Ch0", "Ch1", "Ch2"],
        )

        with tables.open_file(str(path), "r") as h5:
            # Should have description with 3 channels
            desc = h5.root.description[0]
            assert desc["channels"] == 3

    def test_section_count(self, temp_output_dir, sample_epochs, dt):
        import tables

        path = temp_output_dir / "test.h5"
        save_hdf5(path, sample_epochs, dt, channel_names=["Response"])

        with tables.open_file(str(path), "r") as h5:
            # Find the channel group and check section count
            ch_group = h5.root.Response
            ch_desc = ch_group.description[0]
            assert ch_desc["n_sections"] == sample_epochs.shape[0]
