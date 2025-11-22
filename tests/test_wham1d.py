import math
from pathlib import Path

import numpy as np
import pytest

from pywham.wham1d import (
    DEGREES,
    RADIANS,
    Wham1D,
    Wham1DConfig,
    build_config,
    main,
    parse_periodic,
    parse_units,
)
from pywham.structures import HistGroup1D, Histogram1D


@pytest.fixture
def base_config(tmp_path: Path) -> Wham1DConfig:
    return Wham1DConfig(
        hist_min=0.0,
        hist_max=2.0,
        num_bins=2,
        tolerance=1e-6,
        temperature=300.0,
        numpad=0,
        metadata_path=tmp_path / "meta.txt",
        freefile_path=tmp_path / "free.txt",
        periodic=False,
        period=0.0,
        k_B=0.001,
    )


@pytest.fixture
def wham(base_config: Wham1DConfig) -> Wham1D:
    return Wham1D(base_config)


def test_clear_and_coor(wham: Wham1D) -> None:
    wham.histogram = [1.0, 2.0]
    wham.clear_histogram()
    assert wham.histogram == [0.0, 0.0]
    assert wham.calc_coor(0) == pytest.approx(0.5)
    assert wham.calc_coor(1) == pytest.approx(1.5)


def test_calc_bias_periodic(base_config: Wham1DConfig) -> None:
    base_config.periodic = True
    base_config.period = DEGREES
    wham = Wham1D(base_config)
    group = HistGroup1D(
        num_windows=1,
        bias_locations=[170.0],
        spring_constants=[2.0],
        free_energies=[0.0],
        previous_free_energies=[0.0],
        temperatures=[base_config.kT],
        partitions=[1.0],
        histograms=[Histogram1D(0, 0, 0, 0, data=[0.0], cumulative=[0.0])],
    )
    bias = wham.calc_bias(group, 0, 350.0)
    expected_dx = -20.0  # wrapped from 180 difference
    assert bias == pytest.approx(0.5 * expected_dx * expected_dx * 2.0)


def test_hist_alloc_and_get_histval(wham: Wham1D) -> None:
    hist = wham.hist_alloc(0, 1, 2, 2)
    hist.data[0] = 5.0
    assert hist.first == 0
    assert hist.last == 1
    assert wham.get_histval(hist, 0) == 5.0
    assert wham.get_histval(hist, -1) == 0.0
    assert wham.get_histval(hist, 2) == 0.0


def test_make_hist_group(wham: Wham1D) -> None:
    group = wham.make_hist_group(2)
    assert group.num_windows == 2
    assert len(group.histograms) == 2
    assert len(group.spring_constants) == 2


def test_is_metadata_and_numwindows(wham: Wham1D) -> None:
    lines = ["# comment", "", " file 0 0", "data 0 0 1.0 300"]
    assert not wham.is_metadata(lines[0])
    assert not wham.is_metadata(lines[1])
    assert wham.is_metadata(lines[2])
    assert wham.get_numwindows(lines) == 2


def test_read_data_energy_and_range(tmp_path: Path, wham: Wham1D) -> None:
    datafile = tmp_path / "data.dat"
    datafile.write_text("0 0.5 0\n1 1.5 0\n#2 0.2 0\n", encoding="utf-8")
    count = wham.read_data(datafile, have_energy=True)
    assert count == 2
    assert wham.histogram == [1.0, 1.0]
    assert wham._find_range() == (0, 1)


def test_read_data_invalid_columns(tmp_path: Path, wham: Wham1D) -> None:
    datafile = tmp_path / "bad.dat"
    datafile.write_text("0 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError):
        wham.read_data(datafile, have_energy=True)


def _prepare_metadata_files(tmp_path: Path) -> tuple[Path, Path]:
    datafile = tmp_path / "data.dat"
    datafile.write_text("0 0.5 0\n1 1.5 0\n", encoding="utf-8")
    metafile = tmp_path / "meta.txt"
    metafile.write_text(f"{datafile} 0.0 1.0 1.0 300\n", encoding="utf-8")
    return metafile, datafile


def test_read_metadata_success(base_config: Wham1DConfig, tmp_path: Path) -> None:
    meta, _ = _prepare_metadata_files(tmp_path)
    base_config.metadata_path = meta
    wham = Wham1D(base_config)
    group = wham.make_hist_group(1)
    count, have_temp = wham.read_metadata(meta.read_text().splitlines(), group)
    assert count == 1
    assert have_temp is True
    hist = group.histograms[0]
    assert hist.data == [1.0, 1.0]
    assert group.partitions[0] == pytest.approx(2.0)


def test_read_metadata_inconsistent_temperatures(base_config: Wham1DConfig, tmp_path: Path) -> None:
    datafile = tmp_path / "data.dat"
    datafile.write_text("0 0.5\n", encoding="utf-8")
    meta = tmp_path / "meta.txt"
    meta.write_text(f"{datafile} 0.0 1.0\n{datafile} 0.0 1.0 1.0 300\n", encoding="utf-8")
    base_config.metadata_path = meta
    wham = Wham1D(base_config)
    group = wham.make_hist_group(2)
    with pytest.raises(ValueError):
        wham.read_metadata(meta.read_text().splitlines(), group)


def test_save_free_and_convergence(wham: Wham1D) -> None:
    group = wham.make_hist_group(2)
    group.free_energies = [1.0, 2.0]
    wham.save_free(group)
    assert group.previous_free_energies == [1.0, 2.0]
    group.free_energies = [1.0, 2.0]
    assert wham.is_converged(group)
    group.free_energies[1] = 2.5
    assert not wham.is_converged(group)
    assert wham.average_diff(group) == pytest.approx(0.25)


def test_calc_free() -> None:
    config = Wham1DConfig(
        hist_min=0.0,
        hist_max=1.0,
        num_bins=2,
        tolerance=1e-6,
        temperature=1.0,
        numpad=0,
        metadata_path=Path("meta"),
        freefile_path=Path("free"),
        periodic=False,
        period=0.0,
        k_B=1.0,
    )
    wham = Wham1D(config)
    free, min_bin = wham.calc_free([math.exp(-1), 1.0])
    assert min_bin == 1
    assert free[1] == 0.0


def _single_window_group(config: Wham1DConfig) -> tuple[Wham1D, HistGroup1D, list[float]]:
    wham = Wham1D(config)
    hist = Histogram1D(first=0, last=0, num_points=2, num_mc_samples=2, data=[2.0], cumulative=[0.0])
    group = HistGroup1D(
        num_windows=1,
        bias_locations=[0.0],
        spring_constants=[0.0],
        free_energies=[0.0],
        previous_free_energies=[0.0],
        temperatures=[config.kT],
        partitions=[2.0],
        histograms=[hist],
    )
    probabilities = [0.0]
    return wham, group, probabilities


def test_wham_iteration_updates_free_energy(base_config: Wham1DConfig) -> None:
    wham, group, prob = _single_window_group(base_config)
    wham.wham_iteration(group, prob, have_energy=True)
    assert prob[0] == pytest.approx(1.0)
    assert group.free_energies[0] == pytest.approx(0.0)


def test_run_creates_output(base_config: Wham1DConfig, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    meta, _ = _prepare_metadata_files(tmp_path)
    base_config.metadata_path = meta
    base_config.freefile_path = tmp_path / "free.txt"
    wham = Wham1D(base_config)
    wham.run()
    captured = capsys.readouterr().out
    assert "#Number of windows = 1" in captured
    assert base_config.freefile_path.exists()
    content = base_config.freefile_path.read_text(encoding="utf-8")
    assert "#Window" in content


def test_mk_new_hist_and_random_bin(base_config: Wham1DConfig) -> None:
    wham = Wham1D(base_config)
    cumulative = [0.0, 0.5]
    distribution = [0.0, 0.0]
    generator = np.random.default_rng(123)
    wham.mk_new_hist(cumulative, distribution, 2, 4, generator)
    assert sum(distribution) == 4.0
    assert all(val >= 0 for val in distribution)


def test_parse_units_and_periodic_and_build_config(tmp_path: Path) -> None:
    k_B, remaining = parse_units(["units", "real", "extra"])
    assert k_B != 0.0
    assert remaining == ["extra"]
    periodic, period, consumed = parse_periodic("Ppi")
    assert periodic and period == pytest.approx(RADIANS) and consumed == 1
    periodic, period, consumed = parse_periodic("P180")
    assert periodic and period == pytest.approx(180.0)
    args = [
        "units",
        "real",
        "P",
        "0",
        "1",
        "10",
        "0.1",
        "300",
        "0",
        str(tmp_path / "meta"),
        str(tmp_path / "free"),
        "2",
        "5",
    ]
    config = build_config(args)
    assert config.periodic is True
    assert config.period == pytest.approx(DEGREES)
    assert config.num_mc_runs == 2
    assert config.mc_seed == -5


def test_main_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    meta, _ = _prepare_metadata_files(tmp_path)
    args = ["0", "2", "2", "0.1", "300", "0", str(meta), str(tmp_path / "free.txt")]
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    main(args)
    output = capsys.readouterr().out
    assert "#Number of windows = 1" in output
    assert (tmp_path / "free.txt").exists()
