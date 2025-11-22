import math
import sys
from pathlib import Path
import bisect
import random
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback for offline environments
    numpy_stub = types.ModuleType("numpy")

    def searchsorted(a, v, side="left"):
        if side == "right":
            return bisect.bisect_right(a, v)
        return bisect.bisect_left(a, v)

    def isscalar(value):
        return isinstance(value, (int, float))

    class ndarray(list):
        pass

    class _Generator:
        def __init__(self, seed=None):
            self._rng = random.Random(seed)

        def random(self):
            return self._rng.random()

    rng_module = types.ModuleType("numpy.random")
    rng_module.Generator = _Generator

    def default_rng(seed=None):
        return _Generator(seed)

    rng_module.default_rng = default_rng

    numpy_stub.random = rng_module
    numpy_stub.searchsorted = searchsorted
    numpy_stub.isscalar = isscalar
    numpy_stub.ndarray = ndarray
    numpy_stub.bool_ = bool
    sys.modules["numpy"] = numpy_stub
    sys.modules["numpy.random"] = rng_module
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


# Helper to build a simple configuration for tests

def make_config(tmp_path: Path, **overrides):
    base = dict(
        hist_min=0.0,
        hist_max=2.0,
        num_bins=2,
        tolerance=1e-6,
        temperature=1.0,
        numpad=0,
        metadata_path=tmp_path / "metadata.dat",
        freefile_path=tmp_path / "freefile.dat",
        periodic=False,
        period=0.0,
        k_B=0.0019829237,
        num_mc_runs=0,
        mc_seed=None,
    )
    base.update(overrides)
    return Wham1DConfig(**base)


def test_calc_bias_periodic_wrap(tmp_path):
    config = make_config(tmp_path, periodic=True, period=10.0)
    wham = Wham1D(config)
    group = wham.make_hist_group(1)
    group.spring_constants[0] = 2.0
    group.bias_locations[0] = 1.0

    bias = wham.calc_bias(group, 0, 9.0)
    assert math.isclose(bias, 4.0)


def test_hist_alloc_and_get_histval(tmp_path):
    wham = Wham1D(make_config(tmp_path))
    hist = wham.hist_alloc(0, 2, 3, 2)
    hist.data = [1.0, 2.0, 3.0]

    assert wham.get_histval(hist, 0) == 1.0
    assert wham.get_histval(hist, 2) == 3.0
    assert wham.get_histval(hist, 5) == 0.0


def test_is_metadata_and_get_numwindows(tmp_path):
    lines = ["# comment", "", "file1 0.0 1.0", "file2 0.0 1.0"]
    wham = Wham1D(make_config(tmp_path))
    assert wham.get_numwindows(lines) == 2


def test_read_data_counts_and_weights(tmp_path):
    datafile = tmp_path / "data.dat"
    datafile.write_text("# header\n0 0.5 1.0\n1 1.5 2.0\n")
    config = make_config(tmp_path)
    wham = Wham1D(config)

    count = wham.read_data(datafile, have_energy=True)
    assert count == 2
    # weights should reflect exp(-energy/kT)
    expected_weight = math.exp(-1.0 / config.kT)
    assert wham.histogram[0] == pytest.approx(expected_weight)


def test_read_data_missing_energy_raises(tmp_path):
    datafile = tmp_path / "bad.dat"
    datafile.write_text("0 0.5\n")
    wham = Wham1D(make_config(tmp_path))

    with pytest.raises(ValueError):
        wham.read_data(datafile, have_energy=True)


def test_read_metadata_temperature_mismatch(tmp_path):
    good_data = tmp_path / "data.dat"
    good_data.write_text("0 1.0\n")
    config = make_config(tmp_path)
    wham = Wham1D(config)
    group = wham.make_hist_group(2)
    lines = [
        f"{good_data} 0.0 1.0 1.0 300.0",
        f"{good_data} 0.0 1.0",
    ]

    with pytest.raises(ValueError):
        wham.read_metadata(lines, group)


def test_read_metadata_range_error(tmp_path):
    outside = tmp_path / "outside.dat"
    outside.write_text("0 -1.0\n1 -1.0\n")
    config = make_config(tmp_path)
    wham = Wham1D(config)
    group = wham.make_hist_group(1)

    with pytest.raises(ValueError):
        wham.read_metadata([f"{outside} 0.0 1.0"], group)


def test_save_free_and_convergence_checks(tmp_path):
    wham = Wham1D(make_config(tmp_path))
    group = wham.make_hist_group(2)
    group.free_energies = [1.0, 3.0]

    wham.save_free(group)
    assert group.previous_free_energies == [1.0, 3.0]
    assert not wham.is_converged(group)
    assert wham.average_diff(group) == pytest.approx(2.0)


def test_calc_free(tmp_path):
    config = make_config(tmp_path)
    wham = Wham1D(config)
    free, min_bin = wham.calc_free([0.25, 0.75])
    assert min_bin == 1
    assert free[1] == 0.0
    assert free[0] == pytest.approx(config.kT * math.log(3))


def test_wham_iteration_updates_probabilities(tmp_path):
    config = make_config(tmp_path)
    wham = Wham1D(config)
    group = wham.make_hist_group(1)
    hist = wham.hist_alloc(0, 1, 4, 4)
    hist.data = [2.0, 2.0]
    hist.num_points = 4
    group.histograms[0] = hist
    group.partitions[0] = hist.num_points
    group.temperatures[0] = 1.0

    probabilities = [0.0, 0.0]
    wham.wham_iteration(group, probabilities, have_energy=False)
    assert probabilities == pytest.approx([0.5, 0.5])
    assert group.free_energies[0] == pytest.approx(0.0)


def test_mk_new_hist_and_get_rand_bin(tmp_path):
    wham = Wham1D(make_config(tmp_path))
    cumulative = [0.0, 0.5]
    distribution = [5.0, 5.0]
    generator = np.random.default_rng(0)

    wham.mk_new_hist(cumulative, distribution, num_bins=2, num_points=4, generator=generator)
    assert sum(distribution) == 4.0
    assert all(value >= 0 for value in distribution)
    assert any(value != 0 for value in distribution)


def test_parse_units_and_periodic():
    kb, remaining = parse_units(["units", "real", "rest"])
    assert kb != 0.0
    assert remaining == ["rest"]

    periodic, period, consumed = parse_periodic("Ppi")
    assert periodic and math.isclose(period, RADIANS)
    assert consumed == 1

    periodic, period, consumed = parse_periodic("P180")
    assert periodic and math.isclose(period, 180.0)
    assert consumed == 1

    periodic, _, consumed = parse_periodic("none")
    assert not periodic and consumed == 0


def test_build_config_and_main(tmp_path, monkeypatch):
    metadata = tmp_path / "meta.dat"
    freefile = tmp_path / "free.dat"
    args = ["P", "0", "2", "2", "0.1", "1.0", "0", str(metadata), str(freefile)]

    config = build_config(args)
    assert config.periodic is True
    assert config.num_bins == 2

    called = {}

    def fake_run(self):
        called["ran"] = True

    monkeypatch.setattr(Wham1D, "run", fake_run)
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(sys, "argv", ["wham"] + args)

    main(args)
    assert called["ran"]


def test_run_creates_output_file(tmp_path, capsys):
    # Build minimal metadata and data files
    data1 = tmp_path / "window1.dat"
    data2 = tmp_path / "window2.dat"
    data1.write_text("0 0.5\n1 0.6\n")
    data2.write_text("0 1.4\n1 1.5\n")

    metadata = tmp_path / "metadata.dat"
    metadata.write_text(f"{data1} 0.0 1.0\n{data2} 1.0 1.0\n")

    config = make_config(tmp_path, metadata_path=metadata, freefile_path=tmp_path / "freefile.dat")
    wham = Wham1D(config)

    wham.run()

    captured = capsys.readouterr().out
    assert "#Number of windows" in captured
    assert config.freefile_path.exists()
    content = config.freefile_path.read_text()
    assert "#Coor" in content
    assert "#Window" in content


def test_build_config_with_seed(tmp_path):
    metadata = tmp_path / "meta.dat"
    freefile = tmp_path / "free.dat"
    args = ["0", "2", "2", "0.1", "1.0", "0", str(metadata), str(freefile), "5", "10"]
    config = build_config(args)
    assert config.num_mc_runs == 5
    assert config.mc_seed == -10
