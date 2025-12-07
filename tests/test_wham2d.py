import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from pywham.wham2d import (
    DEGREES,
    MASKED,
    RADIANS,
    Wham2D,
    Wham2DConfig,
    build_config,
    main,
    parse_periodic,
    parse_units,
    _run_bootstrap_trial,
)
from pywham.structures import HistGroup2D, Histogram2D


@pytest.fixture
def basic_config(tmp_path: Path) -> Wham2DConfig:
    return Wham2DConfig(
        hist_min_x=0.0,
        hist_max_x=2.0,
        num_bins_x=2,
        hist_min_y=0.0,
        hist_max_y=2.0,
        num_bins_y=2,
        tolerance=0.1,
        temperature=2.0,
        numpad=0,
        metadata_path=tmp_path / "metadata.txt",
        freefile_path=tmp_path / "free.txt",
        use_mask=False,
        periodic_x=True,
        period_x=10.0,
        periodic_y=True,
        period_y=10.0,
        k_B=1.0,
    )


def test_config_properties_and_clear_histogram(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    assert wham.config.bin_width_x == 1.0
    assert wham.config.bin_width_y == 1.0
    assert wham.config.kT == pytest.approx(2.0)

    wham.histogram[0][0] = 5.0
    wham.clear_histogram()
    assert wham.histogram == [[0.0, 0.0], [0.0, 0.0]]


def test_calc_coor_and_bias_periodic(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    coor = wham.calc_coor(1, 0)
    assert coor == (1.5, 0.5)

    hist_group = wham.make_hist_group(1)
    hist_group.bias_locations[0] = [0.0, 0.0]
    hist_group.spring_x[0] = 2.0
    hist_group.spring_y[0] = 1.0

    bias = wham.calc_bias(hist_group, 0, (9.0, 0.0))
    assert bias == pytest.approx(1.0)  # wrapped dx becomes -1, spring_x=2 -> 0.5*2*1^2

    hist_group.bias_locations[0] = [9.0, 9.0]
    bias_no_wrap = wham.calc_bias(hist_group, 0, (8.0, 8.0))
    assert bias_no_wrap == pytest.approx(1.5)


def test_hist_alloc_and_make_hist_group(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    hist = wham.hist_alloc(0, 1, 0, 1, 5, 3)
    assert hist.data == [[0.0, 0.0], [0.0, 0.0]]
    assert len(hist.cumulative) == 5  # (2*2)+1

    group = wham.make_hist_group(2)
    assert isinstance(group, HistGroup2D)
    assert len(group.bias_locations) == 2
    assert all(value == 1.0 for value in group.free_energies)


def test_metadata_helpers() -> None:
    wham = Wham2D(
        Wham2DConfig(
            hist_min_x=0.0,
            hist_max_x=1.0,
            num_bins_x=1,
            hist_min_y=0.0,
            hist_max_y=1.0,
            num_bins_y=1,
            tolerance=0.1,
            temperature=1.0,
            numpad=0,
            metadata_path=Path("meta"),
            freefile_path=Path("free"),
            use_mask=False,
            periodic_x=False,
            period_x=0.0,
            periodic_y=False,
            period_y=0.0,
            k_B=1.0,
        )
    )

    assert not wham.is_metadata("# comment")
    assert not wham.is_metadata("   \t ")
    assert wham.is_metadata("file 0 0 1 1")

    lines = ["# header", "file 0 0 1 1", "file2 0 0 1 1"]
    assert wham.get_numwindows(lines) == 2


def test_read_data_with_energy_and_mask(tmp_path: Path, basic_config: Wham2DConfig) -> None:
    data_file = tmp_path / "data.txt"
    data_file.write_text("""
# header line
0 0.5 0.5 1.0
1 1.5 1.5 0.0
2 0.2 0.2 0.5
    """.strip())

    wham = Wham2D(basic_config)
    mask = [[0, 0], [0, 0]]
    count = wham.read_data(data_file, have_energy=True, use_mask=True, mask=mask)

    expected_weight = math.exp(-1.0 / wham.config.kT) + math.exp(-0.5 / wham.config.kT)
    assert count == 3
    assert wham.histogram[0][0] == pytest.approx(expected_weight)
    assert wham.histogram[1][1] == pytest.approx(1.0)
    assert mask == [[1, 0], [0, 1]]

    malformed = tmp_path / "malformed.txt"
    malformed.write_text("0 0.1 0.2\n")
    with pytest.raises(ValueError):
        wham.read_data(malformed, have_energy=True, use_mask=False, mask=None)


def test_read_metadata_and_range(tmp_path: Path) -> None:
    data_file = tmp_path / "data.txt"
    data_file.write_text("0 0.5 0.5 0.0\n")
    metadata_line = f"{data_file} 0.5 0.5 1.0 1.0 1.0 1.0"

    config = Wham2DConfig(
        hist_min_x=0.0,
        hist_max_x=1.0,
        num_bins_x=1,
        hist_min_y=0.0,
        hist_max_y=1.0,
        num_bins_y=1,
        tolerance=0.01,
        temperature=1.0,
        numpad=0,
        metadata_path=tmp_path / "metadata.txt",
        freefile_path=tmp_path / "free.txt",
        use_mask=False,
        periodic_x=False,
        period_x=0.0,
        periodic_y=False,
        period_y=0.0,
        k_B=1.0,
    )

    wham = Wham2D(config)
    group = wham.make_hist_group(1)
    count, have_temp, entries = wham.read_metadata([metadata_line], group, False, None)

    assert count == 1
    assert have_temp is True
    assert len(entries) == 1
    assert group.histograms[0].data == [[1.0]]
    assert group.partitions[0] == pytest.approx(1.0)

    mixed_group = wham.make_hist_group(2)
    mixed_lines = [metadata_line, f"{data_file} 0.5 0.5 1 1 1"]
    with pytest.raises(ValueError):
        wham.read_metadata(mixed_lines, mixed_group, False, None)


def test_find_range_and_get_histval(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    wham.histogram[1][0] = 1.0
    wham.histogram[1][1] = 2.0
    wham.histogram[0][1] = 3.0
    assert wham._find_range() == (0, 1, 0, 1)

    hist = Histogram2D(0, 1, 0, 1, 0, 0, data=[[4.0, 5.0], [6.0, 7.0]])
    assert wham.get_histval(hist, 1, 0) == 6.0
    assert wham.get_histval(hist, -1, 0) == 0.0


def test_aux_data_written(tmp_path: Path, basic_config: Wham2DConfig) -> None:
    data_file = tmp_path / "data.dat"
    data_file.write_text("0 0.25 0.75\n0 0.5 0.5\n", encoding="utf-8")
    metadata_path = tmp_path / "metadata.txt"
    metadata_path.write_text(f"{data_file} 0.0 0.0 1.0 1.0\n", encoding="utf-8")
    basic_config.metadata_path = metadata_path
    basic_config.aux_data_path = tmp_path / "aux.yaml"
    wham = Wham2D(basic_config)
    group = wham.make_hist_group(1)
    _, _, entries = wham.read_metadata(metadata_path.read_text().splitlines(), group, False, None)
    map_values = [1.0 for _ in entries]
    mh_samples: list[list[float]] = []
    wham._write_aux_data(entries, group, map_values, mh_samples)
    aux = yaml.safe_load(basic_config.aux_data_path.read_text(encoding="utf-8"))
    assert aux["dim_umbrella"] == 2
    assert aux["windows"][0]["trajectory"] == str(data_file)
    assert aux["output_dir"] == str(basic_config.freefile_path.parent)


def test_save_convergence_and_average_diff(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    group = wham.make_hist_group(2)
    group.free_energies = [2.0, 4.0]
    wham.save_free(group)
    assert group.previous_free_energies == [2.0, 4.0]
    assert group.free_energies == [0.0, 0.0]

    logged_current = [0.5, 0.6]
    logged_previous = [0.45, 0.55]
    assert wham.is_converged(group, logged_current, logged_previous)
    assert wham.average_diff(logged_current, logged_previous) == pytest.approx(0.05)

    logged_current[1] = 0.0
    assert not wham.is_converged(group, logged_current, logged_previous)


def test_calc_free_with_mask(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    prob = [[0.2, 0.3], [0.4, 0.5]]
    mask = [[1, 0], [1, 1]]
    free = wham.calc_free(prob, use_mask=True, mask=mask)

    assert free[0][1] == MASKED
    assert prob[0][1] == 0.0
    assert free[1][1] == pytest.approx(0.0)


def test_wham_iteration_updates_prob_and_free(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    group = wham.make_hist_group(1)
    group.histograms[0] = Histogram2D(0, 0, 0, 0, 5, 5, data=[[5.0]])
    group.free_energies = [1.0]
    group.previous_free_energies = [1.0]
    group.temperatures = [1.0]
    wham.save_free(group)

    prob = np.zeros((2, 2))
    bias_lookup = np.ones((2, 2, 1))
    num_lookup = np.array([[10.0, 0.0], [0.0, 0.0]])

    wham.wham_iteration(group, prob, have_energy=False, use_mask=False, mask=None, bias_lookup=bias_lookup, num_lookup=num_lookup)
    assert prob[0][0] == pytest.approx(2.0)
    assert group.free_energies[0] == pytest.approx(0.5)


def test_wham_iteration_stabilizes_near_zero_denominator(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    group = wham.make_hist_group(1)
    group.histograms[0] = Histogram2D(0, 0, 0, 0, 0, 0, data=[[0.0]])
    group.free_energies = [0.0]
    group.previous_free_energies = [0.0]
    group.temperatures = [1.0]

    prob = np.zeros((2, 2))
    # Extremely small bias values drive the denominator toward zero
    bias_lookup = np.full((2, 2, 1), 1e-300)
    num_lookup = np.zeros((2, 2))

    wham.wham_iteration(group, prob, have_energy=False, use_mask=False, mask=None, bias_lookup=bias_lookup, num_lookup=num_lookup)

    assert all(math.isfinite(value) for row in prob for value in row)
    assert all(value >= 0.0 for row in prob for value in row)
    assert all(math.isfinite(value) and value > 0.0 for value in group.free_energies)


def test_run_writes_freefile(tmp_path: Path) -> None:
    data_file = tmp_path / "data.txt"
    data_file.write_text("0 0.5 0.5 0.0\n")
    metadata_line = f"{data_file} 0.5 0.5 1.0 1.0 1.0 1.0"
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.write_text(metadata_line + "\n")

    config = Wham2DConfig(
        hist_min_x=0.0,
        hist_max_x=1.0,
        num_bins_x=1,
        hist_min_y=0.0,
        hist_max_y=1.0,
        num_bins_y=1,
        tolerance=0.01,
        temperature=1.0,
        numpad=0,
        metadata_path=metadata_file,
        freefile_path=tmp_path / "free.txt",
        use_mask=False,
        periodic_x=False,
        period_x=0.0,
        periodic_y=False,
        period_y=0.0,
        k_B=1.0,
    )

    wham = Wham2D(config)
    wham.run()

    free_contents = config.freefile_path.read_text().strip().splitlines()
    assert free_contents[0].startswith("#X")
    assert len(free_contents) == 2
    fields = free_contents[1].split("\t")
    assert float(fields[0]) == pytest.approx(0.5)
    assert float(fields[1]) == pytest.approx(0.5)


def test_parse_periodic_and_units() -> None:
    config = {"periodic_x": True, "periodic_y": False, "period_x": "pi", "period_y": 0.0}
    periodic_x, period_x = parse_periodic(config, "x")
    periodic_y, period_y = parse_periodic(config, "y")

    assert periodic_x is True
    assert period_x == pytest.approx(RADIANS)
    assert periodic_y is False
    assert period_y == 0.0

    assert pytest.approx(parse_units("real")) == 0.0019872067
    with pytest.raises(ValueError):
        parse_units("units")


def test_bootstrap_handles_sparse_histograms(basic_config: Wham2DConfig) -> None:
    wham = Wham2D(basic_config)
    hist = Histogram2D(0, 0, 0, 0, 0, 0, data=[[0.0]], cumulative=[0.0, 1.0])
    base_group = wham.make_hist_group(1)
    base_group.histograms[0] = hist
    base_group.temperatures = [1.0]

    result = _run_bootstrap_trial(
        trial_index=0,
        config=basic_config,
        base_hist_group=base_group,
        have_energy=False,
        mask=None,
        use_mask=False,
        seed=1,
    )

    assert not result.too_many_iterations
    assert all(math.isfinite(val) and val >= 0.0 for val in result.free_energies)
    assert all(math.isfinite(cell) for row in result.probabilities for cell in row)


def test_build_config_and_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_file = tmp_path / "data.txt"
    data_file.write_text("0 0.5 0.5 0.0\n")
    metadata_file = tmp_path / "metadata.txt"
    metadata_file.write_text(f"{data_file} 0.5 0.5 1.0 1.0 1.0 1.0\n")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hist_min_x": 0.0,
                "hist_max_x": 1.0,
                "num_bins_x": 1,
                "hist_min_y": 0.0,
                "hist_max_y": 1.0,
                "num_bins_y": 1,
                "tolerance": 0.01,
                "temperature": 1.0,
                "numpad": 0,
                "metadata_file": str(metadata_file),
                "freefile": str(tmp_path / "free.txt"),
                "use_mask": False,
                "periodic_x": True,
                "period_x": 0.0,
                "periodic_y": True,
                "period_y": 0.0,
            }
        ),
        encoding="utf-8",
    )

    config = build_config(config_path)
    assert config.periodic_x and config.periodic_y
    assert config.num_bins_x == 1
    assert config.hist_min_x == 0.0

    main([str(config_path)])
    captured = capsys.readouterr()
    assert "#Number of windows = 1" in captured.out
    assert (tmp_path / "free.txt").exists()
