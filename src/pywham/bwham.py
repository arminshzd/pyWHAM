"""Bayesian WHAM driver built on the WHAM translations."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import yaml

from .structures import HistGroup1D, HistGroup2D, Histogram1D, Histogram2D
from .wham1d import Wham1D, Wham1DConfig, parse_periodic as parse_periodic_1d, parse_units
from .wham2d import Wham2D, Wham2DConfig, parse_periodic as parse_periodic_2d


@dataclass
class BayesWhamConfig:
    """Configuration for running BayesWHAM."""

    dimension: int
    base_config: Wham1DConfig | Wham2DConfig
    num_samples: int = 200
    burn_in: int = 50
    thinning: int = 1
    dirichlet_alpha: float = 1.0


@dataclass
class BayesResult1D:
    coordinate: List[float]
    map_free_energy: List[float]
    map_probabilities: List[float]
    mean_free_energy: List[float]
    std_free_energy: List[float]
    mean_probabilities: List[float]
    std_probabilities: List[float]
    map_window_free: List[float]
    mean_window_free: List[float]
    std_window_free: List[float]


@dataclass
class BayesResult2D:
    coordinate_x: List[List[float]]
    coordinate_y: List[List[float]]
    map_free_energy: List[List[float]]
    map_probabilities: List[List[float]]
    mean_free_energy: List[List[float]]
    std_free_energy: List[List[float]]
    mean_probabilities: List[List[float]]
    std_probabilities: List[List[float]]
    map_window_free: List[float]
    mean_window_free: List[float]
    std_window_free: List[float]


class BayesWHAM:
    """Lightweight Bayesian wrapper over the WHAM translations.

    The implementation follows the workflow in the reference BayesWHAM script
    by drawing Dirichlet-resampled histograms to approximate the posterior over
    the free energy surface. We reuse the ingestion and iteration logic from the
    WHAM translators to preserve trajectory handling and outputs.
    """

    def __init__(self, config: BayesWhamConfig):
        self.config = config

    def _posterior_samples(self) -> int:
        usable = max(self.config.num_samples - self.config.burn_in, 0)
        if usable == 0:
            return 0
        return (usable + self.config.thinning - 1) // self.config.thinning

    # --- 1D helpers -----------------------------------------------------
    def _resample_hist1d(self, hist: Histogram1D, rng: np.random.Generator) -> Histogram1D:
        size = hist.last - hist.first + 1
        alpha = np.asarray(hist.data, dtype=float) + float(self.config.dirichlet_alpha)
        weights = rng.dirichlet(alpha, size=1)[0]
        counts = weights * float(hist.num_points)
        return Histogram1D(
            first=hist.first,
            last=hist.last,
            num_points=hist.num_points,
            num_mc_samples=hist.num_mc_samples,
            data=counts.tolist(),
            cumulative=[0.0 for _ in range(size)],
        )

    def _clone_group_1d(self, base: HistGroup1D) -> HistGroup1D:
        return HistGroup1D(
            num_windows=base.num_windows,
            bias_locations=list(base.bias_locations),
            spring_constants=list(base.spring_constants),
            free_energies=list(base.free_energies),
            previous_free_energies=list(base.previous_free_energies),
            temperatures=list(base.temperatures),
            partitions=list(base.partitions),
            histograms=[Histogram1D(0, 0, 0, 0) for _ in base.histograms],
        )

    def _load_1d(self, wham: Wham1D) -> tuple[HistGroup1D, bool, list]:
        lines = wham.config.metadata_path.read_text(encoding="utf-8").splitlines()
        num_windows = wham.get_numwindows(lines)
        print(f"#Number of windows = {num_windows}")
        hist_group = wham.make_hist_group(num_windows)
        count_windows, have_temp, entries = wham.read_metadata(lines, hist_group)
        assert count_windows == hist_group.num_windows
        if not have_temp:
            for i in range(hist_group.num_windows):
                hist_group.temperatures[i] = wham.config.kT
        return hist_group, have_temp, entries

    def _run_wham_1d(self, wham: Wham1D, hist_group: HistGroup1D, have_energy: bool) -> tuple[List[float], List[float]]:
        probabilities = [0.0 for _ in range(wham.config.num_bins)]
        iteration = 0
        first = True
        while not wham.is_converged(hist_group) or first:
            first = False
            wham.save_free(hist_group)
            wham.wham_iteration(hist_group, probabilities, have_energy)
            iteration += 1
            if iteration >= 100000:
                print(f"Too many iterations: {iteration}")
                break
        total = sum(probabilities)
        if total:
            probabilities = [p / total for p in probabilities]
        free_energy, _ = wham.calc_free(probabilities)
        return free_energy, probabilities

    def _bayes_samples_1d(
        self,
        wham: Wham1D,
        base_group: HistGroup1D,
        have_energy: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_count = self._posterior_samples()
        if sample_count == 0:
            return np.empty((0, wham.config.num_bins)), np.empty((0, wham.config.num_bins)), np.empty((0, base_group.num_windows))
        rng = np.random.default_rng()
        free_samples = np.zeros((sample_count, wham.config.num_bins), dtype=float)
        prob_samples = np.zeros_like(free_samples)
        window_samples = np.zeros((sample_count, base_group.num_windows), dtype=float)

        base_histograms = list(base_group.histograms)
        idx = 0
        for draw in range(self.config.num_samples):
            resampled = self._clone_group_1d(base_group)
            resampled.histograms = [self._resample_hist1d(hist, rng) for hist in base_histograms]
            free_energy, probabilities = self._run_wham_1d(wham, resampled, have_energy)
            if draw < self.config.burn_in or (draw - self.config.burn_in) % self.config.thinning != 0:
                continue
            free_samples[idx, :] = np.asarray(free_energy, dtype=float)
            prob_samples[idx, :] = np.asarray(probabilities, dtype=float)
            window_samples[idx, :] = np.asarray(resampled.free_energies, dtype=float)
            idx += 1
        return free_samples[:idx, :], prob_samples[:idx, :], window_samples[:idx, :]

    def _format_output_1d(self, wham: Wham1D, result: BayesResult1D) -> None:
        with wham.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write("#Coor\tFree\tProb\tMeanFree\tStdFree\tMeanProb\tStdProb\n")
            for coor, f_map, p_map, f_mean, f_std, p_mean, p_std in zip(
                result.coordinate,
                result.map_free_energy,
                result.map_probabilities,
                result.mean_free_energy,
                result.std_free_energy,
                result.mean_probabilities,
                result.std_probabilities,
            ):
                freefile.write(
                    f"{coor:.6f}\t{f_map:.6f}\t{p_map:.6e}\t{f_mean:.6f}\t{f_std:.6f}\t{p_mean:.6e}\t{p_std:.6e}\n"
                )
            freefile.write("\n#Window\tFree (free energy units)\tMean\tStd\n")
            for idx, (f_map, f_mean, f_std) in enumerate(
                zip(result.map_window_free, result.mean_window_free, result.std_window_free)
            ):
                freefile.write(f"#{idx}\t{f_map:.6f}\t{f_mean:.6f}\t{f_std:.6f}\n")

    # --- 2D helpers -----------------------------------------------------
    def _resample_hist2d(self, hist: Histogram2D, rng: np.random.Generator) -> Histogram2D:
        size_x = hist.last_x - hist.first_x + 1
        size_y = hist.last_y - hist.first_y + 1
        flat = np.asarray(hist.data, dtype=float).reshape(size_x * size_y)
        alpha = flat + float(self.config.dirichlet_alpha)
        weights = rng.dirichlet(alpha, size=1)[0]
        counts = (weights * float(hist.num_points)).reshape((size_x, size_y))
        return Histogram2D(
            first_x=hist.first_x,
            last_x=hist.last_x,
            first_y=hist.first_y,
            last_y=hist.last_y,
            num_points=hist.num_points,
            num_mc_samples=hist.num_mc_samples,
            data=counts.tolist(),
            cumulative=[0.0 for _ in range(size_x * size_y)],
        )

    def _clone_group_2d(self, base: HistGroup2D) -> HistGroup2D:
        return HistGroup2D(
            num_windows=base.num_windows,
            bias_locations=[list(row) for row in base.bias_locations],
            spring_x=list(base.spring_x),
            spring_y=list(base.spring_y),
            free_energies=list(base.free_energies),
            previous_free_energies=list(base.previous_free_energies),
            temperatures=list(base.temperatures),
            partitions=list(base.partitions),
            histograms=[Histogram2D(0, 0, 0, 0, 0, 0) for _ in base.histograms],
        )

    def _load_2d(self, wham: Wham2D):
        lines = wham.config.metadata_path.read_text(encoding="utf-8").splitlines()
        num_windows = wham.get_numwindows(lines)
        print(f"#Number of windows = {num_windows}")

        mask = None
        if wham.config.use_mask:
            mask = [[0 for _ in range(wham.config.num_bins_y)] for _ in range(wham.config.num_bins_x)]

        hist_group = wham.make_hist_group(num_windows)
        count_windows, have_temp, entries = wham.read_metadata(lines, hist_group, wham.config.use_mask, mask)
        assert count_windows == hist_group.num_windows
        if not have_temp:
            for i in range(hist_group.num_windows):
                hist_group.temperatures[i] = wham.config.kT
        for i in range(hist_group.num_windows):
            hist_group.free_energies[i] = 1.0
            hist_group.previous_free_energies[i] = 1.0
        return hist_group, have_temp, entries, mask

    def _run_wham_2d(
        self, wham: Wham2D, hist_group: HistGroup2D, have_energy: bool, mask: list[list[int]] | None
    ) -> tuple[List[List[float]], List[List[float]]]:
        dtype = np.float32 if wham.config.use_float32 else np.float64
        x_grid, y_grid = wham._coordinate_grids(dtype)

        num_lookup = np.zeros((wham.config.num_bins_x, wham.config.num_bins_y), dtype=dtype)
        for hist in hist_group.histograms:
            if hist.num_points == 0:
                continue
            x_slice = slice(hist.first_x, hist.last_x + 1)
            y_slice = slice(hist.first_y, hist.last_y + 1)
            num_lookup[x_slice, y_slice] += np.asarray(hist.data, dtype=dtype)

        bias_lookup = wham._build_bias_lookup(hist_group, x_grid, y_grid, dtype)

        prob = np.zeros((wham.config.num_bins_x, wham.config.num_bins_y), dtype=dtype)
        iteration = 0
        first = True
        converged = False
        while not converged or first:
            first = False
            wham.save_free(hist_group)
            wham.wham_iteration(hist_group, prob, have_energy, wham.config.use_mask, mask, bias_lookup, num_lookup)
            iteration += 1

            epsilon = float(np.finfo(dtype).tiny)
            hist_group.free_energies = [max(val, epsilon) for val in hist_group.free_energies]
            hist_group.previous_free_energies = [max(val, epsilon) for val in hist_group.previous_free_energies]
            logged_current = [
                hist_group.temperatures[i] * math.log(hist_group.free_energies[i]) for i in range(hist_group.num_windows)
            ]
            logged_previous = [
                hist_group.temperatures[i] * math.log(hist_group.previous_free_energies[i])
                for i in range(hist_group.num_windows)
            ]
            converged = wham.is_converged(hist_group, logged_current, logged_previous)
            if iteration >= 100000:
                print(f"Too many iterations: {iteration}")
                break

        free_energy = np.asarray(wham.calc_free(prob.tolist(), wham.config.use_mask, mask), dtype=prob.dtype)
        total = float(np.sum(prob))
        if total > 0:
            prob /= total
        return free_energy.tolist(), prob.tolist()

    def _bayes_samples_2d(
        self, wham: Wham2D, base_group: HistGroup2D, have_energy: bool, mask: list[list[int]] | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sample_count = self._posterior_samples()
        if sample_count == 0:
            return np.empty((0, wham.config.num_bins_x, wham.config.num_bins_y)), np.empty(
                (0, wham.config.num_bins_x, wham.config.num_bins_y)
            ), np.empty((0, base_group.num_windows))
        rng = np.random.default_rng()
        free_samples = np.zeros((sample_count, wham.config.num_bins_x, wham.config.num_bins_y), dtype=float)
        prob_samples = np.zeros_like(free_samples)
        window_samples = np.zeros((sample_count, base_group.num_windows), dtype=float)

        base_histograms = list(base_group.histograms)
        idx = 0
        for draw in range(self.config.num_samples):
            resampled = self._clone_group_2d(base_group)
            resampled.histograms = [self._resample_hist2d(hist, rng) for hist in base_histograms]
            free_energy, probabilities = self._run_wham_2d(wham, resampled, have_energy, mask)
            if draw < self.config.burn_in or (draw - self.config.burn_in) % self.config.thinning != 0:
                continue
            free_samples[idx, :, :] = np.asarray(free_energy, dtype=float)
            prob_samples[idx, :, :] = np.asarray(probabilities, dtype=float)
            window_samples[idx, :] = np.asarray(resampled.free_energies, dtype=float)
            idx += 1
        return free_samples[:idx, :, :], prob_samples[:idx, :, :], window_samples[:idx, :]

    def _format_output_2d(self, wham: Wham2D, result: BayesResult2D) -> None:
        with wham.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write("#X\tY\tFree\tProb\tMeanFree\tStdFree\tMeanProb\tStdProb\n")
            for i in range(wham.config.num_bins_x):
                for j in range(wham.config.num_bins_y):
                    freefile.write(
                        f"{result.coordinate_x[i][j]:.6f}\t{result.coordinate_y[i][j]:.6f}\t{result.map_free_energy[i][j]:.6f}\t"
                        f"{result.map_probabilities[i][j]:.6e}\t{result.mean_free_energy[i][j]:.6f}\t{result.std_free_energy[i][j]:.6f}\t"
                        f"{result.mean_probabilities[i][j]:.6e}\t{result.std_probabilities[i][j]:.6e}\n"
                    )
            freefile.write("\n#Window\tFree (free energy units)\tMean\tStd\n")
            for idx, (f_map, f_mean, f_std) in enumerate(
                zip(result.map_window_free, result.mean_window_free, result.std_window_free)
            ):
                freefile.write(f"#{idx}\t{f_map:.6f}\t{f_mean:.6f}\t{f_std:.6f}\n")

    # --- User facing ----------------------------------------------------
    def run(self) -> None:
        if self.config.dimension == 1:
            wham = Wham1D(self.config.base_config)  # type: ignore[arg-type]
            hist_group, have_energy, _ = self._load_1d(wham)
            map_free, map_prob = self._run_wham_1d(wham, hist_group, have_energy)
            samples_free, samples_prob, window_samples = self._bayes_samples_1d(wham, hist_group, have_energy)
            mean_free = samples_free.mean(axis=0) if samples_free.size else np.asarray(map_free, dtype=float)
            std_free = samples_free.std(axis=0) if samples_free.size else np.zeros_like(map_free, dtype=float)
            mean_prob = samples_prob.mean(axis=0) if samples_prob.size else np.asarray(map_prob, dtype=float)
            std_prob = samples_prob.std(axis=0) if samples_prob.size else np.zeros_like(map_prob, dtype=float)
            mean_window = window_samples.mean(axis=0) if window_samples.size else np.asarray(hist_group.free_energies)
            std_window = window_samples.std(axis=0) if window_samples.size else np.zeros_like(hist_group.free_energies)

            coordinates = [wham.calc_coor(i) for i in range(wham.config.num_bins)]
            result = BayesResult1D(
                coordinate=coordinates,
                map_free_energy=map_free,
                map_probabilities=map_prob,
                mean_free_energy=mean_free.tolist(),
                std_free_energy=std_free.tolist(),
                mean_probabilities=mean_prob.tolist(),
                std_probabilities=std_prob.tolist(),
                map_window_free=list(hist_group.free_energies),
                mean_window_free=mean_window.tolist(),
                std_window_free=std_window.tolist(),
            )
            self._format_output_1d(wham, result)
            return

        wham2d = Wham2D(self.config.base_config)  # type: ignore[arg-type]
        hist_group2d, have_energy2d, _, mask = self._load_2d(wham2d)
        map_free_2d, map_prob_2d = self._run_wham_2d(wham2d, hist_group2d, have_energy2d, mask)
        samples_free_2d, samples_prob_2d, window_samples_2d = self._bayes_samples_2d(
            wham2d, hist_group2d, have_energy2d, mask
        )

        mean_free_2d = samples_free_2d.mean(axis=0) if samples_free_2d.size else np.asarray(map_free_2d, dtype=float)
        std_free_2d = samples_free_2d.std(axis=0) if samples_free_2d.size else np.zeros_like(map_free_2d, dtype=float)
        mean_prob_2d = samples_prob_2d.mean(axis=0) if samples_prob_2d.size else np.asarray(map_prob_2d, dtype=float)
        std_prob_2d = samples_prob_2d.std(axis=0) if samples_prob_2d.size else np.zeros_like(map_prob_2d, dtype=float)
        mean_window_2d = (
            window_samples_2d.mean(axis=0) if window_samples_2d.size else np.asarray(hist_group2d.free_energies, dtype=float)
        )
        std_window_2d = window_samples_2d.std(axis=0) if window_samples_2d.size else np.zeros_like(hist_group2d.free_energies)

        x_coords = []
        y_coords = []
        for i in range(wham2d.config.num_bins_x):
            row_x = []
            row_y = []
            for j in range(wham2d.config.num_bins_y):
                coor = wham2d.calc_coor(i, j)
                row_x.append(coor[0])
                row_y.append(coor[1])
            x_coords.append(row_x)
            y_coords.append(row_y)

        result2d = BayesResult2D(
            coordinate_x=x_coords,
            coordinate_y=y_coords,
            map_free_energy=map_free_2d,
            map_probabilities=map_prob_2d,
            mean_free_energy=mean_free_2d.tolist(),
            std_free_energy=std_free_2d.tolist(),
            mean_probabilities=mean_prob_2d.tolist(),
            std_probabilities=std_prob_2d.tolist(),
            map_window_free=list(hist_group2d.free_energies),
            mean_window_free=mean_window_2d.tolist(),
            std_window_free=std_window_2d.tolist(),
        )
        self._format_output_2d(wham2d, result2d)


# --- configuration ------------------------------------------------------


def build_config(yaml_path: Path) -> BayesWhamConfig:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

    config_raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config_raw, dict):
        raise ValueError("YAML configuration must define a mapping of parameters")

    num_samples = int(config_raw.get("num_samples", 200))
    burn_in = int(config_raw.get("burn_in", 50))
    thinning = int(config_raw.get("thinning", 1))
    dirichlet_alpha = float(config_raw.get("dirichlet_alpha", 1.0))

    if "hist_min_y" in config_raw:
        k_B = parse_units(config_raw.get("units"))
        periodic_x, period_x = parse_periodic_2d(config_raw, "x")
        periodic_y, period_y = parse_periodic_2d(config_raw, "y")
        required_fields = [
            "hist_min_x",
            "hist_max_x",
            "num_bins_x",
            "hist_min_y",
            "hist_max_y",
            "num_bins_y",
            "tolerance",
            "temperature",
            "numpad",
            "metadata_file",
            "freefile",
            "use_mask",
        ]
        for field in required_fields:
            if field not in config_raw:
                raise ValueError(f"Missing required configuration field: {field}")
        base = Wham2DConfig(
            hist_min_x=float(config_raw["hist_min_x"]),
            hist_max_x=float(config_raw["hist_max_x"]),
            num_bins_x=int(config_raw["num_bins_x"]),
            hist_min_y=float(config_raw["hist_min_y"]),
            hist_max_y=float(config_raw["hist_max_y"]),
            num_bins_y=int(config_raw["num_bins_y"]),
            tolerance=float(config_raw["tolerance"]),
            temperature=float(config_raw["temperature"]),
            numpad=int(config_raw["numpad"]),
            metadata_path=Path(config_raw["metadata_file"]),
            freefile_path=Path(config_raw["freefile"]),
            use_mask=bool(config_raw["use_mask"]),
            periodic_x=periodic_x,
            period_x=period_x,
            periodic_y=period_y,
            period_y=period_y,
            k_B=k_B,
            use_float32=bool(config_raw.get("use_float32", False)),
            bias_chunk_size=(int(config_raw["bias_chunk_size"]) if "bias_chunk_size" in config_raw else None),
            num_mc_runs=0,
            mc_seed=None,
            mc_workers=None,
            freefile_error_path=None,
            aux_data_path=Path(config_raw["aux_data_file"]) if "aux_data_file" in config_raw else None,
        )
        return BayesWhamConfig(
            dimension=2,
            base_config=base,
            num_samples=num_samples,
            burn_in=burn_in,
            thinning=thinning,
            dirichlet_alpha=dirichlet_alpha,
        )

    k_B = parse_units(config_raw.get("units"))
    periodic, period = parse_periodic_1d(config_raw)
    required_fields = [
        "hist_min",
        "hist_max",
        "num_bins",
        "tolerance",
        "temperature",
        "numpad",
        "metadata_file",
        "freefile",
    ]
    for field in required_fields:
        if field not in config_raw:
            raise ValueError(f"Missing required configuration field: {field}")
    base = Wham1DConfig(
        hist_min=float(config_raw["hist_min"]),
        hist_max=float(config_raw["hist_max"]),
        num_bins=int(config_raw["num_bins"]),
        tolerance=float(config_raw["tolerance"]),
        temperature=float(config_raw["temperature"]),
        numpad=int(config_raw["numpad"]),
        metadata_path=Path(config_raw["metadata_file"]),
        freefile_path=Path(config_raw["freefile"]),
        periodic=periodic,
        period=period,
        k_B=k_B,
        num_mc_runs=0,
        mc_seed=None,
        mc_workers=None,
        ingest_workers=None,
        aux_data_path=None,
    )
    return BayesWhamConfig(
        dimension=1,
        base_config=base,
        num_samples=num_samples,
        burn_in=burn_in,
        thinning=thinning,
        dirichlet_alpha=dirichlet_alpha,
    )


# --- CLI ---------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        raise ValueError("bwham expects a single argument: path to a YAML configuration file")
    yaml_path = Path(args[0])
    print(f"# Loading configuration from {yaml_path}")
    config = build_config(yaml_path)
    bwham = BayesWHAM(config)
    bwham.run()


if __name__ == "__main__":
    main()
