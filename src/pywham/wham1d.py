"""Python translation of the 1D WHAM executable."""

from __future__ import annotations

import concurrent.futures
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml

import numpy as np
from numpy.random import Generator
from .structures import HistGroup1D, Histogram1D

DEGREES = 360.0
RADIANS = 6.28318530717959
k_B_DEFAULT = 0.0019829237


@dataclass
class Wham1DConfig:
    hist_min: float
    hist_max: float
    num_bins: int
    tolerance: float
    temperature: float
    numpad: int
    metadata_path: Path
    freefile_path: Path
    periodic: bool
    period: float
    k_B: float
    num_mc_runs: int = 0
    mc_seed: int | None = None
    mc_workers: int | None = None
    ingest_workers: int | None = None

    @property
    def bin_width(self) -> float:
        return (self.hist_max - self.hist_min) / float(self.num_bins)

    @property
    def kT(self) -> float:
        return self.temperature * self.k_B


class Wham1D:
    def __init__(self, config: Wham1DConfig):
        self.config = config
        self.histogram: List[float] = [0.0 for _ in range(config.num_bins)]

    def clear_histogram(self) -> None:
        for i in range(self.config.num_bins):
            self.histogram[i] = 0.0

    def calc_coor(self, index: int) -> float:
        return self.config.hist_min + self.config.bin_width * (float(index) + 0.5)

    def calc_bias(self, hist_group: HistGroup1D, index: int, coor: float) -> float:
        spring = hist_group.spring_constants[index]
        loc = hist_group.bias_locations[index]
        dx = coor - loc
        if self.config.periodic:
            dx = abs(dx)
            if dx > self.config.period / 2.0:
                dx -= self.config.period
        return 0.5 * dx * dx * spring

    def hist_alloc(self, first: int, last: int, num_points: int, num_mc_samples: int) -> Histogram1D:
        size = last - first + 1
        return Histogram1D(
            first=first,
            last=last,
            num_points=num_points,
            num_mc_samples=num_mc_samples,
            data=[0.0 for _ in range(size)],
            cumulative=[0.0 for _ in range(size)],
        )

    def make_hist_group(self, num_windows: int) -> HistGroup1D:
        return HistGroup1D(
            num_windows=num_windows,
            bias_locations=[0.0 for _ in range(num_windows)],
            spring_constants=[0.0 for _ in range(num_windows)],
            free_energies=[0.0 for _ in range(num_windows)],
            previous_free_energies=[0.0 for _ in range(num_windows)],
            temperatures=[0.0 for _ in range(num_windows)],
            partitions=[0.0 for _ in range(num_windows)],
            histograms=[Histogram1D(0, 0, 0, 0) for _ in range(num_windows)],
        )

    @staticmethod
    def is_metadata(line: str) -> bool:
        if not line:
            return False
        if line.startswith("#"):
            return False
        return any(not ch.isspace() for ch in line)

    def get_numwindows(self, metadata: Iterable[str]) -> int:
        return sum(1 for line in metadata if self.is_metadata(line))

    @staticmethod
    def read_data(filename: Path, have_energy: bool, config: Wham1DConfig) -> Tuple[list[float], int]:
        histogram = [0.0 for _ in range(config.num_bins)]
        count = 0
        with filename.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.startswith("#"):
                    continue
                parts = raw.strip().split()
                if have_energy:
                    if len(parts) < 3:
                        raise ValueError(f"failure reading {filename}: missing energy value")
                    _, value_s, energy_s = parts[:3]
                    value = float(value_s)
                    energy = float(energy_s)
                else:
                    if len(parts) < 2:
                        raise ValueError(f"failure reading {filename}: missing position value")
                    _, value_s = parts[:2]
                    value = float(value_s)
                    energy = 0.0
                if config.hist_min < value < config.hist_max:
                    index = int((value - config.hist_min) / config.bin_width)
                    if have_energy:
                        histogram[index] += math.exp(-energy / config.kT)
                    else:
                        histogram[index] += 1.0
                    count += 1
        return histogram, count

    def read_metadata(self, lines: Iterable[str], hist_group: HistGroup1D) -> Tuple[int, bool]:
        entries: list[MetadataEntry] = []
        have_temp = False
        have_notemp = False

        for line in lines:
            if not self.is_metadata(line):
                continue
            tokens = line.split()
            filename = Path(tokens[0])
            loc = float(tokens[1])
            spring = float(tokens[2])
            correl_time = float(tokens[3]) if len(tokens) >= 4 else 1.0
            temp = float(tokens[4]) if len(tokens) >= 5 else -1.0

            if len(tokens) > 4:
                have_temp = True
            else:
                have_notemp = True

            entries.append(
                MetadataEntry(
                    index=len(entries),
                    filename=filename,
                    loc=loc,
                    spring=spring,
                    correl_time=correl_time,
                    temp=temp,
                )
            )

        if (have_temp and have_notemp) or (not have_temp and not have_notemp):
            raise ValueError("Some but not all metadata lines specify a temperature")

        worker_count = self._ingest_worker_count()
        have_energy = have_temp

        results: list[WindowLoadResult] = []
        if worker_count > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
                future_map = {
                    pool.submit(_load_window_data, entry, have_energy, self.config): entry
                    for entry in entries
                }
                for future in concurrent.futures.as_completed(future_map):
                    entry = future_map[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive
                        raise RuntimeError(f"Failed to load {entry.filename}: {exc}") from exc
        else:
            for entry in entries:
                results.append(_load_window_data(entry, have_energy, self.config))

        results.sort(key=lambda item: item.index)

        for entry, result in zip(entries, results):
            hist_group.bias_locations[entry.index] = entry.loc
            hist_group.spring_constants[entry.index] = entry.spring
            if have_energy:
                hist_group.temperatures[entry.index] = entry.temp * self.config.k_B
            else:
                hist_group.temperatures[entry.index] = -1.0
            hist_group.histograms[entry.index] = result.histogram
            hist_group.partitions[entry.index] = result.partition
            for warning in result.warnings:
                print(warning)

        return len(entries), have_temp

    @staticmethod
    def _find_range(histogram: list[float]) -> Tuple[int, int]:
        min_nonzero = 0
        max_nonzero = len(histogram) - 1

        still_zero = True
        i = 0
        while still_zero and i < len(histogram):
            if histogram[i] > 0:
                still_zero = False
            else:
                i += 1
        min_nonzero = i

        still_zero = True
        i = len(histogram) - 1
        while still_zero and i >= min_nonzero:
            if histogram[i] > 0:
                still_zero = False
            else:
                i -= 1
        max_nonzero = i

        return min_nonzero, max_nonzero

    def _ingest_worker_count(self) -> int:
        if os.getenv("PYWHAM_DISABLE_PARALLEL"):
            return 1
        if self.config.ingest_workers is not None:
            return max(int(self.config.ingest_workers), 1)
        env_workers = os.getenv("PYWHAM_INGEST_WORKERS")
        if env_workers:
            try:
                return max(int(env_workers), 1)
            except ValueError:
                pass
        return max(os.cpu_count() or 1, 1)


    def get_histval(self, hist: Histogram1D, index: int) -> float:
        if index < hist.first or index > hist.last:
            return 0.0
        return hist.data[index - hist.first]

    def save_free(self, hist_group: HistGroup1D) -> None:
        for i in range(hist_group.num_windows):
            hist_group.previous_free_energies[i] = hist_group.free_energies[i]
            hist_group.free_energies[i] = 0.0

    def is_converged(self, hist_group: HistGroup1D) -> bool:
        current = np.asarray(hist_group.free_energies, dtype=float)
        previous = np.asarray(hist_group.previous_free_energies, dtype=float)
        return bool(np.all(np.abs(current - previous) <= self.config.tolerance))

    def average_diff(self, hist_group: HistGroup1D) -> float:
        current = np.asarray(hist_group.free_energies, dtype=float)
        previous = np.asarray(hist_group.previous_free_energies, dtype=float)
        return float(np.mean(np.abs(current - previous)))

    def _write_iteration_snapshot(
        self, iteration: int, free_energy: list[float], probabilities: list[float], free_energies: list[float]
    ) -> None:
        with self.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write(f"# Iteration {iteration}\n")
            freefile.write("#Coor\tFree\tProb\n")
            for i in range(self.config.num_bins):
                coor = self.calc_coor(i)
                freefile.write(f"{coor:.6f}\t{free_energy[i]:.6f}\t{probabilities[i]:.6e}\n")
            freefile.write("\n# Window\tFree (free energy units)\n")
            for idx, value in enumerate(free_energies):
                freefile.write(f"#{idx}\t{value:.6f}\n")


    def calc_free(self, probabilities: List[float]) -> Tuple[List[float], int]:
        free = [-self.config.kT * math.log(p) for p in probabilities]
        min_val = min(free)
        bin_min = free.index(min_val)
        adjusted = [f - min_val for f in free]
        return adjusted, bin_min

    def wham_iteration(self, hist_group: HistGroup1D, prob: List[float], have_energy: bool) -> None:
        num_windows = hist_group.num_windows
        num_bins = self.config.num_bins

        probabilities = np.asarray(prob, dtype=float)
        bias_locations = np.asarray(hist_group.bias_locations, dtype=float)
        spring_constants = np.asarray(hist_group.spring_constants, dtype=float)
        previous_free_energies = np.asarray(hist_group.previous_free_energies, dtype=float)
        temperatures = np.asarray(hist_group.temperatures, dtype=float)
        partitions = np.asarray(hist_group.partitions, dtype=float)
        num_points = np.array([hist_group.histograms[i].num_points for i in range(num_windows)], dtype=float)

        coordinates = self.config.hist_min + self.config.bin_width * (np.arange(num_bins) + 0.5)
        dx = coordinates[None, :] - bias_locations[:, None]
        if self.config.periodic:
            dx = np.abs(dx)
            dx = np.where(dx > self.config.period / 2.0, dx - self.config.period, dx)
        bias_lookup = 0.5 * spring_constants[:, None] * dx * dx

        hist_matrix = np.zeros((num_windows, num_bins), dtype=float)
        for window_index, hist in enumerate(hist_group.histograms):
            if hist.first <= hist.last:
                start = hist.first
                end = hist.last + 1
                hist_matrix[window_index, start:end] = hist.data

        numerator = hist_matrix.sum(axis=0)
        bias_factor = np.exp((previous_free_energies[:, None] - bias_lookup) / temperatures[:, None])
        denom_weights = partitions if have_energy else num_points
        denom = (denom_weights[:, None] * bias_factor).sum(axis=0)
        probabilities = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom != 0.0)

        free_energy_terms = np.exp(-bias_lookup / temperatures[:, None]) * probabilities
        free_energies = free_energy_terms.sum(axis=1)
        free_energies = -temperatures * np.log(free_energies)
        free_energies = free_energies - free_energies[0]

        prob[:] = probabilities.tolist()
        hist_group.free_energies = free_energies.tolist()

    def run(self) -> None:
        lines = self.config.metadata_path.read_text(encoding="utf-8").splitlines()
        num_windows = self.get_numwindows(lines)
        print(f"#Number of windows = {num_windows}")

        hist_group = self.make_hist_group(num_windows)
        count_windows, have_temp = self.read_metadata(lines, hist_group)
        assert count_windows == hist_group.num_windows

        if have_temp:
            have_energy = True
        else:
            have_energy = False
            for i in range(hist_group.num_windows):
                hist_group.temperatures[i] = self.config.kT

        final_f = [0.0 for _ in range(hist_group.num_windows)]

        probabilities = [0.0 for _ in range(self.config.num_bins)]
        free_energy = [0.0 for _ in range(self.config.num_bins)]
        final_prob = [0.0 for _ in range(self.config.num_bins)]
        ave_p = [0.0 for _ in range(self.config.num_bins)]
        ave_p2 = [0.0 for _ in range(self.config.num_bins)]
        ave_pdf = [0.0 for _ in range(self.config.num_bins)]
        ave_pdf2 = [0.0 for _ in range(self.config.num_bins)]
        ave_F = [0.0 for _ in range(hist_group.num_windows)]
        ave_F2 = [0.0 for _ in range(hist_group.num_windows)]

        iteration = 0
        first = True
        while not self.is_converged(hist_group) or first:
            first = False
            self.save_free(hist_group)
            self.wham_iteration(hist_group, probabilities, have_energy)
            iteration += 1
            if iteration % 10 == 0:
                error = self.average_diff(hist_group)
                print(f"# Iteration {iteration:8d} | error {error:12.6e}")
            if iteration % 100 == 0:
                free_energy, _ = self.calc_free(probabilities)
                self._write_iteration_snapshot(iteration, free_energy, probabilities, hist_group.free_energies)
                for j in range(hist_group.num_windows):
                    final_f[j] = hist_group.free_energies[j]
            if iteration >= 100000:
                print(f"Too many iterations: {iteration}")
                break

        for j in range(hist_group.num_windows):
            final_f[j] = hist_group.free_energies[j]

        total = sum(probabilities)
        for i in range(self.config.num_bins):
            probabilities[i] /= total
            final_prob[i] = probabilities[i]

        free_energy, _ = self.calc_free(probabilities)

        ave_p = [0.0 for _ in ave_p]
        ave_p2 = [0.0 for _ in ave_p2]
        ave_pdf = [0.0 for _ in ave_pdf]
        ave_pdf2 = [0.0 for _ in ave_pdf2]
        ave_F = [0.0 for _ in ave_F]
        ave_F2 = [0.0 for _ in ave_F2]

        if self.config.num_mc_runs > 0:
            base_hist_group = _clone_hist_group(hist_group)
            seed_value = self.config.mc_seed if self.config.mc_seed is not None else 1
            seed_sequence = np.random.SeedSequence(seed_value)
            seeds = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(self.config.num_mc_runs)]

            workers = self.config.mc_workers
            if workers is None:
                workers = os.cpu_count() or 1
            workers = max(1, min(workers, self.config.num_mc_runs))

            if workers == 1:
                results = [
                    _run_bootstrap_trial(i, self.config, base_hist_group, have_energy, seeds[i])
                    for i in range(self.config.num_mc_runs)
                ]
            else:
                results: List[BootstrapResult] = [None for _ in range(self.config.num_mc_runs)]  # type: ignore
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(
                            _run_bootstrap_trial,
                            i,
                            self.config,
                            base_hist_group,
                            have_energy,
                            seeds[i],
                        ): i
                        for i in range(self.config.num_mc_runs)
                    }
                    for future in concurrent.futures.as_completed(futures):
                        idx = futures[future]
                        results[idx] = future.result()

            for result in results:
                if result.too_many_iterations:
                    print(f"Too many iterations: {result.iterations}")
                print(f"#MC trial {result.trial_index}: {result.iterations} iterations")
                print("#PMF values")

                for j in range(self.config.num_bins):
                    probability = result.probabilities[j]
                    pdf = result.pdf[j]
                    ave_p[j] += probability
                    ave_pdf[j] += pdf
                    ave_p2[j] += probability * probability
                    ave_pdf2[j] += pdf * pdf
                for j in range(hist_group.num_windows):
                    ave_F[j] += result.free_energies[j] - result.free_energies[0]
                    ave_F2[j] += result.free_energies[j] * result.free_energies[j]

            for i in range(self.config.num_bins):
                ave_p[i] /= float(self.config.num_mc_runs)
                ave_p2[i] /= float(self.config.num_mc_runs)
                ave_p2[i] = math.sqrt(max(ave_p2[i] - ave_p[i] * ave_p[i], 0.0))
                ave_pdf[i] /= float(self.config.num_mc_runs)
                ave_pdf2[i] /= float(self.config.num_mc_runs)
                ave_pdf2[i] = math.sqrt(max(ave_pdf2[i] - ave_pdf[i] * ave_pdf[i], 0.0))

            for i in range(hist_group.num_windows):
                ave_F[i] /= float(self.config.num_mc_runs)
                ave_F2[i] /= float(self.config.num_mc_runs)
                ave_F2[i] = math.sqrt(max(ave_F2[i] - ave_F[i] * ave_F[i], 0.0))
        else:
            print("# No MC error analysis requested")

        with self.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write("#Coor\t\tFree\t+/-\t\tProb\t\t+/-\n")
            for i in range(-self.config.numpad, 0):
                coor = self.calc_coor(i)
                freefile.write(
                    f"{coor}\t{free_energy[self.config.num_bins + i]}\t{ave_pdf2[self.config.num_bins + i]}\t"
                    f"{final_prob[self.config.num_bins + i]}\t{ave_p2[self.config.num_bins + i]}\n"
                )
            for i in range(self.config.num_bins):
                coor = self.calc_coor(i)
                freefile.write(
                    f"{coor}\t{free_energy[i]}\t{ave_pdf2[i]}\t{final_prob[i]}\t{ave_p2[i]}\n"
                )
            for i in range(self.config.numpad):
                coor = self.calc_coor(self.config.num_bins + i)
                freefile.write(
                    f"{coor}\t{free_energy[i]}\t{ave_pdf2[i]}\t{final_prob[i]}\t{ave_p2[i]}\n"
                )
            freefile.write("#Window\t\tFree\t+/-\t\n")
            for i in range(hist_group.num_windows):
                freefile.write(f"#{i}\t{final_f[i]}\t{ave_F2[i]}\n")

    def mk_new_hist(
        self, cumulative: List[float], distribution: List[float], num_bins: int, num_points: int, generator: Generator
    ) -> None:
        for i in range(num_bins):
            distribution[i] = 0.0
        for _ in range(num_points):
            j = self.get_rand_bin(cumulative, num_bins, generator)
            distribution[j] += 1.0

    def get_rand_bin(self, cumulative: List[float], num_bins: int, generator: Generator) -> int:
        value = generator.random()
        index = np.searchsorted(cumulative, value, side="right") - 1
        return min(max(index, 0), num_bins - 1)


@dataclass
class MetadataEntry:
    index: int
    filename: Path
    loc: float
    spring: float
    correl_time: float
    temp: float


@dataclass
class WindowLoadResult:
    index: int
    histogram: Histogram1D
    partition: float
    min_nonzero: int
    max_nonzero: int
    warnings: list[str]


def _build_histogram(
    histogram: list[float],
    num_points: int,
    correl_time: float,
    config: Wham1DConfig,
    source: Path,
) -> tuple[Histogram1D, float, list[str]]:
    min_nonzero, max_nonzero = Wham1D._find_range(histogram)
    if min_nonzero > max_nonzero:
        raise ValueError(
            "No data points within histogram bounds "
            f"[{config.hist_min}, {config.hist_max}]"
        )

    mc_samples = int(num_points / correl_time)
    warnings: list[str] = []
    if mc_samples < 1:
        warnings.append(f"# Correl time is too big for {source}")
        warnings.append(f"# You have {num_points} points, correl time {correl_time}")
        warnings.append("# Bootstrap error analysis will crash")

    num_used = max_nonzero - min_nonzero
    trimmed = Histogram1D(
        first=min_nonzero,
        last=max_nonzero,
        num_points=num_points,
        num_mc_samples=mc_samples,
        data=[0.0 for _ in range(num_used + 1)],
        cumulative=[0.0 for _ in range(num_used + 1)],
    )

    for i in range(min_nonzero, max_nonzero + 1):
        trimmed.data[i - min_nonzero] = histogram[i]
        if i == min_nonzero:
            trimmed.cumulative[0] = 0.0
        else:
            trimmed.cumulative[i - min_nonzero] = trimmed.cumulative[i - min_nonzero - 1] + histogram[i - 1]

    total = trimmed.cumulative[num_used] + histogram[max_nonzero]
    tiny = float(np.finfo(float).tiny)
    if total <= tiny:
        warnings.append(
            "# Warning: Window"
            f" {source} has near-zero counts/partition; consider removing it from the metadata"
            " or rerunning with a softer spring."
        )
    safe_total = max(total, tiny)
    for i in range(num_used + 1):
        trimmed.cumulative[i] /= safe_total

    return trimmed, total, warnings


def _load_window_data(entry: MetadataEntry, have_energy: bool, config: Wham1DConfig) -> WindowLoadResult:
    histogram, num_points = Wham1D.read_data(entry.filename, have_energy, config)
    if num_points < 0:
        raise OSError(f"Error trying to read {entry.filename}")

    trimmed, partition, warnings = _build_histogram(
        histogram, num_points, entry.correl_time, config, entry.filename
    )

    return WindowLoadResult(
        index=entry.index,
        histogram=trimmed,
        partition=partition,
        min_nonzero=trimmed.first,
        max_nonzero=trimmed.last,
        warnings=warnings,
    )


@dataclass
class BootstrapResult:
    trial_index: int
    iterations: int
    probabilities: List[float]
    pdf: List[float]
    free_energies: List[float]
    too_many_iterations: bool = False


def _clone_hist_group(base: HistGroup1D) -> HistGroup1D:
    histograms = [
        Histogram1D(
            first=hist.first,
            last=hist.last,
            num_points=hist.num_points,
            num_mc_samples=hist.num_mc_samples,
            data=list(hist.data),
            cumulative=list(hist.cumulative),
        )
        for hist in base.histograms
    ]
    return HistGroup1D(
        num_windows=base.num_windows,
        bias_locations=list(base.bias_locations),
        spring_constants=list(base.spring_constants),
        free_energies=[0.0 for _ in base.free_energies],
        previous_free_energies=[0.0 for _ in base.previous_free_energies],
        temperatures=list(base.temperatures),
        partitions=list(base.partitions),
        histograms=histograms,
    )


def _run_bootstrap_trial(
    trial_index: int,
    config: Wham1DConfig,
    base_hist_group: HistGroup1D,
    have_energy: bool,
    seed: int,
) -> BootstrapResult:
    wham = Wham1D(config)
    hist_group = _clone_hist_group(base_hist_group)
    generator = np.random.default_rng(seed)
    probabilities = [0.0 for _ in range(config.num_bins)]

    for j in range(hist_group.num_windows):
        hist = hist_group.histograms[j]
        num_used = hist.last - hist.first + 1
        wham.mk_new_hist(hist.cumulative, hist.data, num_used, hist.num_mc_samples, generator)
        hist_group.previous_free_energies[j] = 0.0
        hist_group.free_energies[j] = 0.0

    iteration = 0
    first = True
    too_many_iterations = False
    while not wham.is_converged(hist_group) or first:
        first = False
        wham.save_free(hist_group)
        wham.wham_iteration(hist_group, probabilities, have_energy)
        iteration += 1
        if iteration >= 100000:
            too_many_iterations = True
            break

    total = sum(probabilities)
    if total:
        probabilities = [p / total for p in probabilities]

    pdf = [-config.kT * math.log(p) if p > 0.0 else float("inf") for p in probabilities]

    return BootstrapResult(
        trial_index=trial_index,
        iterations=iteration,
        probabilities=probabilities,
        pdf=pdf,
        free_energies=list(hist_group.free_energies),
        too_many_iterations=too_many_iterations,
    )


def parse_units(units: str | None) -> float:
    if units is None:
        return k_B_DEFAULT
    if units == "lj":
        k_B = 1.0
    elif units == "real":
        k_B = 0.0019872067
    elif units == "metal":
        k_B = 8.617343e-5
    elif units == "si":
        k_B = 1.3806504e-23
    elif units == "cgs":
        k_B = 1.3806504e-16
    elif units == "electron":
        k_B = 3.16681534e-6
    elif units == "micro":
        k_B = 1.3806504e-8
    elif units == "nano":
        k_B = 0.013806504
    elif units == "default":
        k_B = k_B_DEFAULT
    else:
        raise ValueError(f"Unknown unit style: {units}\n")
    print(f"# Setting value of k_B to = {k_B:.15g}")
    return k_B


def parse_periodic(config: dict) -> tuple[bool, float]:
    periodic = bool(config.get("periodic", False))
    if not periodic:
        return False, 0.0
    period_value = config.get("period", DEGREES)
    if isinstance(period_value, str) and period_value.lower() == "pi":
        period = RADIANS
    else:
        period = float(period_value)
    print(f"#Turning on periodicity with period = {period}")
    return True, period


def build_config(yaml_path: Path) -> Wham1DConfig:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must define a mapping of parameters")

    k_B = parse_units(config.get("units"))
    periodic, period = parse_periodic(config)

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
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    num_mc = int(config.get("num_mc_runs", 0))
    seed = config.get("mc_seed")
    if seed is not None:
        seed = int(seed)
        if seed > 0:
            seed = -seed
    workers = config.get("mc_workers")
    if workers is not None:
        workers = int(workers)
        if workers < 1:
            workers = 1
    ingest_workers = config.get("ingest_workers")
    if ingest_workers is not None:
        ingest_workers = int(ingest_workers)
        if ingest_workers < 1:
            ingest_workers = 1

    return Wham1DConfig(
        hist_min=float(config["hist_min"]),
        hist_max=float(config["hist_max"]),
        num_bins=int(config["num_bins"]),
        tolerance=float(config["tolerance"]),
        temperature=float(config["temperature"]),
        numpad=int(config["numpad"]),
        metadata_path=Path(config["metadata_file"]),
        freefile_path=Path(config["freefile"]),
        periodic=periodic,
        period=period,
        k_B=k_B,
        num_mc_runs=num_mc,
        mc_seed=seed,
        mc_workers=workers,
        ingest_workers=ingest_workers,
    )


def main(argv: List[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError("wham now expects a single argument: path to a YAML configuration file")
    yaml_path = Path(args[0])
    print(f"# Loading configuration from {yaml_path}")
    config = build_config(yaml_path)
    wham = Wham1D(config)
    wham.run()


if __name__ == "__main__":
    main()
