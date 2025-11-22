"""Python translation of the 1D WHAM executable."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

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

    def read_data(self, filename: Path, have_energy: bool) -> int:
        self.clear_histogram()
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
                if self.config.hist_min < value < self.config.hist_max:
                    index = int((value - self.config.hist_min) / self.config.bin_width)
                    if have_energy:
                        self.histogram[index] += math.exp(-energy / self.config.kT)
                    else:
                        self.histogram[index] += 1.0
                    count += 1
        return count

    def read_metadata(self, lines: Iterable[str], hist_group: HistGroup1D) -> Tuple[int, bool]:
        current_window = 0
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

            hist_group.bias_locations[current_window] = loc
            hist_group.spring_constants[current_window] = spring

            if len(tokens) > 4:
                hist_group.temperatures[current_window] = temp * self.config.k_B
                have_temp = True
            else:
                hist_group.temperatures[current_window] = -1.0
                have_notemp = True

            if (have_temp and have_notemp) or (not have_temp and not have_notemp):
                raise ValueError("Some but not all metadata lines specify a temperature")

            num_points = self.read_data(filename, have_temp)
            if num_points < 0:
                raise OSError(f"Error trying to read {filename}")

            mc_samples = int(num_points / correl_time)
            if mc_samples < 1:
                print(f"# Correl time is too big for {filename}")
                print(f"# You have {num_points} points, correl time {correl_time}")
                print("# Bootstrap error analysis will crash")

            min_nonzero, max_nonzero = self._find_range()
            if min_nonzero > max_nonzero:
                raise ValueError(
                    "No data points within histogram bounds "
                    f"[{self.config.hist_min}, {self.config.hist_max}]"
                )

            trimmed = self.hist_alloc(min_nonzero, max_nonzero, num_points, mc_samples)
            for i in range(min_nonzero, max_nonzero + 1):
                trimmed.data[i - min_nonzero] = self.histogram[i]
                if i == min_nonzero:
                    trimmed.cumulative[0] = 0.0
                else:
                    trimmed.cumulative[i - min_nonzero] = (
                        trimmed.cumulative[i - min_nonzero - 1] + self.histogram[i - 1]
                    )

            num_used = max_nonzero - min_nonzero
            total = trimmed.cumulative[num_used] + self.histogram[max_nonzero]
            for i in range(num_used + 1):
                trimmed.cumulative[i] /= total
            hist_group.histograms[current_window] = trimmed
            hist_group.partitions[current_window] = total
            current_window += 1

        return current_window, have_temp

    def _find_range(self) -> Tuple[int, int]:
        min_nonzero = 0
        max_nonzero = self.config.num_bins - 1

        still_zero = True
        i = 0
        while still_zero and i < self.config.num_bins:
            if self.histogram[i] > 0:
                still_zero = False
            else:
                i += 1
        min_nonzero = i

        still_zero = True
        i = self.config.num_bins - 1
        while still_zero and i >= min_nonzero:
            if self.histogram[i] > 0:
                still_zero = False
            else:
                i -= 1
        max_nonzero = i

        return min_nonzero, max_nonzero

    def get_histval(self, hist: Histogram1D, index: int) -> float:
        if index < hist.first or index > hist.last:
            return 0.0
        return hist.data[index - hist.first]

    def save_free(self, hist_group: HistGroup1D) -> None:
        for i in range(hist_group.num_windows):
            hist_group.previous_free_energies[i] = hist_group.free_energies[i]
            hist_group.free_energies[i] = 0.0

    def is_converged(self, hist_group: HistGroup1D) -> bool:
        for i in range(hist_group.num_windows):
            error = abs(hist_group.free_energies[i] - hist_group.previous_free_energies[i])
            if error > self.config.tolerance:
                return False
        return True

    def average_diff(self, hist_group: HistGroup1D) -> float:
        error = 0.0
        for i in range(hist_group.num_windows):
            error += abs(hist_group.free_energies[i] - hist_group.previous_free_energies[i])
        return error / float(hist_group.num_windows)

    def calc_free(self, probabilities: List[float]) -> Tuple[List[float], int]:
        free = [-self.config.kT * math.log(p) for p in probabilities]
        min_val = min(free)
        bin_min = free.index(min_val)
        adjusted = [f - min_val for f in free]
        return adjusted, bin_min

    def wham_iteration(self, hist_group: HistGroup1D, prob: List[float], have_energy: bool) -> None:
        for i in range(self.config.num_bins):
            coor = self.calc_coor(i)
            num = 0.0
            denom = 0.0
            for j in range(hist_group.num_windows):
                num += self.get_histval(hist_group.histograms[j], i)
                bias = self.calc_bias(hist_group, j, coor)
                bf = math.exp((hist_group.previous_free_energies[j] - bias) / hist_group.temperatures[j])
                if have_energy:
                    denom += hist_group.partitions[j] * bf
                else:
                    denom += hist_group.histograms[j].num_points * bf
            prob[i] = num / denom
            for j in range(hist_group.num_windows):
                bias = self.calc_bias(hist_group, j, coor)
                bf = math.exp(-bias / hist_group.temperatures[j]) * prob[i]
                hist_group.free_energies[j] += bf

        for j in range(hist_group.num_windows):
            hist_group.free_energies[j] = -hist_group.temperatures[j] * math.log(hist_group.free_energies[j])
        for j in range(hist_group.num_windows - 1, -1, -1):
            hist_group.free_energies[j] -= hist_group.free_energies[0]

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
                print(f"#Iteration {iteration}:  {error}")
            if iteration % 100 == 0:
                free_energy, _ = self.calc_free(probabilities)
                for i in range(self.config.num_bins):
                    coor = self.calc_coor(i)
                    print(f"{coor}\t{free_energy[i]}\t{probabilities[i]}")
                print()
                print("# Dumping simulation biases, in the metadata file order ")
                print("# Window  F (free energy units)")
                for j in range(hist_group.num_windows):
                    print(f"# {j}\t{hist_group.free_energies[j]}")
                    final_f[j] = hist_group.free_energies[j]
            if iteration >= 100000:
                print(f"Too many iterations: {iteration}")
                break

        print("# Dumping simulation biases, in the metadata file order ")
        print("# Window  F (free energy units)")
        for j in range(hist_group.num_windows):
            print(f"# {j}\t{hist_group.free_energies[j]}")
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
            generator = np.random.default_rng(self.config.mc_seed if self.config.mc_seed is not None else 1)
            for i in range(self.config.num_mc_runs):
                for j in range(hist_group.num_windows):
                    hist = hist_group.histograms[j]
                    num_used = hist.last - hist.first + 1
                    self.mk_new_hist(hist.cumulative, hist.data, num_used, hist.num_mc_samples, generator)
                    hist_group.previous_free_energies[j] = 0.0
                    hist_group.free_energies[j] = 0.0

                iteration = 0
                first = True
                while not self.is_converged(hist_group) or first:
                    first = False
                    self.save_free(hist_group)
                    self.wham_iteration(hist_group, probabilities, have_energy)
                    iteration += 1
                    if iteration >= 100000:
                        print(f"Too many iterations: {iteration}")
                        break
                print(f"#MC trial {i}: {iteration} iterations")
                print("#PMF values")

                total = sum(probabilities)
                for j in range(self.config.num_bins):
                    probabilities[j] /= total

                for j in range(self.config.num_bins):
                    pdf = -self.config.kT * math.log(probabilities[j])
                    ave_p[j] += probabilities[j]
                    ave_pdf[j] += pdf
                    ave_p2[j] += probabilities[j] * probabilities[j]
                    ave_pdf2[j] += pdf * pdf
                for j in range(hist_group.num_windows):
                    ave_F[j] += hist_group.free_energies[j] - hist_group.free_energies[0]
                    ave_F2[j] += hist_group.free_energies[j] * hist_group.free_energies[j]

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


def parse_units(args: List[str]) -> Tuple[float, List[str]]:
    if args and args[0] == "units":
        if len(args) < 2:
            raise ValueError("Command line: wham [units <real|metal|lj|...>] [P|Ppi|Pval]  hist_min hist_max num_bins tol temperature numpad metadatafile freefile [num_MC_trials randSeed]\n")
        units = args[1]
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
        return k_B, args[2:]
    return k_B_DEFAULT, args


def parse_periodic(arg: str) -> Tuple[bool, float, int]:
    periodic = False
    period = 0.0
    consumed = 0
    if arg.upper().startswith("P"):
        periodic = True
        suffix = arg[1:]
        if not suffix:
            period = DEGREES
        else:
            suffix = suffix.upper()
            if suffix.startswith("PI"):
                period = RADIANS
            else:
                period = float(suffix)
        print(f"#Turning on periodicity with period = {period}")
        consumed = 1
    return periodic, period, consumed


def build_config(argv: List[str]) -> Wham1DConfig:
    args = argv[:]
    k_B, args = parse_units(args)
    periodic, period, consumed = parse_periodic(args[0])
    if consumed:
        args = args[1:]
    if len(args) not in (8, 10):
        raise ValueError(
            "Command line: wham [units <real|metal|lj|...>] [P|Ppi|Pval]  hist_min hist_max num_bins tol temperature numpad metadatafile freefile [num_MC_trials randSeed]\n"
        )

    hist_min = float(args[0])
    hist_max = float(args[1])
    num_bins = int(args[2])
    tol = float(args[3])
    temperature = float(args[4])
    numpad = int(args[5])
    metadata = Path(args[6])
    freefile = Path(args[7])

    num_mc = 0
    seed = None
    if len(args) == 10:
        num_mc = int(args[8])
        seed = int(args[9])
        if seed > 0:
            seed = -seed
    return Wham1DConfig(
        hist_min=hist_min,
        hist_max=hist_max,
        num_bins=num_bins,
        tolerance=tol,
        temperature=temperature,
        numpad=numpad,
        metadata_path=metadata,
        freefile_path=freefile,
        periodic=periodic,
        period=period,
        k_B=k_B,
        num_mc_runs=num_mc,
        mc_seed=seed,
    )


def main(argv: List[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    print("#" + " ".join(args))
    config = build_config(args)
    wham = Wham1D(config)
    wham.run()


if __name__ == "__main__":
    main()
