"""Python translation of the 2D WHAM executable."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from .structures import HistGroup2D, Histogram2D

DEGREES = 360.0
RADIANS = 6.28318530717959
k_B_DEFAULT = 0.0019829237
MASKED = 9999999.0


@dataclass
class Wham2DConfig:
    hist_min_x: float
    hist_max_x: float
    num_bins_x: int
    hist_min_y: float
    hist_max_y: float
    num_bins_y: int
    tolerance: float
    temperature: float
    numpad: int
    metadata_path: Path
    freefile_path: Path
    use_mask: bool
    periodic_x: bool
    period_x: float
    periodic_y: bool
    period_y: float
    k_B: float

    @property
    def bin_width_x(self) -> float:
        return (self.hist_max_x - self.hist_min_x) / float(self.num_bins_x)

    @property
    def bin_width_y(self) -> float:
        return (self.hist_max_y - self.hist_min_y) / float(self.num_bins_y)

    @property
    def kT(self) -> float:
        return self.temperature * self.k_B


class Wham2D:
    def __init__(self, config: Wham2DConfig):
        self.config = config
        self.histogram: List[List[float]] = [
            [0.0 for _ in range(config.num_bins_y)] for _ in range(config.num_bins_x)
        ]

    def clear_histogram(self) -> None:
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                self.histogram[i][j] = 0.0

    def calc_coor(self, i: int, j: int) -> Tuple[float, float]:
        return (
            self.config.hist_min_x + self.config.bin_width_x * (float(i) + 0.5),
            self.config.hist_min_y + self.config.bin_width_y * (float(j) + 0.5),
        )

    def calc_bias(self, hist_group: HistGroup2D, index: int, coor: Tuple[float, float]) -> float:
        spring_x = hist_group.spring_x[index]
        spring_y = hist_group.spring_y[index]
        dx = coor[0] - hist_group.bias_locations[index][0]
        dy = coor[1] - hist_group.bias_locations[index][1]
        half_px = self.config.period_x / 2.0 if self.config.periodic_x else None
        half_py = self.config.period_y / 2.0 if self.config.periodic_y else None
        if half_px is not None:
            dx = abs(dx)
            if dx > half_px:
                dx -= self.config.period_x
        if half_py is not None:
            dy = abs(dy)
            if dy > half_py:
                dy -= self.config.period_y
        return 0.5 * (spring_x * dx * dx + spring_y * dy * dy)

    def hist_alloc(
        self, min_nonzero_x: int, max_nonzero_x: int, min_nonzero_y: int, max_nonzero_y: int, num_points: int, num_mc_samples: int
    ) -> Histogram2D:
        data = [
            [0.0 for _ in range(max_nonzero_y - min_nonzero_y + 1)]
            for _ in range(max_nonzero_x - min_nonzero_x + 1)
        ]
        # cumulative array has one extra slot as in the C code
        cumulative = [0.0 for _ in range((max_nonzero_x - min_nonzero_x + 1) * (max_nonzero_y - min_nonzero_y + 1) + 1)]
        return Histogram2D(
            first_x=min_nonzero_x,
            last_x=max_nonzero_x,
            first_y=min_nonzero_y,
            last_y=max_nonzero_y,
            num_points=num_points,
            num_mc_samples=num_mc_samples,
            data=data,
            cumulative=cumulative,
        )

    def make_hist_group(self, num_windows: int) -> HistGroup2D:
        return HistGroup2D(
            num_windows=num_windows,
            bias_locations=[[0.0, 0.0] for _ in range(num_windows)],
            spring_x=[0.0 for _ in range(num_windows)],
            spring_y=[0.0 for _ in range(num_windows)],
            free_energies=[1.0 for _ in range(num_windows)],
            previous_free_energies=[1.0 for _ in range(num_windows)],
            temperatures=[0.0 for _ in range(num_windows)],
            partitions=[0.0 for _ in range(num_windows)],
            histograms=[Histogram2D(0, 0, 0, 0, 0, 0) for _ in range(num_windows)],
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

    def read_data(self, filename: Path, have_energy: bool, use_mask: bool, mask: List[List[int]] | None) -> int:
        self.clear_histogram()
        count = 0
        with filename.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.startswith("#"):
                    continue
                parts = raw.strip().split()
                if have_energy:
                    if len(parts) < 4:
                        raise ValueError(f"Failure reading {filename}: missing energy value")
                    _, value_x_s, value_y_s, energy_s = parts[:4]
                    value_x = float(value_x_s)
                    value_y = float(value_y_s)
                    energy = float(energy_s)
                else:
                    if len(parts) < 3:
                        continue
                    _, value_x_s, value_y_s = parts[:3]
                    value_x = float(value_x_s)
                    value_y = float(value_y_s)
                    energy = 0.0
                if (
                    self.config.hist_min_x < value_x < self.config.hist_max_x
                    and self.config.hist_min_y < value_y < self.config.hist_max_y
                ):
                    index_x = int((value_x - self.config.hist_min_x) / self.config.bin_width_x)
                    index_y = int((value_y - self.config.hist_min_y) / self.config.bin_width_y)
                    if have_energy:
                        self.histogram[index_x][index_y] += math.exp(-energy / self.config.kT)
                    else:
                        self.histogram[index_x][index_y] += 1.0
                    count += 1
                    if use_mask and mask is not None:
                        mask[index_x][index_y] = 1
        return count

    def read_metadata(
        self, lines: Iterable[str], hist_group: HistGroup2D, use_mask: bool, mask: List[List[int]] | None
    ) -> Tuple[int, bool]:
        current_window = 0
        have_temp = False
        have_notemp = False

        for line in lines:
            if not self.is_metadata(line):
                continue
            tokens = line.split()
            filename = Path(tokens[0])
            locx = float(tokens[1])
            locy = float(tokens[2])
            springx = float(tokens[3])
            springy = float(tokens[4])
            correl_time = float(tokens[5]) if len(tokens) >= 6 else 1.0
            temp = float(tokens[6]) if len(tokens) >= 7 else -1.0

            hist_group.bias_locations[current_window][0] = locx
            hist_group.bias_locations[current_window][1] = locy
            hist_group.spring_x[current_window] = springx
            hist_group.spring_y[current_window] = springy

            if len(tokens) > 6:
                hist_group.temperatures[current_window] = temp * self.config.k_B
                have_temp = True
            else:
                hist_group.temperatures[current_window] = -1.0
                have_notemp = True

            if (have_temp and have_notemp) or (not have_temp and not have_notemp):
                raise ValueError("Some but not all metadata lines specify a temperature")

            num_points = self.read_data(filename, have_temp, use_mask, mask)
            if num_points < 0:
                raise OSError(f"Error trying to read {filename}")

            mc_samples = int(num_points / correl_time)
            if mc_samples < 1:
                print(f"# Correl time is too big for {filename}:")
                print(f"# You have {num_points} points, correl time {correl_time}")
                print("# Bootstrap error analysis will crash")

            min_nonzero_x, max_nonzero_x, min_nonzero_y, max_nonzero_y = self._find_range()
            if min_nonzero_x > max_nonzero_x or min_nonzero_y > max_nonzero_y:
                raise ValueError(
                    "No data points within the histogram bounds:\n"
                    f"X\t[{self.config.hist_min_x}, {self.config.hist_max_x}]\n"
                    f"Y\t[{self.config.hist_min_y}, {self.config.hist_max_y}]"
                )

            trimmed = self.hist_alloc(min_nonzero_x, max_nonzero_x, min_nonzero_y, max_nonzero_y, num_points, mc_samples)
            bin_index = 0
            trimmed.cumulative[0] = 0.0
            for i in range(min_nonzero_x, max_nonzero_x + 1):
                for j in range(min_nonzero_y, max_nonzero_y + 1):
                    bin_index += 1
                    trimmed.cumulative[bin_index] = trimmed.cumulative[bin_index - 1] + self.histogram[i][j]
                    trimmed.data[i - min_nonzero_x][j - min_nonzero_y] = self.histogram[i][j]
            total = trimmed.cumulative[bin_index]
            last_bin = bin_index
            for bin_index in range(last_bin + 1):
                trimmed.cumulative[bin_index] /= total
            hist_group.histograms[current_window] = trimmed
            hist_group.partitions[current_window] = total
            current_window += 1

        return current_window, have_temp

    def _find_range(self) -> Tuple[int, int, int, int]:
        min_nonzero_x = 0
        max_nonzero_x = self.config.num_bins_x - 1
        min_nonzero_y = 0
        max_nonzero_y = self.config.num_bins_y - 1

        found = False
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                if self.histogram[i][j] > 0:
                    min_nonzero_x = i
                    found = True
                    break
            if found:
                break

        found = False
        for i in range(self.config.num_bins_x - 1, -1, -1):
            for j in range(self.config.num_bins_y):
                if self.histogram[i][j] > 0:
                    max_nonzero_x = i
                    found = True
                    break
            if found:
                break

        min_nonzero_y = self.config.num_bins_y - 1
        max_nonzero_y = 0
        for i in range(min_nonzero_x, max_nonzero_x + 1):
            for j in range(self.config.num_bins_y):
                if self.histogram[i][j] > 0:
                    min_nonzero_y = min(min_nonzero_y, j)
                    max_nonzero_y = max(max_nonzero_y, j)

        return min_nonzero_x, max_nonzero_x, min_nonzero_y, max_nonzero_y

    def get_histval(self, hist: Histogram2D, i: int, j: int) -> float:
        if i < hist.first_x or i > hist.last_x or j < hist.first_y or j > hist.last_y:
            return 0.0
        return hist.data[i - hist.first_x][j - hist.first_y]

    def save_free(self, hist_group: HistGroup2D) -> None:
        for i in range(hist_group.num_windows):
            hist_group.previous_free_energies[i] = hist_group.free_energies[i]
            hist_group.free_energies[i] = 0.0

    def is_converged(self, hist_group: HistGroup2D, logged_current: List[float], logged_previous: List[float]) -> bool:
        for cur, prev in zip(logged_current, logged_previous):
            if abs(cur - prev) > self.config.tolerance:
                return False
        return True

    def average_diff(self, logged_current: List[float], logged_previous: List[float]) -> float:
        error = 0.0
        for cur, prev in zip(logged_current, logged_previous):
            error += abs(cur - prev)
        return error / float(len(logged_current)) if logged_current else 0.0

    def calc_free(self, prob: List[List[float]], use_mask: bool, mask: List[List[int]] | None) -> List[List[float]]:
        free = [[0.0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)]
        min_val = 1e50
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                if use_mask and mask is not None and not mask[i][j]:
                    prob[i][j] = 0.0
                    free[i][j] = MASKED
                else:
                    free[i][j] = -self.config.kT * math.log(prob[i][j])
                    if free[i][j] < min_val:
                        min_val = free[i][j]
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                if not (use_mask and mask is not None and not mask[i][j]):
                    free[i][j] -= min_val
        return free

    def wham_iteration(
        self,
        hist_group: HistGroup2D,
        prob: List[List[float]],
        have_energy: bool,
        use_mask: bool,
        mask: List[List[int]] | None,
        bias_lookup: List[List[List[float]]],
        num_lookup: List[List[float]],
    ) -> None:
        for i in range(self.config.num_bins_x):
            for k in range(self.config.num_bins_y):
                if use_mask and mask is not None and not mask[i][k]:
                    continue
                denom = 0.0
                for j in range(hist_group.num_windows):
                    bf = hist_group.previous_free_energies[j] * bias_lookup[i][k][j]
                    if have_energy:
                        denom += hist_group.partitions[j] * bf
                    else:
                        denom += hist_group.histograms[j].num_points * bf
                prob[i][k] = num_lookup[i][k] / denom
                for j in range(hist_group.num_windows):
                    bf = bias_lookup[i][k][j] * prob[i][k]
                    hist_group.free_energies[j] += bf
        for j in range(hist_group.num_windows):
            hist_group.free_energies[j] = 1.0 / hist_group.free_energies[j]

    def run(self) -> None:
        lines = self.config.metadata_path.read_text(encoding="utf-8").splitlines()
        num_windows = self.get_numwindows(lines)
        print(f"#Number of windows = {num_windows}")

        mask = None
        if self.config.use_mask:
            mask = [[0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)]

        hist_group = self.make_hist_group(num_windows)
        count_windows, have_temp = self.read_metadata(lines, hist_group, self.config.use_mask, mask)
        assert count_windows == hist_group.num_windows

        if have_temp:
            have_energy = True
        else:
            have_energy = False
            for i in range(hist_group.num_windows):
                hist_group.temperatures[i] = self.config.kT

        for i in range(hist_group.num_windows):
            hist_group.free_energies[i] = 1.0
            hist_group.previous_free_energies[i] = 1.0

        num_lookup = [
            [0.0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)
        ]
        bias_lookup = [
            [[0.0 for _ in range(hist_group.num_windows)] for _ in range(self.config.num_bins_y)]
            for _ in range(self.config.num_bins_x)
        ]

        for i in range(self.config.num_bins_x):
            for k in range(self.config.num_bins_y):
                coor = self.calc_coor(i, k)
                num_lookup[i][k] = 0.0
                for j in range(hist_group.num_windows):
                    num_lookup[i][k] += self.get_histval(hist_group.histograms[j], i, k)
                    bias_lookup[i][k][j] = math.exp(-self.calc_bias(hist_group, j, coor) / hist_group.temperatures[j])

        prob = [
            [0.0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)
        ]
        final_prob = [
            [0.0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)
        ]
        free_ene = [
            [0.0 for _ in range(self.config.num_bins_y)] for _ in range(self.config.num_bins_x)
        ]

        iteration = 0
        first = True
        converged = False
        while not converged or first:
            first = False
            self.save_free(hist_group)
            self.wham_iteration(hist_group, prob, have_energy, self.config.use_mask, mask, bias_lookup, num_lookup)
            iteration += 1

            logged_current = [
                hist_group.temperatures[i] * math.log(hist_group.free_energies[i]) for i in range(hist_group.num_windows)
            ]
            logged_previous = [
                hist_group.temperatures[i] * math.log(hist_group.previous_free_energies[i]) for i in range(hist_group.num_windows)
            ]
            converged = self.is_converged(hist_group, logged_current, logged_previous)
            if iteration % 10 == 0:
                error = self.average_diff(logged_current, logged_previous)
                print(f"#Iteration {iteration}:  {error}")
            if iteration % 100 == 0:
                free_ene = self.calc_free(prob, self.config.use_mask, mask)
                for i in range(self.config.num_bins_x):
                    for j in range(self.config.num_bins_y):
                        coor = self.calc_coor(i, j)
                        print(f"{coor[0]}\t{coor[1]}\t{free_ene[i][j]}\t{prob[i][j]}")
                print("# Dumping simulation biases, in the metadata file order ")
                print("# Window  F (free energy units)")
                for j in range(hist_group.num_windows):
                    print(f"# {j}\t{hist_group.free_energies[j]}")
            if iteration >= 100000:
                print(f"Too many iterations: {iteration}")
                break

        print("# Dumping simulation biases, in the metadata file order ")
        print("# Window  F (free energy units)")
        for j in range(hist_group.num_windows):
            print(f"# {j}\t{hist_group.free_energies[j] - hist_group.free_energies[0]}")

        free_ene = self.calc_free(prob, self.config.use_mask, mask)
        total = sum(sum(row) for row in prob)
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                prob[i][j] /= total
                final_prob[i][j] = prob[i][j]

        if self.config.use_mask and mask is not None:
            for i in range(self.config.num_bins_x):
                for j in range(self.config.num_bins_y):
                    if not mask[i][j]:
                        free_ene[i][j] = MASKED

        with self.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write("#X\t\tY\t\tFree\t\tPro\n")
            for i in range(-self.config.numpad, 0):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[self.config.num_bins_x + i][self.config.num_bins_y + j]}\t{final_prob[self.config.num_bins_x + i][self.config.num_bins_y + j]}\n"
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[self.config.num_bins_x + i][j]}\t{final_prob[self.config.num_bins_x + i][j]}\n"
                    )
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[self.config.num_bins_x + i][j - self.config.num_bins_y]}\t{final_prob[self.config.num_bins_x + i][j - self.config.num_bins_y]}\n"
                    )
            for i in range(self.config.num_bins_x):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[i][self.config.num_bins_y + j]}\t{final_prob[i][self.config.num_bins_y + j]}\n"
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(f"{coor[0]}\t{coor[1]}\t{free_ene[i][j]}\t{final_prob[i][j]}\n")
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[i][j - self.config.num_bins_y]}\t{final_prob[i][j - self.config.num_bins_y]}\n"
                    )
            for i in range(self.config.num_bins_x, self.config.num_bins_x + self.config.numpad):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[i - self.config.num_bins_x][self.config.num_bins_y + j]}\t{final_prob[i - self.config.num_bins_x][self.config.num_bins_y + j]}\n"
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[i - self.config.num_bins_x][j]}\t{final_prob[i - self.config.num_bins_x][j]}\n"
                    )
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        f"{coor[0]}\t{coor[1]}\t{free_ene[i - self.config.num_bins_x][j - self.config.num_bins_y]}\t{final_prob[i - self.config.num_bins_x][j - self.config.num_bins_y]}\n"
                    )

def parse_periodic(token: str) -> Tuple[bool, float]:
    if not token.upper().startswith("P"):
        raise ValueError(
            "Command line:  wham-2d [units <real|metal|lj|...>] Px[=0|pi|val] hist_min_x hist_max_x num_bins_x Py[=0|pi|val] hist_min_y hist_max_y num_bins_y tol temperature numpad metadatafile freefile use_mask\n"
        )
    if len(token) == 2:
        return True, DEGREES
    suffix = token[3:]
    if not suffix:
        return True, DEGREES
    if suffix[0] == "0":
        return False, 0.0
    if suffix[0].isalpha():
        if suffix.upper().startswith("PI"):
            return True, RADIANS
        raise ValueError(
            "Command line:  wham-2d [units <real|metal|lj|...>] Px[=0|pi|val] hist_min_x hist_max_x num_bins_x Py[=0|pi|val] hist_min_y hist_max_y num_bins_y tol temperature numpad metadatafile freefile use_mask\n"
        )
    return True, float(suffix)


def parse_units(args: List[str]) -> Tuple[float, List[str]]:
    if args and args[0] == "units":
        if len(args) < 2:
            raise ValueError(
                "Command line:  wham-2d [units <real|metal|lj|...>] Px[=0|pi|val] hist_min_x hist_max_x num_bins_x Py[=0|pi|val] hist_min_y hist_max_y num_bins_y tol temperature numpad metadatafile freefile use_mask\n"
            )
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


def build_config(argv: List[str]) -> Wham2DConfig:
    args = argv[:]
    k_B, args = parse_units(args)
    if len(args) != 14:
        raise ValueError(
            "Command line:  wham-2d [units <real|metal|lj|...>] Px[=0|pi|val] hist_min_x hist_max_x num_bins_x Py[=0|pi|val] hist_min_y hist_max_y num_bins_y tol temperature numpad metadatafile freefile use_mask\n"
        )
    periodic_x, period_x = parse_periodic(args[0])
    hist_min_x = float(args[1])
    hist_max_x = float(args[2])
    num_bins_x = int(args[3])

    periodic_y, period_y = parse_periodic(args[4])
    hist_min_y = float(args[5])
    hist_max_y = float(args[6])
    num_bins_y = int(args[7])

    tolerance = float(args[8])
    temperature = float(args[9])
    numpad = int(args[10])
    metadata = Path(args[11])
    freefile = Path(args[12])
    use_mask = bool(int(args[13]))

    return Wham2DConfig(
        hist_min_x=hist_min_x,
        hist_max_x=hist_max_x,
        num_bins_x=num_bins_x,
        hist_min_y=hist_min_y,
        hist_max_y=hist_max_y,
        num_bins_y=num_bins_y,
        tolerance=tolerance,
        temperature=temperature,
        numpad=numpad,
        metadata_path=metadata,
        freefile_path=freefile,
        use_mask=use_mask,
        periodic_x=periodic_x,
        period_x=period_x,
        periodic_y=periodic_y,
        period_y=period_y,
        k_B=k_B,
    )


def main(argv: List[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    print("#" + " ".join(args))
    config = build_config(args)
    wham = Wham2D(config)
    wham.run()


if __name__ == "__main__":
    main()
