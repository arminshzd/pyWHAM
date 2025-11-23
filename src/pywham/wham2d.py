"""Python translation of the 2D WHAM executable."""

from __future__ import annotations

import concurrent.futures
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from numpy.random import Generator
import yaml

from .structures import HistGroup2D, Histogram2D

DEGREES = 360.0
RADIANS = 6.28318530717959
k_B_DEFAULT = 0.0019829237
MASKED = np.inf


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
    use_float32: bool = False
    bias_chunk_size: int | None = None
    num_mc_runs: int = 0
    mc_seed: int | None = None
    mc_workers: int | None = None
    freefile_error_path: Path | None = None

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
        epsilon = float(np.finfo(float).tiny)
        min_val = 1e50
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                if use_mask and mask is not None and not mask[i][j]:
                    prob[i][j] = 0.0
                    free[i][j] = MASKED
                else:
                    free[i][j] = -self.config.kT * math.log(max(prob[i][j], epsilon))
                    if free[i][j] < min_val:
                        min_val = free[i][j]
        for i in range(self.config.num_bins_x):
            for j in range(self.config.num_bins_y):
                if not (use_mask and mask is not None and not mask[i][j]):
                    free[i][j] -= min_val
        return free

    def mk_new_hist(
        self,
        cumulative: List[float],
        distribution: List[List[float]],
        first_x: int,
        last_x: int,
        first_y: int,
        last_y: int,
        num_points: int,
        generator: Generator,
    ) -> None:
        num_x = last_x - first_x + 1
        num_y = last_y - first_y + 1
        for i in range(num_x):
            for j in range(num_y):
                distribution[i][j] = 0.0
        for _ in range(num_points):
            bin_index = self.get_rand_bin(cumulative, num_x * num_y, generator)
            x_offset = bin_index // num_y
            y_offset = bin_index % num_y
            distribution[x_offset][y_offset] += 1.0

    def get_rand_bin(self, cumulative: List[float], num_bins: int, generator: Generator) -> int:
        value = generator.random()
        index = np.searchsorted(cumulative, value, side="right") - 1
        return min(max(int(index), 0), num_bins - 1)

    def wham_iteration(
        self,
        hist_group: HistGroup2D,
        prob: np.ndarray,
        have_energy: bool,
        use_mask: bool,
        mask: List[List[int]] | None,
        bias_lookup: np.ndarray,
        num_lookup: np.ndarray,
    ) -> None:
        dtype = prob.dtype
        finfo = np.finfo(dtype)
        mask_arr = np.asarray(mask, dtype=bool) if use_mask and mask is not None else None
        factors = np.asarray(
            hist_group.partitions if have_energy else [h.num_points for h in hist_group.histograms],
            dtype=dtype,
        )
        weight = np.asarray(hist_group.previous_free_energies, dtype=dtype) * factors
        with np.errstate(over="ignore"):
            denom = np.tensordot(bias_lookup, weight, axes=([2], [0]))
        tiny = finfo.tiny
        if mask_arr is not None:
            denom = np.where(mask_arr, denom, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            safe_denom = np.maximum(denom, tiny)
            prob[:] = num_lookup / safe_denom
        prob[:] = np.nan_to_num(prob, nan=0.0, posinf=finfo.max, neginf=0.0)
        prob[:] = np.minimum(prob, finfo.max)
        if mask_arr is not None:
            prob *= mask_arr

        bias_prob = prob[..., None] * bias_lookup
        bias_sum = bias_prob.sum(axis=(0, 1))
        bias_sum = np.nan_to_num(bias_sum, nan=0.0, posinf=finfo.max, neginf=0.0)
        bias_sum = np.maximum(bias_sum, tiny)
        updated = np.clip(1.0 / bias_sum, tiny, finfo.max).tolist()
        for idx, val in enumerate(updated):
            hist_group.free_energies[idx] = val

    def _coordinate_grids(self, dtype: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
        x_centers = self.config.hist_min_x + self.config.bin_width_x * (
            np.arange(self.config.num_bins_x, dtype=dtype) + 0.5
        )
        y_centers = self.config.hist_min_y + self.config.bin_width_y * (
            np.arange(self.config.num_bins_y, dtype=dtype) + 0.5
        )
        return np.meshgrid(x_centers, y_centers, indexing="ij")

    def _build_bias_lookup(
        self,
        hist_group: HistGroup2D,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        num_windows = hist_group.num_windows
        bias_lookup = np.empty(
            (self.config.num_bins_x, self.config.num_bins_y, num_windows), dtype=dtype
        )
        bias_locations = np.asarray(hist_group.bias_locations, dtype=dtype)
        spring_x = np.asarray(hist_group.spring_x, dtype=dtype)
        spring_y = np.asarray(hist_group.spring_y, dtype=dtype)
        temperatures = np.asarray(hist_group.temperatures, dtype=dtype)
        chunk_size = self.config.bias_chunk_size or num_windows

        for start in range(0, num_windows, chunk_size):
            end = min(start + chunk_size, num_windows)
            bx = bias_locations[start:end, 0][:, None, None]
            by = bias_locations[start:end, 1][:, None, None]
            dx = x_grid[None, :, :] - bx
            dy = y_grid[None, :, :] - by

            if self.config.periodic_x:
                dx = np.abs(dx)
                dx = np.where(dx > self.config.period_x / 2.0, dx - self.config.period_x, dx)
            if self.config.periodic_y:
                dy = np.abs(dy)
                dy = np.where(dy > self.config.period_y / 2.0, dy - self.config.period_y, dy)

            bias_energy = 0.5 * (
                spring_x[start:end, None, None] * dx * dx
                + spring_y[start:end, None, None] * dy * dy
            )
            bias_energy /= temperatures[start:end, None, None]
            bias_lookup[:, :, start:end] = np.exp(-bias_energy.transpose(1, 2, 0)).astype(
                dtype, copy=False
            )

        return bias_lookup

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

        dtype = np.float32 if self.config.use_float32 else np.float64
        x_grid, y_grid = self._coordinate_grids(dtype)

        num_lookup = np.zeros(
            (self.config.num_bins_x, self.config.num_bins_y), dtype=dtype
        )
        for hist in hist_group.histograms:
            if hist.num_points == 0:
                continue
            x_slice = slice(hist.first_x, hist.last_x + 1)
            y_slice = slice(hist.first_y, hist.last_y + 1)
            num_lookup[x_slice, y_slice] += np.asarray(hist.data, dtype=dtype)

        bias_lookup = self._build_bias_lookup(hist_group, x_grid, y_grid, dtype)

        prob = np.zeros(
            (self.config.num_bins_x, self.config.num_bins_y), dtype=dtype
        )
        final_prob = np.zeros_like(prob)
        free_ene = np.zeros_like(prob)

        iteration = 0
        first = True
        converged = False
        while not converged or first:
            first = False
            self.save_free(hist_group)
            self.wham_iteration(hist_group, prob, have_energy, self.config.use_mask, mask, bias_lookup, num_lookup)
            iteration += 1

            epsilon = float(np.finfo(dtype).tiny)
            hist_group.free_energies = [max(val, epsilon) for val in hist_group.free_energies]
            hist_group.previous_free_energies = [max(val, epsilon) for val in hist_group.previous_free_energies]
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

        free_ene = np.asarray(self.calc_free(prob, self.config.use_mask, mask), dtype=prob.dtype)
        total = float(np.sum(prob))
        if total > 0:
            prob /= total
        final_prob[:] = prob

        if self.config.use_mask and mask is not None:
            for i in range(self.config.num_bins_x):
                for j in range(self.config.num_bins_y):
                    if not mask[i][j]:
                        free_ene[i][j] = MASKED

        prob_std = np.zeros_like(final_prob)
        free_std = np.zeros_like(final_prob)
        window_std = np.zeros(hist_group.num_windows, dtype=final_prob.dtype)

        if self.config.num_mc_runs > 0:
            base_hist_group = _clone_hist_group(hist_group)
            seed_value = self.config.mc_seed if self.config.mc_seed is not None else 1
            seed_sequence = np.random.SeedSequence(seed_value)
            seeds = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(self.config.num_mc_runs)]

            workers = self.config.mc_workers
            if workers is None:
                workers = os.cpu_count() or 1
            workers = max(1, min(workers, self.config.num_mc_runs))

            results: list[BootstrapResult | None] = [None for _ in range(self.config.num_mc_runs)]
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        _run_bootstrap_trial,
                        i,
                        self.config,
                        base_hist_group,
                        have_energy,
                        mask,
                        self.config.use_mask,
                        seeds[i],
                    ): i
                    for i in range(self.config.num_mc_runs)
                }
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()

            ave_prob = np.zeros_like(final_prob)
            ave_prob2 = np.zeros_like(final_prob)
            ave_free = np.zeros_like(final_prob)
            ave_free2 = np.zeros_like(final_prob)
            free_counts = np.zeros_like(final_prob, dtype=np.int32)
            ave_window = np.zeros(hist_group.num_windows, dtype=final_prob.dtype)
            ave_window2 = np.zeros(hist_group.num_windows, dtype=final_prob.dtype)

            for result in results:
                assert result is not None
                probabilities = np.asarray(result.probabilities, dtype=final_prob.dtype)
                free_surface = np.asarray(result.free_energy, dtype=final_prob.dtype)
                free_biases = np.asarray(result.free_energies, dtype=final_prob.dtype)

                ave_prob += probabilities
                ave_prob2 += probabilities * probabilities
                free_mask = np.isfinite(free_surface)
                ave_free += np.where(free_mask, free_surface, 0)
                ave_free2 += np.where(free_mask, free_surface * free_surface, 0)
                free_counts += free_mask.astype(np.int32)
                ave_window += free_biases
                ave_window2 += free_biases * free_biases

            count = float(self.config.num_mc_runs)
            ave_prob /= count
            ave_prob2 /= count
            prob_std = np.sqrt(np.maximum(ave_prob2 - ave_prob * ave_prob, 0.0))

            valid_counts = np.maximum(free_counts, 1)
            ave_free = ave_free / valid_counts
            ave_free2 = ave_free2 / valid_counts
            free_std = np.sqrt(np.maximum(ave_free2 - ave_free * ave_free, 0.0))

            ave_window /= count
            ave_window2 /= count
            window_std = np.sqrt(np.maximum(ave_window2 - ave_window * ave_window, 0.0))
        else:
            print("# No MC error analysis requested")

        header = "#X\t\tY\t\tFree\t\tPro"
        if self.config.num_mc_runs > 0:
            header += "\t\tFreeErr\t\tProErr"

        with self.config.freefile_path.open("w", encoding="utf-8") as freefile:
            freefile.write(f"{header}\n")
            for i in range(-self.config.numpad, 0):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[self.config.num_bins_x + i][self.config.num_bins_y + j],
                            final_prob[self.config.num_bins_x + i][self.config.num_bins_y + j],
                            free_std[self.config.num_bins_x + i][self.config.num_bins_y + j],
                            prob_std[self.config.num_bins_x + i][self.config.num_bins_y + j],
                        )
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[self.config.num_bins_x + i][j],
                            final_prob[self.config.num_bins_x + i][j],
                            free_std[self.config.num_bins_x + i][j],
                            prob_std[self.config.num_bins_x + i][j],
                        )
                    )
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[self.config.num_bins_x + i][j - self.config.num_bins_y],
                            final_prob[self.config.num_bins_x + i][j - self.config.num_bins_y],
                            free_std[self.config.num_bins_x + i][j - self.config.num_bins_y],
                            prob_std[self.config.num_bins_x + i][j - self.config.num_bins_y],
                        )
                    )
            for i in range(self.config.num_bins_x):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i][self.config.num_bins_y + j],
                            final_prob[i][self.config.num_bins_y + j],
                            free_std[i][self.config.num_bins_y + j],
                            prob_std[i][self.config.num_bins_y + j],
                        )
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i][j],
                            final_prob[i][j],
                            free_std[i][j],
                            prob_std[i][j],
                        )
                    )
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i][j - self.config.num_bins_y],
                            final_prob[i][j - self.config.num_bins_y],
                            free_std[i][j - self.config.num_bins_y],
                            prob_std[i][j - self.config.num_bins_y],
                        )
                    )
            for i in range(self.config.num_bins_x, self.config.num_bins_x + self.config.numpad):
                for j in range(-self.config.numpad, 0):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i - self.config.num_bins_x][self.config.num_bins_y + j],
                            final_prob[i - self.config.num_bins_x][self.config.num_bins_y + j],
                            free_std[i - self.config.num_bins_x][self.config.num_bins_y + j],
                            prob_std[i - self.config.num_bins_x][self.config.num_bins_y + j],
                        )
                    )
                for j in range(self.config.num_bins_y):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i - self.config.num_bins_x][j],
                            final_prob[i - self.config.num_bins_x][j],
                            free_std[i - self.config.num_bins_x][j],
                            prob_std[i - self.config.num_bins_x][j],
                        )
                    )
                for j in range(self.config.num_bins_y, self.config.num_bins_y + self.config.numpad):
                    coor = self.calc_coor(i, j)
                    freefile.write(
                        self._format_free_line(
                            coor,
                            free_ene[i - self.config.num_bins_x][j - self.config.num_bins_y],
                            final_prob[i - self.config.num_bins_x][j - self.config.num_bins_y],
                            free_std[i - self.config.num_bins_x][j - self.config.num_bins_y],
                            prob_std[i - self.config.num_bins_x][j - self.config.num_bins_y],
                        )
                    )

        if self.config.num_mc_runs > 0 and self.config.freefile_error_path is not None:
            with self.config.freefile_error_path.open("w", encoding="utf-8") as errorfile:
                errorfile.write("#Window\t\tFreeErr\n")
                for i in range(hist_group.num_windows):
                    errorfile.write(f"#{i}\t{window_std[i]}\n")

    def _format_free_line(
        self,
        coor: Tuple[float, float],
        free_val: float,
        prob_val: float,
        free_err: float,
        prob_err: float,
    ) -> str:
        if self.config.num_mc_runs > 0:
            return f"{coor[0]}\t{coor[1]}\t{free_val}\t{prob_val}\t{free_err}\t{prob_err}\n"
        return f"{coor[0]}\t{coor[1]}\t{free_val}\t{prob_val}\n"


@dataclass
class BootstrapResult:
    trial_index: int
    iterations: int
    probabilities: list[list[float]]
    free_energy: list[list[float]]
    free_energies: list[float]
    too_many_iterations: bool = False


def _clone_hist_group(base: HistGroup2D) -> HistGroup2D:
    histograms = [
        Histogram2D(
            first_x=hist.first_x,
            last_x=hist.last_x,
            first_y=hist.first_y,
            last_y=hist.last_y,
            num_points=hist.num_points,
            num_mc_samples=hist.num_mc_samples,
            data=[list(row) for row in hist.data],
            cumulative=list(hist.cumulative),
        )
        for hist in base.histograms
    ]
    return HistGroup2D(
        num_windows=base.num_windows,
        bias_locations=[list(loc) for loc in base.bias_locations],
        spring_x=list(base.spring_x),
        spring_y=list(base.spring_y),
        free_energies=[1.0 for _ in base.free_energies],
        previous_free_energies=[1.0 for _ in base.previous_free_energies],
        temperatures=list(base.temperatures),
        partitions=list(base.partitions),
        histograms=histograms,
    )


def _run_bootstrap_trial(
    trial_index: int,
    config: Wham2DConfig,
    base_hist_group: HistGroup2D,
    have_energy: bool,
    mask: list[list[int]] | None,
    use_mask: bool,
    seed: int,
) -> BootstrapResult:
    wham = Wham2D(config)
    hist_group = _clone_hist_group(base_hist_group)
    generator = np.random.default_rng(seed)
    dtype = np.float32 if config.use_float32 else np.float64

    prob = np.zeros((config.num_bins_x, config.num_bins_y), dtype=dtype)

    for j in range(hist_group.num_windows):
        hist = hist_group.histograms[j]
        num_x = hist.last_x - hist.first_x + 1
        num_y = hist.last_y - hist.first_y + 1
        wham.mk_new_hist(
            hist.cumulative,
            hist.data,
            hist.first_x,
            hist.last_x,
            hist.first_y,
            hist.last_y,
            hist.num_mc_samples,
            generator,
        )
        hist_group.previous_free_energies[j] = 1.0
        hist_group.free_energies[j] = 1.0

    x_grid, y_grid = wham._coordinate_grids(dtype)
    bias_lookup = wham._build_bias_lookup(hist_group, x_grid, y_grid, dtype)

    num_lookup = np.zeros((config.num_bins_x, config.num_bins_y), dtype=dtype)
    for hist in hist_group.histograms:
        if hist.num_points == 0:
            continue
        x_slice = slice(hist.first_x, hist.last_x + 1)
        y_slice = slice(hist.first_y, hist.last_y + 1)
        num_lookup[x_slice, y_slice] += np.asarray(hist.data, dtype=dtype)

    iteration = 0
    first = True
    converged = False
    while not converged or first:
        first = False
        wham.save_free(hist_group)
        wham.wham_iteration(hist_group, prob, have_energy, use_mask, mask, bias_lookup, num_lookup)
        iteration += 1

        epsilon = float(np.finfo(dtype).tiny)
        hist_group.free_energies = [max(val, epsilon) for val in hist_group.free_energies]
        hist_group.previous_free_energies = [max(val, epsilon) for val in hist_group.previous_free_energies]
        logged_current = [
            hist_group.temperatures[i] * math.log(hist_group.free_energies[i]) for i in range(hist_group.num_windows)
        ]
        logged_previous = [
            hist_group.temperatures[i] * math.log(hist_group.previous_free_energies[i]) for i in range(hist_group.num_windows)
        ]
        converged = wham.is_converged(hist_group, logged_current, logged_previous)
        if iteration >= 100000:
            break

    free_energy = wham.calc_free(prob, use_mask, mask)
    total = float(np.sum(prob))
    if total > 0:
        prob = prob / total

    free_biases = [val - hist_group.free_energies[0] for val in hist_group.free_energies]

    return BootstrapResult(
        trial_index=trial_index,
        iterations=iteration,
        probabilities=prob.tolist(),
        free_energy=free_energy,
        free_energies=free_biases,
        too_many_iterations=iteration >= 100000,
    )

def parse_periodic(config: dict, suffix: str) -> Tuple[bool, float]:
    periodic = bool(config.get(f"periodic_{suffix}", False))
    if not periodic:
        return False, 0.0
    period_value = config.get(f"period_{suffix}", DEGREES)
    if isinstance(period_value, str) and period_value.lower() == "pi":
        period = RADIANS
    else:
        period = float(period_value)
    print(f"#Turning on periodicity for {suffix.upper()} with period = {period}")
    return True, period


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


def build_config(yaml_path: Path) -> Wham2DConfig:
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("YAML configuration must define a mapping of parameters")

    k_B = parse_units(config.get("units"))
    periodic_x, period_x = parse_periodic(config, "x")
    periodic_y, period_y = parse_periodic(config, "y")

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
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")

    return Wham2DConfig(
        hist_min_x=float(config["hist_min_x"]),
        hist_max_x=float(config["hist_max_x"]),
        num_bins_x=int(config["num_bins_x"]),
        hist_min_y=float(config["hist_min_y"]),
        hist_max_y=float(config["hist_max_y"]),
        num_bins_y=int(config["num_bins_y"]),
        tolerance=float(config["tolerance"]),
        temperature=float(config["temperature"]),
        numpad=int(config["numpad"]),
        metadata_path=Path(config["metadata_file"]),
        freefile_path=Path(config["freefile"]),
        use_mask=bool(config["use_mask"]),
        periodic_x=periodic_x,
        period_x=period_x,
        periodic_y=periodic_y,
        period_y=period_y,
        k_B=k_B,
        use_float32=bool(config.get("use_float32", False)),
        bias_chunk_size=(int(config["bias_chunk_size"]) if "bias_chunk_size" in config else None),
        num_mc_runs=int(config.get("num_mc_runs", 0)),
        mc_seed=(int(config["mc_seed"]) if "mc_seed" in config else None),
        mc_workers=(int(config["mc_workers"]) if "mc_workers" in config else None),
        freefile_error_path=(Path(config["freefile_error"]) if "freefile_error" in config else None),
    )


def main(argv: List[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError("wham-2d now expects a single argument: path to a YAML configuration file")
    yaml_path = Path(args[0])
    print(f"# Loading configuration from {yaml_path}")
    config = build_config(yaml_path)
    wham = Wham2D(config)
    wham.run()


if __name__ == "__main__":
    main()
