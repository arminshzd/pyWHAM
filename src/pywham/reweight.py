"""Projection of WHAM umbrella sampling results into auxiliary collective variables."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import yaml


@dataclass
class WindowRecord:
    trajectory: Path
    bias_center: List[float]
    spring_constants: List[float]
    num_samples: int


@dataclass
class AuxData:
    dim_umbrella: int
    temperature: float
    k_B: float
    periodicity: List[bool]
    periods: List[float | None]
    histogram_edges: List[np.ndarray]
    windows: List[WindowRecord]
    map_values: np.ndarray
    mh_samples: np.ndarray
    projection_hist_edges: List[np.ndarray]
    projection_metadata: Path | None
    output_dir: Path


@dataclass
class ReweightResult:
    bin_centers: List[np.ndarray]
    bin_widths: List[np.ndarray]
    probabilities_map: np.ndarray
    probability_density_map: np.ndarray
    free_energy_map: np.ndarray
    probabilities_mh: np.ndarray
    probability_density_mh: np.ndarray
    free_energy_mh: np.ndarray

    def write(self, output_file: Path) -> None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bin_centers": [array.tolist() for array in self.bin_centers],
            "bin_widths": [array.tolist() for array in self.bin_widths],
            "map": {
                "probabilities": self.probabilities_map.tolist(),
                "pdf": self.probability_density_map.tolist(),
                "free_energy": self.free_energy_map.tolist(),
            },
            "mh_samples": {
                "probabilities": self.probabilities_mh.tolist(),
                "pdf": self.probability_density_mh.tolist(),
                "free_energy": self.free_energy_mh.tolist(),
            },
        }
        output_file.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class Reweighter:
    def __init__(self, aux_data: AuxData):
        self.aux = aux_data

    def run(self) -> ReweightResult:
        edges_umb = self.aux.histogram_edges
        dim = self.aux.dim_umbrella
        centers_umb = [0.5 * (edges[:-1] + edges[1:]) for edges in edges_umb]
        periods = np.array([period if period is not None else math.nan for period in self.aux.periods], dtype=float)

        umb_centers = np.asarray([record.bias_center for record in self.aux.windows], dtype=float)
        umb_forces = np.asarray([record.spring_constants for record in self.aux.windows], dtype=float)
        if umb_centers.shape[1] != dim or umb_forces.shape[1] != dim:
            raise ValueError("Mismatch between auxiliary data dimensionality and stored bias parameters")

        num_bins_umb = [len(edge) - 1 for edge in edges_umb]
        hist_centers = centers_umb
        hist_widths = [edges[1:] - edges[:-1] for edges in edges_umb]
        total_bins_umb = int(np.prod(num_bins_umb))

        N_i = np.asarray([record.num_samples for record in self.aux.windows], dtype=float)
        if np.any(N_i <= 0):
            raise ValueError("Auxiliary data windows must include positive num_samples")

        f_map = self.aux.map_values
        f_mh = self.aux.mh_samples

        traj = self._load_umbrella_trajectories()
        traj_proj = self._load_projection_trajectories()
        edges_proj = self.aux.projection_hist_edges
        num_bins_proj = [len(edge) - 1 for edge in edges_proj]
        total_bins_proj = int(np.prod(num_bins_proj))

        bin_volumes_proj = _bin_volumes([edges[1:] - edges[:-1] for edges in edges_proj])
        bias_lookup = self._bias_lookup(umb_centers, umb_forces, hist_centers, periods)

        if np.any(f_map <= 0):
            raise ValueError("map_values must be positive to compute exp(+beta F_i)")
        map_prefactors = np.divide(N_i, f_map)
        if f_mh.size > 0:
            if np.any(f_mh <= 0):
                raise ValueError("mh_samples must be positive to compute exp(+beta F_i)")
            mh_prefactors = np.divide(N_i[None, :], f_mh)
        else:
            mh_prefactors = f_mh

        p_map = np.zeros(total_bins_proj, dtype=float)
        p_mh = np.zeros((total_bins_proj, f_mh.shape[0]), dtype=float)

        for sim_index, (traj_i, traj_proj_i) in enumerate(zip(traj, traj_proj), start=1):
            for sample_idx in range(traj_i.shape[0]):
                proj_coords = traj_proj_i[sample_idx]
                umb_coords = traj_i[sample_idx]
                sub_proj = _locate_bin(proj_coords, edges_proj)
                if sub_proj is None:
                    continue
                sub_umb = _locate_bin(umb_coords, edges_umb)
                if sub_umb is None:
                    continue
                idx_proj = np.ravel_multi_index(sub_proj, tuple(num_bins_proj), order="C")
                idx_umb = np.ravel_multi_index(sub_umb, tuple(num_bins_umb), order="C")

                weights = bias_lookup[:, idx_umb]
                denom_map = float(np.dot(map_prefactors, weights))
                if denom_map == 0.0:
                    raise ZeroDivisionError("Encountered zero denominator while computing MAP weight")
                p_map[idx_proj] += 1.0 / denom_map

                if f_mh.size > 0:
                    denom_mh = mh_prefactors @ weights
                    if np.any(denom_mh == 0.0):
                        raise ZeroDivisionError("Encountered zero denominator while computing MH weights")
                    p_mh[idx_proj, :] += 1.0 / denom_mh

            print(f"# Completed projection for simulation {sim_index}")

        p_map = _normalize_vector(p_map)
        p_mh = _normalize_matrix_columns(p_mh)

        pdf_map = np.divide(p_map, bin_volumes_proj, out=np.zeros_like(p_map), where=bin_volumes_proj > 0)
        betaF_map = np.full_like(pdf_map, np.nan)
        mask = pdf_map > 0
        betaF_map[mask] = -np.log(pdf_map[mask])
        finite_betaF = betaF_map[np.isfinite(betaF_map)]
        if finite_betaF.size:
            betaF_map = betaF_map - np.mean(finite_betaF)

        pdf_mh = np.zeros_like(p_mh)
        betaF_mh = np.full_like(p_mh, np.nan)
        if f_mh.size > 0:
            for idx in range(total_bins_proj):
                volume = bin_volumes_proj[idx]
                if volume <= 0:
                    continue
                pdf_mh[idx, :] = p_mh[idx, :] / volume
            positive = pdf_mh > 0
            betaF_mh[positive] = -np.log(pdf_mh[positive])
            with np.errstate(all="ignore"):
                means = np.nanmean(betaF_mh, axis=0)
            betaF_mh = betaF_mh - means

        result = ReweightResult(
            bin_centers=[0.5 * (edges[:-1] + edges[1:]) for edges in edges_proj],
            bin_widths=[edges[1:] - edges[:-1] for edges in edges_proj],
            probabilities_map=p_map,
            probability_density_map=pdf_map,
            free_energy_map=betaF_map,
            probabilities_mh=p_mh,
            probability_density_mh=pdf_mh,
            free_energy_mh=betaF_mh,
        )
        result_path = self.aux.output_dir / "reweight_output.yaml"
        result.write(result_path)
        return result

    def _load_umbrella_trajectories(self) -> List[np.ndarray]:
        trajectories: List[np.ndarray] = []
        dim = self.aux.dim_umbrella
        for record in self.aux.windows:
            data = np.loadtxt(record.trajectory, dtype=float)
            data = np.atleast_2d(data)
            if data.shape[1] < dim + 1:
                raise ValueError(
                    f"{record.trajectory} must contain at least {dim + 1} columns "
                    "(an index/time column plus the umbrella coordinates)"
                )
            if record.num_samples > 0 and data.shape[0] != record.num_samples:
                raise ValueError(
                    f"{record.trajectory} contains {data.shape[0]} samples, expected {record.num_samples}"
                )
            trajectories.append(data[:, 1 : dim + 1])
        return trajectories

    def _load_projection_trajectories(self) -> List[np.ndarray]:
        metadata_path = self.aux.projection_metadata
        if metadata_path is None:
            raise ValueError("Auxiliary data must define projection_metadata")
        base_dir = metadata_path.parent
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        trajectories: List[np.ndarray] = []
        dim_proj = len(self.aux.projection_hist_edges)
        for entry in lines:
            entry = entry.strip()
            if not entry or entry.startswith("#"):
                continue
            parts = entry.split()
            path = Path(parts[0])
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            data = np.loadtxt(path, dtype=float)
            data = np.atleast_2d(data)
            if data.shape[1] < dim_proj + 1:
                raise ValueError(
                    f"{path} must contain at least {dim_proj + 1} columns "
                    "(an index/time column plus the projection coordinates)"
                )
            trajectories.append(data[:, 1 : dim_proj + 1])
        if len(trajectories) != len(self.aux.windows):
            raise ValueError("Number of projection trajectories does not match number of umbrella windows")
        return trajectories

    def _bias_lookup(
        self,
        centers: np.ndarray,
        forces: np.ndarray,
        bin_centers: Sequence[np.ndarray],
        periods: np.ndarray,
    ) -> np.ndarray:
        grid = _grid_from_vectors(bin_centers)
        delta = grid[None, :, :] - centers[:, None, :]
        for dim_index, periodic in enumerate(self.aux.periodicity):
            diff = delta[:, :, dim_index]
            if periodic:
                period = periods[dim_index]
                diff = np.abs(diff)
                delta[:, :, dim_index] = np.minimum.reduce(
                    [diff, np.abs(diff + period), np.abs(diff - period)]
                )
            else:
                delta[:, :, dim_index] = np.abs(diff)
        energy = 0.5 * np.sum(forces[:, None, :] * delta * delta, axis=2)
        beta = 1.0 / (self.aux.k_B * self.aux.temperature)
        return np.exp(-beta * energy)


def load_aux_data(yaml_path: Path) -> AuxData:
    if not yaml_path.exists():
        raise FileNotFoundError(f"Auxiliary data file not found: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Auxiliary data file must define a mapping of parameters")

    base_dir = yaml_path.parent
    dim = int(config.get("dim_umbrella", 0))
    if dim < 1:
        raise ValueError("dim_umbrella must be >= 1")

    histogram_edges_data = config.get("histogram_edges")
    if not histogram_edges_data:
        raise ValueError("histogram_edges section missing from auxiliary data")
    histogram_edges = [_to_float_array(row) for row in histogram_edges_data]
    if len(histogram_edges) != dim:
        raise ValueError("Number of histogram edge arrays does not match dim_umbrella")

    periodicity_raw = config.get("periodicity")
    if periodicity_raw is None or len(periodicity_raw) != dim:
        raise ValueError("periodicity must be provided for each umbrella dimension")
    periodicity = [bool(item) for item in periodicity_raw]
    periods_raw = config.get("periods") or [None for _ in range(dim)]
    if len(periods_raw) != dim:
        raise ValueError("periods list must match dim_umbrella length")
    periods = [float(value) if value is not None else None for value in periods_raw]

    windows_raw = config.get("windows")
    if not windows_raw:
        raise ValueError("windows list is required in auxiliary data")
    windows: List[WindowRecord] = []
    for entry in windows_raw:
        trajectory = _resolve_path(base_dir, entry.get("trajectory"))
        bias_center = _ensure_float_list(entry.get("bias_center"), dim, "bias_center")
        springs = _ensure_float_list(entry.get("spring_constants"), dim, "spring_constants")
        num_samples = int(entry.get("num_samples", 0))
        windows.append(WindowRecord(trajectory, bias_center, springs, num_samples))

    map_values_raw = config.get("map_values")
    if map_values_raw is None:
        raise ValueError("Auxiliary data is missing map_values")
    map_values = np.asarray(map_values_raw, dtype=float).reshape(-1)
    if map_values.size != len(windows):
        raise ValueError("map_values length must match number of windows")
    mh_raw = config.get("mh_samples", [])
    mh_samples = np.asarray(mh_raw, dtype=float)
    if mh_samples.size == 0:
        mh_samples = np.zeros((0, len(windows)), dtype=float)
    else:
        mh_samples = np.atleast_2d(mh_samples.astype(float))
        if mh_samples.shape[1] != len(windows):
            raise ValueError("mh_samples must have one column per window")

    projection_edges = _parse_projection_edges(config, base_dir)
    proj_meta_value = config.get("projection_metadata")
    if proj_meta_value is None:
        raise ValueError("projection_metadata must be provided in auxiliary data")
    projection_metadata = _resolve_path(base_dir, proj_meta_value)

    output_dir_value = config.get("output_dir")
    output_dir = _resolve_path(base_dir, output_dir_value) if output_dir_value else yaml_path.parent

    temperature = float(config.get("temperature", 0.0))
    k_B = float(config.get("k_B", 0.0))
    if temperature <= 0 or k_B <= 0:
        raise ValueError("temperature and k_B must be positive values")

    return AuxData(
        dim_umbrella=dim,
        temperature=temperature,
        k_B=k_B,
        periodicity=periodicity,
        periods=periods,
        histogram_edges=histogram_edges,
        windows=windows,
        map_values=map_values,
        mh_samples=mh_samples,
        projection_hist_edges=projection_edges,
        projection_metadata=projection_metadata,
        output_dir=output_dir,
    )


def _resolve_path(base_dir: Path, value: str | None) -> Path:
    if value is None:
        raise ValueError("Expected a path value in auxiliary data but found null")
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _ensure_float_list(values: Sequence[float] | None, dim: int, field: str) -> List[float]:
    if values is None or len(values) != dim:
        raise ValueError(f"{field} must provide {dim} entries per window")
    return [float(value) for value in values]


def _to_float_array(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("Histogram edge arrays must be one-dimensional and contain at least two entries")
    return array


def _parse_projection_edges(config: dict, base_dir: Path) -> List[np.ndarray]:
    ranges = config.get("projection_bins")
    if ranges:
        parsed: list[np.ndarray] = []
        for spec in ranges:
            try:
                hist_min = float(spec["min"])
                hist_max = float(spec["max"])
                num_bins = int(spec["num_bins"])
            except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError("Each projection bin spec must define min, max, and num_bins") from exc
            if num_bins < 1:
                raise ValueError("projection bin counts must be >= 1")
            parsed.append(np.linspace(hist_min, hist_max, num_bins + 1, dtype=float))
        return parsed
    edges = config.get("projection_hist_edges")
    if edges is not None:
        return [_to_float_array(row) for row in edges]
    edges_file = config.get("projection_hist_edges_file")
    if edges_file:
        return _load_bin_edges(_resolve_path(base_dir, edges_file))
    raise ValueError("Provide projection_bins, projection_hist_edges, or projection_hist_edges_file")


def _load_bin_edges(path: Path) -> List[np.ndarray]:
    edges: List[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            values = np.fromstring(line, sep=" ")
            if values.size == 0:
                continue
            edges.append(values)
    if not edges:
        raise ValueError(f"No bin edges found in {path}")
    return edges


def _grid_from_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*vectors, indexing="ij")
    flat = [axis.reshape(-1) for axis in mesh]
    return np.stack(flat, axis=-1)


def _bin_volumes(widths: Sequence[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*widths, indexing="ij")
    volume = np.ones(mesh[0].shape, dtype=float)
    for axis in mesh:
        volume *= axis
    return volume.reshape(-1)


def _locate_bin(values: np.ndarray, edges: Sequence[np.ndarray]) -> tuple[int, ...] | None:
    subs: List[int] = []
    for coord, axis_edges in zip(values, edges):
        idx = int(np.searchsorted(axis_edges, coord, side="right") - 1)
        if idx < 0 or idx >= len(axis_edges) - 1:
            return None
        subs.append(idx)
    return tuple(subs)


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("Probability vector is empty; cannot normalize")
    return values / total


def _normalize_matrix_columns(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    totals = matrix.sum(axis=0)
    for idx, total in enumerate(totals):
        if total > 0:
            matrix[:, idx] /= total
    return matrix


def main(argv: List[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise ValueError("reweight expects a single argument: path to aux_data YAML file")
    yaml_path = Path(args[0])
    print(f"# Loading auxiliary data from {yaml_path}")
    aux_data = load_aux_data(yaml_path)
    reweighter = Reweighter(aux_data)
    reweighter.run()


if __name__ == "__main__":
    main()
