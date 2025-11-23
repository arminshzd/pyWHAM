"""
2D Langevin umbrella sampling demo for pyWHAM using the Müller-Brown surface.

Dependencies
------------
* NumPy
* PyYAML

Configurable parameters (YAML)
------------------------------
* `centers`: list of `[x, y]` umbrella centers
* `spring_constant_x`: harmonic spring constant along X
* `spring_constant_y`: harmonic spring constant along Y
* `temperature`: thermodynamic temperature (k_B = 1)
* `friction`: friction coefficient
* `time_step`: integrator time step
* `steps`: total simulation steps
* `stride`: output stride for saved frames
* `correlation_time`: correlation time reported to WHAM2D
* `output_dir`: directory for trajectory and metadata files
* `metadata_name`: output metadata filename
* `seed`: RNG seed

Example usage
-------------
```
python examples/langevin_umbrella_2d.py --config examples/umbrella_config_2d.yaml

python -m pywham.wham2d examples/wham2d_config.yaml
```

Example WHAM2D configuration (`examples/wham2d_config.yaml`):
```
hist_min_x: -1.5
hist_max_x: 1.2
num_bins_x: 120
hist_min_y: -0.2
hist_max_y: 2.0
num_bins_y: 120
tolerance: 1e-5
temperature: 1.0
numpad: 0
metadata_file: examples/output_2d/umbrella_metadata_2d.txt
freefile: examples/output_2d/free_energy_2d.dat
use_mask: false
periodic_x: false
period_x: 0
periodic_y: false
period_y: 0
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import yaml

k_B = 1.0


def potential(x: float, y: float) -> float:
    """Müller-Brown potential energy surface."""

    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])
    A = np.array([-200.0, -100.0, -170.0, 15.0])

    dx = x - x0
    dy = y - y0
    exponent = a * dx**2 + b * dx * dy + c * dy**2
    return float(np.sum(A * np.exp(exponent)))


def potential_force(x: float, y: float) -> Tuple[float, float]:
    """Force corresponding to the Müller-Brown potential (-∇U)."""

    a = np.array([-1.0, -1.0, -6.5, 0.7])
    b = np.array([0.0, 0.0, 11.0, 0.6])
    c = np.array([-10.0, -10.0, -6.5, 0.7])
    x0 = np.array([1.0, 0.0, -0.5, -1.0])
    y0 = np.array([0.0, 0.5, 1.5, 1.0])
    A = np.array([-200.0, -100.0, -170.0, 15.0])

    dx = x - x0
    dy = y - y0
    exponent = a * dx**2 + b * dx * dy + c * dy**2
    prefactor = A * np.exp(exponent)

    d_dx = np.sum(prefactor * (2.0 * a * dx + b * dy))
    d_dy = np.sum(prefactor * (b * dx + 2.0 * c * dy))
    return -float(d_dx), -float(d_dy)


def umbrella_energy(
    x: float, y: float, center: Sequence[float], spring_constant_x: float, spring_constant_y: float
) -> float:
    """Harmonic umbrella potential energy."""

    dx = x - center[0]
    dy = y - center[1]
    return 0.5 * spring_constant_x * dx * dx + 0.5 * spring_constant_y * dy * dy


def umbrella_force(
    x: float, y: float, center: Sequence[float], spring_constant_x: float, spring_constant_y: float
) -> Tuple[float, float]:
    """Force from the harmonic umbrella (-∇U)."""

    return (
        -spring_constant_x * (x - center[0]),
        -spring_constant_y * (y - center[1]),
    )


def langevin_integrator(
    x0: float,
    y0: float,
    center: Sequence[float],
    spring_constant_x: float,
    spring_constant_y: float,
    *,
    temperature: float,
    friction: float,
    time_step: float,
    steps: int,
    stride: int,
    random_state: np.random.Generator,
) -> List[Tuple[int, float, float, float]]:
    r"""Run overdamped Langevin dynamics with 2D umbrellas.

    The update follows the overdamped equation ``dx = (F/γ) dt + \sqrt{2 k_B T / γ} dW``
    with independent noise in each dimension.
    """

    x = float(x0)
    y = float(y0)
    diffusion = (2.0 * k_B * temperature) / friction
    noise_scale = np.sqrt(diffusion * time_step)

    records: List[Tuple[int, float, float, float]] = []
    for step in range(steps):
        fx, fy = potential_force(x, y)
        ux, uy = umbrella_force(x, y, center, spring_constant_x, spring_constant_y)
        deterministic_x = (fx + ux) / friction * time_step
        deterministic_y = (fy + uy) / friction * time_step
        stochastic_x = noise_scale * random_state.normal()
        stochastic_y = noise_scale * random_state.normal()
        x += deterministic_x + stochastic_x
        y += deterministic_y + stochastic_y

        if step % stride == 0:
            bias_energy = umbrella_energy(x, y, center, spring_constant_x, spring_constant_y)
            records.append((step, x, y, bias_energy))

    return records


def save_trajectory(path: Path, records: Iterable[Tuple[int, float, float, float]]) -> None:
    """Save <t_step> <x> <y> <bias_energy> lines for WHAM2D."""

    with path.open("w", encoding="utf-8") as handle:
        for step, x, y, bias_energy in records:
            handle.write(f"{step} {x:.8f} {y:.8f} {bias_energy:.8f}\n")


def write_metadata_line(
    handle,
    traj_path: Path,
    center: Sequence[float],
    spring_constant_x: float,
    spring_constant_y: float,
    correlation_time: float,
    temperature: float,
) -> None:
    """Emit a metadata line matching the README's 2D format."""

    handle.write(
        f"{traj_path} {center[0]:.6f} {center[1]:.6f} {spring_constant_x:.6f} {spring_constant_y:.6f} "
        f"{correlation_time:.6f} {temperature:.6f}\n"
    )


def parse_config() -> dict:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 2D umbrella trajectories with bias energies and metadata "
            "compatible with Wham2D.read_data(..., have_energy=True)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/umbrella_config_2d.yaml"),
        help="Path to a YAML configuration file.",
    )

    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    defaults = {
        "centers": [
            [-1.0, 0.0],
            [-1.0, 1.0],
            [-0.5, 0.5],
            [-0.5, 1.5],
            [0.0, 0.5],
            [0.0, 1.5],
        ],
        "spring_constant_x": 25.0,
        "spring_constant_y": 25.0,
        "temperature": 1.0,
        "friction": 1.0,
        "time_step": 0.001,
        "steps": 200_000,
        "stride": 100,
        "correlation_time": 10.0,
        "output_dir": "examples/output_2d",
        "metadata_name": "umbrella_metadata_2d.txt",
        "seed": 1234,
    }

    merged = {**defaults, **config}
    merged["output_dir"] = Path(merged["output_dir"])
    merged["centers"] = [list(pair) for pair in merged.get("centers", [])]
    merged["metadata_name"] = str(merged.get("metadata_name", defaults["metadata_name"]))
    return merged


def main() -> None:
    cfg = parse_config()
    rng = np.random.default_rng(cfg["seed"])

    output_dir: Path = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / cfg["metadata_name"]

    with metadata_path.open("w", encoding="utf-8") as meta_handle:
        for idx, center in enumerate(cfg["centers"]):
            traj_path = output_dir / f"traj2d_window_{idx:02d}.dat"
            trajectory = langevin_integrator(
                x0=center[0],
                y0=center[1],
                center=center,
                spring_constant_x=cfg["spring_constant_x"],
                spring_constant_y=cfg["spring_constant_y"],
                temperature=cfg["temperature"],
                friction=cfg["friction"],
                time_step=cfg["time_step"],
                steps=cfg["steps"],
                stride=cfg["stride"],
                random_state=rng,
            )
            save_trajectory(traj_path, trajectory)
            write_metadata_line(
                meta_handle,
                traj_path,
                center,
                cfg["spring_constant_x"],
                cfg["spring_constant_y"],
                cfg["correlation_time"],
                cfg["temperature"],
            )

    print(f"Generated 2D trajectories in: {output_dir.resolve()}")
    print(f"Metadata file: {metadata_path.resolve()}")
    print(
        "README: run `python -m pywham.wham2d examples/wham2d_config.yaml` where the YAML "
        "sets histogram bounds, bin counts, and metadata_file to "
        f"{metadata_path}` to perform WHAM2D analysis."
    )


if __name__ == "__main__":
    main()
