"""
Langevin umbrella sampling demo for pyWHAM.

Dependencies
------------
* NumPy
* PyYAML

Configurable parameters (YAML)
------------------------------
* `centers`: list of umbrella centers
* `spring_constant`: harmonic spring constant
* `temperature`: thermodynamic temperature (k_B = 1)
* `friction`: friction coefficient
* `time_step`: integrator time step
* `steps`: total simulation steps
* `stride`: output stride for saved frames
* `sub_steps`: number of Euler-Maruyama sub-steps per saved integration step
* `correlation_time`: correlation time reported to WHAM
* `output_dir`: directory for trajectory and metadata files
* `metadata_name`: output metadata filename
* `seed`: RNG seed

Example usage
-------------
```
python examples/langevin_umbrella.py --config examples/umbrella_config_1d.yaml

python -m pywham.wham1d examples/wham1d_config.yaml
```

Example YAML configuration (`examples/umbrella_config_1d.yaml`):
```
centers: [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
spring_constant: 10.0
temperature: 1.0
friction: 1.0
time_step: 0.001
steps: 200000
stride: 100
output_dir: output_1D
metadata_name: umbrella_metadata.txt
```
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import yaml

k_B = 1.0


def potential(x: float) -> float:
    """Test potential used for the sampling."""

    return (
        np.exp(2.5 * x**32)
        - 1.0
        + 0.5 * x**2
        - 2.0 * np.exp(-1000.0 * x**2)
        - 1.75 * np.exp(-500.0 * (x - 0.5) ** 2)
    )


def potential_force(x: float) -> float:
    """Force corresponding to the test potential (-dU/dx)."""

    derivative = (
        80.0 * x**31 * np.exp(2.5 * x**32)
        + x
        + 4000.0 * x * np.exp(-1000.0 * x**2)
        + 1750.0 * (x - 0.5) * np.exp(-500.0 * (x - 0.5) ** 2)
    )
    return -derivative


def umbrella_energy(x: float, center: float, spring_constant: float) -> float:
    """Harmonic umbrella potential energy."""

    displacement = x - center
    return 0.5 * spring_constant * displacement * displacement


def umbrella_force(x: float, center: float, spring_constant: float) -> float:
    """Force from the harmonic umbrella (-dU/dx)."""

    return -spring_constant * (x - center)


def langevin_integrator(
    x0: float,
    center: float,
    spring_constant: float,
    *,
    temperature: float,
    friction: float,
    time_step: float,
    steps: int,
    stride: int,
    random_state: np.random.Generator,
    sub_steps: int = 10,
) -> List[Tuple[int, float, float]]:
    """Run overdamped Langevin dynamics with an umbrella restraint.

    The update follows the overdamped equation dx = (F/γ) dt + sqrt(2 k_B T / γ)
    dW and clips positions to [-1, 1] each step to keep the stiff test potential
    numerically stable.  The integration step is internally split into
    ``sub_steps`` smaller Euler–Maruyama updates (dt = time_step / sub_steps) to
    reduce discretization bias for the narrow wells in the test potential.
    """

    x = float(x0)
    if sub_steps <= 0:
        raise ValueError("sub_steps must be a positive integer")

    dt = time_step / float(sub_steps)
    diffusion = (2.0 * k_B * temperature) / friction
    noise_scale = np.sqrt(diffusion * dt)

    records: List[Tuple[int, float, float]] = []
    for step in range(steps):
        for _ in range(sub_steps):
            total_force = potential_force(x) + umbrella_force(
                x, center, spring_constant
            )
            deterministic = (total_force / friction) * dt
            stochastic = noise_scale * random_state.normal()
            x += deterministic + stochastic
            x = float(np.clip(x, -1.0, 1.0))

        if step % stride == 0:
            bias_energy = umbrella_energy(x, center, spring_constant)
            records.append((step, x, bias_energy))

    return records


def save_trajectory(path: Path, records: Iterable[Tuple[int, float, float]]) -> None:
    """Save <t_step> <x> <bias_energy> lines for WHAM consumption."""

    with path.open("w", encoding="utf-8") as handle:
        for step, position, bias_energy in records:
            handle.write(f"{step} {position:.8f} {bias_energy:.8f}\n")


def write_metadata_line(
    handle,
    traj_path: Path,
    center: float,
    spring_constant: float,
    correlation_time: float,
    temperature: float,
) -> None:
    """Emit a single metadata line matching Wham1D.read_metadata expectations."""

    handle.write(
        f"{traj_path} {center:.6f} {spring_constant:.6f} "
        f"{correlation_time:.6f} {temperature:.6f}\n"
    )


def parse_config() -> dict:
    parser = argparse.ArgumentParser(
        description=(
            "Generate umbrella-sampling trajectories that include bias energies "
            "and metadata compatible with Wham1D.read_data(..., have_energy=True)."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/umbrella_config.yaml"),
        help="Path to a YAML configuration file.",
    )

    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    defaults = {
        "centers": [-1.0, -0.5, 0.0, 0.5, 1.0],
        "spring_constant": 10.0,
        "temperature": 1.0,
        "friction": 1.0,
        "time_step": 0.001,
        "steps": 200000,
        "stride": 100,
        "sub_steps": 10,
        "correlation_time": 10.0,
        "output_dir": "examples/output",
        "metadata_name": "umbrella_metadata.txt",
        "seed": 1234,
    }

    merged = {**defaults, **config}
    merged["output_dir"] = Path(merged["output_dir"])
    merged["centers"] = list(merged.get("centers", []))
    merged["sub_steps"] = int(merged.get("sub_steps", 1))
    merged["metadata_name"] = str(merged["metadata_name"])
    return merged


def main() -> None:
    cfg = parse_config()
    rng = np.random.default_rng(cfg["seed"])

    output_dir: Path = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / cfg["metadata_name"]

    with metadata_path.open("w", encoding="utf-8") as meta_handle:
        for idx, center in enumerate(cfg["centers"]):
            traj_path = output_dir / f"traj_window_{idx:02d}.dat"
            trajectory = langevin_integrator(
                x0=center,
                center=center,
                spring_constant=cfg["spring_constant"],
                temperature=cfg["temperature"],
                friction=cfg["friction"],
                time_step=cfg["time_step"],
                steps=cfg["steps"],
                stride=cfg["stride"],
                random_state=rng,
                sub_steps=cfg["sub_steps"],
            )
            save_trajectory(traj_path, trajectory)
            write_metadata_line(
                meta_handle,
                traj_path,
                center,
                cfg["spring_constant"],
                cfg["correlation_time"],
                cfg["temperature"],
            )

    print(f"Generated trajectories in: {output_dir.resolve()}")
    print(f"Metadata file: {metadata_path.resolve()}")
    print(
        "README: run `python -m pywham.wham1d {metadata_path} --have-energy "
        "--hist-min -3.0 --hist-max 3.0 --bin-width 0.02 --temperature "
        f"{cfg['temperature']}` to perform WHAM analysis."
    )


if __name__ == "__main__":
    main()
