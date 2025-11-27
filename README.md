# Python implementation of WHAM

This is a python implementation of the Weighted Histogram Analysis Method (WHAM). The logic of this package is adapted from [Prof. Alan Grossfield's WHAM](http://membrane.urmc.rochester.edu/?page_id=126) code with a few improvement and QoL features.

## Usage

The command line interfaces now accept a single argument: the path to a YAML file
describing all runtime parameters. Examples for the 1D and 2D solvers are shown
below.

### 1D configuration

```yaml
# wham-config.yml
hist_min: -3.14
hist_max: 3.14
num_bins: 200
tolerance: 1e-5
temperature: 300.0
numpad: 0
metadata_file: metadata.txt
freefile: output.free
# optional parameters
units: real               # default uses built-in k_B
periodic: true            # enable periodic coordinates
period: pi                # "pi" uses 2*pi radians; numeric values accepted
num_mc_runs: 0
mc_seed: -12345
```

Key reference for the 1D configuration:

- `hist_min` / `hist_max` (required): Lower/upper coordinate bounds for the histogram axis; values outside this range are ignored. Units match the raw data.
- `num_bins` (required): Number of bins spanning `[hist_min, hist_max]`; sets the bin width used throughout the WHAM iteration.
- `tolerance` (required): Convergence cutoff applied to window free energies; iteration stops when the absolute per-window change is below this value.
- `temperature` (required): Simulation temperature used with Boltzmann constant `k_B` (see `units`) to form `kT` for weighting energies.
- `numpad` (required): Number of bins to duplicate from each edge when writing the output `.free` file; use `0` to omit padding or a positive integer to wrap edge values for easier periodic plotting.
- `metadata_file` (required): Path to the window metadata file.
- `freefile` (required): Destination path for the WHAM output (probabilities, free energies, and optional uncertainties).
- `units` (optional, default uses `k_B = 0.0019829237`): Selects a preset Boltzmann constant (e.g., `real`, `lj`, `metal`, `si`, `cgs`, `electron`, `micro`, `nano`, `default`).
- `periodic` / `period` (optional): Enable periodic coordinates with `periodic: true`; omit or set `periodic: false` for non-periodic systems. When enabled, `period` defaults to `360.0` (degrees). Supplying `period: pi` switches to a `2π` radian period, and numeric values are accepted for custom periods.
- `num_mc_runs` (optional, default `0`): Number of bootstrap trials. Values greater than zero trigger resampling to estimate uncertainties; the `+/-` columns in the `.free` file then report standard deviations of probability and free energy across trials. With `0`, bootstrap is skipped and uncertainty columns remain zero.
- `mc_seed` (optional): Seed for bootstrap resampling; positive seeds are negated internally to mirror the original WHAM behavior. If omitted, a deterministic base seed of `1` is used.
- `mc_workers` (optional): Limits parallel bootstrap workers. When unset, the solver uses the CPU count (capped at `num_mc_runs`); values below `1` are coerced to `1`.
- `ingest_workers` (optional): Controls parallel ingestion of window files. Defaults to the CPU count (also honoring the `PYWHAM_DISABLE_PARALLEL` and `PYWHAM_INGEST_WORKERS` environment variables); values below `1` are coerced to `1`.

Run the solver with:

```bash
python -m pywham.wham1d wham-config.yml
```
or
```bash
python -m pywham.wham2d wham-config.yml
```

#### 1D metadata file

Each non-comment, non-empty line in `metadata_file` should contain:

```
<data_path> <bias_center> <spring_constant> [<correlation_time> [<temperature>]]
```

- `data_path`: path to the trajectory file for this window.
- `bias_center`: biased coordinate center.
- `spring_constant`: harmonic spring constant.
- `correlation_time` (optional, default `1.0`): divides the raw sample count to estimate independent samples.
- `temperature` (optional): if provided on every line, data files must include an energy column and the solver weights
  frames by `exp(-energy / kT)`; omit from all lines to use uniform weights.

Lines may start with `#` for comments. Mixing lines with and without temperature is rejected.

## Example data generation

Sample umbrella-sampling configuration files are included under `examples/`:

```bash
# 1D trajectories and WHAM reconstruction
python examples/langevin_umbrella.py --config examples/umbrella_config_1d.yaml
python -m pywham.wham1d examples/wham1d_config.yaml

# 2D trajectories on the Müller-Brown surface and WHAM2D reconstruction
python examples/langevin_umbrella_2d.py --config examples/umbrella_config_2d.yaml
python -m pywham.wham2d examples/wham2d_config.yaml
```

Running these commands will emit trajectories, metadata, and reconstructed free
energies under `examples/output_1D` and `examples/output_2d`, which can be fed
back into the library for visualization or further analysis.

## Visualization helpers

The `pywham.visualization` submodule provides quick plotting utilities that
uses the WHAM solver outputs.

- **Automatic plotting from a freefile**: ``save_free_energy_plots`` detects the
  1D vs 2D output format and writes PNGs next to the input file (or into a
  custom directory). For example, after running the 1D example above you can do:

  ```python
  from pywham import save_free_energy_plots

  outputs = save_free_energy_plots("examples/output_1D/output.free")
  print(outputs["free_energy"])
  ```

  This function expects the standard WHAM freefile layout (five columns for 1D:
  coordinate, free energy, uncertainty, probability, probability uncertainty;
  four columns for 2D: x, y, free energy, probability) and raises
  ``ValueError`` if the shape does not match the expected format.

- **Direct plotting**: ``plot_free_energy_1d`` and ``plot_free_energy_2d`` take
  coordinates and arrays directly and optionally save or show the figure:

  ```python
  import numpy as np
  from pywham import plot_free_energy_1d, plot_free_energy_2d

  # 1D profile with uncertainty band
  x = np.linspace(-3.14, 3.14, 200)
  free = np.loadtxt("examples/output_1D/output.free")[:, 1]
  err = np.loadtxt("examples/output_1D/output.free")[:, 2]
  plot_free_energy_1d(x, free, err, output_path="free_energy.png")

  # 2D surface from a grid-shaped array
  data = np.loadtxt("examples/output_2d/output-2d.free")
  x_vals = np.unique(data[:, 0])
  y_vals = np.unique(data[:, 1])
  free_surface = data[:, 2].reshape(len(x_vals), len(y_vals))
  plot_free_energy_2d(x_vals, y_vals, free_surface, output_path="free_energy_2d.png")
  ```

### 2D configuration

```yaml
# wham2d-config.yml
hist_min_x: -3.14
hist_max_x: 3.14
num_bins_x: 50
hist_min_y: -2.0
hist_max_y: 2.0
num_bins_y: 40
tolerance: 1e-5
temperature: 300.0
numpad: 0
metadata_file: metadata-2d.txt
freefile: output-2d.free
use_mask: false
units: real
periodic_x: true
period_x: pi
periodic_y: false
period_y: 0
```

Required keys mirror the arguments consumed by `build_config`: histogram bounds and bin counts for each axis (`hist_min_x`, `hist_max_x`, `num_bins_x`, `hist_min_y`, `hist_max_y`, `num_bins_y`), solver controls (`tolerance`, `temperature`, and `numpad` padding), file paths for input/output (`metadata_file`, `freefile`), and the boolean `use_mask`. Optional keys include energy-unit conversion (`units`), per-axis periodicity toggles and periods (`periodic_x`/`period_x`, `periodic_y`/`period_y`), memory/precision controls (`use_float32`, `bias_chunk_size`), bootstrap settings (`num_mc_runs`, `mc_seed`, `mc_workers`), and an optional error output `freefile_error` for window-level bootstrap uncertainties. When `use_mask` is true, bins not visited in the histogram are masked so they do not participate in normalization or smoothing; leave it false to treat unvisited bins as zero-probability padding. Enabling `use_float32` and/or a positive `bias_chunk_size` can significantly reduce memory footprint (and potentially speed up I/O-bound runs) at the cost of floating-point precision or small overhead per chunk; disable them to favor accuracy. Bootstrap outputs are appended as extra free-energy/probability columns in `freefile`, and if `freefile_error` is set a separate file listing per-window errors is written after the main surface output.

Run the solver with:

```bash
python -m pywham.wham2d wham2d-config.yml
```

#### 2D metadata file

Each non-comment, non-empty line in `metadata_file` should contain:

```
<data_path> <bias_center_x> <bias_center_y> <spring_x> <spring_y> [<correlation_time> [<temperature>]]
```

- `data_path`: path to the trajectory file for this window.
- `bias_center_x` / `bias_center_y`: biased coordinate centers for each dimension.
- `spring_x` / `spring_y`: harmonic spring constants along X and Y.
- `correlation_time` (optional, default `1.0`): divides the raw sample count to estimate independent samples.
- `temperature` (optional): if provided on every line, data files must include an energy column and the solver weights
  frames by `exp(-energy / kT)`; omit from all lines to use uniform weights.

Lines may start with `#` for comments. Mixing lines with and without temperature is rejected.
