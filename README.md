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
