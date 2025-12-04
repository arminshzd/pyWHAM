# Python implementation of WHAM

This is a python implementation of the Weighted Histogram Analysis Method (WHAM). The logic of this package is adapted from Prof. Alan [Grossfield's WHAM](http://membrane.urmc.rochester.edu/?page_id=126) code with a few improvement and QoL features.

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
aux_data_file: aux_data.yaml  # optional; emit inputs required for reweighting
```

Run the solver with:

```bash
wham wham-config.yml
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
All metadata-referenced trajectories (for WHAM or projection runs) may use relative
paths; they are resolved against the directory that contains the metadata file, so you
can keep each metadata bundle self-contained regardless of the working directory.

## Bayesian reweighting into auxiliary CVs

To reuse WHAM trajectories and metadata for Bayesian projections, add `aux_data_file`
to the 1D/2D configuration. When `wham` finishes it writes a self-contained
`aux_data.yaml` file describing the umbrella histograms, bias parameters, raw
trajectories, and the MAP/Monte-Carlo estimates of the partition-function ratios
`f_i = Z/Z_i`. A minimal 1D example looks like:

```yaml
dim_umbrella: 1
temperature: 300.0
k_B: 0.0019872067
periodicity: [false]
periods: [null]
histogram_edges:
  - [-3.14, -1.0, 1.0, 3.14]
windows:
  - trajectory: traj_1.txt
    bias_center: [0.0]
    spring_constants: [2.0]
    num_samples: 10000
map_values: [1.0]       # populated when aux_data_file is set; edit to override
mh_samples: []          # optional MH/bootstrapped samples, one row per draw
projection_hist_edges: null    # supply the projection bin edges or a file path
projection_bins:
  - {min: -3.14, max: 3.14, num_bins: 200}
projection_metadata: proj_metadata.txt   # mirrors WHAM metadata order; list one trajectory path per line
output_dir: reweight_output
```

Before invoking the reweighter, edit this file to provide the projection histogram
bins (using `{min, max, num_bins}` entries or an explicit edge list/file) and the synchronized
auxiliary trajectories collected during umbrella sampling. The `map_values` and
`mh_samples` entries are pre-filled by `wham`/`wham-2d` when `aux_data_file` is set,
so most workflows only need to supply the projection data and adjust `output_dir`.

Key sections:

- `windows`: copied directly from the WHAM metadata (trajectory paths, bias centers, springs, sample counts).
- `map_values`/`mh_samples`: MAP partition ratios and optional samples written by the solver; leave them alone unless you have external post-processing.
- `projection_bins`: per-dimension `{min, max, num_bins}` specs describing the auxiliary CV histogram (or set `projection_hist_edges` / `projection_hist_edges_file` if you need irregular spacing).
- `projection_metadata`: metadata file (same number and ordering of entries as the WHAM metadata) listing the projection trajectory paths; each line is resolved relative to the metadata file location (or absolute paths may be used).

Run the projector with:

```bash
reweight aux_data.yaml
```

The solver reconstructs the umbrella histogram grid, pre-computes the harmonic biases,
and iterates through all synchronized umbrella/projection samples. Each projected bin
accumulates MAP and MH weights using the unbiased probabilities. Results now land in a
single `reweight_output.yaml` file under `output_dir` containing:

- `bin_centers`, `bin_widths`
- `map`: probabilities, pdf, and free energy for the MAP estimate
- `mh_samples`: per-sample probabilities, pdf, and free energy arrays

These match the legacy Bayes script’s content while being easier to parse.

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
consume the WHAM solver outputs.

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
aux_data_file: aux_data_2d.yaml
```

Run the solver with:

```bash
wham-2d wham2d-config.yml
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
