# Python implementation of WHAM

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
