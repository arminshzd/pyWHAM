import yaml
import pytest
from pathlib import Path

from pywham.bwham import BayesWHAM, build_config


def _write_simple_metadata(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    data_file = base_dir / "traj.dat"
    data_file.write_text("0 0.5 0\n1 1.5 0\n", encoding="utf-8")
    metadata = base_dir / "metadata.txt"
    metadata.write_text("traj.dat 0.0 1.0 1.0 300\n", encoding="utf-8")
    return metadata


def test_build_config_resolves_paths_1d(tmp_path: Path) -> None:
    metadata_path = _write_simple_metadata(tmp_path / "data1d")
    config_path = tmp_path / "config1d.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hist_min": 0.0,
                "hist_max": 2.0,
                "num_bins": 2,
                "tolerance": 1e-6,
                "temperature": 300.0,
                "numpad": 0,
                "metadata_file": "data1d/metadata.txt",
                "freefile": "free/output.txt",
                "aux_data_file": "aux/aux.yaml",
            }
        ),
        encoding="utf-8",
    )

    config = build_config(config_path)

    metadata_resolved = metadata_path.resolve()
    base_dir = metadata_resolved.parent
    assert config.base_config.metadata_path == metadata_resolved
    assert config.base_config.freefile_path == (base_dir / "free/output.txt").resolve()
    assert config.base_config.aux_data_path == (base_dir / "aux/aux.yaml").resolve()


def test_build_config_resolves_paths_2d(tmp_path: Path) -> None:
    metadata_path = _write_simple_metadata(tmp_path / "data2d")
    config_path = tmp_path / "config2d.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hist_min_x": 0.0,
                "hist_max_x": 1.0,
                "num_bins_x": 2,
                "hist_min_y": 0.0,
                "hist_max_y": 1.0,
                "num_bins_y": 2,
                "tolerance": 1e-6,
                "temperature": 300.0,
                "numpad": 0,
                "metadata_file": "data2d/metadata.txt",
                "freefile": "free2d.txt",
                "use_mask": False,
                "aux_data_file": "aux/out.yaml",
            }
        ),
        encoding="utf-8",
    )

    config = build_config(config_path)

    metadata_resolved = metadata_path.resolve()
    base_dir = metadata_resolved.parent
    assert config.base_config.metadata_path == metadata_resolved
    assert config.base_config.freefile_path == (base_dir / "free2d.txt").resolve()
    assert config.base_config.aux_data_path == (base_dir / "aux/out.yaml").resolve()


def test_bwham_writes_aux_data(tmp_path: Path) -> None:
    metadata_path = _write_simple_metadata(tmp_path / "bayes")
    config_path = tmp_path / "bayes.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "hist_min": 0.0,
                "hist_max": 2.0,
                "num_bins": 2,
                "tolerance": 1e-6,
                "temperature": 300.0,
                "numpad": 0,
                "metadata_file": str(metadata_path.relative_to(config_path.parent)),
                "freefile": "free.txt",
                "aux_data_file": "aux/aux.yaml",
                "num_samples": 2,
                "burn_in": 0,
                "thinning": 1,
            }
        ),
        encoding="utf-8",
    )

    config = build_config(config_path)
    bwham_runner = BayesWHAM(config)
    bwham_runner.run()

    aux_path = config.base_config.aux_data_path
    assert aux_path is not None and aux_path.exists()

    aux = yaml.safe_load(aux_path.read_text(encoding="utf-8"))
    assert aux["metadata_file"] == str(config.base_config.metadata_path)
    assert aux["map_values"] == pytest.approx([0.0])
    assert len(aux["mh_samples"]) == 2
    assert aux["windows"][0]["trajectory"].endswith("traj.dat")

