import numpy as np
import pytest
import yaml
from pathlib import Path

import pywham.reweight as rw
from pywham.reweight import ReweightResult, Reweighter, load_aux_data


def _create_auxiliary_setup(base_dir: Path) -> Path:
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()

    umbrella_coords = np.array([0.25, 0.75, 1.25, 1.75])
    projection_coords = np.array([-0.5, -0.3, 0.2, 0.8])
    umbrella_traj = np.column_stack((np.arange(len(umbrella_coords)), umbrella_coords))
    proj_traj = np.column_stack((np.arange(len(projection_coords)), projection_coords))
    umb_path = base_dir / "traj_1.txt"
    proj_path = base_dir / "proj_1.txt"
    np.savetxt(umb_path, umbrella_traj, fmt="%.4f")
    np.savetxt(proj_path, proj_traj, fmt="%.4f")

    aux_dict = {
        "dim_umbrella": 1,
        "temperature": 1.0,
        "k_B": 1.0,
        "periodicity": [False],
        "periods": [None],
        "histogram_edges": [[0.0, 1.0, 2.0]],
        "windows": [
            {
                "trajectory": umb_path.name,
                "bias_center": [0.0],
                "spring_constants": [0.0],
                "num_samples": len(umbrella_traj),
            }
        ],
        "map_values": [1.0],
        "mh_samples": [[1.0], [2.0]],
        "projection_bins": [
            {"min": -1.0, "max": 1.0, "num_bins": 2},
        ],
        "projection_metadata": "projection_meta.txt",
        "output_dir": output_dir.name,
    }
    (base_dir / "projection_meta.txt").write_text(f"{proj_path.name}\n", encoding="utf-8")
    aux_path = base_dir / "aux_data.yaml"
    aux_path.write_text(yaml.safe_dump(aux_dict), encoding="utf-8")
    return aux_path


def test_normalization_helpers() -> None:
    with pytest.raises(ValueError):
        rw._normalize_vector(np.zeros(2))

    matrix = np.array([[0.0, 1.0], [0.0, 1.0]])
    normalized = rw._normalize_matrix_columns(matrix.copy())
    assert normalized[0, 0] == 0.0
    assert normalized[1, 0] == 0.0
    assert normalized[:, 1] == pytest.approx([0.5, 0.5])


def test_locate_bin_and_edges(tmp_path: Path) -> None:
    edges = [np.array([0.0, 1.0, 2.0])]
    assert rw._locate_bin(np.array([0.5]), edges) == (0,)
    assert rw._locate_bin(np.array([-0.2]), edges) is None

    empty_edges = tmp_path / "edges.txt"
    empty_edges.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        rw._load_bin_edges(empty_edges)


def test_reweight_result_write(tmp_path: Path) -> None:
    result = ReweightResult(
        bin_centers=[np.array([0.0, 1.0])],
        bin_widths=[np.array([0.5, 0.5])],
        probabilities_map=np.array([0.6, 0.4]),
        probability_density_map=np.array([0.6, 0.4]),
        free_energy_map=np.array([0.0, 1.0]),
        probabilities_mh=np.array([[0.6], [0.4]]),
        probability_density_mh=np.array([[0.6], [0.4]]),
        free_energy_mh=np.array([[0.0], [1.0]]),
    )
    output_file = tmp_path / "reweight.yaml"
    result.write(output_file)
    payload = yaml.safe_load(output_file.read_text(encoding="utf-8"))
    assert payload["map"]["probabilities"] == pytest.approx([0.6, 0.4])
    assert np.asarray(payload["mh_samples"]["probabilities"]) == pytest.approx(np.array([[0.6], [0.4]]))


def test_load_aux_data_success(tmp_path: Path) -> None:
    aux_path = _create_auxiliary_setup(tmp_path / "aux")
    aux = load_aux_data(aux_path)
    assert aux.dim_umbrella == 1
    assert aux.temperature == pytest.approx(1.0)
    assert len(aux.windows) == 1
    assert aux.windows[0].num_samples == 4
    assert aux.map_values.shape == (1,)
    assert aux.mh_samples.shape == (2, 1)
    assert len(aux.projection_hist_edges) == 1

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_aux_data(bad_yaml)


def test_reweighter_run_end_to_end(tmp_path: Path) -> None:
    aux_path = _create_auxiliary_setup(tmp_path / "end_to_end")
    aux = load_aux_data(aux_path)
    reweighter = Reweighter(aux)
    result = reweighter.run()

    assert np.allclose(result.bin_centers[0], [-0.5, 0.5])
    assert np.allclose(result.probabilities_map, [0.5, 0.5])
    assert np.allclose(result.probabilities_mh, [[0.5, 0.5], [0.5, 0.5]])

    output_file = aux.output_dir / "reweight_output.yaml"
    assert output_file.exists()
    payload = yaml.safe_load(output_file.read_text(encoding="utf-8"))
    assert payload["map"]["probabilities"] == pytest.approx([0.5, 0.5])
    assert np.asarray(payload["mh_samples"]["probabilities"]) == pytest.approx(np.array([[0.5, 0.5], [0.5, 0.5]]))


def test_reweighter_respects_positive_beta_free_energy(tmp_path: Path) -> None:
    base_dir = tmp_path / "weighted"
    base_dir.mkdir()
    output_dir = base_dir / "output"
    output_dir.mkdir()

    # Two windows with distinct free energies; the second window has F = ln 2
    umb1 = np.array([[0.0, -1.0], [1.0, -1.0]])
    umb2 = np.array([[0.0, 1.0], [1.0, 1.0]])
    proj1 = umb1.copy()
    proj2 = umb2.copy()
    umb1_path = base_dir / "traj_1.txt"
    umb2_path = base_dir / "traj_2.txt"
    proj1_path = base_dir / "proj_1.txt"
    proj2_path = base_dir / "proj_2.txt"
    np.savetxt(umb1_path, umb1, fmt="%.6f")
    np.savetxt(umb2_path, umb2, fmt="%.6f")
    np.savetxt(proj1_path, proj1, fmt="%.6f")
    np.savetxt(proj2_path, proj2, fmt="%.6f")

    map_values = [1.0, 0.5]  # exp(-beta F) with F = ln 2 for the second window
    mh_samples = [map_values]
    aux_dict = {
        "dim_umbrella": 1,
        "temperature": 1.0,
        "k_B": 1.0,
        "periodicity": [False],
        "periods": [None],
        "histogram_edges": [[-2.0, 0.0, 2.0]],
        "windows": [
            {
                "trajectory": umb1_path.name,
                "bias_center": [-1.0],
                "spring_constants": [1.0],
                "num_samples": umb1.shape[0],
            },
            {
                "trajectory": umb2_path.name,
                "bias_center": [1.0],
                "spring_constants": [1.0],
                "num_samples": umb2.shape[0],
            },
        ],
        "map_values": map_values,
        "mh_samples": mh_samples,
        "projection_hist_edges": [[-2.0, 0.0, 2.0]],
        "projection_metadata": "projection_meta.txt",
        "output_dir": output_dir.name,
    }

    (base_dir / "projection_meta.txt").write_text(
        f"{proj1_path.name}\n{proj2_path.name}\n", encoding="utf-8"
    )
    aux_path = base_dir / "aux_data.yaml"
    aux_path.write_text(yaml.safe_dump(aux_dict), encoding="utf-8")

    aux = load_aux_data(aux_path)
    reweighter = Reweighter(aux)
    result = reweighter.run()

    expected = [0.6269, 0.3731]
    assert result.probabilities_map == pytest.approx(expected, rel=2e-3)
    assert result.probabilities_mh[:, 0] == pytest.approx(expected, rel=2e-3)


def test_reweight_cli_main(tmp_path: Path) -> None:
    aux_path = _create_auxiliary_setup(tmp_path / "cli")
    rw.main([str(aux_path)])
    aux = load_aux_data(aux_path)
    assert (aux.output_dir / "reweight_output.yaml").exists()
