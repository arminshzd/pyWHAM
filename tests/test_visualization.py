import matplotlib
from pathlib import Path

import numpy as np

matplotlib.use("Agg")

from pywham.visualization import save_free_energy_plots


ROOT = Path(__file__).resolve().parent.parent


def test_save_free_energy_plots_1d(tmp_path):
    freefile = ROOT / "examples" / "output_1D" / "output.free"

    outputs = save_free_energy_plots(freefile, output_dir=tmp_path)

    assert "free_energy" in outputs
    out_path = outputs["free_energy"]
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_free_energy_plots_2d(tmp_path):
    freefile = ROOT / "examples" / "output_2d" / "output.free"

    outputs = save_free_energy_plots(freefile, output_dir=tmp_path)

    assert "free_energy" in outputs
    out_path = outputs["free_energy"]
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_free_energy_plots_caps_free_energy_1d(monkeypatch, tmp_path):
    freefile = tmp_path / "custom_1d.free"
    freefile.write_text("\n".join(["0 1 0 0 0", "1 10.5 0 0 0"]))

    captured = {}

    def fake_plot_free_energy_1d(x, free_energy, errors, *, output_path=None, show=False):
        captured["free_energy"] = np.asarray(list(free_energy))
        return None, None

    monkeypatch.setattr("pywham.visualization.plot_free_energy_1d", fake_plot_free_energy_1d)

    save_free_energy_plots(freefile, output_dir=tmp_path, max_F=10)

    assert np.isinf(captured["free_energy"][1])


def test_save_free_energy_plots_caps_free_energy_2d(monkeypatch, tmp_path):
    freefile = tmp_path / "custom_2d.free"
    freefile.write_text("\n".join(["0 0 1 0", "1 0 11 0"]))

    captured = {}

    def fake_plot_free_energy_2d(x, y, free_energy, *, levels=15, cmap="viridis", output_path=None, show=False):
        captured["free_energy"] = np.asarray(free_energy)
        return (None, None), (None, None)

    monkeypatch.setattr("pywham.visualization.plot_free_energy_2d", fake_plot_free_energy_2d)

    save_free_energy_plots(freefile, output_dir=tmp_path, max_F=10)

    assert np.isinf(captured["free_energy"].max())
