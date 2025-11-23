import matplotlib
from pathlib import Path

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
