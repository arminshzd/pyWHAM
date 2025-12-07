"""Visualization helpers for WHAM outputs.

This module provides convenience plotting functions for 1D and 2D
free energy surfaces produced by the WHAM solvers. The 1D plot uses a
line for the free energy profile and, when bootstrap errors are
available, an uncertainty band rendered via ``fill_between``. The 2D
plots render contour and filled contour overlays for the free energy
and, optionally, the associated bootstrap error surface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_free_energy_1d(
    x: Iterable[float],
    free_energy: Iterable[float],
    errors: Iterable[float] | None = None,
    *,
    output_path: str | Path | None = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a 1D free energy profile.

    Parameters
    ----------
    x:
        Coordinates associated with each bin.
    free_energy:
        Free energy values (typically in units of ``kT``) for each bin.
    errors:
        Optional bootstrap standard deviations for each bin. When
        provided, the uncertainty band is drawn with ``fill_between``.
    output_path:
        If provided, the figure is saved to this path.
    show:
        Whether to call :func:`matplotlib.pyplot.show` before returning.

    Returns
    -------
    (Figure, Axes)
        The figure and axes containing the plot, useful for further
        customization by callers.
    """

    x_vals = np.asarray(list(x), dtype=float)
    free_vals = np.asarray(list(free_energy), dtype=float)

    if x_vals.shape != free_vals.shape:
        raise ValueError("x and free_energy must have the same shape")

    fig, ax = plt.subplots()
    ax.plot(x_vals, free_vals, label="Free energy", color="C0")

    if errors is not None:
        err_vals = np.asarray(list(errors), dtype=float)
        if err_vals.shape != free_vals.shape:
            raise ValueError("errors must match the shape of free_energy")
        lower = free_vals - err_vals
        upper = free_vals + err_vals
        ax.fill_between(
            x_vals,
            lower,
            upper,
            color="C0",
            alpha=0.3,
            label="Bootstrap uncertainty",
        )

    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Free energy")
    ax.set_title("1D WHAM free energy profile")
    ax.grid(True, linestyle=":", alpha=0.5)
    if errors is not None:
        ax.legend()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def _mesh_from_axes(x: Iterable[float], y: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x_vals = np.asarray(list(x), dtype=float)
    y_vals = np.asarray(list(y), dtype=float)
    return np.meshgrid(x_vals, y_vals, indexing="xy")


def _validate_grid(data: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray, label: str) -> None:
    if data.shape != x_grid.shape:
        raise ValueError(f"{label} must have shape {x_grid.shape}, got {data.shape}")
    if data.shape != y_grid.shape:
        raise ValueError(f"{label} must have shape {y_grid.shape}, got {data.shape}")


def _contour_plot(
    ax: plt.Axes,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    surface: np.ndarray,
    *,
    cmap: str,
    levels: int,
    label: str,
) -> None:
    contourf = ax.contourf(x_grid, y_grid, surface, levels=levels, cmap=cmap)
    contour = ax.contour(x_grid, y_grid, surface, levels=levels, colors="k", linewidths=0.6)
    ax.clabel(contour, fmt="%.2f", fontsize=8)
    ax.figure.colorbar(contourf, ax=ax, label=label)


def _preserve_unique(values: np.ndarray) -> np.ndarray:
    """Return unique values while preserving their first-seen order."""

    unique, indices = np.unique(values, return_index=True)
    order = np.argsort(indices)
    return unique[order]


def plot_free_energy_2d(
    x: Iterable[float],
    y: Iterable[float],
    free_energy: Iterable[Iterable[float]],
    errors: Iterable[Iterable[float]] | None = None,
    *,
    levels: int = 15,
    cmap: str = "viridis",
    output_path: str | Path | None = None,
    error_output_path: str | Path | None = None,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes] | Tuple[Tuple[plt.Figure, plt.Axes], Tuple[plt.Figure, plt.Axes]]:
    """Plot 2D WHAM free energy surfaces.

    The primary plot is a contour/contourf overlay of the free energy
    surface. When ``errors`` are provided, a secondary figure is
    produced with the same contour style to visualize the uncertainty
    landscape.

    Parameters
    ----------
    x, y:
        One-dimensional coordinate arrays defining the grid.
    free_energy:
        2D free energy array shaped like ``(len(x), len(y))``.
    errors:
        Optional 2D bootstrap error array with the same shape as
        ``free_energy``.
    levels:
        Number of contour levels to render.
    cmap:
        Colormap used for the filled contours.
    output_path:
        Destination for saving the free energy figure, when provided.
    error_output_path:
        Destination for saving the error figure, when provided.
    show:
        Whether to call :func:`matplotlib.pyplot.show` before returning.

    Returns
    -------
    Figure/Axes tuple or pair of tuples
        Always returns the free energy figure/axes. When ``errors`` are
        provided, also returns the error figure/axes as the second
        element of a tuple.
    """

    x_grid, y_grid = _mesh_from_axes(x, y)
    fe_surface = np.asarray(list(map(list, free_energy)), dtype=float)
    _validate_grid(fe_surface, x_grid, y_grid, "free_energy")

    fe_fig, fe_ax = plt.subplots()
    _contour_plot(
        fe_ax,
        x_grid,
        y_grid,
        fe_surface,
        cmap=cmap,
        levels=levels,
        label="Free energy",
    )
    fe_ax.set_xlabel("X coordinate")
    fe_ax.set_ylabel("Y coordinate")
    fe_ax.set_title("2D WHAM free energy surface")

    if output_path is not None:
        fe_fig.savefig(output_path, bbox_inches="tight")

    if errors is None:
        if show:
            plt.show()
        return fe_fig, fe_ax

    err_surface = np.asarray(list(map(list, errors)), dtype=float)
    _validate_grid(err_surface, x_grid, y_grid, "errors")

    err_fig, err_ax = plt.subplots()
    _contour_plot(
        err_ax,
        x_grid,
        y_grid,
        err_surface,
        cmap=cmap,
        levels=levels,
        label="Uncertainty",
    )
    err_ax.set_xlabel("X coordinate")
    err_ax.set_ylabel("Y coordinate")
    err_ax.set_title("2D WHAM free energy uncertainty")

    if error_output_path is not None:
        err_fig.savefig(error_output_path, bbox_inches="tight")

    if show:
        plt.show()

    return (fe_fig, fe_ax), (err_fig, err_ax)


def save_free_energy_plots(
    freefile_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_F: float | None = None,
    levels: int = 15,
    cmap: str = "viridis",
    show: bool = False,
) -> dict[str, Path]:
    """Create and save WHAM free energy plots from a solver output file.

    The helper inspects the column structure of ``freefile_path`` to dispatch to
    the appropriate 1D or 2D plotting routine. Files generated by
    :mod:`pywham.wham1d` contain five columns (coordinate, free energy,
    uncertainty, probability, probability uncertainty), while 2D outputs from
    :mod:`pywham.wham2d` contain four columns (x, y, free energy, probability).

    Parameters
    ----------
    freefile_path:
        Path to the WHAM ``freefile`` output (for example, ``output.free``).
    output_dir:
        Directory for saving the generated plots. Defaults to the directory of
        ``freefile_path``.
    max_F:
        Optional cap for the free energy values. Energies greater than this
        threshold are replaced with ``np.inf`` after parsing the freefile.
    levels:
        Number of contour levels to render for 2D plots.
    cmap:
        Matplotlib colormap used for 2D filled contours.
    show:
        Whether to call :func:`matplotlib.pyplot.show` before returning.

    Returns
    -------
    dict
        Mapping from plot label to the output file path used for saving.

    Raises
    ------
    ValueError
        If the file contents do not resemble WHAM 1D or 2D outputs.
    """

    freefile = Path(freefile_path)
    output_directory = Path(output_dir) if output_dir is not None else freefile.parent
    output_directory.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(freefile)
    data = np.atleast_2d(data)

    if data.ndim != 2 or data.shape[1] not in (4, 5, 6):
        raise ValueError(
            "freefile must contain five columns (1D WHAM) or four/six columns (2D WHAM)"
        )

    if max_F is not None:
        free_energy_col = 1 if data.shape[1] == 5 else 2
        data[:, free_energy_col] = np.where(data[:, free_energy_col] > max_F, np.inf, data[:, free_energy_col])

    outputs: dict[str, Path] = {}

    if data.shape[1] == 5:
        coordinates = data[:, 0]
        free_energy = data[:, 1]
        errors = data[:, 2]
        output_path = output_directory / f"{freefile.stem}_free_energy.png"
        plot_free_energy_1d(coordinates, free_energy, errors, output_path=output_path, show=show)
        outputs["free_energy"] = output_path
        return outputs

    x_vals = _preserve_unique(data[:, 0])
    y_vals = _preserve_unique(data[:, 1])

    if data.shape[0] != len(x_vals) * len(y_vals):
        raise ValueError(
            "2D freefile does not contain a complete grid of (x, y, free_energy) samples"
        )

    fe_surface = np.empty((len(x_vals), len(y_vals)))
    x_index = {value: idx for idx, value in enumerate(x_vals)}
    y_index = {value: idx for idx, value in enumerate(y_vals)}
    for x_val, y_val, free_val, *_ in data:
        fe_surface[x_index[x_val], y_index[y_val]] = free_val

    output_path = output_directory / f"{freefile.stem}_free_energy.png"
    plot_free_energy_2d(
        x_vals,
        y_vals,
        fe_surface,
        levels=levels,
        cmap=cmap,
        output_path=output_path,
        show=show,
    )
    outputs["free_energy"] = output_path
    return outputs

