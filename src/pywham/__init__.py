"""Python translation of the WHAM C implementations."""

from importlib import import_module
from typing import Any

__all__ = [
    "BayesWHAM",
    "BayesWhamConfig",
    "build_bwham_config",
    "plot_free_energy_1d",
    "plot_free_energy_2d",
    "save_free_energy_plots",
]

_LAZY_ATTRS = {
    "BayesWHAM": (".bwham", "BayesWHAM"),
    "BayesWhamConfig": (".bwham", "BayesWhamConfig"),
    "build_bwham_config": (".bwham", "build_config"),
    "plot_free_energy_1d": (".visualization", "plot_free_energy_1d"),
    "plot_free_energy_2d": (".visualization", "plot_free_energy_2d"),
    "save_free_energy_plots": (".visualization", "save_free_energy_plots"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(f"module 'pywham' has no attribute {name!r}") from None
    module = import_module(module_name, __name__)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(__all__)
