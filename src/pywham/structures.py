"""Data structures shared by the WHAM translations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Histogram1D:
    first: int
    last: int
    num_points: int
    num_mc_samples: int
    data: List[float] = field(default_factory=list)
    cumulative: List[float] = field(default_factory=list)


@dataclass
class HistGroup1D:
    num_windows: int
    bias_locations: List[float] = field(default_factory=list)
    spring_constants: List[float] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    previous_free_energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    partitions: List[float] = field(default_factory=list)
    histograms: List[Histogram1D] = field(default_factory=list)


@dataclass
class Histogram2D:
    first_x: int
    last_x: int
    first_y: int
    last_y: int
    num_points: int
    num_mc_samples: int
    data: List[List[float]] = field(default_factory=list)
    cumulative: List[float] = field(default_factory=list)


@dataclass
class HistGroup2D:
    num_windows: int
    bias_locations: List[List[float]] = field(default_factory=list)
    spring_x: List[float] = field(default_factory=list)
    spring_y: List[float] = field(default_factory=list)
    free_energies: List[float] = field(default_factory=list)
    previous_free_energies: List[float] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    partitions: List[float] = field(default_factory=list)
    histograms: List[Histogram2D] = field(default_factory=list)
