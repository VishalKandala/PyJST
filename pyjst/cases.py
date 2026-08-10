"""Definitions and body-fitted geometry for physical validation cases."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tan, pi

import numpy as np

from .benchmarks import ObliqueShockSolution, weak_oblique_shock
from .config import BoundaryCondition, BoundaryKind, Freestream, GasModel, GridSpec, SolverCase
from .grid import BodyFittedGrid


@dataclass(frozen=True)
class CompressionCornerCase:
    """Supersonic compression corner with a single lower-wall turn.

    The mesh has vertical index lines and is linearly blended from the wedge
    wall to a constant upper farfield. It is intended for an all-supersonic
    inflow/outflow treatment and a lower slip wall once the curvilinear finite
    volume operator is enabled.
    """

    mach: float = 2.0
    deflection_degrees: float = 10.0
    corner_x: float = 1.0
    length: float = 4.0
    farfield_height: float = 1.5
    nx: int = 160
    ny: int = 80
    pressure: float = 101_325.0
    density: float = 1.225
    gas: GasModel = GasModel()

    def __post_init__(self) -> None:
        for name, value in (
            ("mach", self.mach), ("corner_x", self.corner_x), ("length", self.length),
            ("farfield_height", self.farfield_height), ("pressure", self.pressure), ("density", self.density),
        ):
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.corner_x < self.length:
            raise ValueError("corner_x must lie within the domain")
        if not 0.0 < self.deflection_degrees < 45.0:
            raise ValueError("deflection_degrees must lie between 0 and 45")
        if self.nx < 4 or self.ny < 4:
            raise ValueError("nx and ny must each be at least 4")
        if self.wall_height(self.length) >= self.farfield_height:
            raise ValueError("farfield_height must exceed the downstream wedge height")

    @property
    def freestream(self) -> Freestream:
        return Freestream(self.mach, self.pressure, self.density)

    @property
    def reference(self) -> ObliqueShockSolution:
        return weak_oblique_shock(self.mach, self.deflection_degrees, self.gas.gamma)

    def wall_height(self, x: float | np.ndarray) -> float | np.ndarray:
        """Lower wall elevation for scalar or array ``x`` coordinates."""
        return np.maximum(np.asarray(x) - self.corner_x, 0.0) * tan(self.deflection_degrees * pi / 180.0)

    def vertices(self) -> np.ndarray:
        """Return body-fitted vertices with shape ``(nx+1, ny+1, 2)``."""
        x = np.linspace(0.0, self.length, self.nx + 1)
        wall = self.wall_height(x)
        eta = np.linspace(0.0, 1.0, self.ny + 1)
        vertices = np.empty((self.nx + 1, self.ny + 1, 2), dtype=float)
        vertices[..., 0] = x[:, None]
        vertices[..., 1] = wall[:, None] + eta[None, :] * (self.farfield_height - wall[:, None])
        return vertices

    def grid(self) -> BodyFittedGrid:
        """Return the body-fitted finite-volume geometry for this case."""
        return BodyFittedGrid(self.vertices())

    def solver_case(self) -> SolverCase:
        """Return boundary/numerical settings matched to :meth:`grid`.

        The rectangular ``GridSpec`` supplies only cell counts and ghost-cell
        policy here; physical face geometry is supplied by ``BodyFittedGrid``.
        """
        return SolverCase(
            name="mach-2-ten-degree-compression-corner",
            gas=self.gas,
            freestream=self.freestream,
            grid=GridSpec(self.nx, self.ny, 0.0, self.length, 0.0, self.farfield_height),
            boundaries={
                "left": BoundaryCondition(BoundaryKind.SUPERSONIC_INFLOW),
                "right": BoundaryCondition(BoundaryKind.SUPERSONIC_OUTFLOW),
                "bottom": BoundaryCondition(BoundaryKind.SLIP_WALL),
                "top": BoundaryCondition(BoundaryKind.SUPERSONIC_OUTFLOW),
            },
        )


def compression_corner_case() -> CompressionCornerCase:
    """Return the baseline Mach-2, 10-degree validation case."""
    return CompressionCornerCase()
