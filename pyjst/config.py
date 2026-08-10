"""Validated, SI-unit configuration objects for PyJST cases.

The first solver version is deliberately structured-grid only.  These
definitions form the stable public interface; numerical kernels should receive
one :class:`SolverCase` rather than reach into module-level globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, isfinite, pi, sin, sqrt


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive; got {value!r}")


class BoundaryKind(str, Enum):
    """Supported physical boundary-condition categories."""

    SUPERSONIC_INFLOW = "supersonic_inflow"
    SUPERSONIC_OUTFLOW = "supersonic_outflow"
    SLIP_WALL = "slip_wall"
    FARFIELD = "farfield"
    PERIODIC = "periodic"


@dataclass(frozen=True)
class GasModel:
    """Calorically perfect gas model in SI units."""

    gamma: float = 1.4
    gas_constant: float = 287.05

    def __post_init__(self) -> None:
        _positive("gamma", self.gamma)
        _positive("gas_constant", self.gas_constant)
        if self.gamma <= 1.0:
            raise ValueError("gamma must be greater than 1")

    def sound_speed(self, pressure: float, density: float) -> float:
        """Return ``sqrt(gamma * pressure / density)``."""
        _positive("pressure", pressure)
        _positive("density", density)
        return sqrt(self.gamma * pressure / density)


@dataclass(frozen=True)
class Freestream:
    """Uniform upstream state, specified by Mach, static pressure and density."""

    mach: float
    pressure: float
    density: float
    angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        _positive("mach", self.mach)
        _positive("pressure", self.pressure)
        _positive("density", self.density)
        if not isfinite(self.angle_degrees):
            raise ValueError("angle_degrees must be finite")

    def primitive(self, gas: GasModel) -> tuple[float, float, float, float]:
        """Return ``(rho, u, v, p)`` for this freestream."""
        speed = self.mach * gas.sound_speed(self.pressure, self.density)
        angle = self.angle_degrees * pi / 180.0
        return self.density, speed * cos(angle), speed * sin(angle), self.pressure


@dataclass(frozen=True)
class GridSpec:
    """Uniform Cartesian structured grid, stored as interior cell counts."""

    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    ghost_cells: int = 2

    def __post_init__(self) -> None:
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must each be at least 2")
        if self.ghost_cells < 2:
            raise ValueError("JST fourth-order dissipation requires at least 2 ghost cells")
        for name, value in (
            ("x_min", self.x_min), ("x_max", self.x_max),
            ("y_min", self.y_min), ("y_max", self.y_max),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("grid maximum coordinates must exceed minimum coordinates")

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        return (self.y_max - self.y_min) / self.ny

    @property
    def cell_shape(self) -> tuple[int, int]:
        return self.nx + 2 * self.ghost_cells, self.ny + 2 * self.ghost_cells


@dataclass(frozen=True)
class JSTParameters:
    """Numerical controls for the four-stage JST pseudo-time integrator."""

    cfl: float = 0.5
    cfl_initial: float = 0.05
    cfl_ramp_iterations: int = 100
    kappa2: float = 0.5
    kappa4: float = 0.02
    residual_tolerance: float = 1.0e-8
    max_iterations: int = 10_000
    rk_coefficients: tuple[float, float, float, float] = (0.25, 1.0 / 3.0, 0.5, 1.0)

    def __post_init__(self) -> None:
        for name, value in (
            ("cfl", self.cfl), ("cfl_initial", self.cfl_initial), ("kappa2", self.kappa2),
            ("kappa4", self.kappa4),
            ("residual_tolerance", self.residual_tolerance),
        ):
            _positive(name, value)
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if self.cfl_initial > self.cfl:
            raise ValueError("cfl_initial must not exceed cfl")
        if self.cfl_ramp_iterations < 0:
            raise ValueError("cfl_ramp_iterations must not be negative")
        if len(self.rk_coefficients) != 4 or any(alpha <= 0.0 for alpha in self.rk_coefficients):
            raise ValueError("rk_coefficients must contain four positive values")


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary type, with optional imposed state reserved for later cases."""

    kind: BoundaryKind


@dataclass(frozen=True)
class SolverCase:
    """Complete immutable description of one solver run."""

    name: str
    gas: GasModel
    freestream: Freestream
    grid: GridSpec
    boundaries: dict[str, BoundaryCondition]
    numerics: JSTParameters = field(default_factory=JSTParameters)

    def __post_init__(self) -> None:
        expected = {"left", "right", "bottom", "top"}
        if set(self.boundaries) != expected:
            raise ValueError(f"boundaries must contain exactly {sorted(expected)}")
        if not self.name.strip():
            raise ValueError("case name must not be blank")


def uniform_supersonic_case() -> SolverCase:
    """Return the first exact-preservation test case for the new solver."""
    return SolverCase(
        name="uniform-supersonic-flow",
        gas=GasModel(),
        freestream=Freestream(mach=2.0, pressure=101_325.0, density=1.225),
        grid=GridSpec(nx=64, ny=32, x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0),
        boundaries={
            "left": BoundaryCondition(BoundaryKind.SUPERSONIC_INFLOW),
            "right": BoundaryCondition(BoundaryKind.SUPERSONIC_OUTFLOW),
            "bottom": BoundaryCondition(BoundaryKind.PERIODIC),
            "top": BoundaryCondition(BoundaryKind.PERIODIC),
        },
    )
