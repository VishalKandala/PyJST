"""Analytic reference solutions used to validate solver benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from math import asin, cos, pi, sin, sqrt, tan


@dataclass(frozen=True)
class ObliqueShockSolution:
    """Weak attached oblique-shock state ratios for a perfect gas."""

    upstream_mach: float
    deflection_degrees: float
    shock_angle_degrees: float
    downstream_mach: float
    pressure_ratio: float
    density_ratio: float
    temperature_ratio: float


def weak_oblique_shock(upstream_mach: float, deflection_degrees: float, gamma: float = 1.4) -> ObliqueShockSolution:
    """Solve the theta-beta-M relation for its weak, attached-shock branch.

    Raises ``ValueError`` if the given deflection has no attached weak-shock
    solution.  The result is an independent reference for the compression
    corner numerical solution, not a solver boundary condition.
    """
    if upstream_mach <= 1.0:
        raise ValueError("an oblique-shock reference requires supersonic upstream Mach number")
    if not 0.0 < deflection_degrees < 90.0:
        raise ValueError("deflection_degrees must lie between 0 and 90")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")

    theta = deflection_degrees * pi / 180.0
    mach_angle = asin(1.0 / upstream_mach)

    def residual(beta: float) -> float:
        rhs = 2.0 / tan(beta) * (upstream_mach**2 * sin(beta) ** 2 - 1.0) / (
            upstream_mach**2 * (gamma + cos(2.0 * beta)) + 2.0
        )
        return rhs - tan(theta)

    # The relation is negative at both limits. Scan from the Mach angle to
    # locate its first sign change, which is the weak branch.
    samples = 20_000
    previous_beta = mach_angle + 1.0e-10
    previous_value = residual(previous_beta)
    bracket: tuple[float, float] | None = None
    for sample in range(1, samples + 1):
        beta = mach_angle + (pi / 2.0 - 1.0e-10 - mach_angle) * sample / samples
        value = residual(beta)
        if previous_value <= 0.0 <= value:
            bracket = previous_beta, beta
            break
        previous_beta, previous_value = beta, value
    if bracket is None:
        raise ValueError("deflection exceeds the attached weak-shock limit for this Mach number")

    low, high = bracket
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if residual(midpoint) >= 0.0:
            high = midpoint
        else:
            low = midpoint
    beta = 0.5 * (low + high)
    normal_mach_1 = upstream_mach * sin(beta)
    pressure_ratio = 1.0 + 2.0 * gamma / (gamma + 1.0) * (normal_mach_1**2 - 1.0)
    density_ratio = (gamma + 1.0) * normal_mach_1**2 / (2.0 + (gamma - 1.0) * normal_mach_1**2)
    normal_mach_2 = sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * normal_mach_1**2)
        / (gamma * normal_mach_1**2 - 0.5 * (gamma - 1.0))
    )
    downstream_mach = normal_mach_2 / sin(beta - theta)
    return ObliqueShockSolution(
        upstream_mach=upstream_mach,
        deflection_degrees=deflection_degrees,
        shock_angle_degrees=beta * 180.0 / pi,
        downstream_mach=downstream_mach,
        pressure_ratio=pressure_ratio,
        density_ratio=density_ratio,
        temperature_ratio=pressure_ratio / density_ratio,
    )
