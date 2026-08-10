"""Explicit steady pseudo-time solver for the Cartesian JST discretization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boundary import apply_boundary_conditions
from .config import SolverCase
from .flux import central_residual
from .grid import BodyFittedGrid, CartesianGrid
from .jst import jst_dissipation
from .state import conservative_to_primitive, uniform_conservative_state


@dataclass(frozen=True)
class SolverResult:
    """State and convergence information returned by :func:`solve`."""

    conservative: np.ndarray
    residual_history: np.ndarray
    absolute_residual_history: np.ndarray
    residual_reference: float
    iterations: int
    converged: bool


def freestream_initial_state(grid: CartesianGrid | BodyFittedGrid, case: SolverCase) -> np.ndarray:
    """Create a uniform conservative state, including all ghost cells."""
    sx, sy = grid.interior
    if (sx.stop - sx.start, sy.stop - sy.start) != (case.grid.nx, case.grid.ny):
        raise ValueError("grid physical dimensions must match case.grid")
    rho, u, v, pressure = case.freestream.primitive(case.gas)
    return uniform_conservative_state(grid.shape, case.gas, rho, u, v, pressure)


def local_time_steps(
    conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, case: SolverCase, cfl: float | None = None
) -> np.ndarray:
    """Return local pseudo-time steps for physical cells.

    Face spectral radii are integrated with the grid's face vectors. The
    returned array has physical-cell shape ``(nx, ny)``.
    """
    primitive = conservative_to_primitive(conservative, case.gas)
    rho, u, v, pressure = np.moveaxis(primitive, -1, 0)
    sound = np.sqrt(case.gas.gamma * pressure / rho)
    sx, sy = grid.interior
    u_x = 0.5 * (u[sx.start - 1 : sx.stop, sy] + u[sx.start : sx.stop + 1, sy])
    v_x = 0.5 * (v[sx.start - 1 : sx.stop, sy] + v[sx.start : sx.stop + 1, sy])
    a_x = 0.5 * (sound[sx.start - 1 : sx.stop, sy] + sound[sx.start : sx.stop + 1, sy])
    u_y = 0.5 * (u[sx, sy.start - 1 : sy.stop] + u[sx, sy.start : sy.stop + 1])
    v_y = 0.5 * (v[sx, sy.start - 1 : sy.stop] + v[sx, sy.start : sy.stop + 1])
    a_y = 0.5 * (sound[sx, sy.start - 1 : sy.stop] + sound[sx, sy.start : sy.stop + 1])
    x_normal, y_normal = grid.x_face_vectors, grid.y_face_vectors
    spectral_x = np.abs(u_x * x_normal[..., 0] + v_x * x_normal[..., 1]) + a_x * np.linalg.norm(x_normal, axis=-1)
    spectral_y = np.abs(u_y * y_normal[..., 0] + v_y * y_normal[..., 1]) + a_y * np.linalg.norm(y_normal, axis=-1)
    spectral_sum = spectral_x[:-1, :] + spectral_x[1:, :] + spectral_y[:, :-1] + spectral_y[:, 1:]
    return (case.numerics.cfl if cfl is None else cfl) * grid.cell_volumes / spectral_sum


def iteration_cfl(case: SolverCase, iteration: int) -> float:
    """Return the CFL value scheduled for one-based outer ``iteration``."""
    if iteration < 1:
        raise ValueError("iteration must be one-based and positive")
    ramp = case.numerics.cfl_ramp_iterations
    if ramp == 0:
        return case.numerics.cfl
    fraction = min((iteration - 1) / ramp, 1.0)
    return case.numerics.cfl_initial + fraction * (case.numerics.cfl - case.numerics.cfl_initial)


def _stage_residual(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, case: SolverCase) -> np.ndarray:
    """Return central residual minus JST dissipation after refreshing ghosts."""
    apply_boundary_conditions(conservative, grid, case)
    return central_residual(conservative, grid, case.gas) - jst_dissipation(
        conservative, grid, case.gas, case.numerics
    )


def advance_one_iteration(
    conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, case: SolverCase, cfl: float | None = None
) -> tuple[np.ndarray, float]:
    """Advance one four-stage JST pseudo-time iteration without mutating input."""
    expected_shape = (*grid.shape, 4)
    if conservative.shape != expected_shape:
        raise ValueError(f"conservative must have shape {expected_shape}; got {conservative.shape}")

    base = conservative.copy()
    apply_boundary_conditions(base, grid, case)
    time_step = local_time_steps(base, grid, case, cfl)
    stage = base.copy()
    sx, sy = grid.interior
    cell_volumes = grid.cell_volumes
    first_residual_norm = 0.0

    # JST's multistage form always updates from the start-of-iteration state,
    # while evaluating each stage residual from the previous stage state.
    for stage_number, alpha in enumerate(case.numerics.rk_coefficients):
        residual = _stage_residual(stage, grid, case)
        if stage_number == 0:
            first_residual_norm = float(np.max(np.abs(residual[sx, sy, :]) / cell_volumes[..., None]))
        next_stage = stage.copy()
        next_stage[sx, sy, :] = base[sx, sy, :] - alpha * time_step[..., None] * residual[sx, sy, :] / cell_volumes[..., None]
        stage = next_stage

    apply_boundary_conditions(stage, grid, case)
    return stage, first_residual_norm


def solve(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, case: SolverCase) -> SolverResult:
    """Pseudo-time march to the configured residual tolerance.

    The input is never mutated.  The recorded residual is the maximum absolute
    first-stage residual per cell area for each outer iteration.
    """
    state = conservative.copy()
    normalized_history: list[float] = []
    absolute_history: list[float] = []
    reference: float | None = None
    for iteration in range(1, case.numerics.max_iterations + 1):
        state, absolute_residual = advance_one_iteration(state, grid, case, iteration_cfl(case, iteration))
        if reference is None:
            reference = absolute_residual
        normalized_residual = absolute_residual / reference if reference > 0.0 else 0.0
        absolute_history.append(absolute_residual)
        normalized_history.append(normalized_residual)
        if normalized_residual <= case.numerics.residual_tolerance:
            return SolverResult(
                state, np.asarray(normalized_history), np.asarray(absolute_history), reference, iteration, True
            )
    return SolverResult(
        state, np.asarray(normalized_history), np.asarray(absolute_history), reference or 0.0,
        case.numerics.max_iterations, False,
    )
