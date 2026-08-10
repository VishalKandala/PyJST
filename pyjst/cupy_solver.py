"""Optional CuPy implementation of the structured JST pseudo-time solver.

This module is deliberately isolated from the NumPy backend.  CuPy is imported
only when ``backend=\"cupy\"`` is selected, and static mesh data are copied to
the device once per solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BoundaryKind, SolverCase
from .grid import BodyFittedGrid, CartesianGrid


Grid = CartesianGrid | BodyFittedGrid


def _cupy():
    try:
        import cupy as cp
    except ImportError as error:
        raise ImportError("The CuPy backend requires CuPy; install pyjst[gpu] on a CUDA 12 system.") from error
    return cp


@dataclass(frozen=True)
class DeviceGeometry:
    """Static grid geometry stored on a CuPy device."""

    x_face_vectors: object
    y_face_vectors: object
    cell_volumes: object

    @classmethod
    def from_grid(cls, grid: Grid, cp):
        return cls(cp.asarray(grid.x_face_vectors), cp.asarray(grid.y_face_vectors), cp.asarray(grid.cell_volumes))


def _primitive(state, gamma: float, cp):
    rho = state[..., 0]
    u = state[..., 1] / rho
    v = state[..., 2] / rho
    pressure = (gamma - 1.0) * (state[..., 3] - 0.5 * rho * (u * u + v * v))
    return rho, u, v, pressure


def _apply_boundaries(state, grid: Grid, case: SolverCase, geometry: DeviceGeometry, cp) -> None:
    sx, sy = grid.interior
    g = sx.start
    rho, u, v, pressure = case.freestream.primitive(case.gas)
    freestream = cp.asarray((rho, rho * u, rho * v, pressure / (case.gas.gamma - 1.0) + 0.5 * rho * (u * u + v * v)))

    def fill(side: str) -> None:
        kind = case.boundaries[side].kind
        if kind is BoundaryKind.SUPERSONIC_INFLOW:
            if side == "left": state[:g] = freestream
            elif side == "right": state[sx.stop:] = freestream
            elif side == "bottom": state[:, :g] = freestream
            else: state[:, sy.stop:] = freestream
        elif kind is BoundaryKind.SUPERSONIC_OUTFLOW:
            if side == "left": state[:g] = state[g:g + 1]
            elif side == "right": state[sx.stop:] = state[sx.stop - 1:sx.stop]
            elif side == "bottom": state[:, :g] = state[:, g:g + 1]
            else: state[:, sy.stop:] = state[:, sy.stop - 1:sy.stop]
        elif kind is BoundaryKind.SLIP_WALL:
            if side not in ("bottom", "top"):
                raise NotImplementedError("slip walls are currently implemented on structured y boundaries only")
            normals = geometry.y_face_vectors[:, 0 if side == "bottom" else -1]
            normals = normals / cp.linalg.norm(normals, axis=1)[:, None]
            source = state[sx, sy.start:sy.start + 1] if side == "bottom" else state[sx, sy.stop - 1:sy.stop]
            density, velocity_x, velocity_y, static_pressure = _primitive(source, case.gas.gamma, cp)
            normal_velocity = velocity_x * normals[:, None, 0] + velocity_y * normals[:, None, 1]
            velocity_x = velocity_x - 2.0 * normal_velocity * normals[:, None, 0]
            velocity_y = velocity_y - 2.0 * normal_velocity * normals[:, None, 1]
            reflected = cp.empty_like(source)
            reflected[..., 0] = density
            reflected[..., 1] = density * velocity_x
            reflected[..., 2] = density * velocity_y
            reflected[..., 3] = static_pressure / (case.gas.gamma - 1.0) + 0.5 * density * (velocity_x**2 + velocity_y**2)
            if side == "bottom": state[sx, :g] = reflected
            else: state[sx, sy.stop:] = reflected
        else:
            raise NotImplementedError(f"boundary kind {kind.value!r} is not implemented yet")

    if case.boundaries["left"].kind is BoundaryKind.PERIODIC:
        state[:g] = state[sx.stop - g:sx.stop]
        state[sx.stop:] = state[sx.start:sx.start + g]
    else:
        fill("left"); fill("right")
    if case.boundaries["bottom"].kind is BoundaryKind.PERIODIC:
        state[:, :g] = state[:, sy.stop - g:sy.stop]
        state[:, sy.stop:] = state[:, sy.start:sy.start + g]
    else:
        fill("bottom"); fill("top")


def _residual(state, grid: Grid, case: SolverCase, geometry: DeviceGeometry, cp):
    rho, u, v, pressure = _primitive(state, case.gas.gamma, cp)
    flux_x = cp.empty_like(state); flux_y = cp.empty_like(state)
    flux_x[..., 0] = rho * u; flux_x[..., 1] = rho * u * u + pressure; flux_x[..., 2] = rho * u * v; flux_x[..., 3] = (state[..., 3] + pressure) * u
    flux_y[..., 0] = rho * v; flux_y[..., 1] = rho * u * v; flux_y[..., 2] = rho * v * v + pressure; flux_y[..., 3] = (state[..., 3] + pressure) * v
    sx, sy = grid.interior
    face_x_f = 0.5 * (flux_x[:-1] + flux_x[1:]); face_x_g = 0.5 * (flux_y[:-1] + flux_y[1:])
    face_y_f = 0.5 * (flux_x[:, :-1] + flux_x[:, 1:]); face_y_g = 0.5 * (flux_y[:, :-1] + flux_y[:, 1:])
    x_f = face_x_f[sx.start - 1:sx.stop, sy]; x_g = face_x_g[sx.start - 1:sx.stop, sy]
    y_f = face_y_f[sx, sy.start - 1:sy.stop]; y_g = face_y_g[sx, sy.start - 1:sy.stop]
    x_integrated = x_f * geometry.x_face_vectors[..., 0, None] + x_g * geometry.x_face_vectors[..., 1, None]
    y_integrated = y_f * geometry.y_face_vectors[..., 0, None] + y_g * geometry.y_face_vectors[..., 1, None]
    residual = cp.zeros_like(state)
    residual[sx, sy] = x_integrated[1:] - x_integrated[:-1] + y_integrated[:, 1:] - y_integrated[:, :-1]
    return residual


def _dissipation(state, grid: Grid, case: SolverCase, geometry: DeviceGeometry, cp):
    rho, u, v, pressure = _primitive(state, case.gas.gamma, cp)
    sound = cp.sqrt(case.gas.gamma * pressure / rho)
    sensor_x = cp.zeros_like(pressure); sensor_y = cp.zeros_like(pressure)
    sensor_x[1:-1] = cp.abs(pressure[2:] - 2.0 * pressure[1:-1] + pressure[:-2]) / (pressure[2:] + 2.0 * pressure[1:-1] + pressure[:-2])
    sensor_y[:, 1:-1] = cp.abs(pressure[:, 2:] - 2.0 * pressure[:, 1:-1] + pressure[:, :-2]) / (pressure[:, 2:] + 2.0 * pressure[:, 1:-1] + pressure[:, :-2])
    sx, sy = grid.interior
    ux = 0.5 * (u[:-1] + u[1:]); vx = 0.5 * (v[:-1] + v[1:]); ax = 0.5 * (sound[:-1] + sound[1:])
    uy = 0.5 * (u[:, :-1] + u[:, 1:]); vy = 0.5 * (v[:, :-1] + v[:, 1:]); ay = 0.5 * (sound[:, :-1] + sound[:, 1:])
    nx = cp.zeros((*ux.shape, 2)); ny = cp.zeros((*uy.shape, 2))
    nx[sx.start - 1:sx.stop, sy] = geometry.x_face_vectors; ny[sx, sy.start - 1:sy.stop] = geometry.y_face_vectors
    lambda_x = cp.abs(ux * nx[..., 0] + vx * nx[..., 1]) + ax * cp.linalg.norm(nx, axis=-1)
    lambda_y = cp.abs(uy * ny[..., 0] + vy * ny[..., 1]) + ay * cp.linalg.norm(ny, axis=-1)
    eps2_x = case.numerics.kappa2 * cp.maximum(sensor_x[:-1], sensor_x[1:]); eps2_y = case.numerics.kappa2 * cp.maximum(sensor_y[:, :-1], sensor_y[:, 1:])
    eps4_x = cp.maximum(0.0, case.numerics.kappa4 - eps2_x); eps4_y = cp.maximum(0.0, case.numerics.kappa4 - eps2_y)
    fd_x = cp.zeros((state.shape[0] - 1, state.shape[1], 4), dtype=state.dtype); fd_y = cp.zeros((state.shape[0], state.shape[1] - 1, 4), dtype=state.dtype)
    delta_x = state[2:-1] - state[1:-2]; third_x = state[3:] - 3.0 * state[2:-1] + 3.0 * state[1:-2] - state[:-3]
    fd_x[1:-1] = lambda_x[1:-1, :, None] * (eps2_x[1:-1, :, None] * delta_x - eps4_x[1:-1, :, None] * third_x)
    delta_y = state[:, 2:-1] - state[:, 1:-2]; third_y = state[:, 3:] - 3.0 * state[:, 2:-1] + 3.0 * state[:, 1:-2] - state[:, :-3]
    fd_y[:, 1:-1] = lambda_y[:, 1:-1, None] * (eps2_y[:, 1:-1, None] * delta_y - eps4_y[:, 1:-1, None] * third_y)
    fd_x[sx.start - 1, sy] = 0.0; fd_x[sx.stop - 1, sy] = 0.0; fd_y[sx, sy.start - 1] = 0.0; fd_y[sx, sy.stop - 1] = 0.0
    dissipation = cp.zeros_like(state)
    dissipation[sx, sy] = fd_x[sx, sy] - fd_x[sx.start - 1:sx.stop - 1, sy] + fd_y[sx, sy] - fd_y[sx, sy.start - 1:sy.stop - 1]
    return dissipation


def _time_steps(state, grid: Grid, case: SolverCase, geometry: DeviceGeometry, cp, cfl: float):
    rho, u, v, pressure = _primitive(state, case.gas.gamma, cp); sound = cp.sqrt(case.gas.gamma * pressure / rho); sx, sy = grid.interior
    ux = 0.5 * (u[sx.start - 1:sx.stop, sy] + u[sx.start:sx.stop + 1, sy]); vx = 0.5 * (v[sx.start - 1:sx.stop, sy] + v[sx.start:sx.stop + 1, sy]); ax = 0.5 * (sound[sx.start - 1:sx.stop, sy] + sound[sx.start:sx.stop + 1, sy])
    uy = 0.5 * (u[sx, sy.start - 1:sy.stop] + u[sx, sy.start:sy.stop + 1]); vy = 0.5 * (v[sx, sy.start - 1:sy.stop] + v[sx, sy.start:sy.stop + 1]); ay = 0.5 * (sound[sx, sy.start - 1:sy.stop] + sound[sx, sy.start:sy.stop + 1])
    spectral_x = cp.abs(ux * geometry.x_face_vectors[..., 0] + vx * geometry.x_face_vectors[..., 1]) + ax * cp.linalg.norm(geometry.x_face_vectors, axis=-1)
    spectral_y = cp.abs(uy * geometry.y_face_vectors[..., 0] + vy * geometry.y_face_vectors[..., 1]) + ay * cp.linalg.norm(geometry.y_face_vectors, axis=-1)
    return cfl * geometry.cell_volumes / (spectral_x[:-1] + spectral_x[1:] + spectral_y[:, :-1] + spectral_y[:, 1:])


def advance_one_iteration_cupy(state, grid: Grid, case: SolverCase, geometry: DeviceGeometry, cfl: float, cp):
    """Advance a device-resident state by one four-stage iteration."""
    base = state.copy(); _apply_boundaries(base, grid, case, geometry, cp); time_step = _time_steps(base, grid, case, geometry, cp, cfl); stage = base.copy(); sx, sy = grid.interior; residual_norm = 0.0
    for stage_number, alpha in enumerate(case.numerics.rk_coefficients):
        _apply_boundaries(stage, grid, case, geometry, cp); residual = _residual(stage, grid, case, geometry, cp) - _dissipation(stage, grid, case, geometry, cp)
        if stage_number == 0: residual_norm = float(cp.max(cp.abs(residual[sx, sy]) / geometry.cell_volumes[..., None]).get())
        next_stage = stage.copy(); next_stage[sx, sy] = base[sx, sy] - alpha * time_step[..., None] * residual[sx, sy] / geometry.cell_volumes[..., None]; stage = next_stage
    _apply_boundaries(stage, grid, case, geometry, cp)
    return stage, residual_norm


def benchmark_cupy_iterations(conservative: np.ndarray, grid: Grid, case: SolverCase, *, repeats: int, warmup_iterations: int) -> float:
    """Return median GPU iteration time, excluding setup and host/device copies."""
    from time import perf_counter
    cp = _cupy(); geometry = DeviceGeometry.from_grid(grid, cp); state = cp.asarray(conservative, dtype=cp.float64); cfl = case.numerics.cfl_initial
    for _ in range(warmup_iterations): advance_one_iteration_cupy(state, grid, case, geometry, cfl, cp)
    cp.cuda.Stream.null.synchronize(); durations = []
    for _ in range(repeats):
        started = perf_counter(); advance_one_iteration_cupy(state, grid, case, geometry, cfl, cp); cp.cuda.Stream.null.synchronize(); durations.append(perf_counter() - started)
    return float(np.median(durations))


def solve_cupy(conservative: np.ndarray, grid: Grid, case: SolverCase):
    """Solve on a CUDA device and return the standard host-resident result."""
    cp = _cupy(); geometry = DeviceGeometry.from_grid(grid, cp); state = cp.asarray(conservative, dtype=cp.float64)
    if state.shape != (*grid.shape, 4): raise ValueError(f"conservative must have shape {(*grid.shape, 4)}")
    normalized, absolute, reference = [], [], None
    for iteration in range(1, case.numerics.max_iterations + 1):
        state, residual_norm = advance_one_iteration_cupy(state, grid, case, geometry, _iteration_cfl(case, iteration), cp)
        if reference is None: reference = residual_norm
        value = residual_norm / reference if reference > 0.0 else 0.0; absolute.append(residual_norm); normalized.append(value)
        if value <= case.numerics.residual_tolerance: return _result(cp.asnumpy(state), normalized, absolute, reference, iteration, True)
    return _result(cp.asnumpy(state), normalized, absolute, reference or 0.0, case.numerics.max_iterations, False)


def _iteration_cfl(case: SolverCase, iteration: int) -> float:
    ramp = case.numerics.cfl_ramp_iterations
    return case.numerics.cfl if ramp == 0 else case.numerics.cfl_initial + min((iteration - 1) / ramp, 1.0) * (case.numerics.cfl - case.numerics.cfl_initial)


def _result(state, normalized, absolute, reference, iterations, converged):
    from .solver import SolverResult
    return SolverResult(state, np.asarray(normalized), np.asarray(absolute), reference, iterations, converged)
