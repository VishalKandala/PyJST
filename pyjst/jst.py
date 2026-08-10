"""Jameson--Schmidt--Turkel artificial dissipation on Cartesian grids."""

from __future__ import annotations

import numpy as np

from .config import GasModel, JSTParameters
from .grid import BodyFittedGrid, CartesianGrid
from .state import conservative_to_primitive


def _require_grid_state(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid) -> None:
    expected_shape = (*grid.shape, 4)
    if conservative.shape != expected_shape:
        raise ValueError(f"conservative must have shape {expected_shape}; got {conservative.shape}")


def pressure_sensors(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, gas: GasModel) -> tuple[np.ndarray, np.ndarray]:
    """Return JST pressure sensors at cell centres in the x and y directions.

    The sensor is ``abs(p[i+1] - 2*p[i] + p[i-1]) / (p[i+1] + 2*p[i] + p[i-1])``.
    It is zero in the outermost ghost row because that row cannot support the
    centred stencil.  Two ghost layers make every face of a physical cell
    valid for the subsequent dissipation operator.
    """
    _require_grid_state(conservative, grid)
    pressure = conservative_to_primitive(conservative, gas)[..., 3]
    sensor_x = np.zeros_like(pressure)
    sensor_y = np.zeros_like(pressure)

    sensor_x[1:-1, :] = np.abs(pressure[2:, :] - 2.0 * pressure[1:-1, :] + pressure[:-2, :]) / (
        pressure[2:, :] + 2.0 * pressure[1:-1, :] + pressure[:-2, :]
    )
    sensor_y[:, 1:-1] = np.abs(pressure[:, 2:] - 2.0 * pressure[:, 1:-1] + pressure[:, :-2]) / (
        pressure[:, 2:] + 2.0 * pressure[:, 1:-1] + pressure[:, :-2]
    )
    return sensor_x, sensor_y


def jst_dissipation(
    conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, gas: GasModel, parameters: JSTParameters
) -> np.ndarray:
    """Return the integrated JST dissipation contribution for physical cells.

    For the face between cells ``i`` and ``i+1`` this implementation uses

    ``d = lambda * (eps2 * (Q[i+1]-Q[i]) - eps4 * d3Q)``,

    where ``d3Q = Q[i+2]-3Q[i+1]+3Q[i]-Q[i-1]``.  ``lambda`` is the
    arithmetic-average Euler spectral radius at the face.  The returned value
    is the conservative divergence ``d_e - d_w + d_n - d_s`` multiplied by
    face length.  It must be *subtracted* from the central convective residual
    in the pseudo-time update.
    """
    _require_grid_state(conservative, grid)
    primitive = conservative_to_primitive(conservative, gas)
    rho, u, v, pressure = np.moveaxis(primitive, -1, 0)
    sound = np.sqrt(gas.gamma * pressure / rho)
    sensor_x, sensor_y = pressure_sensors(conservative, grid, gas)

    # Face i lies between cells i and i+1.  The end faces remain zero because
    # their fourth-order stencils are unavailable; physical-cell faces do not
    # use them when GridSpec has two ghost layers.
    u_x = 0.5 * (u[:-1, :] + u[1:, :])
    v_x = 0.5 * (v[:-1, :] + v[1:, :])
    a_x = 0.5 * (sound[:-1, :] + sound[1:, :])
    u_y = 0.5 * (u[:, :-1] + u[:, 1:])
    v_y = 0.5 * (v[:, :-1] + v[:, 1:])
    a_y = 0.5 * (sound[:, :-1] + sound[:, 1:])
    normal_x = np.zeros((*u_x.shape, 2))
    normal_y = np.zeros((*u_y.shape, 2))
    sx, sy = grid.interior
    normal_x[sx.start - 1 : sx.stop, sy, :] = grid.x_face_vectors
    normal_y[sx, sy.start - 1 : sy.stop, :] = grid.y_face_vectors
    lambda_x = np.abs(u_x * normal_x[..., 0] + v_x * normal_x[..., 1]) + a_x * np.linalg.norm(normal_x, axis=-1)
    lambda_y = np.abs(u_y * normal_y[..., 0] + v_y * normal_y[..., 1]) + a_y * np.linalg.norm(normal_y, axis=-1)
    eps2_x = parameters.kappa2 * np.maximum(sensor_x[:-1, :], sensor_x[1:, :])
    eps2_y = parameters.kappa2 * np.maximum(sensor_y[:, :-1], sensor_y[:, 1:])
    eps4_x = np.maximum(0.0, parameters.kappa4 - eps2_x)
    eps4_y = np.maximum(0.0, parameters.kappa4 - eps2_y)

    face_diss_x = np.zeros((conservative.shape[0] - 1, conservative.shape[1], 4), dtype=float)
    face_diss_y = np.zeros((conservative.shape[0], conservative.shape[1] - 1, 4), dtype=float)

    delta_x = conservative[2:-1, :, :] - conservative[1:-2, :, :]
    third_x = (
        (conservative[3:, :, :] - conservative[2:-1, :, :])
        - 2.0 * (conservative[2:-1, :, :] - conservative[1:-2, :, :])
        + (conservative[1:-2, :, :] - conservative[:-3, :, :])
    )
    face_diss_x[1:-1, :, :] = lambda_x[1:-1, :, None] * (
        eps2_x[1:-1, :, None] * delta_x - eps4_x[1:-1, :, None] * third_x
    )

    delta_y = conservative[:, 2:-1, :] - conservative[:, 1:-2, :]
    third_y = (
        (conservative[:, 3:, :] - conservative[:, 2:-1, :])
        - 2.0 * (conservative[:, 2:-1, :] - conservative[:, 1:-2, :])
        + (conservative[:, 1:-2, :] - conservative[:, :-3, :])
    )
    face_diss_y[:, 1:-1, :] = lambda_y[:, 1:-1, None] * (
        eps2_y[:, 1:-1, None] * delta_y - eps4_y[:, 1:-1, None] * third_y
    )

    # A physical boundary does not provide a full centred stencil. The
    # boundary condition supplies its physical flux, so JST dissipation is
    # applied only on interior faces.
    face_diss_x[sx.start - 1, sy, :] = 0.0
    face_diss_x[sx.stop - 1, sy, :] = 0.0
    face_diss_y[sx, sy.start - 1, :] = 0.0
    face_diss_y[sx, sy.stop - 1, :] = 0.0

    dissipation = np.zeros_like(conservative)
    dissipation[sx, sy, :] = (
        face_diss_x[sx, sy, :] - face_diss_x[slice(sx.start - 1, sx.stop - 1), sy, :]
        + face_diss_y[sx, sy, :] - face_diss_y[sx, slice(sy.start - 1, sy.stop - 1), :]
    )
    return dissipation
