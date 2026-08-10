"""Physical Euler fluxes and a central Cartesian finite-volume residual."""

from __future__ import annotations

import numpy as np

from .config import GasModel
from .grid import BodyFittedGrid, CartesianGrid
from .state import conservative_to_primitive


def euler_flux(conservative: np.ndarray, gas: GasModel) -> tuple[np.ndarray, np.ndarray]:
    """Return physical Euler fluxes ``(F, G)`` in the x and y directions.

    ``conservative`` has components ``[rho, rho*u, rho*v, rho*E]``.  The
    returned arrays have the same shape and contain the conservative fluxes
    through faces with positive x and y normals respectively.
    """
    primitive = conservative_to_primitive(conservative, gas)
    rho, u, v, pressure = np.moveaxis(primitive, -1, 0)
    rho_energy = conservative[..., 3]

    flux_x = np.empty_like(conservative)
    flux_x[..., 0] = rho * u
    flux_x[..., 1] = rho * u * u + pressure
    flux_x[..., 2] = rho * u * v
    flux_x[..., 3] = (rho_energy + pressure) * u

    flux_y = np.empty_like(conservative)
    flux_y[..., 0] = rho * v
    flux_y[..., 1] = rho * u * v
    flux_y[..., 2] = rho * v * v + pressure
    flux_y[..., 3] = (rho_energy + pressure) * v
    return flux_x, flux_y


def central_residual(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, gas: GasModel) -> np.ndarray:
    """Return the integrated central inviscid residual for physical cells.

    The residual is the outward face-flux integral. Grid face vectors carry
    both orientation and length, so the same operator works for Cartesian and
    body-fitted structured meshes.

    Ghost-cell residuals are set to zero. Boundary conditions must fill the
    conservative ghost cells before this operator is called.
    """
    expected_shape = (*grid.shape, 4)
    if conservative.shape != expected_shape:
        raise ValueError(f"conservative must have shape {expected_shape}; got {conservative.shape}")

    flux_x, flux_y = euler_flux(conservative, gas)
    # Face index i is the face between physical cells i-1 and i, except at
    # index 0/nx where one side is a ghost cell.
    face_x_f = 0.5 * (flux_x[:-1, :, :] + flux_x[1:, :, :])
    face_x_g = 0.5 * (flux_y[:-1, :, :] + flux_y[1:, :, :])
    face_y_f = 0.5 * (flux_x[:, :-1, :] + flux_x[:, 1:, :])
    face_y_g = 0.5 * (flux_y[:, :-1, :] + flux_y[:, 1:, :])

    residual = np.zeros_like(conservative)
    sx, sy = grid.interior
    x_start, x_stop = sx.start, sx.stop
    y_start, y_stop = sy.start, sy.stop
    x_flux_f = face_x_f[x_start - 1 : x_stop, y_start:y_stop, :]
    x_flux_g = face_x_g[x_start - 1 : x_stop, y_start:y_stop, :]
    y_flux_f = face_y_f[x_start:x_stop, y_start - 1 : y_stop, :]
    y_flux_g = face_y_g[x_start:x_stop, y_start - 1 : y_stop, :]
    x_normal = grid.x_face_vectors
    y_normal = grid.y_face_vectors
    x_integrated = x_flux_f * x_normal[..., 0, None] + x_flux_g * x_normal[..., 1, None]
    y_integrated = y_flux_f * y_normal[..., 0, None] + y_flux_g * y_normal[..., 1, None]
    residual[sx, sy, :] = x_integrated[1:, :, :] - x_integrated[:-1, :, :] + y_integrated[:, 1:, :] - y_integrated[:, :-1, :]
    return residual
