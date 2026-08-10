"""Conservative-state utilities for the two-dimensional Euler equations."""

from __future__ import annotations

import numpy as np

from .config import GasModel


N_VARIABLES = 4


def _require_state_shape(state: np.ndarray, name: str) -> None:
    if state.ndim < 1 or state.shape[-1] != N_VARIABLES:
        raise ValueError(f"{name} must have final dimension {N_VARIABLES}; got {state.shape}")


def primitive_to_conservative(primitive: np.ndarray, gas: GasModel) -> np.ndarray:
    """Convert ``[..., rho, u, v, p]`` to ``[..., rho, rho*u, rho*v, rho*E]``.

    The returned array is a new floating-point array.  Pressure and density
    are checked because an Euler state with either non-positive is invalid.
    """
    primitive = np.asarray(primitive, dtype=float)
    _require_state_shape(primitive, "primitive")
    rho, u, v, pressure = np.moveaxis(primitive, -1, 0)
    if not np.all(np.isfinite(primitive)):
        raise ValueError("primitive state must be finite")
    if np.any(rho <= 0.0) or np.any(pressure <= 0.0):
        raise ValueError("primitive density and pressure must be positive")

    conservative = np.empty_like(primitive)
    conservative[..., 0] = rho
    conservative[..., 1] = rho * u
    conservative[..., 2] = rho * v
    conservative[..., 3] = pressure / (gas.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
    return conservative


def conservative_to_primitive(conservative: np.ndarray, gas: GasModel) -> np.ndarray:
    """Convert ``[..., rho, rho*u, rho*v, rho*E]`` to ``[..., rho, u, v, p]``."""
    conservative = np.asarray(conservative, dtype=float)
    _require_state_shape(conservative, "conservative")
    rho, rho_u, rho_v, rho_energy = np.moveaxis(conservative, -1, 0)
    if not np.all(np.isfinite(conservative)):
        raise ValueError("conservative state must be finite")
    if np.any(rho <= 0.0):
        raise ValueError("conservative density must be positive")

    u = rho_u / rho
    v = rho_v / rho
    pressure = (gas.gamma - 1.0) * (rho_energy - 0.5 * rho * (u * u + v * v))
    if np.any(pressure <= 0.0):
        raise ValueError("conservative state yields non-positive pressure")

    primitive = np.empty_like(conservative)
    primitive[..., 0] = rho
    primitive[..., 1] = u
    primitive[..., 2] = v
    primitive[..., 3] = pressure
    return primitive


def uniform_conservative_state(
    shape: tuple[int, int], gas: GasModel, rho: float, u: float, v: float, pressure: float
) -> np.ndarray:
    """Create a state array filled with one uniform physical state."""
    primitive = np.empty((*shape, N_VARIABLES), dtype=float)
    primitive[..., 0] = rho
    primitive[..., 1] = u
    primitive[..., 2] = v
    primitive[..., 3] = pressure
    return primitive_to_conservative(primitive, gas)
