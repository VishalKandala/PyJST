"""Ghost-cell boundary conditions for the first structured-grid cases."""

from __future__ import annotations

import numpy as np

from .config import BoundaryKind, GasModel, SolverCase
from .grid import BodyFittedGrid, CartesianGrid
from .state import conservative_to_primitive, primitive_to_conservative, uniform_conservative_state


def _freestream_state(case: SolverCase) -> np.ndarray:
    rho, u, v, pressure = case.freestream.primitive(case.gas)
    return uniform_conservative_state((1, 1), case.gas, rho, u, v, pressure)[0, 0]


def _require_periodic_pair(case: SolverCase, first: str, second: str) -> None:
    first_periodic = case.boundaries[first].kind is BoundaryKind.PERIODIC
    second_periodic = case.boundaries[second].kind is BoundaryKind.PERIODIC
    if first_periodic != second_periodic:
        raise ValueError(f"{first} and {second} must both be periodic or both be non-periodic")


def _fill_side(
    conservative: np.ndarray,
    side: str,
    kind: BoundaryKind,
    grid: CartesianGrid | BodyFittedGrid,
    freestream: np.ndarray,
    gas: GasModel,
) -> None:
    """Fill one non-periodic ghost side in place."""
    sx, sy = grid.interior
    g = sx.start
    if kind is BoundaryKind.SUPERSONIC_INFLOW:
        if side == "left":
            conservative[:g, :, :] = freestream
        elif side == "right":
            conservative[sx.stop :, :, :] = freestream
        elif side == "bottom":
            conservative[:, :g, :] = freestream
        else:
            conservative[:, sy.stop :, :] = freestream
    elif kind is BoundaryKind.SUPERSONIC_OUTFLOW:
        if side == "left":
            conservative[:g, :, :] = conservative[g : g + 1, :, :]
        elif side == "right":
            conservative[sx.stop :, :, :] = conservative[sx.stop - 1 : sx.stop, :, :]
        elif side == "bottom":
            conservative[:, :g, :] = conservative[:, g : g + 1, :]
        else:
            conservative[:, sy.stop :, :] = conservative[:, sy.stop - 1 : sy.stop, :]
    elif kind is BoundaryKind.SLIP_WALL:
        sx, sy = grid.interior
        if side not in ("bottom", "top"):
            raise NotImplementedError("slip walls are currently implemented on structured y boundaries only")
        face_normal = grid.y_face_vectors[:, 0 if side == "bottom" else -1, :]
        face_normal = face_normal / np.linalg.norm(face_normal, axis=1)[:, None]
        source_y = slice(sy.start, sy.start + 1) if side == "bottom" else slice(sy.stop - 1, sy.stop)
        primitive = conservative_to_primitive(conservative[sx, source_y, :], gas)
        normal_velocity = primitive[..., 1] * face_normal[:, None, 0] + primitive[..., 2] * face_normal[:, None, 1]
        primitive[..., 1] -= 2.0 * normal_velocity * face_normal[:, None, 0]
        primitive[..., 2] -= 2.0 * normal_velocity * face_normal[:, None, 1]
        reflected = primitive_to_conservative(primitive, gas)
        if side == "bottom":
            conservative[sx, :g, :] = reflected
        else:
            conservative[sx, sy.stop :, :] = reflected
    else:
        raise NotImplementedError(f"boundary kind {kind.value!r} is not implemented yet")


def apply_boundary_conditions(conservative: np.ndarray, grid: CartesianGrid | BodyFittedGrid, case: SolverCase) -> None:
    """Fill all ghost cells of ``conservative`` in place.

    The current baseline supports periodic boundaries and strictly supersonic
    flow through a domain.  A supersonic inflow prescribes all conservative
    variables from the case freestream; a supersonic outflow extrapolates the
    adjacent physical cell. A structured lower or upper slip wall reflects
    normal velocity while retaining density, pressure, and tangential velocity.
    Subsonic characteristic conditions remain deferred.
    """
    expected_shape = (*grid.shape, 4)
    if conservative.shape != expected_shape:
        raise ValueError(f"conservative must have shape {expected_shape}; got {conservative.shape}")
    sx, sy = grid.interior
    if (sx.stop - sx.start, sy.stop - sy.start) != (case.grid.nx, case.grid.ny):
        raise ValueError("grid physical dimensions must match case.grid")

    _require_periodic_pair(case, "left", "right")
    _require_periodic_pair(case, "bottom", "top")
    g = sx.start
    freestream = _freestream_state(case)

    # x sides are filled first. Filling y sides afterwards makes their corners
    # consistent with y-periodicity or their corresponding y-side condition.
    if case.boundaries["left"].kind is BoundaryKind.PERIODIC:
        conservative[:g, :, :] = conservative[sx.stop - g : sx.stop, :, :]
        conservative[sx.stop :, :, :] = conservative[sx.start : sx.start + g, :, :]
    else:
        _fill_side(conservative, "left", case.boundaries["left"].kind, grid, freestream, case.gas)
        _fill_side(conservative, "right", case.boundaries["right"].kind, grid, freestream, case.gas)

    if case.boundaries["bottom"].kind is BoundaryKind.PERIODIC:
        conservative[:, :g, :] = conservative[:, sy.stop - g : sy.stop, :]
        conservative[:, sy.stop :, :] = conservative[:, sy.start : sy.start + g, :]
    else:
        _fill_side(conservative, "bottom", case.boundaries["bottom"].kind, grid, freestream, case.gas)
        _fill_side(conservative, "top", case.boundaries["top"].kind, grid, freestream, case.gas)
