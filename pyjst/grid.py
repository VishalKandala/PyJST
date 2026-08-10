"""Structured Cartesian finite-volume geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import GridSpec


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform cell-centred grid, including the requested ghost-cell layers.

    Array axis 0 is the streamwise/x direction and axis 1 is the y direction.
    Cell states consequently use the shape ``(nx_total, ny_total, 4)``.
    """

    spec: GridSpec

    @property
    def shape(self) -> tuple[int, int]:
        """Total cell shape including ghost cells."""
        return self.spec.cell_shape

    @property
    def interior(self) -> tuple[slice, slice]:
        """Slices selecting physical (non-ghost) cells."""
        g = self.spec.ghost_cells
        return slice(g, g + self.spec.nx), slice(g, g + self.spec.ny)

    @property
    def cell_volume(self) -> float:
        """Area of one physical cell in this two-dimensional discretization."""
        return self.spec.dx * self.spec.dy

    @property
    def cell_volumes(self) -> np.ndarray:
        """Physical-cell areas with shape ``(nx, ny)``."""
        return np.full((self.spec.nx, self.spec.ny), self.cell_volume)

    @property
    def x_face_vectors(self) -> np.ndarray:
        """Positive-x face vectors, shape ``(nx + 1, ny, 2)``."""
        faces = np.zeros((self.spec.nx + 1, self.spec.ny, 2))
        faces[..., 0] = self.spec.dy
        return faces

    @property
    def y_face_vectors(self) -> np.ndarray:
        """Positive-y face vectors, shape ``(nx, ny + 1, 2)``."""
        faces = np.zeros((self.spec.nx, self.spec.ny + 1, 2))
        faces[..., 1] = self.spec.dx
        return faces

    def cell_centres(self, include_ghosts: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return two arrays of x/y cell-centre coordinates.

        Ghost centre coordinates are extrapolated uniformly outside the physical
        domain when ``include_ghosts`` is true.
        """
        g = self.spec.ghost_cells if include_ghosts else 0
        x = self.spec.x_min + (np.arange(-g, self.spec.nx + g) + 0.5) * self.spec.dx
        y = self.spec.y_min + (np.arange(-g, self.spec.ny + g) + 0.5) * self.spec.dy
        return np.meshgrid(x, y, indexing="ij")


@dataclass(frozen=True)
class BodyFittedGrid:
    """Cell-centred structured grid derived from physical mesh vertices.

    ``vertices`` has shape ``(nx+1, ny+1, 2)`` and contains physical-domain
    nodes only.  State arrays still carry two ghost-cell layers; geometric
    face vectors are only required on the physical domain boundary and its
    interior faces.
    """

    vertices: np.ndarray
    ghost_cells: int = 2

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        if vertices.ndim != 3 or vertices.shape[-1] != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 3:
            raise ValueError("vertices must have shape (nx+1, ny+1, 2) with nx and ny at least 2")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("vertices must be finite")
        if self.ghost_cells < 2:
            raise ValueError("JST fourth-order dissipation requires at least 2 ghost cells")
        object.__setattr__(self, "vertices", vertices)
        if np.any(self.cell_volumes <= 0.0):
            raise ValueError("all body-fitted cells must have positive area")

    @property
    def nx(self) -> int:
        return self.vertices.shape[0] - 1

    @property
    def ny(self) -> int:
        return self.vertices.shape[1] - 1

    @property
    def shape(self) -> tuple[int, int]:
        return self.nx + 2 * self.ghost_cells, self.ny + 2 * self.ghost_cells

    @property
    def interior(self) -> tuple[slice, slice]:
        g = self.ghost_cells
        return slice(g, g + self.nx), slice(g, g + self.ny)

    @property
    def cell_volumes(self) -> np.ndarray:
        lower_left = self.vertices[:-1, :-1, :]
        lower_right = self.vertices[1:, :-1, :]
        upper_right = self.vertices[1:, 1:, :]
        upper_left = self.vertices[:-1, 1:, :]
        twice_area = (
            lower_left[..., 0] * lower_right[..., 1] - lower_left[..., 1] * lower_right[..., 0]
            + lower_right[..., 0] * upper_right[..., 1] - lower_right[..., 1] * upper_right[..., 0]
            + upper_right[..., 0] * upper_left[..., 1] - upper_right[..., 1] * upper_left[..., 0]
            + upper_left[..., 0] * lower_left[..., 1] - upper_left[..., 1] * lower_left[..., 0]
        )
        return 0.5 * twice_area

    @property
    def cell_volume(self) -> float:
        """Compatibility scalar; body-fitted operations use ``cell_volumes``."""
        return float(np.mean(self.cell_volumes))

    @property
    def x_face_vectors(self) -> np.ndarray:
        """Positive computational-x face vectors, shape ``(nx+1, ny, 2)``."""
        edge = self.vertices[:, 1:, :] - self.vertices[:, :-1, :]
        return np.stack((edge[..., 1], -edge[..., 0]), axis=-1)

    @property
    def y_face_vectors(self) -> np.ndarray:
        """Positive computational-y face vectors, shape ``(nx, ny+1, 2)``."""
        edge = self.vertices[1:, :, :] - self.vertices[:-1, :, :]
        return np.stack((-edge[..., 1], edge[..., 0]), axis=-1)
