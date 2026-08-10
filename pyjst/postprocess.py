"""Post-processing helpers for inspecting and exporting PyJST solutions.

The plotting dependency is deliberately imported only when a figure is
requested, so running the solver does not require Matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .cases import CompressionCornerCase
from .config import GridSpec, JSTParameters, SolverCase, uniform_supersonic_case
from .grid import BodyFittedGrid, CartesianGrid
from .solver import SolverResult, freestream_initial_state, solve
from .state import conservative_to_primitive

if TYPE_CHECKING:
    from matplotlib.figure import Figure


Grid = CartesianGrid | BodyFittedGrid


def physical_primitive(conservative: np.ndarray, grid: Grid, case: SolverCase) -> np.ndarray:
    """Return ``[rho, u, v, p]`` in physical cells, excluding ghost cells."""
    expected_shape = (*grid.shape, 4)
    if conservative.shape != expected_shape:
        raise ValueError(f"conservative must have shape {expected_shape}; got {conservative.shape}")
    sx, sy = grid.interior
    return conservative_to_primitive(conservative, case.gas)[sx, sy, :]


def pressure_ratio(conservative: np.ndarray, grid: Grid, case: SolverCase) -> np.ndarray:
    """Return physical-cell static pressure normalized by freestream pressure."""
    return physical_primitive(conservative, grid, case)[..., 3] / case.freestream.pressure


def mach_number(conservative: np.ndarray, grid: Grid, case: SolverCase) -> np.ndarray:
    """Return the physical-cell Mach-number magnitude."""
    primitive = physical_primitive(conservative, grid, case)
    velocity = np.hypot(primitive[..., 1], primitive[..., 2])
    sound_speed = np.sqrt(case.gas.gamma * primitive[..., 3] / primitive[..., 0])
    return velocity / sound_speed


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Figure generation requires Matplotlib; install pyjst[viz].") from error
    return plt


def _field_mesh(grid: Grid) -> tuple[np.ndarray, np.ndarray, bool]:
    if isinstance(grid, BodyFittedGrid):
        return grid.vertices[..., 0], grid.vertices[..., 1], True
    x, y = grid.cell_centres()
    return x, y, False


def plot_solution(
    result: SolverResult, grid: Grid, case: SolverCase, *, title: str | None = None
) -> Figure:
    """Create a pressure-ratio and residual-history summary figure.

    The left panel uses physical cell locations (or body-fitted vertices) and
    the right panel records the normalized residual from the solver result.
    """
    plt = _pyplot()
    x, y, has_vertices = _field_mesh(grid)
    values = pressure_ratio(result.conservative, grid, case)
    figure, (field_axis, residual_axis) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    if has_vertices:
        image = field_axis.pcolormesh(x.T, y.T, values.T, shading="flat", cmap="viridis")
        field_axis.plot(grid.vertices[:, 0, 0], grid.vertices[:, 0, 1], color="white", linewidth=1.2)
    else:
        image = field_axis.pcolormesh(x.T, y.T, values.T, shading="nearest", cmap="viridis")

    figure.colorbar(image, ax=field_axis, label=r"$p / p_\infty$")
    field_axis.set(xlabel="x", ylabel="y", title="Static pressure ratio")
    field_axis.set_aspect("equal")

    iterations = np.arange(1, result.residual_history.size + 1)
    if np.all(result.residual_history == 0.0):
        residual_axis.text(0.5, 0.5, "Exact zero residual", ha="center", va="center",
                           transform=residual_axis.transAxes, fontsize=13)
        residual_axis.set_xlim(0.5, max(1.5, result.residual_history.size + 0.5))
        residual_axis.set_ylim(0.0, 1.0)
    else:
        residual_axis.semilogy(iterations, result.residual_history, color="#c44e52")
    residual_axis.set(xlabel="Pseudo-time iteration", ylabel="Normalized residual", title="Convergence history")
    residual_axis.grid(True, which="both", alpha=0.3)
    if title:
        figure.suptitle(title)
    return figure


def save_solution_figure(
    result: SolverResult, grid: Grid, case: SolverCase, output: str | Path, *, title: str | None = None, dpi: int = 300
) -> Path:
    """Save :func:`plot_solution` as a high-resolution PNG and return its path."""
    if dpi < 1:
        raise ValueError("dpi must be positive")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_solution(result, grid, case, title=title)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    _pyplot().close(figure)
    return destination.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PyJST case and save a solution-summary figure.")
    parser.add_argument("--case", choices=("straight-channel", "compression-corner"), default="straight-channel")
    parser.add_argument("--output", type=Path, required=True, help="PNG output path")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--nx", type=int)
    parser.add_argument("--ny", type=int)
    return parser


def main() -> None:
    """Run a representative case and write its pressure/convergence figure."""
    arguments = _parser().parse_args()
    numerics = JSTParameters(max_iterations=arguments.iterations, cfl=0.4, cfl_initial=0.05, cfl_ramp_iterations=100)
    if arguments.case == "straight-channel":
        case = uniform_supersonic_case()
        if arguments.nx is not None or arguments.ny is not None:
            case = replace(case, grid=GridSpec(arguments.nx or case.grid.nx, arguments.ny or case.grid.ny,
                                                case.grid.x_min, case.grid.x_max, case.grid.y_min, case.grid.y_max))
        case = replace(case, numerics=numerics)
        grid: Grid = CartesianGrid(case.grid)
        title = "Straight channel: uniform Mach-2 flow"
    else:
        definition = CompressionCornerCase(nx=arguments.nx or 160, ny=arguments.ny or 80)
        case = replace(definition.solver_case(), numerics=numerics)
        grid = definition.grid()
        title = "Compression corner: Mach 2, 10° deflection"
    result = solve(freestream_initial_state(grid, case), grid, case)
    print(save_solution_figure(result, grid, case, arguments.output, title=title))


if __name__ == "__main__":
    main()
