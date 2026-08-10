"""Command-line entry point for PyJST demonstration cases."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .cases import CompressionCornerCase
from .config import GridSpec, JSTParameters
from .grid import CartesianGrid
from .solver import freestream_initial_state, solve
from .config import uniform_supersonic_case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a PyJST steady Euler demonstration case.")
    parser.add_argument("--case", choices=("uniform", "compression-corner"), default="uniform")
    parser.add_argument("--iterations", type=int, default=500, help="maximum pseudo-time iterations")
    parser.add_argument("--cfl", type=float, default=0.4, help="final CFL number")
    parser.add_argument("--cfl-initial", type=float, default=0.05, help="initial CFL number")
    parser.add_argument("--cfl-ramp", type=int, default=100, help="iterations over which CFL is ramped")
    parser.add_argument("--tolerance", type=float, default=1.0e-6, help="normalized residual tolerance")
    parser.add_argument("--nx", type=int, help="physical cells in the streamwise direction")
    parser.add_argument("--ny", type=int, help="physical cells in the cross-stream direction")
    parser.add_argument("--mach", type=float, help="upstream Mach number for the compression-corner case")
    parser.add_argument("--deflection", type=float, help="wedge deflection in degrees for the compression-corner case")
    return parser


def main() -> None:
    """Parse arguments, run the requested case, and print convergence data."""
    arguments = _parser().parse_args()
    numerics = JSTParameters(
        cfl=arguments.cfl,
        cfl_initial=arguments.cfl_initial,
        cfl_ramp_iterations=arguments.cfl_ramp,
        max_iterations=arguments.iterations,
        residual_tolerance=arguments.tolerance,
    )

    if arguments.case == "uniform":
        case = uniform_supersonic_case()
        if arguments.nx is not None or arguments.ny is not None:
            case = replace(
                case,
                grid=GridSpec(
                    nx=arguments.nx or case.grid.nx,
                    ny=arguments.ny or case.grid.ny,
                    x_min=case.grid.x_min,
                    x_max=case.grid.x_max,
                    y_min=case.grid.y_min,
                    y_max=case.grid.y_max,
                ),
            )
        case = replace(case, numerics=numerics)
        grid = CartesianGrid(case.grid)
    else:
        options: dict[str, float | int] = {}
        for name in ("nx", "ny", "mach"):
            value = getattr(arguments, name)
            if value is not None:
                options[name] = value
        if arguments.deflection is not None:
            options["deflection_degrees"] = arguments.deflection
        definition = CompressionCornerCase(**options)
        case = replace(definition.solver_case(), numerics=numerics)
        grid = definition.grid()

    result = solve(freestream_initial_state(grid, case), grid, case)
    print(f"case: {case.name}")
    print(f"grid: {case.grid.nx} x {case.grid.ny} physical cells")
    print(f"iterations: {result.iterations}")
    print(f"converged: {result.converged}")
    print(f"normalized residual: {result.residual_history[-1]:.6e}")
    print(f"absolute residual: {result.absolute_residual_history[-1]:.6e}")


if __name__ == "__main__":
    main()
