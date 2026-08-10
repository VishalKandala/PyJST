"""Reproducible CPU baseline benchmarks for PyJST pseudo-time iterations."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np

from .cases import CompressionCornerCase
from .config import GridSpec, SolverCase, uniform_supersonic_case
from .grid import BodyFittedGrid, CartesianGrid
from .solver import advance_one_iteration, freestream_initial_state

if TYPE_CHECKING:
    from matplotlib.figure import Figure


CaseName = Literal["straight-channel", "compression-corner"]
Grid = CartesianGrid | BodyFittedGrid


@dataclass(frozen=True)
class PerformanceSample:
    """Median wall-clock cost of one four-stage pseudo-time iteration."""

    case: CaseName
    nx: int
    ny: int
    seconds_per_iteration: float
    cell_updates_per_second: float
    backend: str = "numpy"


def _build_case(case_name: CaseName, resolution: int) -> tuple[Grid, SolverCase]:
    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    if case_name == "straight-channel":
        case = uniform_supersonic_case()
        case = replace(
            case,
            grid=GridSpec(resolution, resolution, case.grid.x_min, case.grid.x_max, case.grid.y_min, case.grid.y_max),
        )
        return CartesianGrid(case.grid), case
    if case_name == "compression-corner":
        definition = CompressionCornerCase(nx=resolution, ny=resolution)
        return definition.grid(), definition.solver_case()
    raise ValueError(f"unsupported benchmark case {case_name!r}")


def benchmark_case(
    case_name: CaseName, resolution: int, *, repeats: int = 3, warmup_iterations: int = 1, backend: str = "numpy"
) -> PerformanceSample:
    """Measure a case at one square resolution without including setup time."""
    if repeats < 1 or warmup_iterations < 0:
        raise ValueError("repeats must be positive and warmup_iterations must be non-negative")
    grid, case = _build_case(case_name, resolution)
    initial_state = freestream_initial_state(grid, case)
    if backend == "cupy":
        from .cupy_solver import benchmark_cupy_iterations
        seconds = benchmark_cupy_iterations(initial_state, grid, case, repeats=repeats, warmup_iterations=warmup_iterations)
    elif backend == "numpy":
        for _ in range(warmup_iterations): advance_one_iteration(initial_state, grid, case)
        durations = []
        for _ in range(repeats):
            started = perf_counter(); advance_one_iteration(initial_state, grid, case); durations.append(perf_counter() - started)
        seconds = float(np.median(durations))
    else:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    cells = case.grid.nx * case.grid.ny
    return PerformanceSample(case_name, case.grid.nx, case.grid.ny, seconds, cells / seconds, backend)


def benchmark_suite(
    resolutions: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048), *, repeats: int = 2, warmup_iterations: int = 1, backend: str = "numpy"
) -> list[PerformanceSample]:
    """Benchmark both representative cases across a sequence of square grids."""
    return [
        benchmark_case(case_name, resolution, repeats=repeats, warmup_iterations=warmup_iterations, backend=backend)
        for case_name in ("straight-channel", "compression-corner")
        for resolution in resolutions
    ]


def write_benchmark_results(
    samples: list[PerformanceSample], output: str | Path, *, repeats: int | None = None, warmup_iterations: int | None = None
) -> Path:
    """Write measurements and runtime metadata to a JSON file."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "benchmark": "PyJST four-stage pseudo-time iteration (median wall-clock time)",
        "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "samples": [asdict(sample) for sample in samples],
    }
    if repeats is not None and warmup_iterations is not None:
        payload["measurement"] = {"repeats": repeats, "warmup_iterations": warmup_iterations}
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return destination.resolve()


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Performance plots require Matplotlib; install pyjst[viz].") from error
    return plt


def plot_benchmark_results(samples: list[PerformanceSample]) -> Figure:
    """Plot backend-aware iteration cost and throughput against cell count."""
    if not samples:
        raise ValueError("samples must not be empty")
    plt = _pyplot()
    figure, (time_axis, throughput_axis) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    labels = {"straight-channel": "Straight channel", "compression-corner": "Compression corner"}
    for (backend, case_name), color in zip(sorted({(sample.backend, sample.case) for sample in samples}), ("#4c72b0", "#c44e52", "#55a868", "#8172b2")):
        case_samples = sorted((sample for sample in samples if sample.case == case_name and sample.backend == backend), key=lambda item: item.nx * item.ny)
        if not case_samples:
            continue
        cells = [sample.nx * sample.ny for sample in case_samples]
        label = f"{labels[case_name]} ({backend.upper()})"
        time_axis.loglog(cells, [sample.seconds_per_iteration for sample in case_samples], "o-", color=color, label=label)
        throughput_axis.semilogx(cells, [sample.cell_updates_per_second / 1.0e6 for sample in case_samples], "o-", color=color, label=label)
    time_axis.set(xlabel="Physical cells", ylabel="Seconds per pseudo-time iteration", title="Iteration cost")
    throughput_axis.set(xlabel="Physical cells", ylabel="Million cell updates / s", title="Throughput")
    for axis in (time_axis, throughput_axis):
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
    return figure


def save_benchmark_figure(samples: list[PerformanceSample], output: str | Path, *, dpi: int = 300) -> Path:
    """Save a backend-aware benchmark plot as a PNG and return its path."""
    if dpi < 1:
        raise ValueError("dpi must be positive")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_benchmark_results(samples)
    figure.savefig(destination, dpi=dpi, bbox_inches="tight", facecolor="white")
    _pyplot().close(figure)
    return destination.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PyJST iteration scaling on a selected backend.")
    parser.add_argument("--resolutions", type=int, nargs="+", default=(64, 128, 256, 512, 1024, 2048))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    return parser


def main() -> None:
    """Run the baseline suite and save its machine-readable results and plot."""
    arguments = _parser().parse_args()
    samples = benchmark_suite(tuple(arguments.resolutions), repeats=arguments.repeats,
                              warmup_iterations=arguments.warmup_iterations, backend=arguments.backend)
    print(write_benchmark_results(
        samples, arguments.results, repeats=arguments.repeats, warmup_iterations=arguments.warmup_iterations
    ))
    print(save_benchmark_figure(samples, arguments.figure))


if __name__ == "__main__":
    main()
