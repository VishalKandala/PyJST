import unittest

import numpy as np

from pyjst import CartesianGrid
from pyjst.boundary import apply_boundary_conditions
from pyjst.config import BoundaryCondition, BoundaryKind, SolverCase, uniform_supersonic_case
from pyjst.state import uniform_conservative_state


class BoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = uniform_supersonic_case()
        self.grid = CartesianGrid(self.case.grid)

    def test_supersonic_inflow_and_outflow_fill_x_ghosts(self) -> None:
        state = uniform_conservative_state(self.grid.shape, self.case.gas, 0.9, 10.0, 0.0, 80000.0)
        sx, sy = self.grid.interior
        state[sx, sy, 0] = np.arange(self.case.grid.nx)[:, None] + 2.0
        apply_boundary_conditions(state, self.grid, self.case)
        rho_inf = self.case.freestream.density
        np.testing.assert_allclose(state[:2, 3, 0], rho_inf)
        np.testing.assert_allclose(
            state[sx.stop :, 3, :], np.repeat(state[sx.stop - 1 : sx.stop, 3, :], 2, axis=0)
        )

    def test_periodic_y_copies_opposite_physical_rows(self) -> None:
        state = uniform_conservative_state(self.grid.shape, self.case.gas, 1.0, 1.0, 0.0, 1.0)
        sx, sy = self.grid.interior
        state[sx, sy.start : sy.start + 2, 0] = 11.0
        state[sx, sy.stop - 2 : sy.stop, 0] = 22.0
        apply_boundary_conditions(state, self.grid, self.case)
        np.testing.assert_array_equal(state[sx, :2, 0], 22.0)
        np.testing.assert_array_equal(state[sx, sy.stop :, 0], 11.0)

    def test_rejects_unpaired_periodic_boundary(self) -> None:
        bad_boundaries = dict(self.case.boundaries)
        bad_boundaries["top"] = BoundaryCondition(BoundaryKind.SUPERSONIC_OUTFLOW)
        bad_case = SolverCase(
            self.case.name, self.case.gas, self.case.freestream, self.case.grid, bad_boundaries
        )
        state = uniform_conservative_state(self.grid.shape, self.case.gas, 1.0, 1.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            apply_boundary_conditions(state, self.grid, bad_case)


if __name__ == "__main__":
    unittest.main()
