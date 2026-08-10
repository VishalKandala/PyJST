import unittest

import numpy as np

from pyjst import CartesianGrid, GasModel, GridSpec
from pyjst.flux import central_residual, euler_flux
from pyjst.state import uniform_conservative_state


class FluxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gas = GasModel()

    def test_euler_flux_has_pressure_work_in_energy_component(self) -> None:
        rho, u, v, pressure = 2.0, 3.0, -4.0, 5.0
        energy = pressure / (self.gas.gamma - 1.0) + 0.5 * rho * (u * u + v * v)
        state = np.array([rho, rho * u, rho * v, energy])
        flux_x, flux_y = euler_flux(state, self.gas)
        self.assertAlmostEqual(flux_x[0], rho * u)
        self.assertAlmostEqual(flux_x[1], rho * u * u + pressure)
        self.assertAlmostEqual(flux_x[2], rho * u * v)
        self.assertAlmostEqual(flux_x[3], (energy + pressure) * u)
        self.assertAlmostEqual(flux_y[3], (energy + pressure) * v)

    def test_uniform_state_has_zero_interior_residual(self) -> None:
        grid = CartesianGrid(GridSpec(6, 5, 0.0, 3.0, 0.0, 2.0))
        state = uniform_conservative_state(grid.shape, self.gas, 1.225, 300.0, 10.0, 101325.0)
        residual = central_residual(state, grid, self.gas)
        sx, sy = grid.interior
        np.testing.assert_array_equal(residual[sx, sy], 0.0)

    def test_residual_is_integrated_outward_flux(self) -> None:
        grid = CartesianGrid(GridSpec(2, 2, 0.0, 2.0, 0.0, 2.0))
        state = uniform_conservative_state(grid.shape, self.gas, 1.0, 0.0, 0.0, 1.0)
        # Change the eastern neighbor of the first interior cell only.
        state[3, 2, 3] += 2.5
        residual = central_residual(state, grid, self.gas)
        # The x-momentum component changes by the central east-face pressure flux.
        self.assertAlmostEqual(residual[2, 2, 1], 0.5 * 1.0 * grid.spec.dy)


if __name__ == "__main__":
    unittest.main()
